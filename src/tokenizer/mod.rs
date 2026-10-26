//! A BPE tokenizer that reads Hugging Face `tokenizer.json` files and
//! produces the same token ids as the `tokenizers` library.
//!
//! Supported: byte-level BPE (GPT-2, SmolLM, Llama 3, Qwen) with the GPT-2,
//! Llama 3 and Qwen pre-tokenizer patterns and digit splitting, NFC
//! normalization, and SentencePiece-style BPE (Llama 2, TinyLlama) with
//! `▁` spaces and byte fallback. Added/special tokens are matched in the
//! input first, as the reference library does.

pub mod pretok;
pub mod unicode;
mod unicode_tables;

use crate::json::Json;
use pretok::Pattern;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub special: bool,
}

#[derive(Clone, Debug)]
enum NormStep {
    Nfc,
    Prepend(String),
    Replace(String, String),
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// (left, right) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    added: Vec<AddedToken>,
    added_by_first: HashMap<char, Vec<usize>>,
    norm: Vec<NormStep>,
    /// transformers' non-legacy Llama mode: the Prepend step only applies
    /// to text at the very start of the input, not after special tokens.
    prepend_first_only: bool,
    split_digits: bool,
    pattern: Option<Pattern>,
    byte_level: bool,
    byte_fallback: Option<[u32; 256]>,
    unk: Option<u32>,
    fuse_unk: bool,
    ignore_merges: bool,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    strip_leading_space: bool,
    token_bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
}

/// GPT-2's reversible mapping from bytes to printable characters.
fn byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            extra += 1;
            char::from_u32(255 + extra).unwrap()
        };
    }
    table
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

pub enum Piece<'a> {
    Added(u32),
    Text(&'a str),
}

impl Tokenizer {
    /// Load a model directory's tokenizer the way transformers'
    /// `AutoTokenizer` does: `tokenizer.json` plus the `legacy` flag from
    /// `tokenizer_config.json`.
    pub fn load_dir(dir: &Path) -> Result<Tokenizer, String> {
        let mut tk = Tokenizer::load(&dir.join("tokenizer.json"))?;
        if let Ok(text) = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            && let Ok(cfg) = Json::parse(&text)
            && cfg.get("legacy").and_then(Json::as_bool) == Some(false)
        {
            tk.prepend_first_only = tk.norm.iter().any(|n| matches!(n, NormStep::Prepend(_)));
        }
        Ok(tk)
    }

    pub fn load(path: &Path) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j = Json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Tokenizer::from_json(&j).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn from_json(j: &Json) -> Result<Tokenizer, String> {
        let model = j.get("model").ok_or("no model section")?;
        if model.get("type").and_then(Json::as_str) != Some("BPE") {
            return err("only BPE tokenizers are supported");
        }
        let mut vocab = HashMap::new();
        for (tok, id) in model.get("vocab").and_then(Json::as_object).ok_or("no vocab")? {
            vocab.insert(tok.clone(), id.as_i64().ok_or("bad vocab id")? as u32);
        }

        let mut merges = HashMap::new();
        for (rank, m) in model.get("merges").and_then(Json::as_array).ok_or("no merges")?.iter().enumerate() {
            // Either "a b" or ["a", "b"].
            let (a, b): (Cow<str>, Cow<str>) = match m {
                Json::String(s) => {
                    let (a, b) = s.split_once(' ').ok_or("bad merge")?;
                    (a.into(), b.into())
                }
                Json::Array(pair) if pair.len() == 2 => {
                    (pair[0].as_str().ok_or("bad merge")?.into(), pair[1].as_str().ok_or("bad merge")?.into())
                }
                _ => return err("bad merge"),
            };
            let (Some(&ia), Some(&ib)) = (vocab.get(a.as_ref()), vocab.get(b.as_ref())) else { continue };
            let Some(&merged) = vocab.get(&format!("{a}{b}")) else { continue };
            merges.entry((ia, ib)).or_insert((rank as u32, merged));
        }

        let mut added = Vec::new();
        for t in j.get("added_tokens").and_then(Json::as_array).unwrap_or(&[]) {
            added.push(AddedToken {
                content: t.get("content").and_then(Json::as_str).ok_or("bad added token")?.to_string(),
                id: t.get("id").and_then(Json::as_i64).ok_or("bad added token")? as u32,
                special: t.get("special").and_then(Json::as_bool).unwrap_or(false),
            });
        }
        let mut added_by_first: HashMap<char, Vec<usize>> = HashMap::new();
        for (i, t) in added.iter().enumerate() {
            if let Some(c) = t.content.chars().next() {
                added_by_first.entry(c).or_default().push(i);
            }
        }
        for v in added_by_first.values_mut() {
            v.sort_by_key(|&i| Reverse(added[i].content.len()));
        }

        let mut norm = Vec::new();
        if let Some(n) = j.get("normalizer").filter(|n| !n.is_null()) {
            parse_normalizer(n, &mut norm)?;
        }

        let mut tk = Tokenizer {
            vocab,
            merges,
            added,
            added_by_first,
            norm,
            prepend_first_only: false,
            split_digits: false,
            pattern: None,
            byte_level: false,
            byte_fallback: None,
            unk: None,
            fuse_unk: model.get("fuse_unk").and_then(Json::as_bool).unwrap_or(false),
            ignore_merges: model.get("ignore_merges").and_then(Json::as_bool).unwrap_or(false),
            prefix: vec![],
            suffix: vec![],
            strip_leading_space: false,
            token_bytes: vec![],
            special: vec![],
            cache: Mutex::new(HashMap::new()),
        };
        if let Some(p) = j.get("pre_tokenizer").filter(|p| !p.is_null()) {
            tk.parse_pretokenizer(p)?;
        }
        if let Some(unk) = model.get("unk_token").and_then(Json::as_str) {
            tk.unk = tk.vocab.get(unk).copied();
        }
        if model.get("byte_fallback").and_then(Json::as_bool) == Some(true) {
            let mut ids = [0u32; 256];
            for (b, id) in ids.iter_mut().enumerate() {
                *id = *tk.vocab.get(&format!("<0x{b:02X}>")).ok_or("byte_fallback without <0xNN> tokens")?;
            }
            tk.byte_fallback = Some(ids);
        }
        if let Some(pp) = j.get("post_processor").filter(|p| !p.is_null()) {
            tk.parse_post_processor(pp)?;
        }
        if let Some(d) = j.get("decoder") {
            tk.strip_leading_space = decoder_strips_space(d);
        }
        tk.build_decode_table();
        Ok(tk)
    }

    fn parse_pretokenizer(&mut self, p: &Json) -> Result<(), String> {
        match p.get("type").and_then(Json::as_str) {
            Some("Sequence") => {
                for sub in p.get("pretokenizers").and_then(Json::as_array).unwrap_or(&[]) {
                    self.parse_pretokenizer(sub)?;
                }
            }
            Some("Digits") => {
                if p.get("individual_digits").and_then(Json::as_bool) != Some(true) {
                    return err("only Digits(individual_digits=true) is supported");
                }
                self.split_digits = true;
            }
            Some("Split") => {
                let re = p.path(&["pattern", "Regex"]).and_then(Json::as_str).ok_or("Split without a regex")?;
                self.pattern =
                    Some(Pattern::from_regex(re).ok_or_else(|| format!("unsupported pre-tokenizer regex {re:?}"))?);
                if p.get("behavior").and_then(Json::as_str) != Some("Isolated")
                    || p.get("invert").and_then(Json::as_bool) == Some(true)
                {
                    return err("only Split(behavior=Isolated) is supported");
                }
            }
            Some("ByteLevel") => {
                self.byte_level = true;
                if p.get("add_prefix_space").and_then(Json::as_bool) == Some(true) {
                    return err("ByteLevel add_prefix_space is not supported");
                }
                if p.get("use_regex").and_then(Json::as_bool).unwrap_or(true) {
                    if self.pattern.is_some() {
                        return err("two regex splits");
                    }
                    self.pattern = Some(Pattern::Gpt2);
                }
            }
            other => return err(format!("unsupported pre-tokenizer {other:?}")),
        }
        Ok(())
    }

    fn parse_post_processor(&mut self, pp: &Json) -> Result<(), String> {
        match pp.get("type").and_then(Json::as_str) {
            Some("TemplateProcessing") => {
                let mut seen_seq = false;
                for item in pp.get("single").and_then(Json::as_array).unwrap_or(&[]) {
                    if item.get("Sequence").is_some() {
                        seen_seq = true;
                    } else if let Some(name) = item.path(&["SpecialToken", "id"]).and_then(Json::as_str) {
                        let ids = pp
                            .path(&["special_tokens", name, "ids"])
                            .and_then(Json::as_array)
                            .map(|a| a.iter().filter_map(Json::as_i64).map(|x| x as u32).collect::<Vec<_>>())
                            .or_else(|| self.token_id(name).map(|id| vec![id]))
                            .ok_or_else(|| format!("unknown special token {name}"))?;
                        if seen_seq { self.suffix.extend(ids) } else { self.prefix.extend(ids) }
                    }
                }
                Ok(())
            }
            Some("ByteLevel") => Ok(()),
            Some("Sequence") => {
                for sub in pp.get("processors").and_then(Json::as_array).unwrap_or(&[]) {
                    self.parse_post_processor(sub)?;
                }
                Ok(())
            }
            other => err(format!("unsupported post-processor {other:?}")),
        }
    }

    fn build_decode_table(&mut self) {
        let max_id = self.vocab.values().chain(self.added.iter().map(|t| &t.id)).copied().max().unwrap_or(0);
        let mut bytes = vec![Vec::new(); max_id as usize + 1];
        let mut special = vec![false; max_id as usize + 1];
        let mut char_to_byte = HashMap::new();
        for (b, c) in byte_to_char().iter().enumerate() {
            char_to_byte.insert(*c, b as u8);
        }
        for (tok, &id) in &self.vocab {
            bytes[id as usize] = if self.byte_level {
                let mut v = Vec::with_capacity(tok.len());
                for c in tok.chars() {
                    match char_to_byte.get(&c) {
                        Some(&b) => v.push(b),
                        None => v.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                    }
                }
                v
            } else if let Some(b) = parse_byte_token(tok) {
                vec![b]
            } else {
                tok.replace('\u{2581}', " ").into_bytes()
            };
        }
        for t in &self.added {
            bytes[t.id as usize] = t.content.clone().into_bytes();
            special[t.id as usize] = t.special;
        }
        self.token_bytes = bytes;
        self.special = special;
    }

    /// Ids above this are padding rows in the embedding matrix.
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.added.iter().find(|t| t.content == token).map(|t| t.id).or_else(|| self.vocab.get(token).copied())
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added
    }

    /// The raw bytes a token stands for (special tokens: their text).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.token_bytes.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Split text around added tokens (leftmost-longest match).
    pub fn split_added<'a>(&self, text: &'a str) -> Vec<Piece<'a>> {
        let mut out = Vec::new();
        let mut start = 0;
        let mut iter = text.char_indices();
        while let Some((i, c)) = iter.next() {
            let Some(cands) = self.added_by_first.get(&c) else { continue };
            if let Some(&k) = cands.iter().find(|&&k| text[i..].starts_with(&self.added[k].content)) {
                if i > start {
                    out.push(Piece::Text(&text[start..i]));
                }
                out.push(Piece::Added(self.added[k].id));
                start = i + self.added[k].content.len();
                // Skip the rest of the matched token.
                while iter.clone().next().is_some_and(|(j, _)| j < start) {
                    iter.next();
                }
            }
        }
        if start < text.len() {
            out.push(Piece::Text(&text[start..]));
        }
        out
    }

    /// Encode text. With `add_special`, the post-processor's tokens (such as
    /// Llama's `<s>`) are added, like `tokenizer.encode(text)` in Python.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_special {
            ids.extend_from_slice(&self.prefix);
        }
        for piece in self.split_added(text) {
            match piece {
                Piece::Added(id) => ids.push(id),
                Piece::Text(s) => {
                    let at_start = std::ptr::eq(s.as_ptr(), text.as_ptr());
                    self.encode_segment(s, at_start, &mut ids)
                }
            }
        }
        if add_special {
            ids.extend_from_slice(&self.suffix);
        }
        ids
    }

    /// Encode text that contains no added tokens.
    pub fn encode_text(&self, text: &str, ids: &mut Vec<u32>) {
        self.encode_segment(text, true, ids)
    }

    fn encode_segment(&self, text: &str, at_start: bool, ids: &mut Vec<u32>) {
        let mut s: Cow<str> = Cow::Borrowed(text);
        for step in &self.norm {
            s = match step {
                NormStep::Nfc => match unicode::nfc(&s) {
                    Cow::Borrowed(_) => s,
                    Cow::Owned(o) => Cow::Owned(o),
                },
                NormStep::Prepend(p) if !s.is_empty() && (at_start || !self.prepend_first_only) => {
                    Cow::Owned(format!("{p}{s}"))
                }
                NormStep::Prepend(_) => s,
                NormStep::Replace(a, b) => Cow::Owned(s.replace(a.as_str(), b)),
            };
        }
        let s = s.as_ref();
        let ranges = if self.split_digits { pretok::split_digits(s) } else { vec![(0, s.len())] };
        let mut pieces = Vec::new();
        for (a, b) in ranges {
            match self.pattern {
                Some(p) => p.split(&s[a..b], a, &mut pieces),
                None => pieces.push((a, b)),
            }
        }
        for (a, b) in pieces {
            self.encode_piece(&s[a..b], ids);
        }
    }

    fn encode_piece(&self, piece: &str, ids: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let cacheable = piece.len() < 64;
        if cacheable && let Some(hit) = self.cache.lock().unwrap().get(piece) {
            ids.extend_from_slice(hit);
            return;
        }
        let word: Cow<str> = if self.byte_level {
            let table = byte_to_char_cached();
            Cow::Owned(piece.bytes().map(|b| table[b as usize]).collect())
        } else {
            Cow::Borrowed(piece)
        };
        let out = self.bpe(&word);
        if cacheable {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() > 100_000 {
                cache.clear();
            }
            cache.insert(piece.to_string(), out.clone());
        }
        ids.extend_from_slice(&out);
    }

    fn bpe(&self, word: &str) -> Vec<u32> {
        if self.ignore_merges
            && let Some(&id) = self.vocab.get(word)
        {
            return vec![id];
        }
        const NONE: u32 = u32::MAX;
        let mut ids: Vec<u32> = Vec::with_capacity(word.len());
        let mut buf = [0u8; 4];
        for c in word.chars() {
            let s = c.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(&*s) {
                ids.push(id);
            } else if let Some(fallback) = &self.byte_fallback {
                ids.extend(s.bytes().map(|b| fallback[b as usize]));
            } else if let Some(unk) = self.unk {
                if !(self.fuse_unk && ids.last() == Some(&unk)) {
                    ids.push(unk);
                }
            } else {
                ids.push(NONE);
            }
        }
        // Repeatedly merge the lowest-ranked adjacent pair, leftmost first.
        let n = ids.len();
        let mut next: Vec<i32> = (1..=n as i32).collect();
        next[n - 1] = -1;
        let mut prev: Vec<i32> = (-1..n as i32 - 1).collect();
        let mut heap = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some(&(rank, new)) = self.merges.get(&(ids[i], ids[i + 1])) {
                heap.push(Reverse((rank, i as u32, new)));
            }
        }
        while let Some(Reverse((rank, pos, new))) = heap.pop() {
            let pos = pos as usize;
            if ids[pos] == NONE || next[pos] < 0 {
                continue;
            }
            let nx = next[pos] as usize;
            if self.merges.get(&(ids[pos], ids[nx])) != Some(&(rank, new)) {
                continue;
            }
            ids[pos] = new;
            ids[nx] = NONE;
            next[pos] = next[nx];
            if next[nx] >= 0 {
                prev[next[nx] as usize] = pos as i32;
            }
            if prev[pos] >= 0 {
                let p = prev[pos] as usize;
                if let Some(&(r, m)) = self.merges.get(&(ids[p], new)) {
                    heap.push(Reverse((r, p as u32, m)));
                }
            }
            if next[pos] >= 0
                && let Some(&(r, m)) = self.merges.get(&(new, ids[next[pos] as usize]))
            {
                heap.push(Reverse((r, pos as u32, m)));
            }
        }
        let mut out = Vec::new();
        let mut i = 0i32;
        while i >= 0 {
            if ids[i as usize] != NONE {
                out.push(ids[i as usize]);
            }
            i = next[i as usize];
        }
        out
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            bytes.extend_from_slice(self.token_bytes(id));
        }
        if self.strip_leading_space && bytes.first() == Some(&b' ') {
            bytes.remove(0);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn stream_decoder(&self) -> StreamDecoder<'_> {
        StreamDecoder { tk: self, pending: Vec::new(), started: false }
    }
}

fn byte_to_char_cached() -> &'static [char; 256] {
    static TABLE: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(byte_to_char)
}

fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 { u8::from_str_radix(hex, 16).ok() } else { None }
}

fn parse_normalizer(n: &Json, out: &mut Vec<NormStep>) -> Result<(), String> {
    match n.get("type").and_then(Json::as_str) {
        Some("Sequence") => {
            for sub in n.get("normalizers").and_then(Json::as_array).unwrap_or(&[]) {
                parse_normalizer(sub, out)?;
            }
        }
        Some("NFC") => out.push(NormStep::Nfc),
        Some("Prepend") => {
            out.push(NormStep::Prepend(n.get("prepend").and_then(Json::as_str).ok_or("bad Prepend")?.into()))
        }
        Some("Replace") => {
            let pat = n
                .path(&["pattern", "String"])
                .and_then(Json::as_str)
                .ok_or("only string Replace patterns are supported")?;
            let content = n.get("content").and_then(Json::as_str).ok_or("bad Replace")?;
            out.push(NormStep::Replace(pat.into(), content.into()));
        }
        other => return err(format!("unsupported normalizer {other:?}")),
    }
    Ok(())
}

/// SentencePiece-style decoders end with Strip(" ", start=1).
fn decoder_strips_space(d: &Json) -> bool {
    match d.get("type").and_then(Json::as_str) {
        Some("Sequence") => d.get("decoders").and_then(Json::as_array).unwrap_or(&[]).iter().any(decoder_strips_space),
        Some("Strip") => {
            d.get("content").and_then(Json::as_str) == Some(" ") && d.get("start").and_then(Json::as_i64) == Some(1)
        }
        _ => false,
    }
}

/// Turns a stream of token ids into text, holding back bytes until they
/// form complete UTF-8 characters.
pub struct StreamDecoder<'a> {
    tk: &'a Tokenizer,
    pending: Vec<u8>,
    started: bool,
}

impl StreamDecoder<'_> {
    pub fn push(&mut self, id: u32) -> String {
        if self.tk.is_special(id) {
            return String::new();
        }
        let mut b = self.tk.token_bytes(id);
        if !self.started {
            if self.tk.strip_leading_space && b.first() == Some(&b' ') {
                b = &b[1..];
            }
            self.started = !b.is_empty() || self.started;
        }
        self.pending.extend_from_slice(b);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(), // an incomplete character at the end
            Err(_) => self.pending.len(),                         // invalid bytes: emit lossily
        };
        let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        out
    }

    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}
