//! Pre-tokenizers: split text into the pieces BPE runs on.
//!
//! Hugging Face tokenizers describe this split with a regex. The patterns
//! used by GPT-2-style and Llama-3/Qwen-style tokenizers need Unicode
//! classes and a `(?!\S)` lookahead, so instead of a regex engine they are
//! implemented here as hand-written matchers with exactly the regexes'
//! leftmost-first semantics. `tokenizer.json` patterns are recognised by
//! their text.

use super::unicode::{is_letter, is_number, is_space};

/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
pub const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Llama 3 (digits in groups of up to three).
pub const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen2 / Qwen3 (single digits).
pub const QWEN2: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Gpt2,
    /// Llama 3 and Qwen; the field is the longest run of digits per piece.
    Llama3 {
        max_digits: usize,
    },
}

impl Pattern {
    pub fn from_regex(re: &str) -> Option<Pattern> {
        match re {
            GPT2 => Some(Pattern::Gpt2),
            LLAMA3 => Some(Pattern::Llama3 { max_digits: 3 }),
            QWEN2 => Some(Pattern::Llama3 { max_digits: 1 }),
            _ => None,
        }
    }

    /// Append (start, end) byte ranges of the pieces of `text` to `out`.
    pub fn split(self, text: &str, base: usize, out: &mut Vec<(usize, usize)>) {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut i = 0;
        while i < chars.len() {
            let n = match self {
                Pattern::Gpt2 => gpt2_at(&chars, i),
                Pattern::Llama3 { max_digits } => llama3_at(&chars, i, max_digits),
            };
            debug_assert!(n > 0);
            let start = chars[i].0;
            let end = if i + n < chars.len() { chars[i + n].0 } else { text.len() };
            out.push((base + start, base + end));
            i += n;
        }
    }
}

fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn run(chars: &[(usize, char)], from: usize, pred: impl Fn(char) -> bool) -> usize {
    chars[from..].iter().take_while(|(_, c)| pred(*c)).count()
}

/// The contractions alternative; `ci` makes it case-insensitive.
fn contraction(chars: &[(usize, char)], i: usize, ci: bool) -> usize {
    if chars[i].1 != '\'' {
        return 0;
    }
    for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
        let ok = suffix.chars().enumerate().all(|(k, want)| {
            chars.get(i + 1 + k).is_some_and(|&(_, c)| if ci { c.to_ascii_lowercase() == want } else { c == want })
        });
        if ok {
            return 1 + suffix.len();
        }
    }
    0
}

/// `\s+(?!\S)` falling back to `\s+`, at a whitespace character.
fn whitespace(chars: &[(usize, char)], i: usize) -> usize {
    let n = run(chars, i, is_space);
    if i + n == chars.len() || n == 1 { n } else { n - 1 }
}

fn gpt2_at(chars: &[(usize, char)], i: usize) -> usize {
    let n = contraction(chars, i, false);
    if n > 0 {
        return n;
    }
    let c = chars[i].1;
    let j = if c == ' ' { i + 1 } else { i };
    if let Some(&(_, d)) = chars.get(j) {
        if is_letter(d) {
            return j - i + run(chars, j, is_letter);
        }
        if is_number(d) {
            return j - i + run(chars, j, is_number);
        }
        if is_other(d) {
            return j - i + run(chars, j, is_other);
        }
    }
    if is_space(c) {
        return whitespace(chars, i);
    }
    1
}

fn llama3_at(chars: &[(usize, char)], i: usize, max_digits: usize) -> usize {
    let n = contraction(chars, i, true);
    if n > 0 {
        return n;
    }
    let c = chars[i].1;
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if is_letter(c) {
        return run(chars, i, is_letter);
    }
    if !is_newline(c) && !is_number(c) && chars.get(i + 1).is_some_and(|&(_, d)| is_letter(d)) {
        return 1 + run(chars, i + 1, is_letter);
    }
    // \p{N}{1,max}
    if is_number(c) {
        return run(chars, i, is_number).min(max_digits);
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    let j = if c == ' ' && chars.get(i + 1).is_some_and(|&(_, d)| is_other(d)) { i + 1 } else { i };
    if is_other(chars[j].1) {
        let k = j + run(chars, j, is_other);
        return k - i + run(chars, k, is_newline);
    }
    // \s*[\r\n]+ : up to and including the last newline of the whitespace run
    if is_space(c) {
        let n = run(chars, i, is_space);
        if let Some(last_nl) = (i..i + n).rev().find(|&k| is_newline(chars[k].1)) {
            return last_nl + 1 - i;
        }
        return whitespace(chars, i);
    }
    1
}

/// Split around every numeric character (the `Digits` pre-tokenizer with
/// `individual_digits`), returning the pieces in order.
pub fn split_digits(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c.is_numeric() {
            if i > start {
                out.push((start, i));
            }
            out.push((i, i + c.len_utf8()));
            start = i + c.len_utf8();
        }
    }
    if start < text.len() {
        out.push((start, text.len()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(p: Pattern, s: &str) -> Vec<&str> {
        let mut v = Vec::new();
        p.split(s, 0, &mut v);
        v.iter().map(|&(a, b)| &s[a..b]).collect()
    }

    #[test]
    fn gpt2() {
        assert_eq!(
            pieces(Pattern::Gpt2, "Hello world, it's 2024!"),
            ["Hello", " world", ",", " it", "'s", " 2024", "!"]
        );
        assert_eq!(pieces(Pattern::Gpt2, "a   b  "), ["a", "  ", " b", "  "]);
        assert_eq!(pieces(Pattern::Gpt2, "x\n\ny"), ["x", "\n", "\n", "y"]);
    }

    #[test]
    fn qwen() {
        let q = Pattern::Llama3 { max_digits: 1 };
        assert_eq!(
            pieces(q, "I'M 123 (ok)\n\n  next"),
            ["I", "'M", " ", "1", "2", "3", " (", "ok", ")\n\n", " ", " next"]
        );
        assert_eq!(pieces(Pattern::Llama3 { max_digits: 3 }, "12345"), ["123", "45"]);
        assert_eq!(pieces(q, "a \n b"), ["a", " \n", " b"]);
    }
}
