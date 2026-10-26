//! Loading a model directory and generating text from it.

use crate::chat::ChatFormat;
use crate::model::{Format, Model, State};
use crate::pool::Pool;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Engine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub chat: ChatFormat,
    pub pool: Pool,
    pub dir: PathBuf,
    pub name: String,
    pub load_ms: f64,
}

impl Engine {
    pub fn load(dir: &Path, format: Format, threads: usize) -> Result<Engine, String> {
        let t0 = Instant::now();
        let tokenizer = Tokenizer::load_dir(dir)?;
        let model = Model::load(dir, format)?;
        let chat = ChatFormat::detect(&tokenizer, dir);
        let name = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        Ok(Engine {
            model,
            tokenizer,
            chat,
            pool: Pool::new(threads),
            dir: dir.to_path_buf(),
            name,
            load_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Tokens that end a reply: the config's EOS ids plus the chat
    /// format's end-of-turn tokens.
    pub fn stop_tokens(&self) -> Vec<u32> {
        let mut ids = self.model.cfg.eos_tokens.clone();
        for t in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "</s>", "<|end_of_text|>"] {
            if let Some(id) = self.tokenizer.token_id(t)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        ids
    }

    pub fn default_threads() -> usize {
        // Performance cores matter most; leave headroom on big machines.
        std::thread::available_parallelism().map(|n| n.get().clamp(1, 8)).unwrap_or(4)
    }
}

#[derive(Clone, Debug)]
pub struct GenOptions {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    pub stop_strings: Vec<String>,
}

impl Default for GenOptions {
    fn default() -> Self {
        GenOptions { max_tokens: 256, sampler: SamplerConfig::default(), stop_strings: vec![] }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct GenResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub finish: Finish,
    pub prompt_ms: f64,
    pub gen_ms: f64,
}

impl GenResult {
    pub fn prompt_tps(&self) -> f64 {
        (self.prompt_tokens - self.reused_tokens) as f64 / (self.prompt_ms / 1e3).max(1e-9)
    }
    pub fn gen_tps(&self) -> f64 {
        self.tokens.len().saturating_sub(1) as f64 / (self.gen_ms / 1e3).max(1e-9)
    }
}

/// A sequence with its KV cache. Consecutive requests that share a prefix
/// (a chat that grows turn by turn) only process the new tokens.
pub struct Session<'e> {
    pub engine: &'e Engine,
    state: State,
    tokens: Vec<u32>,
}

const CHUNK: usize = 256;

impl<'e> Session<'e> {
    pub fn new(engine: &'e Engine, context: usize) -> Session<'e> {
        Session { engine, state: State::new(&engine.model, context), tokens: Vec::new() }
    }

    pub fn context(&self) -> usize {
        self.state.cache.capacity
    }

    /// Make the cache hold exactly `prompt` and return the logits after its
    /// last token, plus how many tokens were already cached.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<(Vec<f32>, usize), String> {
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() > self.context() {
            return Err(format!("prompt is {} tokens, context is {}", prompt.len(), self.context()));
        }
        let mut keep = self.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        if keep == prompt.len() {
            keep -= 1; // we still need the last token's logits
        }
        self.tokens.truncate(keep);
        self.state.cache.len = keep;
        let mut logits = Vec::new();
        for chunk in prompt[keep..].chunks(CHUNK) {
            logits = self.engine.model.forward(&self.engine.pool, &mut self.state, chunk, false);
            self.tokens.extend_from_slice(chunk);
        }
        Ok((logits, keep))
    }

    /// Feed one token, returning the logits for the next.
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        self.tokens.push(token);
        self.engine.model.forward(&self.engine.pool, &mut self.state, &[token], false)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Generate a reply to `prompt`, calling `on_text` with each new piece
    /// of text; it returns false to cancel.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        opts: &GenOptions,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<GenResult, String> {
        let engine = self.engine;
        let stops = engine.stop_tokens();
        let t0 = Instant::now();
        let (mut logits, reused) = self.prefill(prompt)?;
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let room = self.context() - self.len();
        let max_tokens = opts.max_tokens.min(room);
        let mut sampler = Sampler::new(opts.sampler.clone());
        let mut decoder = engine.tokenizer.stream_decoder();
        let mut out = Vec::new();
        let mut text = String::new();
        let mut emitted = 0; // bytes of `text` already passed to on_text
        let hold = opts.stop_strings.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
        let mut finish = Finish::Length;
        while out.len() < max_tokens {
            let history: &[u32] = if opts.sampler.repetition_penalty != 1.0 { &self.tokens } else { &[] };
            let tok = sampler.sample(&mut logits, history);
            if stops.contains(&tok) {
                finish = Finish::Stop;
                break;
            }
            out.push(tok);
            text += &decoder.push(tok);
            if let Some(cut) = opts.stop_strings.iter().filter_map(|s| text.find(s.as_str())).min() {
                text.truncate(cut);
                finish = Finish::Stop;
                break;
            }
            // Hold back a possible prefix of a stop string.
            let mut safe = text.len().saturating_sub(hold);
            while !text.is_char_boundary(safe) {
                safe -= 1;
            }
            if safe > emitted {
                if !on_text(&text[emitted..safe]) {
                    finish = Finish::Cancelled;
                    break;
                }
                emitted = safe;
            }
            if out.len() < max_tokens {
                logits = self.step(tok);
            }
        }
        if finish != Finish::Stop || opts.stop_strings.is_empty() {
            text += &decoder.finish();
        }
        if emitted < text.len() && finish != Finish::Cancelled {
            on_text(&text[emitted..]);
        }
        Ok(GenResult {
            text,
            tokens: out,
            prompt_tokens: prompt.len(),
            reused_tokens: reused,
            finish,
            prompt_ms,
            gen_ms: t1.elapsed().as_secs_f64() * 1e3,
        })
    }
}
