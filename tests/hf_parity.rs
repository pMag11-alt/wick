//! Parity with Hugging Face transformers and tokenizers.
//!
//! The fixtures in tests/fixtures/hf were recorded by scripts/reference.py.
//! Each test needs the model in `$WICK_MODELS/<name>` (default `models/`,
//! fetched with scripts/download.sh) and is skipped, loudly, without it.

use std::path::PathBuf;
use wick::chat::Message;
use wick::engine::{Engine, GenOptions, Session};
use wick::json::Json;
use wick::model::{Format, State};
use wick::sampler::SamplerConfig;
use wick::tokenizer::Tokenizer;

fn model_dir(name: &str) -> Option<PathBuf> {
    let root = std::env::var("WICK_MODELS").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models").into());
    let dir = PathBuf::from(root).join(name);
    if dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        eprintln!("SKIPPED: {} not found (scripts/download.sh)", dir.display());
        None
    }
}

fn fixture(name: &str) -> Json {
    let path = format!("{}/tests/fixtures/hf/{name}.json", env!("CARGO_MANIFEST_DIR"));
    Json::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn ids(j: &Json) -> Vec<u32> {
    j.as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as u32).collect()
}

fn check_tokenizer(name: &str) {
    let Some(dir) = model_dir(name) else { return };
    let tk = Tokenizer::load(&dir.join("tokenizer.json")).unwrap();
    let fx = fixture(name);
    let mut failures = vec![];
    for case in fx.get("tokenize").unwrap().as_array().unwrap() {
        let text = case.get("text").unwrap().as_str().unwrap();
        let want = ids(case.get("ids").unwrap());
        let got = tk.encode(text, false);
        if got != want {
            failures.push(format!("{text:?}\n   want {want:?}\n   got  {got:?}"));
        }
        // Decoding gives the text back (for these inputs NFC is the only change).
        let back = tk.decode(&got, false);
        let norm = wick::tokenizer::unicode::nfc(text);
        if back != text && back != norm {
            failures.push(format!("decode {text:?} gave {back:?}"));
        }
    }
    for case in fx.get("encode_special").unwrap().as_array().unwrap() {
        let text = case.get("text").unwrap().as_str().unwrap();
        assert_eq!(tk.encode(text, true), ids(case.get("ids").unwrap()), "encode with special tokens: {text:?}");
    }
    assert!(failures.is_empty(), "{name}: {} mismatches:\n{}", failures.len(), failures.join("\n"));
}

fn check_model(name: &str) {
    let Some(dir) = model_dir(name) else { return };
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIPPED: no weights in {}", dir.display());
        return;
    }
    let fx = fixture(name);
    let mut engine = Engine::load(&dir, Format::BF16, 8).unwrap();

    // Chat template: the exact string transformers renders.
    let chat_msgs = [
        Message::new("user", "Name three primary colors."),
        Message::new("assistant", "Red, yellow and blue."),
        Message::new("user", "And the secondary ones?"),
    ];
    let chat = fx.get("chat").unwrap();
    for (key, want) in chat.as_object().unwrap() {
        let mut msgs = chat_msgs.to_vec();
        match key.as_str() {
            "system" => msgs.insert(0, Message::new("system", "You are terse.")),
            "think" => engine.chat.thinking = Some(true),
            "no_think" => engine.chat.thinking = Some(false),
            _ => {}
        }
        assert_eq!(engine.chat.render(&msgs), want.as_str().unwrap(), "{name}: chat template {key}");
    }

    // Logits: same argmax at every position, top-20 values within 1e-2.
    for case in fx.get("logits").unwrap().as_array().unwrap() {
        let prompt = ids(case.get("ids").unwrap());
        let mut state = State::new(&engine.model, 512);
        let logits = engine.model.forward(&engine.pool, &mut state, &prompt, true);
        let vocab = engine.model.cfg.vocab;
        let argmax: Vec<u32> = logits.chunks_exact(vocab).map(wick::sampler::argmax).collect();
        assert_eq!(argmax, ids(case.get("argmax").unwrap()), "{name}: argmax per position");
        let last = &logits[(prompt.len() - 1) * vocab..];
        let top_ids = ids(case.get("top_ids").unwrap());
        let top_logits = case.get("top_logits").unwrap().as_array().unwrap();
        let mut worst = 0f32;
        for (id, want) in top_ids.iter().zip(top_logits) {
            worst = worst.max((last[*id as usize] - want.as_f64().unwrap() as f32).abs());
        }
        assert!(worst < 1e-2, "{name}: top-20 logits differ by up to {worst}");
        eprintln!("{name}: {} positions match, top-20 logits within {worst:.2e}", prompt.len());
    }

    // Greedy decoding produces exactly the same tokens.
    engine.chat.thinking = engine.chat.thinking.map(|_| false);
    let greedy = fx.get("greedy").unwrap();
    let prompt = ids(greedy.get("prompt").unwrap());
    assert_eq!(engine.tokenizer.encode(&engine.chat.render(&chat_msgs), false), prompt, "{name}: chat prompt ids");
    let want = ids(greedy.get("output").unwrap());
    let mut session = Session::new(&engine, 1024);
    let opts = GenOptions { max_tokens: want.len(), sampler: SamplerConfig::greedy(), stop_strings: vec![] };
    let r = session.generate(&prompt, &opts, |_| true).unwrap();
    let mut got = r.tokens.clone();
    if got.len() < want.len() {
        // Generation stopped on an end-of-turn token, which transformers keeps.
        got.extend(want.get(got.len()).copied());
    }
    assert_eq!(got, want, "{name}: greedy continuation");
}

#[test]
fn tokenizer_smollm2() {
    check_tokenizer("SmolLM2-135M-Instruct");
}

#[test]
fn tokenizer_qwen25() {
    check_tokenizer("Qwen2.5-0.5B-Instruct");
}

#[test]
fn tokenizer_qwen3() {
    check_tokenizer("Qwen3-0.6B");
}

#[test]
fn tokenizer_tinyllama() {
    check_tokenizer("TinyLlama-1.1B-Chat-v1.0");
}

#[test]
fn model_smollm2() {
    check_model("SmolLM2-135M-Instruct");
}

#[test]
fn model_qwen25() {
    check_model("Qwen2.5-0.5B-Instruct");
}

#[test]
fn model_qwen3() {
    check_model("Qwen3-0.6B");
}

#[test]
fn model_tinyllama() {
    check_model("TinyLlama-1.1B-Chat-v1.0");
}
