//! Tiny random models built and scored by transformers (scripts/make_tiny.py).
//! They run everywhere, CI included, and cover every architecture variant,
//! weight format and both prompt and incremental decoding.

use std::path::PathBuf;
use wick::json::Json;
use wick::model::{Format, Model, State};
use wick::pool::Pool;

const MODELS: [&str; 3] = ["llama3-scaled-f32", "qwen2-bias-bf16", "qwen3-qknorm-f32"];

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny").join(name)
}

struct Expected {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
    greedy_prompt: Vec<u32>,
    greedy: Vec<u32>,
}

fn expected(name: &str) -> Expected {
    let j = Json::parse(&std::fs::read_to_string(dir(name).join("expected.json")).unwrap()).unwrap();
    let ids = |k: &str| j.get(k).unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as u32).collect();
    Expected {
        tokens: ids("tokens"),
        logits: j
            .get("logits")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect(),
        greedy_prompt: ids("greedy_prompt"),
        greedy: ids("greedy"),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn all_logits(model: &Model, pool: &Pool, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut state = State::new(model, 256);
    let flat = model.forward(pool, &mut state, tokens, true);
    flat.chunks_exact(model.cfg.vocab).map(|r| r.to_vec()).collect()
}

#[test]
fn matches_transformers_exactly() {
    let pool = Pool::new(4);
    for name in MODELS {
        let exp = expected(name);
        for format in [Format::BF16, Format::F32] {
            let model = Model::load(&dir(name), format).unwrap();
            let got = all_logits(&model, &pool, &exp.tokens);
            assert_eq!(got.len(), exp.logits.len());
            let worst = got.iter().zip(&exp.logits).map(|(g, e)| max_abs_diff(g, e)).fold(0.0, f32::max);
            assert!(worst < 1e-3, "{name} {format:?}: logits differ by {worst}");
        }
    }
}

#[test]
fn incremental_decoding_matches_one_batch() {
    // Feeding the sequence in uneven chunks must give the same logits as one
    // batch: this checks the KV cache, positions and causal masking.
    let pool = Pool::new(3);
    for name in MODELS {
        let exp = expected(name);
        let model = Model::load(&dir(name), Format::BF16).unwrap();
        let batch = all_logits(&model, &pool, &exp.tokens);
        let mut state = State::new(&model, 256);
        let mut pos = 0;
        for size in [1, 5, 2, 7, 1, 1, 13, 3, 15] {
            let chunk = &exp.tokens[pos..pos + size];
            let flat = model.forward(&pool, &mut state, chunk, true);
            for (i, row) in flat.chunks_exact(model.cfg.vocab).enumerate() {
                let d = max_abs_diff(row, &batch[pos + i]);
                assert!(d < 1e-4, "{name}: position {} differs by {d}", pos + i);
            }
            pos += size;
        }
        assert_eq!(pos, exp.tokens.len());
    }
}

#[test]
fn greedy_continuation_matches() {
    let pool = Pool::new(2);
    for name in MODELS {
        let exp = expected(name);
        let model = Model::load(&dir(name), Format::BF16).unwrap();
        let mut state = State::new(&model, 256);
        let mut logits = model.forward(&pool, &mut state, &exp.greedy_prompt, false);
        let mut out = vec![];
        for _ in 0..exp.greedy.len() {
            let t = wick::sampler::argmax(&logits);
            out.push(t);
            logits = model.forward(&pool, &mut state, &[t], false);
        }
        assert_eq!(out, exp.greedy, "{name}");
    }
}

#[test]
fn quantized_formats_stay_close() {
    // Random weights have no structure for quantization to preserve, so
    // this guards the int8 kernels and block layouts (a layout bug drops
    // the similarity to about zero); model quality is measured with
    // perplexity on real models instead.
    let pool = Pool::new(4);
    for name in MODELS {
        let exp = expected(name);
        for (format, min_mean) in [(Format::Q8, 0.995), (Format::Q4, 0.75)] {
            let model = Model::load(&dir(name), format).unwrap();
            let got = all_logits(&model, &pool, &exp.tokens);
            let mean = got.iter().zip(&exp.logits).map(|(g, e)| cosine(g, e)).sum::<f32>() / got.len() as f32;
            eprintln!("{name} {format:?}: mean cosine similarity {mean:.4}");
            assert!(mean > min_mean, "{name} {format:?}: mean cosine similarity {mean}");
        }
    }
}

#[test]
fn rejects_mismatched_weights() {
    // A config that disagrees with the tensors must fail cleanly.
    let tmp = std::env::temp_dir().join(format!("wick-bad-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = dir("qwen2-bias-bf16");
    std::fs::copy(src.join("model.safetensors"), tmp.join("model.safetensors")).unwrap();
    let cfg =
        std::fs::read_to_string(src.join("config.json")).unwrap().replace("\"hidden_size\": 64", "\"hidden_size\": 96");
    std::fs::write(tmp.join("config.json"), cfg).unwrap();
    let err = Model::load(&tmp, Format::BF16).err().expect("load should fail");
    assert!(err.contains("expected shape"), "{err}");
    std::fs::remove_dir_all(&tmp).ok();
}
