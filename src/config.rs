//! Model hyper-parameters from a Hugging Face `config.json`.

use crate::json::Json;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// Llama 2/3, TinyLlama, SmolLM, Mistral: no biases.
    Llama,
    /// Qwen2 / Qwen2.5: biases on the q, k and v projections.
    Qwen2,
    /// Qwen3: per-head RMSNorm on q and k, no biases.
    Qwen3,
}

/// Llama 3 style RoPE frequency scaling.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Arch,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub max_position: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub tie_embeddings: bool,
    pub attn_bias: bool,
    pub qk_norm: bool,
    pub bos_token: Option<u32>,
    pub eos_tokens: Vec<u32>,
}

fn token_ids(v: Option<&Json>) -> Vec<u32> {
    match v {
        Some(Json::Array(a)) => a.iter().filter_map(|x| x.as_i64()).map(|x| x as u32).collect(),
        Some(x) => x.as_i64().map(|x| vec![x as u32]).unwrap_or_default(),
        None => vec![],
    }
}

impl Config {
    pub fn from_json(j: &Json) -> Result<Config, String> {
        let arch_name = j
            .get("architectures")
            .and_then(Json::as_array)
            .and_then(|a| a.first())
            .and_then(Json::as_str)
            .or_else(|| j.get("model_type").and_then(Json::as_str))
            .ok_or("config.json has no architectures")?;
        let arch = match arch_name {
            "LlamaForCausalLM" | "MistralForCausalLM" | "llama" | "mistral" => Arch::Llama,
            "Qwen2ForCausalLM" | "qwen2" => Arch::Qwen2,
            "Qwen3ForCausalLM" | "qwen3" => Arch::Qwen3,
            other => return Err(format!("unsupported architecture {other} (supported: Llama, Mistral, Qwen2, Qwen3)")),
        };
        let need = |k: &str| j.get(k).and_then(Json::as_usize).ok_or_else(|| format!("config.json: missing {k}"));
        let hidden = need("hidden_size")?;
        let heads = need("num_attention_heads")?;
        let kv_heads = j.get("num_key_value_heads").and_then(Json::as_usize).unwrap_or(heads);
        let head_dim = j.get("head_dim").and_then(Json::as_usize).unwrap_or(hidden / heads);
        if let Some(act) = j.get("hidden_act").and_then(Json::as_str)
            && act != "silu"
        {
            return Err(format!("unsupported activation {act}"));
        }
        // transformers 5 writes `rope_parameters` (with rope_theta inside);
        // older configs have top-level `rope_theta` and `rope_scaling`.
        let rope_params = j.get("rope_parameters").filter(|v| !v.is_null());
        let rope_theta = j
            .get("rope_theta")
            .and_then(Json::as_f64)
            .or_else(|| rope_params.and_then(|p| p.get("rope_theta")).and_then(Json::as_f64))
            .unwrap_or(10000.0);
        let rope_scaling = match j.get("rope_scaling").filter(|v| !v.is_null()).or(rope_params) {
            None => None,
            Some(rs) => {
                match rs.get("rope_type").or_else(|| rs.get("type")).and_then(Json::as_str).unwrap_or("default") {
                    "default" => None,
                    "llama3" => {
                        let f = |k: &str| {
                            rs.get(k).and_then(Json::as_f64).ok_or_else(|| format!("rope_scaling: missing {k}"))
                        };
                        Some(RopeScaling {
                            factor: f("factor")? as f32,
                            low_freq_factor: f("low_freq_factor")? as f32,
                            high_freq_factor: f("high_freq_factor")? as f32,
                            original_max_position: f("original_max_position_embeddings")? as usize,
                        })
                    }
                    kind => return Err(format!("unsupported rope_scaling type {kind:?}")),
                }
            }
        };
        let cfg = Config {
            arch,
            hidden,
            intermediate: need("intermediate_size")?,
            layers: need("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim,
            vocab: need("vocab_size")?,
            max_position: j.get("max_position_embeddings").and_then(Json::as_usize).unwrap_or(4096),
            rms_eps: j.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            rope_theta: rope_theta as f32,
            rope_scaling,
            tie_embeddings: j.get("tie_word_embeddings").and_then(Json::as_bool).unwrap_or(false),
            attn_bias: arch == Arch::Qwen2 || j.get("attention_bias").and_then(Json::as_bool).unwrap_or(false),
            qk_norm: arch == Arch::Qwen3,
            bos_token: j.get("bos_token_id").and_then(Json::as_i64).map(|x| x as u32),
            eos_tokens: token_ids(j.get("eos_token_id")),
        };
        if !cfg.heads.is_multiple_of(cfg.kv_heads) {
            return Err("num_attention_heads must be a multiple of num_key_value_heads".into());
        }
        if !cfg.head_dim.is_multiple_of(2) {
            return Err("head_dim must be even for rotary embeddings".into());
        }
        Ok(cfg)
    }

    pub fn load(dir: &Path) -> Result<Config, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j = Json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut cfg = Config::from_json(&j)?;
        // generation_config.json often lists more stop tokens (e.g. Qwen's <|endoftext|>).
        if let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json"))
            && let Ok(g) = Json::parse(&text)
        {
            for id in token_ids(g.get("eos_token_id")) {
                if !cfg.eos_tokens.contains(&id) {
                    cfg.eos_tokens.push(id);
                }
            }
        }
        Ok(cfg)
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Rotary inverse frequencies, with Llama 3 scaling applied if configured.
    pub fn rope_inv_freq(&self) -> Vec<f32> {
        let d = self.head_dim;
        (0..d / 2)
            .map(|i| {
                let inv = 1.0 / (self.rope_theta as f64).powf(2.0 * i as f64 / d as f64);
                let Some(rs) = &self.rope_scaling else { return inv as f32 };
                // Long wavelengths are slowed down by `factor`, short ones kept,
                // and the band in between is interpolated.
                let low_wavelen = rs.original_max_position as f64 / rs.low_freq_factor as f64;
                let high_wavelen = rs.original_max_position as f64 / rs.high_freq_factor as f64;
                let wavelen = 2.0 * std::f64::consts::PI / inv;
                let scaled = if wavelen < high_wavelen {
                    inv
                } else if wavelen > low_wavelen {
                    inv / rs.factor as f64
                } else {
                    let smooth = (rs.original_max_position as f64 / wavelen - rs.low_freq_factor as f64)
                        / (rs.high_freq_factor as f64 - rs.low_freq_factor as f64);
                    (1.0 - smooth) * inv / rs.factor as f64 + smooth * inv
                };
                scaled as f32
            })
            .collect()
    }
}
