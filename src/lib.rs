//! wick: run Llama- and Qwen-family language models on the CPU.
//!
//! Everything is implemented here with only the Rust standard library: the
//! safetensors loader, the BPE tokenizer, the SIMD kernels, quantization,
//! sampling and the HTTP server.

pub mod chat;
pub mod config;
pub mod engine;
pub mod json;
pub mod kernels;
pub mod mmap;
pub mod model;
pub mod pool;
pub mod quant;
pub mod safetensors;
pub mod sampler;
pub mod server;
pub mod tokenizer;
