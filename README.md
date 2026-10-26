# wick

[![CI](https://github.com/pMag11-alt/wick/actions/workflows/ci.yml/badge.svg)](https://github.com/pMag11-alt/wick/actions/workflows/ci.yml)

Run Llama- and Qwen-family language models on your CPU. wick is an inference
engine written in Rust with **no dependencies**: the safetensors loader, the
BPE tokenizer, the SIMD kernels, quantization, sampling and an
OpenAI-compatible HTTP server are all built here on the standard library.
It reproduces Hugging Face transformers **token for token**.

```
$ wick run models/Qwen2.5-0.5B-Instruct -p "What is the capital of France?" -t 0
wick: Qwen2.5-0.5B-Instruct (494M params, 24 layers, qwen2) as q8_0, 0.5 GiB, 8 threads, NEON + dotprod kernels, loaded in 2074 ms
The capital of France is Paris.

[36 prompt tokens (0 cached) at 528.0 tok/s, 7 generated at 59.2 tok/s, Stop]

$ wick serve models/Qwen3-0.6B
wick: serving Qwen3-0.6B on http://127.0.0.1:8080 (OpenAI API at /v1, chat UI at /)
```

## What it does

- **Models.** Llama (Llama 2/3, TinyLlama, SmolLM, Mistral), Qwen2/2.5 and
  Qwen3, including thinking mode. It reads Hugging Face directories directly:
  `config.json` (both the transformers 4 and 5 formats), `tokenizer.json`
  and `model.safetensors`, single-file or sharded, in bf16, f16 or f32.
- **Exact.** The tokenizer gives the same ids as Hugging Face `tokenizers`.
  Chat prompts render byte-for-byte like `apply_chat_template`. Logits match
  transformers' float32 forward pass to within 6×10⁻⁵, and greedy decoding
  produces the same tokens.
- **Fast.** NEON kernels (using the SDOT instruction for int8) on ARM and
  AVX2 kernels on x86. Generation runs at about 90% of llama.cpp's CPU speed
  on the same model and quantization.
- **Quantized.** Q8_0 (8.5 bits/weight, +0.3–0.7% perplexity) and Q4_0
  (4.5 bits/weight), with llama.cpp's block layout, quantized at load time.
  bf16 weights are used zero-copy straight from the memory-mapped file.
- **Serves.** `/v1/chat/completions` and `/v1/completions` with SSE
  streaming, `stop` sequences, seeds and usage stats. It works with the
  official `openai` Python client. A chat that resends its history only
  processes the new tokens (prefix caching), a client that disconnects
  cancels its generation, and there is a built-in web chat UI at `/`.
- **Small.** One 944 KB binary. `cargo tree` shows exactly one crate.

## Correctness

Each check compares wick to Hugging Face transformers 5.20 and tokenizers
0.23 (fixtures recorded by [`scripts/reference.py`](scripts/reference.py)):

| Model | Tokenizer: 28 tricky strings | Chat template | Logits vs fp32 transformers | 24-token greedy reply |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | identical | identical | max diff 4.5×10⁻⁵ | identical |
| Qwen2.5-0.5B-Instruct | identical (incl. NFC) | identical | max diff 6.2×10⁻⁵ | identical |
| Qwen3-0.6B | identical (incl. NFC) | identical (thinking on and off) | max diff 4.7×10⁻⁵ | identical |
| TinyLlama-1.1B-Chat | identical (SentencePiece, byte fallback) | identical | max diff 4.8×10⁻⁵ | identical |

The tricky strings include CJK, Hangul, emoji with ZWJ sequences, combining
marks (NFC), runs of spaces and newlines, CRLF, contractions in any case,
long digit runs and special tokens embedded in text.

**Perplexity on WikiText-2** (test set, first 40 windows of 512 tokens,
computed identically by [`scripts/ppl_reference.py`](scripts/ppl_reference.py)
and `wick ppl`):

| Model | transformers fp32 | wick bf16 | wick Q8_0 | wick Q4_0 |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | 22.6563 | **22.6563** | 22.8065 (+0.7%) | 28.5206 (+26%) |
| Qwen2.5-0.5B-Instruct | 17.9137 | **17.9137** | 17.9753 (+0.3%) | 20.5094 (+14%) |

Q8_0 is nearly lossless, which is why it's the default. Q4_0 hurts
models this small noticeably, as it does in llama.cpp.

## Speed

Measured on an Apple M3 (4 performance + 4 efficiency cores, 16 GB) with
8 threads: `wick bench`, a 128-token prompt and 64 generated tokens, median
of 3 runs. Numbers are tokens/s, prompt / generation.

| Model | bf16 | Q8_0 | Q4_0 |
|---|---|---|---|
| SmolLM2-135M-Instruct | 790 / 153 | 1383 / 226 | 1385 / 278 |
| Qwen2.5-0.5B-Instruct | 194 / 57 | 488 / 88 | 492 / 107 |
| Qwen3-0.6B | 157 / 45 | 356 / 69 | 359 / 78 |
| TinyLlama-1.1B-Chat | 55 / 28 | 140 / 44 | 146 / 52 |

For comparison, `llama-bench -ngl 0 -t 8` from llama.cpp (Homebrew, CPU
only) on Qwen2.5-0.5B-Instruct's official GGUF files:

| | prompt | generation |
|---|---|---|
| llama.cpp Q8_0 | 848 | 101 |
| wick Q8_0 | 488 | 88 (87%) |
| llama.cpp Q4_0 | 826 | 115 |
| wick Q4_0 | 492 | 107 (93%) |

Generation is limited by memory bandwidth, and wick is within about 10%
there. Prompt processing is compute-bound. With `-ngl 0`, llama.cpp still
runs those matrix products through Apple Accelerate, which uses the M3's AMX
matrix unit; wick uses only NEON.

## Usage

```bash
cargo build --release
scripts/download.sh Qwen/Qwen2.5-0.5B-Instruct     # into models/
```

```bash
wick run models/Qwen2.5-0.5B-Instruct -p "Write a haiku about rust"
wick chat models/Qwen3-0.6B --no-think              # interactive; /reset starts over
wick serve models/SmolLM2-135M-Instruct --host 0.0.0.0:8080
wick bench models/TinyLlama-1.1B-Chat-v1.0 -q q4
wick ppl models/Qwen2.5-0.5B-Instruct wiki.test.raw -q q8
wick tokenize models/Qwen3-0.6B "Hello, wick!"
```

Useful options: `-q q8|q4|bf16|f32` (bf16 gives exact transformers
results; q8 is the default), `-t 0` for greedy decoding, `--top-k`,
`--top-p`, `--min-p`, `--repeat-penalty`, `--seed`, `-s` for a system
message, `-c` for the context length, `-j` for threads, and `--raw` for
plain completion without the chat format.

The server speaks the OpenAI API:

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
reply = client.chat.completions.create(model="wick", messages=[{"role": "user", "content": "Hi!"}])
```

## How it works

**Loading.** [`mmap.rs`](src/mmap.rs) maps the safetensors file read-only.
bf16 weights are used in place, so loading SmolLM2 takes 38 ms and the
pages are shared with the OS cache. For Q8_0/Q4_0, weights are converted to
llama.cpp's block layout at load time: 32 weights share one f16 scale, and
Q4_0 packs elements j and j + 16 into one byte. The output projection stays
8-bit even in Q4 mode, because it is the most sensitive matrix.

**The forward pass** ([`model.rs`](src/model.rs)) is the standard
pre-norm decoder: RMSNorm, then q/k/v projections (plus Qwen2's biases and
Qwen3's per-head q/k RMSNorm), rotary embeddings (with Llama 3 frequency
scaling), grouped-query attention against an f32 KV cache, and a SwiGLU
MLP. Prompts run as batches of up to 256 tokens. Matrices that read the same
input (q, k and v; gate and up) share one parallel region.

**Kernels** ([`kernels.rs`](src/kernels.rs)) compute one weight row against
up to four activation rows per pass, so prompt processing converts and loads
each weight once for four tokens. bf16 becomes f32 with a single widening
shift (`vshll_n_u16::<16>`). For the int8 formats, activations are quantized
to 8 bits per block, and each block is two `SDOT` instructions per row.
Weight scales are converted through a 256 KiB lookup table. That change, and
keeping activation scales as f32, doubled single-thread Q4 speed. Every SIMD
kernel has a scalar twin. The tests run the whole suite on both, with
`WICK_SCALAR=1` forcing the scalar path.

**The thread pool** ([`pool.rs`](src/pool.rs)) keeps its workers alive.
Decoding one token is a few hundred matrix-vector products of tens of
microseconds each, so spawning threads per operation would cost more than
the work. Workers spin briefly on an epoch counter, then park on a condvar,
and take chunks of rows from an atomic counter.

**The tokenizer** ([`tokenizer/`](src/tokenizer)) reads `tokenizer.json`.
The pre-tokenizer regexes of GPT-2, Llama 3 and Qwen need `\p{L}`,
`\p{N}` and a `(?!\S)` lookahead, so instead of a regex engine they are
hand-written matchers with the same leftmost-first semantics, selected by
the regex's text. Unicode classes and NFC data are generated from the
Unicode database by [`scripts/gen_unicode.py`](scripts/gen_unicode.py).
BPE merges with a min-heap of (rank, position), as the reference does.

One discrepancy came up: for Llama-2-style tokenizers with
`"legacy": false`, transformers' `AutoTokenizer` overrides `tokenizer.json`
and adds the `▁` prefix only at the very start of the input, never after a
special token. The plain `tokenizers` library doesn't. wick matches whichever
behaviour the model directory asks for, which is how its TinyLlama chat
prompts come out identical.

**Chat formats** ([`chat.rs`](src/chat.rs)) are recognised from the
tokenizer's special tokens (ChatML, Llama 3, Zephyr) instead of running the
Jinja template, and are checked against `apply_chat_template` output. That
includes each family's default system prompt and Qwen3's thinking switch.

## Testing

```bash
cargo test --release                       # everything; skips models you haven't downloaded
WICK_SCALAR=1 cargo test --release         # the same on the portable kernels
```

- **Unit tests:** the JSON parser (malformed inputs, surrogate pairs, depth
  limits), f16 conversion against every one of the 65,536 bit patterns, SIMD
  against scalar kernels, quantization error bounds, NFC, the pre-tokenizer
  splits, the pool and the sampler's distribution.
- **Transformers parity** ([`tests/hf_parity.rs`](tests/hf_parity.rs)):
  the correctness table above, for each downloaded model.
- **Tiny models** ([`tests/tiny_models.rs`](tests/tiny_models.rs)):
  random 2-layer Llama/Qwen2/Qwen3 checkpoints of about 300 KB, generated
  and scored by transformers. They cover Llama 3 RoPE scaling, Qwen2 biases,
  Qwen3 q/k norms with `head_dim ≠ hidden/heads`, f32 and bf16, tied and
  untied heads. The tests also check that feeding tokens in uneven chunks
  gives the same logits as one batch (KV cache and positions), and that
  Q8/Q4 stay close. They found two real bugs: wick was ignoring the
  transformers 5 `rope_parameters` config format, and the reference script
  itself had rounded RoPE frequencies to bf16.
- **Server** ([`tests/server.rs`](tests/server.rs)): real HTTP to a real
  model. It checks that the streamed and non-streamed replies match
  transformers, that a repeated prompt is served from the cache, stop
  sequences, and error responses.

CI runs on Ubuntu (x86-64, AVX2) and macOS (ARM, NEON). It downloads
SmolLM2-135M plus the other three models' tokenizers, and also checks that
the crate has no dependencies.

## Layout

```
src/
  main.rs          the CLI
  engine.rs        loading, sessions with prefix reuse, generation loop
  model.rs         weights, KV cache, forward pass
  kernels.rs       NEON / AVX2 / scalar dot products, norms, softmax
  quant.rs         bf16 / f16 conversion, Q8_0 / Q4_0
  pool.rs          thread pool
  tokenizer/       BPE, pre-tokenizers, Unicode tables, NFC
  chat.rs          chat formats
  sampler.rs       temperature, top-k, top-p, min-p, repetition penalty
  server.rs        HTTP + OpenAI API + SSE; ui.html is the web UI
  json.rs mmap.rs safetensors.rs config.rs
scripts/           downloads, reference fixtures, Unicode tables
tests/             parity, tiny models, server
```

## Limitations

- CPU only, and one sequence at a time: concurrent server requests queue.
- No sliding-window attention, YaRN or other long-context RoPE variants, and
  no Gemma or Phi architectures.
- Chat formats are matched by family, not by running each model's Jinja
  template.
- The KV cache is f32: 24 KB per token for Qwen2.5-0.5B.

## License

MIT
