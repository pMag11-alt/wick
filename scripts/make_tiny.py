#!/usr/bin/env python3
"""Build tiny random models with transformers and record their outputs.

Each fixture in tests/fixtures/tiny/<name>/ is a real Hugging Face model
directory (config.json + model.safetensors) a few hundred KB in size, plus
expected.json with float32 logits for every position of a fixed token
sequence and a greedy continuation. They cover what the big models do not
all exercise in CI: Llama 3 RoPE scaling, Qwen2 biases, Qwen3 q/k norms with
head_dim != hidden / heads, f32 and bf16 checkpoints, tied and untied heads.

    .venv/bin/python scripts/make_tiny.py
"""

import json
import os

import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                          Qwen3Config, Qwen3ForCausalLM)

ROOT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "tiny")
COMMON = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
              rms_norm_eps=1e-5, initializer_range=0.25, bos_token_id=1, eos_token_id=2)

MODELS = {
    "llama3-scaled-f32": (LlamaForCausalLM, LlamaConfig(
        **COMMON, rope_theta=50000.0, tie_word_embeddings=False,
        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 32}), torch.float32),
    "qwen2-bias-bf16": (Qwen2ForCausalLM, Qwen2Config(
        **COMMON, rope_theta=1e6, tie_word_embeddings=True), torch.bfloat16),
    "qwen3-qknorm-f32": (Qwen3ForCausalLM, Qwen3Config(
        **COMMON, head_dim=32, rope_theta=1e6, tie_word_embeddings=True), torch.float32),
}


def main():
    torch.manual_seed(1234)
    tokens = torch.randint(3, 96, (48,)).tolist()
    for name, (cls, cfg, dtype) in MODELS.items():
        model = cls(cfg)
        with torch.no_grad():
            # Random norm weights and biases, so a bug in either shows up.
            for pname, p in model.named_parameters():
                if "norm" in pname:
                    p.copy_(0.5 + torch.rand_like(p))
                elif pname.endswith("bias"):
                    p.copy_(torch.randn_like(p) * 0.5)
        model = model.to(dtype).eval()
        out = os.path.join(ROOT, name)
        model.save_pretrained(out, safe_serialization=True)
        # Score the checkpoint as saved, reloaded in float32. (Casting the
        # in-memory model to bf16 and back would also round RoPE's inv_freq
        # buffer, which a real load never does.)
        ref = cls.from_pretrained(out, dtype=torch.float32).eval()
        with torch.no_grad():
            logits = ref(torch.tensor([tokens])).logits[0]
            gen = ref.generate(torch.tensor([tokens[:8]]), max_new_tokens=16, min_new_tokens=16, do_sample=False,
                               eos_token_id=None, pad_token_id=0)
        expected = {
            "tokens": tokens,
            "logits": [[round(v, 6) for v in row] for row in logits.tolist()],
            "greedy_prompt": tokens[:8],
            "greedy": gen[0, 8:].tolist(),
        }
        with open(os.path.join(out, "expected.json"), "w") as f:
            json.dump(expected, f)
        for extra in ("generation_config.json",):
            path = os.path.join(out, extra)
            if os.path.exists(path):
                os.remove(path)
        size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
        print(f"{name}: {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
