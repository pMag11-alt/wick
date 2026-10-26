#!/usr/bin/env python3
"""Perplexity with transformers, using exactly `wick ppl`'s procedure:
tokenize the whole file without special tokens, cut it into consecutive
non-overlapping windows of --ctx tokens, and score tokens 1..ctx-1 of each
window from a fresh context.

    .venv/bin/python scripts/ppl_reference.py models/Qwen2.5-0.5B-Instruct models/wiki.test.raw --windows 40
"""

import argparse
import math

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("text")
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--windows", type=int, default=0, help="limit to the first N windows")
args = ap.parse_args()

tk = Tokenizer.from_file(f"{args.model}/tokenizer.json")
ids = tk.encode(open(args.text).read(), add_special_tokens=False).ids
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
n = len(ids) // args.ctx
if args.windows:
    n = min(n, args.windows)
nll, count = 0.0, 0
with torch.no_grad():
    for w in range(n):
        x = torch.tensor([ids[w * args.ctx:(w + 1) * args.ctx]])
        logp = torch.log_softmax(model(x).logits[0].double(), -1)
        nll -= logp[:-1].gather(1, x[0, 1:, None]).sum().item()
        count += args.ctx - 1
print(f"{args.model}: perplexity {math.exp(nll / count):.4f} over {count} tokens ({n} windows of {args.ctx})")
