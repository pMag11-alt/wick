#!/usr/bin/env python3
"""Record reference outputs from Hugging Face transformers/tokenizers.

For each model directory this writes tests/fixtures/hf/<name>.json with:
  - token ids for a set of tricky strings (tokenizers library),
  - the rendered chat template (apply_chat_template),
  - float32 logits for two prompts (top-20 at the last position and the
    argmax at every position),
  - a 24-token greedy continuation of a chat prompt.

    .venv/bin/python scripts/reference.py models/SmolLM2-135M-Instruct ...
"""

import json
import os
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = [
    "Hello world",
    "Hello, world! How's it going?",
    "  leading spaces and trailing   ",
    "tabs\tand\nnewlines\n\n\nand\r\nCRLF",
    "I'm, you're, he's, we'll, they've, I'd, CAN'T, DON'T",
    "numbers 1234567890 and 3.14159 and 1,000,000",
    "x=1;y=22;z=333;w=4444",
    "def f(x):\n    return x ** 2  # square\n",
    "Ünïcödé façade naïve résumé",
    "é combining vs é precomposed, Å angstrom",
    "中文分词测试，你好世界！",
    "日本語のテキストと한국어 텍스트",
    "Привет, как дела? Добре.",
    "emoji 😀👍🏽🇮🇳 and ZWJ 👨‍👩‍👧",
    "math ∑∫√∞ ≤ ≥ ≠ ± ×",
    "URL https://example.com/a/b?c=d&e=f#g",
    "email someone@example.org",
    "<b>html</b> &amp; {json: [1, 2]}",
    "a" * 100,
    "     ",
    "\n",
    "!!!???...",
    "The <|im_start|> token inside text",
    "mixedCASE camelCase snake_case kebab-case",
    "ﬁ ligature, ½ fraction, ２ fullwidth",
    "Tab　ideographic space nbsp",
    "'s at start and end'",
    "multiple\n\n\n\nnewlines  \n  with spaces",
]

CHAT = [
    {"role": "user", "content": "Name three primary colors."},
    {"role": "assistant", "content": "Red, yellow and blue."},
    {"role": "user", "content": "And the secondary ones?"},
]

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
]


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "hf")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    for model_dir in sys.argv[1:]:
        name = os.path.basename(os.path.normpath(model_dir))
        print(f"== {name}")
        tk = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        hf_tok = AutoTokenizer.from_pretrained(model_dir)
        fixture = {"model": name, "tokenize": [], "encode_special": [], "chat": {}}

        for t in TEXTS:
            fixture["tokenize"].append({"text": t, "ids": tk.encode(t, add_special_tokens=False).ids})
        for t in TEXTS[:3]:
            fixture["encode_special"].append({"text": t, "ids": tk.encode(t, add_special_tokens=True).ids})
            assert tk.decode(tk.encode(t, add_special_tokens=False).ids) is not None

        kwargs = {}
        if "enable_thinking" in (hf_tok.chat_template or ""):
            for think in (True, False):
                fixture["chat"]["think" if think else "no_think"] = hf_tok.apply_chat_template(
                    CHAT, tokenize=False, add_generation_prompt=True, enable_thinking=think)
            kwargs = {"enable_thinking": False}
        else:
            fixture["chat"]["default"] = hf_tok.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
            with_system = [{"role": "system", "content": "You are terse."}] + CHAT
            fixture["chat"]["system"] = hf_tok.apply_chat_template(with_system, tokenize=False, add_generation_prompt=True)

        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        model.eval()
        fixture["logits"] = []
        with torch.no_grad():
            for p in PROMPTS:
                ids = tk.encode(p, add_special_tokens=True).ids
                logits = model(torch.tensor([ids])).logits[0]
                top = torch.topk(logits[-1], 20)
                fixture["logits"].append({
                    "ids": ids,
                    "argmax": logits.argmax(-1).tolist(),
                    "top_ids": top.indices.tolist(),
                    "top_logits": [round(v, 5) for v in top.values.tolist()],
                })
            chat_ids = hf_tok.apply_chat_template(CHAT, tokenize=True, add_generation_prompt=True, **kwargs)
            if not isinstance(chat_ids, list):
                chat_ids = chat_ids["input_ids"]
            gen = model.generate(torch.tensor([chat_ids]), max_new_tokens=24, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
            fixture["greedy"] = {"prompt": chat_ids, "output": gen[0, len(chat_ids):].tolist()}
        print("  greedy:", repr(hf_tok.decode(fixture["greedy"]["output"])))
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(fixture, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
