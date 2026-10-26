#!/usr/bin/env bash
# Download a model from the Hugging Face Hub into models/<name>/.
#
#   scripts/download.sh HuggingFaceTB/SmolLM2-135M-Instruct
#   scripts/download.sh Qwen/Qwen3-0.6B --tokenizer-only    # skip the weights
set -euo pipefail
repo=$1
files="config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors"
if [ "${2:-}" = "--tokenizer-only" ]; then
  files="config.json tokenizer.json tokenizer_config.json"
fi
dest=models/$(basename "$repo")
mkdir -p "$dest"
for f in $files; do
  [ -s "$dest/$f" ] && continue
  echo "fetching $repo/$f"
  if ! curl -fL --retry 3 -o "$dest/$f.part" "https://huggingface.co/$repo/resolve/main/$f"; then
    rm -f "$dest/$f.part"
    [ "$f" = generation_config.json ] && continue   # optional
    echo "failed: $repo/$f" >&2
    exit 1
  fi
  mv "$dest/$f.part" "$dest/$f"
done
echo "ok: $dest"
