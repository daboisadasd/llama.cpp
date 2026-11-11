#!/usr/bin/env bash
# Download exactly the target and native MTP sidecar from ggml-org's GGUF repo.
set -euo pipefail
: "${MODEL_DIR:=$HOME/models/Qwen3.8-27B}"
: "${HF_REPO:=ggml-org/Qwen3.8-27B-GGUF}"
mkdir -p "$MODEL_DIR"
if ! command -v hf >/dev/null 2>&1; then
  echo 'Install the Hugging Face CLI: python3 -m pip install -U huggingface_hub' >&2
  echo 'Or activate a Python venv that has huggingface_hub installed.' >&2
  exit 1
fi
hf download "$HF_REPO" \
  Qwen3.8-27B-Q4_K_M.gguf \
  mtp-Qwen3.8-27B-Q4_0.gguf \
  --local-dir "$MODEL_DIR"
python3 "$(dirname "$0")/inspect_gguf.py" \
  "$MODEL_DIR/Qwen3.8-27B-Q4_K_M.gguf" \
  "$MODEL_DIR/mtp-Qwen3.8-27B-Q4_0.gguf"
