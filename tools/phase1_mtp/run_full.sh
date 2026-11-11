#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LLAMA_ROOT:=$(cd "$ROOT/../.." && pwd)}"
: "${MODEL_DIR:=$HOME/models/Qwen3.8-27B}"
: "${PRIMARY_DEVICE:=CUDA0}"
: "${DRAFT_DEVICE:=CUDA1}"
: "${TARGET_NGL:=35}"
python3 "$ROOT/source_audit.py" "$LLAMA_ROOT"
python3 "$ROOT/benchmark.py" \
  --server "$LLAMA_ROOT/build/bin/llama-server" \
  --target "$MODEL_DIR/Qwen3.8-27B-Q4_K_M.gguf" \
  --mtp "$MODEL_DIR/mtp-Qwen3.8-27B-Q4_0.gguf" \
  --llama-repo "$LLAMA_ROOT" \
  --primary-device "$PRIMARY_DEVICE" --draft-device "$DRAFT_DEVICE" \
  --target-ngl "$TARGET_NGL" \
  --contexts 4096,16384 --depths 1,2,3,4 \
  --generations 128,512 --prompt-profiles short,long \
  --repeats 3 \
  --output-dir "$ROOT/results_full"
