#!/usr/bin/env bash
# A/B/C run under Nsight Systems. Profiling changes performance; keep separate from timed runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LLAMA_ROOT:=$(cd "$ROOT/../.." && pwd)}"
: "${MODEL_DIR:=$HOME/models/Qwen3.8-27B}"
: "${PRIMARY_DEVICE:=CUDA0}"
: "${DRAFT_DEVICE:=CUDA1}"
: "${TARGET_NGL:=35}"
python3 "$ROOT/benchmark.py" \
  --server "$LLAMA_ROOT/build/bin/llama-server" \
  --target "$MODEL_DIR/Qwen3.8-27B-Q4_K_M.gguf" \
  --mtp "$MODEL_DIR/mtp-Qwen3.8-27B-Q4_0.gguf" \
  --llama-repo "$LLAMA_ROOT" --primary-device "$PRIMARY_DEVICE" \
  --draft-device "$DRAFT_DEVICE" --target-ngl "$TARGET_NGL" \
  --contexts 4096 --depths 3 --generations 256 \
  --modes A,B,C --repeats 1 --nsys --trace \
  --extra-server-arg=--verbosity --extra-server-arg=4 \
  --output-dir "$ROOT/results_profile"
echo 'Inspect A_ctx4096_depth0, B_ctx4096_depth3, C_ctx4096_depth3 .nsys-rep files in Nsight Systems.'
echo 'Inspect GPU identifiers and CUDA kernel/memcopy timelines, not just nvidia-smi utilization.'
