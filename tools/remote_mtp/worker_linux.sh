#!/usr/bin/env bash
# Native GGML RPC worker for a dedicated MTP accelerator (Linux + CUDA).
set -euo pipefail
ACTION="${1:-help}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${WORKER_BUILD_DIR:-$ROOT/build-rpc-linux}"
RPC_BIN="$BUILD_DIR/bin/ggml-rpc-server"
RPC_PORT="${RPC_PORT:-50052}"
WORKER_GPU="${WORKER_GPU:-CUDA0}"
RPC_BIND_IP="${RPC_BIND_IP:-}"
BUILD_JOBS="${BUILD_JOBS:-4}"

case "$ACTION" in
  build)
    command -v cmake >/dev/null || { echo 'Install CMake and C++ build tools.' >&2; exit 2; }
    command -v nvcc >/dev/null || { echo 'Install the NVIDIA CUDA Toolkit (nvcc).' >&2; exit 2; }
    flags=(-S "$ROOT" -B "$BUILD_DIR" -DGGML_CUDA=ON -DGGML_RPC=ON
           -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF)
    if [[ -n "${WORKER_CUDA_ARCH:-}" ]]; then
      flags+=("-DCMAKE_CUDA_ARCHITECTURES=$WORKER_CUDA_ARCH")
    fi
    cmake "${flags[@]}"
    cmake --build "$BUILD_DIR" --target ggml-rpc-server --parallel "$BUILD_JOBS"
    "$RPC_BIN" --help
    ;;
  check)
    [[ -x "$RPC_BIN" ]] || { echo "ERROR: first run $0 build" >&2; exit 2; }
    nvidia-smi -L
    "$RPC_BIN" --help
    echo "Selected worker GPU: $WORKER_GPU"
    git -C "$ROOT" rev-parse HEAD
    ;;
  serve|profile)
    [[ -x "$RPC_BIN" ]] || { echo "ERROR: first run $0 build" >&2; exit 2; }
    [[ -n "$RPC_BIND_IP" ]] || {
      echo "ERROR: set RPC_BIND_IP to the worker's private LAN address. Never publicly expose the RPC port." >&2
      exit 2
    }
    if [[ "$ACTION" == profile ]]; then
      command -v nsys >/dev/null || { echo 'Install NVIDIA Nsight Systems for worker profiling.' >&2; exit 2; }
      mkdir -p "$ROOT/tools/remote_mtp/results_worker"
      exec nsys profile --trace=cuda,nvtx --sample=none \
        "--output=$ROOT/tools/remote_mtp/results_worker/rpc_worker" \
        --force-overwrite=true "$RPC_BIN" \
        --host "$RPC_BIND_IP" --port "$RPC_PORT" --device "$WORKER_GPU"
    fi
    echo "Starting native GGML RPC server on $RPC_BIND_IP:$RPC_PORT, exposing $WORKER_GPU"
    echo "Worker HEAD: $(git -C "$ROOT" rev-parse HEAD)"
    exec "$RPC_BIN" --host "$RPC_BIND_IP" --port "$RPC_PORT" --device "$WORKER_GPU"
    ;;
  monitor)
    command -v nvidia-smi >/dev/null || { echo 'nvidia-smi not installed.' >&2; exit 2; }
    mkdir -p "$ROOT/tools/remote_mtp/results_worker"
    OUT="$ROOT/tools/remote_mtp/results_worker/gpu_monitor.csv"
    echo 'timestamp_utc,index,uuid,name,memory_used_mib,gpu_util_pct,memory_util_pct' >"$OUT"
    echo "Writing worker GPU samples to $OUT (Ctrl+C to stop)"
    while true; do
      now="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
      nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits | while IFS= read -r line; do
          printf '%s,%s\n' "$now" "$line"
        done >>"$OUT"
      sleep .5
    done
    ;;
  update)
    git -C "$ROOT" pull --ff-only origin master
    "$0" build
    ;;
  *)
    cat <<'EOF'
Linux worker: ./tools/remote_mtp/worker_linux.sh ACTION
  build     compile native ggml-rpc-server with CUDA and RPC
  check     show GPUs, server usage and git commit
  serve     run native worker on trusted LAN
  monitor   capture worker GPU VRAM and utilization to CSV
  profile   run worker under NVIDIA Nsight Systems
  update    pull from fork and rebuild the worker

Before serve or profile:
  export RPC_BIND_IP=192.168.1.80  # worker's PRIVATE LAN IP, replace
Optional:
  export RPC_PORT=50052 WORKER_GPU=CUDA0 WORKER_CUDA_ARCH=86
  export GGML_RPC_DEBUG=2
RPC is unauthenticated: firewall it to the trusted host IP.
EOF
    ;;
esac
