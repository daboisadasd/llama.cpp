# Phase 1: native MTP with a dedicated local GPU in llama.cpp

**Native Windows / no WSL:** [Windows setup, CUDA build, downloads, and tests](WINDOWS.md). WSL2 instructions below are optional.

**Status: ready for hardware validation, not yet performance-validated.** This is the in-repository Phase 1 benchmark/test suite. **No C++ patch has been made**, because the existing MTP code appears to support an independent draft model/device; hardware verification remains required. Inspection of the forked upstream code (2026-11-10) found a path to load the native Qwen MTP sidecar independently on another CUDA GPU using **existing** flags. The local source audit in this package must pass; actual CUDA execution and correctness on the user's devices remain to be established. Phase 2 / DSpine and Phase 3 / RPC intentionally have not begun.

## Update, rebuild, rerun (after the initial setup)

No ZIP packages or manually copied scripts are needed. All Phase 1 scripts live in your fork. Each future change will arrive as Git commits to the fork; update your checkout and rebuild:

```bash
cd ~/src/llama.cpp
git pull --ff-only origin master
cmake --build build --target llama-server -j 4
cd tools/phase1_mtp
python3 -m unittest discover -v
./run_smoke.sh
# After smoke correctness: ./run_full.sh; ./run_profile.sh
```

The scripts derive `LLAMA_ROOT` from their location within the repository by default; override it only if you deliberately use a different build tree. Download the model weights once using `./download_models.sh`. Model weights and benchmark output should not be committed to the fork. Pulling updates from the fork does **not** automatically merge new changes from `ggml-org/llama.cpp`; sync upstream separately when desired and rerun `source_audit.py` after syncing.

## Code-path finding

Source paths in this llama.cpp fork (recheck the checkout before using):

- [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) parses `--spec-draft-device` into `params.speculative.draft.devices`.
- [`common/speculative.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) `common_base_params_to_speculative` forwards draft devices to the separate draft model's model params.
- `common_speculative_init_result`: when `--spec-draft-model` is passed, it calls `common_model_load(..., mparams)` and initializes an MTP context on the **separate** draft model. Without a sidecar, `draft-mtp` creates a second context *against the target model*, whose layer buffers are already assigned. Therefore `--spec-draft-device CUDA1` **without** `--spec-draft-model` is **not** sufficient to relocate MTP weights.
- In `common_speculative_impl_draft_mtp`, the target provides `h_nextn` hidden states to the draft. The code attempts a device-resident bridge with `llama_set_embeddings_nextn_dev` + `llama_set_embd_state_src`; when that is not supported it reverts to host hidden-state rows. This is a likely Thunderbolt 4 overhead source, not proof of a working inter-GPU transfer without a run.
- This does **not** split main-model layers across GPUs. `--device` is restricted to **one** CUDA device in A, B and C; `--spec-draft-device` selects the independent sidecar device.

A published upstream bug report about a September 2026 MTP draft tensor split path is [#29044](https://github.com/ggml-org/llama.cpp/issues/29044). Also see the reported quantized-target greedy differences in [#25618](https://github.com/ggml-org/llama.cpp/issues/25618). These are reasons to **test**, not assume compatibility.

## Model pair

Official [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/main):

- `Qwen3.8-27B-Q4_K_M.gguf` — 19 GB target, Qwen3.8-27B, GGUF `qwen35`.
- `mtp-Qwen3.8-27B-Q4_0.gguf` — 1.68 GB **native MTP** sidecar, not an unrelated small language model.
- Alternatives include Q8_0 target (28.6 GB), Q8_0 MTP (3.16 GB), and BF16 files. This package defaults to Q4 to reduce memory pressure.

The `inspect_gguf.py` reader scans only GGUF headers (not model weights), prints metadata including `nextn_predict_layers` and `blk.64.nextn...` tensor names if present; the benchmark rejects missing/mismatched architectures and missing MTP metadata. It does **not** verify the actual tensor values or prove target/draft vocab identity; llama.cpp's loader must do that.

**Important hardware tradeoff:** a 19 GB Q4 target cannot fit wholly into a 16 GB RTX 5080 Laptop GPU, before cache and compute buffers. The scripts use a fixed **35 target GPU layers** by default; remaining target computation can run on CPU/RAM. Select the highest common `TARGET_NGL` that fits **all** A/B/C runs. Do not move target layers to the 3070 Ti to claim a drafting speedup. Adjust CUDA device labels only after checking actual hardware mapping.

## WSL2 setup and build

1. On Windows, update the NVIDIA display driver and `wsl --update`. Inside WSL2, install a modern CUDA **toolkit** (CUDA 12.8+ supports RTX 5080's compute capability 12.0); **do not install a Linux display driver**. NVIDIA's instructions: https://docs.nvidia.com/cuda/wsl-user-guide/ . Verify that *both* the internal 5080 and TB4 eGPU are visible to CUDA inside WSL. A GPU working in Windows is not by itself proof that WSL can access it.
2. In your Ubuntu/WSL terminal, install build requirements and download latest llama.cpp:

    ```bash
    sudo apt update
    sudo apt install -y git cmake ninja-build build-essential python3 python3-venv
    nvcc --version                     # check CUDA toolkit (>= 12.8 recommended)
    mkdir -p ~/src
    git clone https://github.com/daboisadasd/llama.cpp.git ~/src/llama.cpp
    cd ~/src/llama.cpp
    git rev-parse HEAD                  # save the exact commit; master changes rapidly
    cmake -S . -B build -G Ninja \
      -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='86;120'
    cmake --build build --target llama-server -j 4
    ./build/bin/llama-server --list-devices
    ```

    If `nvcc` or a CUDA toolchain isn't installed, use NVIDIA's WSL2 CUDA installation guide rather than installing an Ubuntu Linux NVIDIA driver. The `86;120` flags build for both RTX 3070 Ti (sm_86) and RTX 5080 (sm_120); if your specific CUDA/CMake installation rejects a target, update the toolkit. The resulting `--list-devices` should show CUDA devices and GPU names. If the eGPU is missing, **stop**—do not run the split benchmark with fake or aliased devices.

3. The test tools are already in your clone under `tools/phase1_mtp/`. Download models to disk with at least ~25 GB free:

    ```bash
    python3 -m venv ~/venvs/llama-models
    source ~/venvs/llama-models/bin/activate
    pip install -U huggingface_hub
    cd ~/src/llama.cpp/tools/phase1_mtp
    ./download_models.sh
    ```

    Or download the two matching GGUF files manually from the official repo into `$HOME/models/Qwen3.8-27B/`.

4. Audit actual checkout and run offline tests before GPU work:

    ```bash
    cd ~/src/llama.cpp/tools/phase1_mtp
    python3 -m unittest discover -v
    python3 source_audit.py ~/src/llama.cpp
    ~/src/llama.cpp/build/bin/llama-server --list-devices
    ```

    A failed audit means the upstream source has diverged from the code paths reviewed. It does not necessarily mean the newest llama.cpp is broken; **inspect the changed implementation** before proceeding.

## Exact A/B/C launch examples (manually)

These assume the `--list-devices` output maps `CUDA0` to the laptop RTX 5080 and `CUDA1` to the 3070 Ti. All commands intentionally use `--fit off` and `--n-gpu-layers 35` so llama.cpp cannot silently change model offload per scenario. Run **one server at a time**, on port 8080:

```bash
TARGET="$HOME/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf"
MTP="$HOME/models/Qwen3.8-27B/mtp-Qwen3.8-27B-Q4_0.gguf"
SERVER="$HOME/src/llama.cpp/build/bin/llama-server"

# A. no speculative decoding; target only on CUDA0 (+ CPU spill)
"$SERVER" -m "$TARGET" --device CUDA0 -ngl 35 -c 4096 \
  --flash-attn on --fit off --spec-type none

# B. target on CUDA0, native MTP sidecar also on CUDA0
"$SERVER" -m "$TARGET" --device CUDA0 -ngl 35 -c 4096 \
  --flash-attn on --fit off --spec-type draft-mtp \
  --spec-draft-model "$MTP" --spec-draft-device CUDA0 \
  --spec-draft-ngl all --spec-draft-n-max 3

# C. target on CUDA0, native MTP sidecar exclusively on CUDA1
"$SERVER" -m "$TARGET" --device CUDA0 -ngl 35 -c 4096 \
  --flash-attn on --fit off --spec-type draft-mtp \
  --spec-draft-model "$MTP" --spec-draft-device CUDA1 \
  --spec-draft-ngl all --spec-draft-n-max 3
```

Verify each startup log confirms the selected **separate MTP GGUF**, a working speculative context, correct CUDA buffer assignments, and zero target layers on CUDA1. Merely seeing both GPUs in `nvidia-smi` is **not** proof of where the MTP kernels ran. Run Nsight Systems and correlate named CUDA devices and kernels. If C cannot fit, reduce `-ngl` identically in **A, B and C**, or investigate the standalone sidecar memory requirement.

## Automated correctness/throughput tests

```bash
cd ~/src/llama.cpp/tools/phase1_mtp

# smoke: A, B(depth 1,3), C(depth 1,3), 4096 context, 128 output tokens
./run_smoke.sh

# full: 4096/16384 contexts, depths 1/2/3/4, 128/512 output,
# short/long prompts, 3 repetitions per case
./run_full.sh
```

The default variables can be overridden:

```bash
LLAMA_ROOT="$HOME/src/llama.cpp" \
MODEL_DIR="$HOME/models/Qwen3.8-27B" \
PRIMARY_DEVICE=CUDA0 DRAFT_DEVICE=CUDA1 TARGET_NGL=35 \
./run_smoke.sh
```

If `--list-devices` shows a **reversed** mapping, use `PRIMARY_DEVICE=CUDA1 DRAFT_DEVICE=CUDA0` (or the exact names displayed). Benchmark preflight refuses absent CUDA names and unexpected GPU identities: expected target name contains `5080`, dedicated GPU contains `3070 Ti`. In `benchmark.py` you can override checks with `--expect-primary ... --expect-draft ...` when testing future machines. If primary and secondary are the same name, scenario C is rejected.

For a custom benchmark:

```bash
python3 benchmark.py \
  --server "$HOME/src/llama.cpp/build/bin/llama-server" \
  --target "$HOME/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf" \
  --mtp "$HOME/models/Qwen3.8-27B/mtp-Qwen3.8-27B-Q4_0.gguf" \
  --llama-repo "$HOME/src/llama.cpp" \
  --primary-device CUDA0 --draft-device CUDA1 --target-ngl 35 \
  --contexts 4096,16384 --depths 1,2,3,4 \
  --generations 128,512 --prompt-profiles short,long --repeats 3 \
  --output-dir ./results_full
```

**Collection and interpretation**:

- `summary.csv`: median *actual emitted output* tokens per second, `draft_n_accepted / draft_n` acceptance, A-normalized speedup, model prefill/generation timing. Do not confuse accepted output throughput with draft generation throughput.
- `trials.csv`: request timings, prompt token counts, acceptance, wall-clock, output hash; `mismatches.json` contains deterministic greedy-output differences. Any mismatch makes the program exit nonzero unless `--allow-divergence` is explicitly given for diagnosis. Small quantization differences or backend numerical changes can cause differences, but **must be investigated**; they are not silently waived.
- `*.outputs.json`: complete text, token IDs (when returned), and server timing objects for each trial. `*.server.log`: raw model-load, CUDA memory, and speculative initialization logs. `*.command.json`: the exact launch command.
- `*.gpu.csv`, `*.gpu_summary.json`: sampled GPU memory and utilization. WSL2 driver/NVML limitations may leave utilization empty; a fallback query still attempts to capture memory. Treat empty counters as **unavailable**, not zero.
- `manifest.json`: exact binary version, repository commit when supplied, model metadata, modes, device identities, test arguments. `failures.json` records non-working configurations without inventing numbers.

The long-prompt test **approximates** a fraction of the allocated context using repeated English text; `prompt_n` records actual tokenized size. Thus 16,384 allocated context does not mean the prompt filled 16,384 tokens. To test nearly-full contexts, modify `prompt_for()` based on actual `prompt_n`, leaving room for the generation tokens.

Greedy settings (`temperature=0`, `top_k=1`, fixed seed, `cache_prompt=false`) are used so outputs are comparable. Repetition within a model configuration and reference matching against A are checked. For a rigorous correctness case set, also compare token IDs with representative formatted chat, reasoning-on/off and code prompts; then test with larger sample sets.

## Prove GPU placement and diagnose transfer costs

Install NVIDIA Nsight Systems in a WSL2-compatible environment; it must provide the `nsys` CLI. Trace A/B/C separately from timed tests (tracing can disturb speed):

```bash
./run_profile.sh
# Should create:
# results_profile/A_ctx4096_depth0.nsys-rep
# results_profile/B_ctx4096_depth3.nsys-rep
# results_profile/C_ctx4096_depth3.nsys-rep
```

View each `.nsys-rep` in Nsight Systems, assign trace GPU IDs by device names. Expected qualitative proof: A kernels on target CUDA0 only; B's MTP computations on CUDA0; C adds **real drafting compute kernels** on CUDA1 while target **layer** kernels continue on CUDA0. Examine CUDA `Memcpy`, scheduler gaps, TB4/host-staged transfers, and kernels on each device. Startup CUDA calls do not by themselves prove draft *inference* occurred on the eGPU—focus on the generation period. Also examine `- target h rows on the device: yes/no` in verbose MTP logs; if `no`, host row fallback is a possible contributor to latency. Upstream `common_speculative_print_stats()` exposes cumulative `dur(b,g,a)` in trace logs (begin/draft/accept), including draft time spent in host-side scheduling and synchronization. Extract differences between snapshots with:

```bash
python3 spec_log_report.py results_profile/B_ctx4096_depth3.server.log \
  results_profile/C_ctx4096_depth3.server.log
```

This is **not** a pure GPU draft-kernel latency or the target verification latency; inspect Nsight Systems to isolate those spans.

If `nsys export` supports SQLite in your installed version, convert reports:

```bash
nsys export --type sqlite --output results_profile/A.sqlite \
  results_profile/A_ctx4096_depth0.nsys-rep
nsys export --type sqlite --output results_profile/B.sqlite \
  results_profile/B_ctx4096_depth3.nsys-rep
nsys export --type sqlite --output results_profile/C.sqlite \
  results_profile/C_ctx4096_depth3.nsys-rep
python3 profile_report.py results_profile/A.sqlite \
  results_profile/B.sqlite results_profile/C.sqlite
```

`profile_report.py` reports GPU kernel counts, **summed** kernel times (not wall-clock latencies), total CUDA memcpy bytes, and unknown/missing CUPTI tables. It uses `deviceId` from the CUPTI trace, which must be correlated with actual GPU names. Totals include model load: use the timeline to isolate decode and measure per-iteration draft / target verification / transfer latency. A plain llama-server timing response **does not** expose those as separate fields. Capturing them accurately might require a minimal follow-up instrumentation patch, **only if Phase 1 profiling establishes that they cannot be separated**.

### Troubleshooting priorities

1. `CUDA1` absent in WSL2 → fix Windows driver, WSL update, TB4 eGPU enumeration. Do not claim a split-GPU result; do not substitute `CUDA0` or target-layer sharding.
2. RTX 5080 kernels not buildable → use a CUDA Toolkit that supports sm_120; confirm CMake CUDA architecture list. RTX 3070 Ti is sm_86. Don't install a Linux NVIDIA driver inside WSL2.
3. Target fails with OOM → reduce `TARGET_NGL` for **all** A/B/C, perhaps reduce context for smoke only. If only B fails, also account for draft-sidecar VRAM on CUDA0; use a common target layer count fitting both A and B.
4. `mtp-*.gguf` fails to initialize → review `*.server.log`; check target/draft metadata and `common/speculative.cpp` and inspect upstream [#29044](https://github.com/ggml-org/llama.cpp/issues/29044). Do not automatically switch to `draft-simple`, `DFlash`, or a different model.
5. B/C have zero draft count → speculative path may not have been initialized, even if the server returned text. The benchmark marks this as failure.
6. B/C token IDs differ from A → keep `mismatches.json` and raw results. Verify prompt, sampler settings, quantization, identical target offload, and updated upstream regression reports. A quantitative mismatch on different kernels can be numerical, but do not assume verification is correct without investigating.
7. C slower than B → compare device rows `yes/no`, prefill, synchronized memcopies and CUDA streams. On an eGPU, MTP's ~1.7 GB weights fit, but every draft step may incur expensive hidden-state transfer and synchronization; calculate meaningful **emitted** output tokens/sec, not draft tokens/sec.

## Phase 1 acceptance gates

- [ ] Source audit passes on the exact chosen llama.cpp commit.
- [ ] RTX 5080 and RTX 3070 Ti both appear as different CUDA devices in WSL2.
- [ ] A, B, C all start and run the same quantized target with identical CPU/GPU-layer placement.
- [ ] Nsight traces prove the target runs only on the primary GPU (+ permitted CPU spill) and MTP kernels run on the selected draft GPU.
- [ ] Correct output: acceptable token-ID comparison, draft counters > 0 in B/C, no cache desynchronization, no crashes.
- [ ] Throughput/acceptance benchmark matrix completed with repeated trials and data files.
- [ ] Bottlenecks identified: host vs device hidden-state transfer, draft iteration time, verification time, GPU idle gaps, and eGPU bandwidth/synchronization.
- [ ] Explicit conclusion supported by measured throughput comparing A/B/C.

**Remain in Phase 1 until these gates pass.** No Phase 2 DSpine or Phase 3 RPC functionality is implemented or claimed here.

## What was tested for this deliverable

The development container has **no accessible CUDA GPUs**, **no CUDA toolchain** and **no model GGUF**. The in-repo source was checked through the authenticated GitHub connector, not compiled in the container. Accordingly there is **no claim of compiling or running llama.cpp**, no real GPU trace, no fabricated perf samples. Python-only tests create tiny synthetic GGUF headers and synthetic Nsight SQLite rows to exercise parsers and command/metric construction. They are validation of the *tools*, **not** the model inference engine. Results will exist only when you run the scripts on the two-GPU WSL2 machine.