# Dedicated native MTP on a Linux GPU through GGML RPC

This experiment uses the original llama.cpp decoding engine. The host runs the
target model and verification, while a Linux ggml-rpc-server exposes its
CUDA GPU as RPC0 for the native Qwen3.8 MTP sidecar only.

**Current status:** Source-level placement path found; end-to-end remote MTP
has NOT yet been compiled, run or profiled on both machines. Do not mistake
the successful local eGPU test for an RPC result.

## Network security

The upstream [GGML RPC server](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
is explicitly proof-of-concept software with no authentication. Run it only
on a trusted wired LAN or in a protected tunnel, bind to the Linux worker's
PRIVATE LAN IP, and firewall port 50052 to the host's PRIVATE IP. Never
forward it on a router or expose it to the public Internet.

## 1. Linux machine: worker

Install the NVIDIA Linux driver, CUDA Toolkit, CMake, git, and C/C++ build
tools. Both nvcc and nvidia-smi should work. Then:

~~~bash
mkdir -p ~/src
git clone https://github.com/daboisadasd/llama.cpp.git ~/src/llama.cpp
cd ~/src/llama.cpp
bash ./tools/remote_mtp/worker_linux.sh build
bash ./tools/remote_mtp/worker_linux.sh check
~~~

To build specifically for your Linux GPU, optionally set WORKER_CUDA_ARCH
before the build; for example 86 for Ampere GPUs or 120 for Blackwell.

Obtain the worker's actual private LAN IP using ip -brief address. In one
terminal, using an illustrative address to be replaced with yours:

~~~bash
cd ~/src/llama.cpp
export RPC_BIND_IP=192.168.1.80
export RPC_PORT=50052
export WORKER_GPU=CUDA0
export GGML_RPC_DEBUG=2
bash ./tools/remote_mtp/worker_linux.sh serve
~~~

Leave that command running. To monitor GPU memory and kernel activity in
another terminal:

~~~bash
cd ~/src/llama.cpp
bash ./tools/remote_mtp/worker_linux.sh monitor
~~~

If UFW manages the firewall, restrict access to only your host IP, e.g.:

~~~bash
sudo ufw allow from 192.168.1.10 to any port 50052 proto tcp
~~~

Replace both example IPs; review existing firewall rules to ensure the worker
is not exposed to untrusted systems.

No GGUF model files are needed on the Linux worker for **generic GGML RPC**.
The Windows llama.cpp host loads the local MTP GGUF and transfers the draft
weights over the RPC connection on startup. The target GGUF is not sent to
the worker by configuration. The GGML RPC server runs graphs; it is not
another independent llama.cpp inference engine or HTTP endpoint.

## 2. Windows machine: target and verification

You can unplug your Thunderbolt eGPU. Use Developer PowerShell for VS 2022
on the main Windows machine. This uses your existing cloned fork and model
files from the previous Phase 1 experiment.

~~~powershell
cd C:\src\llama.cpp
git pull --ff-only origin master
.\tools\remote_mtp\windows.ps1 -Action Build
.\tools\remote_mtp\windows.ps1 -Action Test
~~~

The Build step reconfigures your existing build-win tree with
-DGGML_RPC=ON and rebuilds llama-server. It uses -DCMAKE_CUDA_ARCHITECTURES=120
by default (RTX 5080 and RTX 5070 Ti); override -CudaArchitectures 86
if the Windows target is an Ampere GPU.

With the Linux worker running and reachable from Windows, replace the
example worker IP below with yours:

~~~powershell
.\tools\remote_mtp\windows.ps1 -Action Check -Worker 192.168.1.80:50052
~~~

This registers RPC BEFORE listing devices. You should see a distinct
device such as RPC0: 192.168.1.80:50052. If it is missing, stop and send
the output: troubleshoot firewall, network, worker version and the
GGML_RPC host build.

When the check passes:

~~~powershell
.\tools\remote_mtp\windows.ps1 -Action Smoke -Worker 192.168.1.80:50052
~~~

The smoke run compares:

- A: target-only on the host CUDA GPU.
- B: native MTP sidecar on the same host CUDA GPU.
- R: target on host CUDA GPU, native MTP sidecar on the Linux RPC0 GPU.

It uses fixed host target offload, equal sampling parameters and quantization,
and reuses Phase 1's trial recording, output checks and GPU metrics. If B
is out of VRAM, lower the offload budget for all modes, e.g. by adding
-TargetNgl 28 to the Smoke command. It never intentionally shards target
layers onto the remote worker: target --device is explicitly CUDA0.

The same input GGUFs can be kept on your Windows host; no redownload.
The 1.68 GB Q4_0 MTP sidecar requires more VRAM at runtime for caches and
compute. A GPU with just 2 GB VRAM may not be enough. Verify worker VRAM.

For your eventual RTX 5070 Ti target host, substitute
-ExpectPrimary '5070 Ti' if needed. GPU names and CUDA indexes must
be checked on the actual host.

## 3. Results, correctness, and traces

Outputs appear in tools/remote_mtp/results_smoke_fa_on:

- summary.csv / summary.json: useful output tokens per second and draft acceptance.
- failures.json: errors including RPC connection/model initialization failures.
- baseline_mismatches.json: A versus B/R differences (quantized numerical
  divergence already observed locally; not considered resolved).
- local_vs_remote_mismatches.json: B versus R token-hash differences.
- target_placement_audit.json: conservative log scan for unwanted target
  RPC model buffers before the MTP sidecar is loaded.
- Each run's .server.log, .command.json, .gpu.csv and .outputs.json.

**Important:** An exact B/R match is a useful transport control, not
proof of mathematically exact target-only decoding or CUDA placement.
For conclusive placement, profile the worker under Nsight Systems and
inspect the target allocation and worker kernels.

With a working remote smoke test:

~~~powershell
.\tools\remote_mtp\windows.ps1 -Action Trace -Worker 192.168.1.80:50052
~~~

The trace action enables GGML_RPC_DEBUG=3 and verbose logging for a
remote-only run. The rpc_trace_summary.json counts RPC commands seen
in logs; totals include startup, not just generation. Large traces are
not reliable speed benchmarks. Logs are needed to determine if
fine-grained graph execution causes too many round trips.

For CUDA kernel tracing on Linux, stop the ordinary worker, install
Nsight Systems (nsys), and run:

~~~bash
cd ~/src/llama.cpp
export RPC_BIND_IP=192.168.1.80
bash ./tools/remote_mtp/worker_linux.sh profile
~~~

Then repeat a Windows host RPC smoke/trace while the profiler worker
is running. Inspect its nsys-rep output to prove draft GPU execution.

For longer/repeated A/B/R benchmarks:

~~~powershell
.\tools\remote_mtp\windows.ps1 -Action Full -Worker 192.168.1.80:50052
~~~

This runs several draft depths, contexts, lengths and repetitions. It
is deliberately more extensive than the initial smoke run.

## 4. Lightweight repeatability benchmark

The first A/B/R smoke used only one generation per mode. It found roughly
10.392 vs 10.253 output tokens/s at depth 1 and 14.533 vs 14.376 tokens/s
at depth 3 (local vs remote). These near-equal measurements are **not**
proof of negligible network latency: the large, partly CPU-offloaded target
can mask drafting and RPC delays.

Before running the much larger Full matrix, use this targeted repeated
comparison with the same target GPU layers and model quantization:

~~~powershell
cd C:\src\llama.cpp
git pull --ff-only origin master
.\tools\remote_mtp\windows.ps1 -Action Test
.\tools\remote_mtp\windows.ps1 -Action Repeat -Worker 192.168.1.13:50052
~~~

Repeat runs 5 requests at 4096 context and depth 3 for A, B and R with
256-token budgets. It saves fresh results to
tools/remote_mtp/results_repeat_fa_on without overwriting the original
smoke. Reported throughput is the **median across five requests** for
each mode, and the R summary includes throughput_ratio_R_to_B. It still
cannot isolate RPC latency from kernel time; for that inspect Trace and
GPU profiler results.

For raw RPC command counts and dispatcher timings, run Trace. The
rpc_trace_summary.json currently counts messages over the entire
process lifetime; numbers include MTP weight uploads, graph setup and
generated tokens, so they are not per-token network costs.

The Linux worker is unauthenticated: keep its firewall restricted to the
Windows host's trusted private IP (currently 192.168.1.15).

## 5. Updating both machines

Windows host:

~~~powershell
cd C:\src\llama.cpp
.\tools\remote_mtp\windows.ps1 -Action Update
~~~

Linux worker (stop the running RPC process first):

~~~bash
cd ~/src/llama.cpp
bash ./tools/remote_mtp/worker_linux.sh update
bash ./tools/remote_mtp/worker_linux.sh serve
~~~

Keep both ends on the same fork revision whenever possible. No ZIP files,
and no model redownloads are needed for source-only changes.

## 6. What to send back

First send the Windows -Action Check output and the Linux worker startup
log. If Smoke completes, share summary.json, failures.json,
local_vs_remote_mismatches.json and the R_*.server.log if something fails.
No model weights are required for diagnosis.

This is the first attempt using the **existing GGML RPC graph execution**
mechanism. If measurements show excessive graph/network round trips,
the next step is a coarse-grained drafting operation using llama.cpp's
existing RPC connection and device management, not an external HTTP server.
