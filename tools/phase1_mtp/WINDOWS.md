# Native Windows Phase 1: RTX 5080 target + RTX 3070 Ti eGPU MTP

This guide runs the llama.cpp CUDA server, Qwen3.8 model, native MTP sidecar, and benchmark client **entirely on Windows**. No WSL2, Bash, Docker, or ZIP handoffs. Continue only Phase 1 until GPU placement, correct output, and measured performance have been demonstrated.

**Status:** Native Windows launch tooling is committed but **not yet tested with your GPUs**. It is an implementation to validate, not a proven speedup or proof that separate MTP GPU execution works.

## 1. Install dependencies once

On Windows, with the external 3070 Ti connected:

- Install or update the [Windows NVIDIA GPU driver](https://www.nvidia.com/drivers/). Both the laptop RTX 5080 and eGPU RTX 3070 Ti should be visible in Device Manager.
- Install [Visual Studio 2022](https://visualstudio.microsoft.com/vs/community/) (Community or Build Tools) with the **Desktop development with C++** workload, Windows SDK, and CMake tools. Use the Start menu entry **Developer PowerShell for VS 2022** when building.
- Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) **12.8 or newer** compatible with VS 2022. RTX 5080 needs compute architecture 120 and 3070 Ti uses 86. Restart the Developer PowerShell after installation so nvcc is available.
- Install [Git for Windows](https://git-scm.com/downloads/win) and [Python 3.10+ for Windows](https://www.python.org/downloads/windows/) with the Windows Python launcher py.exe.

See the official [llama.cpp CUDA build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) and [NVIDIA Windows CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

## 2. Clone your fork (once)

In **Developer PowerShell for VS 2022**, not WSL/Git Bash:

~~~powershell
New-Item -ItemType Directory -Path C:\src -Force | Out-Null
git clone https://github.com/daboisadasd/llama.cpp.git C:\src\llama.cpp
Set-Location C:\src\llama.cpp
Set-ExecutionPolicy -Scope Process Bypass
~~~

The last command allows local scripts for this PowerShell session only. If preferred, invoke individual scripts through powershell.exe -NoProfile -ExecutionPolicy Bypass -File followed by the script path.

## 3. Build and run the initial test

All steps below use native Windows executables:

~~~powershell
# Check GPU detection, CUDA compiler, Python, Git, and CMake:
.\tools\phase1_mtp\windows.ps1 -Action Check

# Build llama-server.exe for RTX 3070 Ti sm_86 and RTX 5080 sm_120:
.\tools\phase1_mtp\windows.ps1 -Action Build

# Test the benchmark tool without needing the model or GPUs:
.\tools\phase1_mtp\windows.ps1 -Action Test

# Install download helper and fetch just 2 official GGUF files:
py -3 -m pip install -U huggingface_hub
.\tools\phase1_mtp\windows.ps1 -Action Download

# Run A=no MTP, B=MTP on RTX 5080, C=MTP on RTX 3070 Ti:
.\tools\phase1_mtp\windows.ps1 -Action Smoke
~~~

The build uses a separate build-win directory. The Windows server is usually at build-win\bin\Release\llama-server.exe. Models are stored in %USERPROFILE%\models\Qwen3.8-27B. The first model download is approximately 19 GB for the target plus 1.68 GB for the MTP sidecar. Future pulls and rebuilds **do not** redownload these weights.

**IMPORTANT:** CUDA device numbering can differ between Windows and WSL. Read the output from -Action Build, which runs llama-server --list-devices. If it reports CUDA1 as the 5080 and CUDA0 as the 3070 Ti, use:

~~~powershell
.\tools\phase1_mtp\windows.ps1 -Action Smoke -PrimaryDevice CUDA1 -DraftDevice CUDA0
~~~

The runner will reject a mismatched device name instead of silently using the wrong GPU.

The target GGUF is approximately **19 GB** and is unlikely to fit entirely on an RTX 5080 Laptop GPU. These tests use fixed partial CPU offloading, defaulting to **35 GPU layers on the primary 5080** in A, B, and C. This is not target-model sharding to the 3070 Ti. If B runs out of VRAM because target and MTP share the 5080, lower the **same target offload level across all configurations**:

~~~powershell
.\tools\phase1_mtp\windows.ps1 -Action Smoke -TargetNgl 28
~~~

Do not change the GPU layer budget for only one mode. The model load itself may take time, especially if most layers must reside in CPU RAM.

## 4. More complete experiments (only after Smoke passes)

~~~powershell
.\tools\phase1_mtp\windows.ps1 -Action Full
~~~

This uses multiple context lengths, draft depths, short/long generations, and repetitions. Results are written to tools\phase1_mtp\results_windows_full, including summary.csv, summary.json, failures.json, mismatches.json, per-run server logs, and GPU utilization data. The initial smoke results go into results_windows_smoke.

To measure **actual CUDA kernel placement** rather than relying solely on GPU utilization, install [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and ensure nsys.exe is available on PATH:

~~~powershell
nsys --version
.\tools\phase1_mtp\windows.ps1 -Action Profile
~~~

Open the resulting Nsight Systems reports and confirm the intended device executes each draft phase without target-layer allocations. This is optional until basic inference works.

## 5. Every future code update

In Developer PowerShell, inside C:\src\llama.cpp:

~~~powershell
.\tools\phase1_mtp\windows.ps1 -Action Update
.\tools\phase1_mtp\windows.ps1 -Action Test
.\tools\phase1_mtp\windows.ps1 -Action Smoke
~~~

The Update action does **git pull --ff-only origin master** and rebuilds llama-server in the existing build-win configuration; no new ZIP or model download. Re-run Build only if you change CUDA/Visual Studio toolchains.

## What to send back

If a step fails, send the last visible error plus the device list from the Build step. If Smoke completes, send summary.json, failures.json, mismatches.json, and relevant server logs from results_windows_smoke. Do **not** upload the 19 GB model weights.

Native Windows might change eGPU/Thunderbolt transfer behavior relative to WSL2, but no performance conclusion can be made until the A/B/C results and CUDA profiling are collected.
