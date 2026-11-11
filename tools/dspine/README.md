# Native llama.cpp DSpine for Qwen3.8-27B — implementation work

**Status (2026-11-11): checkpoint/configuration preflight and numerical bridge
reference tests are committed, but this is NOT an executable native DSpine
backend yet.** Do not launch the checkpoint using `--spec-type draft-dflash`,
`draft-dspark`, or `draft-mtp`: those algorithms do not implement Tencent's
DSpine code bridge or serial refinement and could produce misleading output.

## Source material (pinned at inspection)

- [Tencent DSpine, main repository](https://github.com/Tencent/DSpine): tree
  `321b0fc1388c4cdf03483ab73856975f7c0c0be7` at inspection.
- [Official Qwen3.8-27B DSpine checkpoint](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16):
  ~4.05 GB BF16 safetensors, ~2.03B parameters. Officially 5-layer,
  `block_size=16`, sliding window 8192.
- DSpine implementation: `sglang_dspine/models/dspine.py`,
  `sglang_dspine/runtime/worker.py`, `sglang_dspine/kernels/dspine.py`,
  and `sglang_dspine/runtime/replayssm`.
- Current llama.cpp native DFlash/DSpark primitives:
  `src/models/dflash.cpp`, `common/speculative.cpp`, `conversion/qwen.py`.

## Why MTP's RPC method cannot directly load this checkpoint

Native MTP uses **one trained NextN head**. Tencent DSpine's draft is
**five causal attention layers** with **learned adjacent-code writes after
layers 1–5**. Its output sampler does a serial 16-way candidate refinement
using a **BF16 target `lm_head` matrix**, even when the target itself runs on
a different GPU. The upstream SGLang implementation currently requires the
unquantized row-contiguous BF16 target projection and uses ReplaySSM and
DSpark verification logic. The original model has additional code bridge
weights and sentinel tensors not represented by llama.cpp's `draft-dflash`
checkpoint/graph formats. Simply converting base DFlash layers and
ignoring the bridge **would be a different algorithm** and is not allowed.

Our previously downloaded `mtp-Qwen3.8-27B-BF16.gguf` contains a candidate
BF16 `output.weight` tensor (5120 × 248320) that may supply the remote
refinement tail, avoiding another large target-weight download. The
preflight validates that tensor's GGUF shape/type, not numerical equality
with Tencent's unquantized target output head. This still must be established.

A 12 GB 3080 Ti worker may have enough VRAM for draft weights + BF16
projection + serial tables and KV cache, but **no claim is made until
measured**. Initial estimates: ~4.05 GB draft weights, ~2.54 GB target
BF16 output head and ~0.76 GB for BF16 top-k lookup tables, plus KV/compute
state. Network transfers are separate from total VRAM.

## Run preflight on the Windows host, no WSL and no archive files

The checkpoint is NOT a runnable llama.cpp draft yet. This is the
preparation and source compatibility gate for the upcoming GGML-native port.

```powershell
cd C:\src\llama.cpp
git pull --ff-only origin master

# Run tests of DSpine model contract + reference bridge math
.\tools\dspine\windows.ps1 -Action Test

# Inspect the existing native DFlash/DSpark code and identify missing pieces
.\tools\dspine\windows.ps1 -Action SourceAudit

# Download/validate only Tencent's tiny config.json first
.\tools\dspine\windows.ps1 -Action Check

# When ready, download the official ~4.05 GB safetensors and validate
.\tools\dspine\windows.ps1 -Action Prepare
```

The `Prepare` command caches the weights under
`%USERPROFILE%\models\Qwen3.8-27B-DSpine-b16` and writes
`tools\dspine\results_preflight\checkpoint_report.json`. It validates
tensor shapes/dtypes using only the safetensors header (no 4 GB RAM copy),
including the learned code bridge and the 5 draft layers. It also checks
the candidate BF16 target lm_head in the MTP sidecar when present. The
safetensors download is **needed once**, not on every build or update.

The command does not launch or benchmark DSpine. It exits rather than
falling back to a different speculative backend.

## New: stream the full DSpine checkpoint to GGUF (experimental, not runnable)

The source checkpoint remains in safetensors. The fork now includes a
**streaming, dependency-free GGUF writer** that preserves the original tensor
names, shapes, dtypes, learned bridge, and version sentinels. It also embeds
the BF16 target output projection from the MTP BF16 GGUF already on your
Windows host, and carries over tokenizer metadata. The 27B target GGUF is not
modified. The output intentionally declares `general.architecture=dspine`,
which the current llama.cpp runtime does not implement; it must not be run
with `--spec-type draft-dflash` or `draft-dspark`.

After your existing `Prepare` step (which you already completed):

```powershell
cd C:\\src\\llama.cpp
git pull --ff-only origin master
.\\tools\\dspine\\windows.ps1 -Action Test
.\\tools\\dspine\\windows.ps1 -Action Convert
```

`Convert` reads from
`%USERPROFILE%\\models\\Qwen3.8-27B-DSpine-b16\\model.safetensors`
without loading 4 GB into memory and writes a new GGUF in that folder.
It requires the previously downloaded
`%USERPROFILE%\\models\\Qwen3.8-27B\\mtp-Qwen3.8-27B-BF16.gguf`
for the BF16 target output head and tokenizer. Expect several GB of new
storage (original ~4 GB draft plus ~2.54 GB head and metadata).
It refuses to overwrite an existing or partially converted artifact;
if an earlier conversion was interrupted, inspect/delete only the
`.partial` output before retrying. Conversion does **not** need CUDA or
rebuilding the server. Results metadata is saved in
`tools\\dspine\\results_preflight\\conversion_report.json`.

This is the first actual native artifact-construction milestone, **not** a
working DSpine graph, and not a benchmark. Do not expose RPC to untrusted
networks.

## Native GGML code bridge CPU smoke test

The first backend-agnostic learned code-bridge graph now lives in
`src/models/dspine-bridge.h`. It constructs code normalization,
anchor/precedessor selection, learned sigmoid gating, residual writes,
and next-layer normalization as GGML ops. The C++ numeric unit test
builds against **CPU GGML only**, checks identity when bridge weights
are zero, and checks one learned adjacent write without cross-block leakage.
It does **not** yet execute the real checkpoint, an entire 5-layer
DSpine drafter, or remote RPC kernels.

On Windows with Visual Studio 2022 Developer PowerShell, from the fork:

```powershell
git pull --ff-only origin master
.\tools\dspine\windows.ps1 -Action Test
.\tools\dspine\windows.ps1 -Action GraphTest
.\tools\dspine\windows.ps1 -Action SourceAudit
```

`GraphTest` uses a separate `build-dspine-bridge-win` tree and does
not change your working `build-win` server. The dedicated
[GitHub Actions preflight](https://github.com/daboisadasd/llama.cpp/actions/workflows/dspine-preflight.yml)
also runs this C++ test; it must pass before the code-bridge graph
can be wired into the model decoder.

## Test actual DSpine bridge weights on the Linux RPC GPU

The experimental `test-dspine-rpc-bridge` executable uses the real
checkpoint-derived GGUF (not synthetic weights). It opens its GGUF header
without loading the entire checkpoint into host RAM, streams **only the
learned last-layer bridge weights** to the Linux RPC GPU, and builds the
same GGML bridge graph as the CPU unit test. It sends deterministic
synthetic BF16 hidden rows through native GGML RPC and reads back the
post-write residuals and RMS values.

After the `Convert` action has successfully produced the GGUF and
the Linux `ggml-rpc-server` is running at 192.168.1.13:50052:

```powershell
cd C:\src\llama.cpp
git pull --ff-only origin master
.\tools\dspine\windows.ps1 -Action GraphTest
.\tools\dspine\windows.ps1 -Action RpcBridgeTest -Worker 192.168.1.13:50052
```

The RPC test compiles in an independent `build-dspine-rpc-win` directory
with `GGML_RPC=ON`, `GGML_CUDA=OFF`, so it requires neither WSL nor
rebuilding your working target-model server. The worker does not require
the GGUF file on its filesystem; weights are uploaded by the host.
Its model GGUF was intentionally marked as the new, **not yet registered**
`dspine` architecture. This test runs the final learned code bridge only;
it does *not* run DSpine's five causal decoder layers, serial 16-way
refinement, ReplaySSM or target verification.

Only run GGML RPC on your trusted LAN, firewall port 50052 to the
Windows host (192.168.1.15), and never expose it on the Internet.

## Native K-way serial refinement and composed proposal tail

The fork now includes src/models/dspine-refine.h: stock GGML operations
compute DSpine's K-by-K exact-sender scores for each proposed position,
argmax transition lanes, and a tiny host chain walker. The graph accepts
the trained BF16 target output projection, normalization, precomputed
TG/TSV lookup tables, last bridge's pre-write state and candidate IDs.
The transition table stays on the selected GPU/RPC device until one small
readback; no per-candidate RPC requests are required.

src/models/dspine-tail.h composes the last learned bridge, target-head
top-K proposals, predecessor candidate lattice and greedy transition
table in one GGML graph. This can be placed on CUDA/RPC once the
trained five-layer backbone and its table initialization are connected.

Four synthetic CPU C++ tests exercise the bridge, BF16 sender-table
precomputation (TG/TSV), serial refinement and the composed proposal tail.
They are executed by the GraphTest action.
These tests do NOT assert that DSpine is ready for user inference:
the actual five attention layers, precomputed learned tables, Qwen3.8
ReplaySSM verification and sampled-temperature correction are pending.

### Real-weight K=16 serial tail over RPC (experimental)

After the lightweight real-weight RpcBridgeTest passes, run its stronger
variant. This additional mode uploads the BF16 target output head already
bundled inside the converted GGUF (approximately 2.54 GB), computes TG
and TSV lookup tables using the trained bridge weights on RPC0, selects
top-16 candidates at each position, and computes a 15-token greedy
serial chain using one GGML graph:

```powershell
.\tools\dspine\windows.ps1 -Action RpcBridgeTest -Worker 192.168.1.13:50052 -Tail
```

This can use several additional gigabytes of temporary GPU memory and
upload time. The production backend will need to persist the TG/TSV
tables instead of recalculating them for every draft. The input to
this harness is deliberately synthetic BF16 hidden states: *it does
not yet exercise the five-layer DSpine backbone or the Qwen3.8 target
model verifier*. A successful run validates only the learned proposal
tail and remote GGML operations. If this test fails on an unsupported
GGML op or insufficient GPU VRAM, do not assume DSpine inference works;
preserve the printed error and worker log.

## Native implementation sequence

1. **Checkpoint import/validation** — inspect exact Tencent model/config
   tensor names and shapes (these scripts, tests). Implement GGUF conversion
   for the five DFlash-derived layers **and all bridge weights**, retaining
   checkpoint version and target-layer metadata.
2. **Bridge graph** — implement learned predecessor causal injection in the
   GGML draft graph: normalize/read code, fuse `[receiver, sender, value]`,
   apply the gated previous-row write, and RMSNorm at each layer. Compare
   intermediate tensors with the upstream DSpine implementation.
3. **Serial refinement** — load the target BF16 `lm_head` onto the drafting
   GPU; implement 16-way candidate selection and serial tables/chain,
   including correct rounding/dtype behavior and target verification.
4. **ReplaySSM and dynamic verification** — preserve Qwen3.8 GDN state on
   verification, then add confidence-based DSpark scheduling and sampled-
   probability correction, not just greedy argmax. Test state integrity.
5. **RPC integration** — reuse `--rpc` and `--spec-draft-device RPC0` to
   run draft weights and bridge entirely remotely, with target layers
   restricted to host CUDA/CPU. Profile worker CUDA kernels and RPC commands.
6. **A/B/C/D/R comparisons** — compare no spec, MTP local, MTP remote,
   DSpine local, and DSpine remote using useful tokens/sec, acceptance,
   output correctness and repeated trials. Record source revisions.

No code above should be represented as passing end-to-end DSpine inference
until its actual graph and verifier exist and run on the user's hardware.
