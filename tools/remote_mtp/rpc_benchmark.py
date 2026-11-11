#!/usr/bin/env python3
"""Native Qwen MTP via GGML RPC: target-only A, host MTP B, remote MTP R.

The original llama.cpp/ggml inference engine performs ALL decoding. The local
GGUF MTP sidecar is uploaded onto the selected RPC GPU by the existing backend.
No independent draft model or HTTP worker is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

# Reuse the Phase 1 benchmark engine and avoid a second inference harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1_mtp"))
import benchmark as bench


DEVICE_LINE = re.compile(r"^\s*([A-Z]+[0-9]+):\s*(.+?)\s*$")
RPC_EVENT = re.compile(r"\]\s+([A-Z][A-Z_]+)\s+\(in:\s*", re.I)
RPC_DONE = re.compile(r"\]\s+([A-Z][A-Z_]+)\s+done in\s+(\d+(?:\.\d+)?)\s+ms", re.I)


def parse_devices(output: str) -> dict[str, str]:
    """Only accept the explicit GGML CUDA0/RPC0 names from --list-devices."""
    found = {}
    for line in output.splitlines():
        match = DEVICE_LINE.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def get_device_listing(server: Path, worker: str | None = None):
    cmd = [str(server)]
    if worker:
        cmd.extend(["--rpc", worker])
    cmd.append("--list-devices")
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    output = run.stdout + "\n" + run.stderr
    if run.returncode:
        raise RuntimeError("llama-server --list-devices failed, RPC=" + str(worker) +
                           ":\n" + output[-4000:] +
                           "\nCheck that BOTH builds enable GGML_RPC, worker is running, and TCP 50052 is reachable.")
    devices = parse_devices(output)
    if not devices:
        raise RuntimeError("No GGML devices reported:\n" + output[-4000:])
    return {"devices": devices, "output": output, "cmd": cmd}


def preflight(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*:[1-9][0-9]{0,4}", args.worker):
        raise ValueError("Worker must be a single host:port endpoint (e.g. 192.168.1.80:50052)")
    port = int(args.worker.rsplit(":", 1)[1])
    if port > 65535:
        raise ValueError("worker port must be <=65535")
    if not re.fullmatch(r"RPC[0-9]+", args.rpc_draft_device):
        raise ValueError("Only RPC devices may draft remotely; expected --rpc-draft-device RPC0")
    if not re.fullmatch(r"CUDA[0-9]+", args.primary_device):
        raise ValueError("Target GPU must be a local CUDA device; never use RPC0 for --primary-device")
    if args.primary_device == args.rpc_draft_device:
        raise ValueError("Target and draft devices may not be identical")
    if not args.server.is_file():
        raise FileNotFoundError("Missing RPC-enabled llama-server: " + str(args.server))

    local = get_device_listing(args.server)
    remote = get_device_listing(args.server, args.worker)
    if args.primary_device not in local["devices"]:
        raise RuntimeError(f"Missing local target GPU {args.primary_device}: {local['devices']}")
    expected = args.expect_primary
    if expected and expected.lower() not in local["devices"][args.primary_device].lower():
        raise RuntimeError(f"Expected local {args.primary_device} to contain {expected!r}, "
                           f"got {local['devices'][args.primary_device]!r}")
    if args.rpc_draft_device not in remote["devices"]:
        raise RuntimeError(f"Missing {args.rpc_draft_device}. Devices after --rpc: "
                           f"{remote['devices']}. Check worker CUDA GPU, RPC build, and server.")
    description = remote["devices"][args.rpc_draft_device]
    if args.worker not in description:
        raise RuntimeError(f"{args.rpc_draft_device} is from {description!r}, not worker {args.worker!r}; "
                           "refusing to draft on a different RPC worker.")
    if args.primary_device not in remote["devices"]:
        raise RuntimeError("Local target GPU disappeared after registering RPC. Refusing load.")
    return {"local": local, "with_rpc": remote,
            "target_device": args.primary_device, "draft_device": args.rpc_draft_device,
            "worker": args.worker, "target_on_rpc": False}


def build_command(a, mode: str, ctx: int, depth: int, port: int):
    if mode == "R":
        # Reuse Phase 1 mode C's separate MTP sidecar/device selection,
        # substituting the remote RPC GPU for the eGPU.
        cmd = bench.build_command(a, "C", ctx, depth, port)
        # IMPORTANT: --rpc must run before --spec-draft-device is parsed.
        cmd[1:1] = ["--rpc", a.worker]
        if cmd[cmd.index("--device") + 1] != a.primary_device:
            raise RuntimeError("Target GPU placement changed unexpectedly")
        if cmd[cmd.index("--spec-draft-device") + 1] != a.rpc_draft_device:
            raise RuntimeError("Draft RPC placement changed unexpectedly")
        if "RPC" in cmd[cmd.index("--device") + 1]:
            raise RuntimeError("Remote GPU must not be included in target --device")
        return cmd
    if mode not in ("A", "B"):
        raise ValueError("Unknown mode: " + mode)
    return bench.build_command(a, mode, ctx, depth, port)


def make_cases(contexts, depths, modes):
    for ctx in contexts:
        if "A" in modes:
            yield bench.Case("A", ctx, 0)
        for depth in depths:
            for mode in ("B", "R"):
                if mode in modes:
                    yield bench.Case(mode, ctx, depth)


def local_vs_remote(rows):
    """Exact token-ID hash comparison B vs R under identical settings."""
    local = {(r["n_ctx"], r["depth"], r["prompt_profile"], r["max_tokens"], r["repeat"]): r
             for r in rows if r["mode"] == "B"}
    mismatches = []
    comparisons = 0
    for row in rows:
        if row["mode"] != "R":
            continue
        key = (row["n_ctx"], row["depth"], row["prompt_profile"], row["max_tokens"], row["repeat"])
        if key not in local:
            continue
        ref = local[key]
        comparisons += 1
        if row["output_sha256"] != ref["output_sha256"]:
            mismatches.append({"context": key[0], "depth": key[1], "prompt": key[2],
                               "max_tokens": key[3], "repeat": key[4],
                               "local_output": ref["output_sha256"],
                               "remote_output": row["output_sha256"],
                               "local_excerpt": ref["output_excerpt"],
                               "remote_excerpt": row["output_excerpt"]})
    return comparisons, mismatches


def add_local_remote_ratios(summary):
    """Annotate every R summary with its measured B throughput ratio, when comparable."""
    locals_by_config = {
        (item["context"], item["depth"], item["prompt"], item["max_tokens"]): item
        for item in summary if item["mode"] == "B"
    }
    # Keep a consistent CSV schema: the original summary writer uses
    # the first row's keys, which may be a target-only A row.
    for item in summary:
        item["throughput_ratio_R_to_B"] = None
        if item["mode"] != "R":
            continue
        key = item["context"], item["depth"], item["prompt"], item["max_tokens"]
        reference = locals_by_config.get(key)
        if reference is None:
            continue
        actual = item["median_accepted_output_tokens_per_sec"]
        local = reference["median_accepted_output_tokens_per_sec"]
        item["throughput_ratio_R_to_B"] = actual / local if actual is not None and local and local > 0 else None
    return summary


def rpc_trace_summary(logs: list[Path]):
    """RPC debug command counts are for WHOLE loads/runs, not per speculative iteration."""
    counts = Counter()
    timings = defaultdict(list)
    for path in logs:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            command = RPC_EVENT.search(line)
            if command:
                counts[command.group(1).upper()] += 1
            done = RPC_DONE.search(line)
            if done:
                timings[done.group(1).upper()].append(float(done.group(2)))
    return {
        "scope": "whole host-server lifetimes; includes model load and shutdown",
        "command_counts": dict(sorted(counts.items())),
        "command_timing_ms": {name: {
            "measurements": len(values), "sum": round(sum(values), 3),
            "median": round(statistics.median(values), 3)}
            for name, values in sorted(timings.items())},
        "warning": "RPC debug logs are not a per-draft-iteration latency trace; do not infer network throughput from these counts alone.",
    }


def check_target_placement(log_path: Path):
    """Look for an RPC target-model buffer before the draft-model load boundary."""
    if not log_path.is_file():
        return {"status": "unknown", "reason": "missing server log"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Only the remote draft case has both stages.
    marker = "loading draft model"
    if marker not in text:
        return {"status": "unknown", "reason": "draft model load boundary not visible in log"}
    before = text.split(marker, 1)[0]
    suspicious = [line.strip() for line in before.splitlines()
                  if re.search(r"RPC\d+.*model buffer|model buffer.*RPC\d+", line, re.I)]
    return {"status": "suspected_target_leak" if suspicious else "no_target_rpc_buffer_seen",
            "suspicious_lines": suspicious[:8],
            "caveat": "This is a log heuristic, not a proof; confirm target placement with profiling."}


def prepare_bench_args(args):
    command = ["--server", str(args.server), "--target", str(args.target),
               "--mtp", str(args.mtp), "--primary-device", args.primary_device,
               "--draft-device", args.rpc_draft_device, "--target-ngl", str(args.target_ngl),
               "--contexts", ",".join(map(str, args.contexts)),
               "--depths", ",".join(map(str, args.depths)),
               "--generations", ",".join(map(str, args.generations)),
               "--prompt-profiles", ",".join(args.prompt_profiles),
               "--repeats", str(args.repeats), "--flash-attn", args.flash_attn,
               "--output-dir", str(args.output_dir),
               "--llama-repo", str(args.llama_repo),
               "--nvidia-smi", args.nvidia_smi]
    if args.trace_rpc:
        command.extend(["--trace", "--extra-server-arg=-lv", "--extra-server-arg=5"])
    params = bench.parse_args(command)
    params.worker = args.worker
    params.rpc_draft_device = args.rpc_draft_device
    return params


def run(args):
    device_info = preflight(args)
    print("REMOTE DEVICE PREFLIGHT PASSED", json.dumps({
        "target": args.primary_device, "draft": args.rpc_draft_device,
        "endpoint": args.worker, "remote_device": device_info["with_rpc"]["devices"][args.rpc_draft_device]
    }), flush=True)
    if args.action == "check":
        print("Local devices:\n" + device_info["local"]["output"])
        print("Devices after RPC registration:\n" + device_info["with_rpc"]["output"])
        print("Preflight confirmed distinct names, not graph execution or weights on GPU.")
        return 0

    if args.action == "trace":
        args.modes = ["R"]
        args.contexts = [4096]
        args.depths = [3]
        args.generations = [128]
        args.prompt_profiles = ["short"]
        args.repeats = 1

    params = prepare_bench_args(args)
    model_info = bench.ensure_models(params)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(args.llama_repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=15)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "device_preflight": device_info,
        "model_info": model_info,
        "modes": args.modes,
        "constraints": "Target --device only includes local CUDA; remote RPC device used solely by --spec-draft-device.",
        "known_issue": "A vs B/R exact equality may fail for quantized Qwen with batched target verification; this is reported, not considered correctness proof.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows, failures, leak_checks = [], [], []
    old_rpc_debug = os.environ.get("GGML_RPC_DEBUG")
    if args.trace_rpc:
        os.environ["GGML_RPC_DEBUG"] = "3"
    try:
        for case in make_cases(args.contexts, args.depths, args.modes):
            try:
                trial = bench.run_server(params, case, args.output_dir, meta, command_builder=build_command)
                rows.extend(trial)
                if case.mode == "R":
                    audit = check_target_placement(args.output_dir / (case.name() + ".server.log"))
                    leak_checks.append({"case": case.name(), **audit})
                    if audit["status"] == "suspected_target_leak":
                        raise RuntimeError(f"Suspected target model weights on RPC: {audit['suspicious_lines']}")
            except Exception as exc:
                failures.append({"case": case.name(), "error": str(exc)})
                print("FAIL", case.name(), str(exc), file=sys.stderr, flush=True)
            finally:
                bench.save_csv(args.output_dir / "trials.csv", rows)
                (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    finally:
        if old_rpc_debug is None:
            os.environ.pop("GGML_RPC_DEBUG", None)
        else:
            os.environ["GGML_RPC_DEBUG"] = old_rpc_debug

    summary = add_local_remote_ratios(bench.summarize(rows))
    bench.save_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    baseline_mismatches = bench.compare_to_baseline(rows)
    b_r_count, b_r_mismatches = local_vs_remote(rows)
    (args.output_dir / "baseline_mismatches.json").write_text(json.dumps(baseline_mismatches, indent=2))
    (args.output_dir / "local_vs_remote_mismatches.json").write_text(json.dumps(b_r_mismatches, indent=2))
    (args.output_dir / "target_placement_audit.json").write_text(json.dumps(leak_checks, indent=2))
    trace_logs = sorted(args.output_dir.glob("R_*.server.log"))
    trace_report = rpc_trace_summary(trace_logs)
    (args.output_dir / "rpc_trace_summary.json").write_text(json.dumps(trace_report, indent=2))
    print("\nA: target-only; B: local native MTP; R: native MTP on RPC0")
    print("mode ctx depth prompt output trials median_output_tok/s acceptance speedup_vs_A remote/local")
    for row in summary:
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(row["mode"], row["context"], row["depth"], row["prompt"], row["max_tokens"],
              row["trials"], fmt(row["median_accepted_output_tokens_per_sec"]),
              fmt(row["acceptance_rate"]), fmt(row["speedup_vs_A"]),
              fmt(row.get("throughput_ratio_R_to_B")))
    print("CASE FAILURES:", len(failures),
          "| A vs B/R mismatches (known unresolved upstream behavior):", len(baseline_mismatches),
          "| B vs R comparable pairs:", b_r_count,
          "| B vs R mismatches:", len(b_r_mismatches))
    print("RPC trace diagnostics:", args.output_dir / "rpc_trace_summary.json")
    print("Target placement audit:", args.output_dir / "target_placement_audit.json")
    print("Results:", args.output_dir.resolve())
    if failures or b_r_mismatches or any(x["status"] == "suspected_target_leak" for x in leak_checks):
        return 2
    if baseline_mismatches:
        print("WARNING: target-only vs MTP output differs. Remote transport may be functional, "
              "but exact greedy correctness is still NOT proven.")
    if args.action != "trace" and "R" in args.modes and not any(row["mode"] == "R" for row in rows):
        return 2
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--action", choices=("check", "smoke", "repeat", "full", "trace"), required=True)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--worker", required=True, help="private LAN endpoint IP:PORT of Linux ggml-rpc-server")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--mtp", type=Path, required=True)
    p.add_argument("--llama-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--primary-device", default="CUDA0")
    p.add_argument("--rpc-draft-device", default="RPC0")
    p.add_argument("--expect-primary", default="5080")
    p.add_argument("--target-ngl", type=int, default=35)
    p.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    p.add_argument("--nvidia-smi", default=shutil.which("nvidia-smi") or "disabled")
    p.add_argument("--trace-rpc", action="store_true")
    args = p.parse_args(argv)
    if args.target_ngl < 0:
        p.error("target-ngl must be >= 0")
    if args.action == "smoke":
        args.contexts, args.depths, args.generations = [4096], [1, 3], [128]
        args.prompt_profiles, args.repeats, args.modes = ["short"], 1, ["A", "B", "R"]
    elif args.action == "repeat":
        # Five steady-state requests per mode, single context and draft depth.
        # Server loads happen outside the timed requests. Keep this lightweight
        # compared to the much larger full matrix.
        args.contexts, args.depths, args.generations = [4096], [3], [256]
        args.prompt_profiles, args.repeats, args.modes = ["short"], 5, ["A", "B", "R"]
    elif args.action == "full":
        args.contexts, args.depths, args.generations = [4096, 16384], [1, 2, 3, 4], [128, 512]
        args.prompt_profiles, args.repeats, args.modes = ["short", "long"], 3, ["A", "B", "R"]
    elif args.action == "trace":
        args.contexts, args.depths, args.generations = [4096], [3], [128]
        args.prompt_profiles, args.repeats, args.modes = ["short"], 1, ["R"]
        args.trace_rpc = True
    else:
        args.contexts, args.depths, args.generations = [4096], [1], [16]
        args.prompt_profiles, args.repeats, args.modes = ["short"], 1, ["A", "B", "R"]
    return args


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        sys.exit("ERROR: " + str(exc))
