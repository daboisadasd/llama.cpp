#!/usr/bin/env python3
"""Short, non-performance A/B/C greedy token-probability probe for native MTP.

Sends the original benchmark prompt to /completion with n_probs enabled.
Asking for probabilities can change backend sampling; do not compare speeds
or assume these results exactly reproduce the earlier no-probability run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError

import benchmark as bench


def first_differing_token(left, right):
    if not isinstance(left, list) or not isinstance(right, list):
        return None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def snapshot(result, index):
    tokens = result.get("tokens")
    probabilities = result.get("completion_probabilities")
    selected = tokens[index] if isinstance(tokens, list) and index < len(tokens) else None
    if not isinstance(probabilities, list) or index >= len(probabilities):
        return {"index": index, "selected_id": selected, "available": False}
    row = probabilities[index]
    if not isinstance(row, dict):
        return {"index": index, "selected_id": selected, "available": False}
    raw = row.get("top_logprobs", [])
    top = [dict(id=x.get("id"), token=x.get("token"), logprob=x.get("logprob"))
           for x in raw if isinstance(x, dict) and isinstance(x.get("logprob"), (int, float))]
    top.sort(key=lambda x: x["logprob"], reverse=True)
    gap = top[0]["logprob"] - top[1]["logprob"] if len(top) >= 2 else None
    return {"index": index, "selected_id": selected, "available": bool(top),
            "top_logprobs": top, "top2_logprob_gap_nats": gap}


def describe(results):
    pairs = (("A", "B"), ("A", "C"), ("B", "C"))
    data = []
    for left, right in pairs:
        if left not in results or right not in results:
            continue
        a, b = results[left], results[right]
        pos = first_differing_token(a.get("tokens"), b.get("tokens"))
        # Always capture position 3 because earlier A/B/C smoke diverged there.
        indices = sorted(set([3] + ([pos] if pos is not None else [])))
        data.append({
            "reference": left, "candidate": right,
            "same_token_ids": isinstance(a.get("tokens"), list) and isinstance(b.get("tokens"), list) and pos is None,
            "first_differing_index_zero_based": pos,
            "reference_windows": [snapshot(a, i) for i in indices],
            "candidate_windows": [snapshot(b, i) for i in indices],
        })
    return data


def run_case(params, mode, depth, output_dir, n_predict, n_probs, timeout):
    context = params.contexts[0]
    case = bench.Case(mode, context, 0 if mode == "A" else depth)
    port = bench.free_port()
    cmd = bench.build_command(params, case.mode, case.n_ctx, case.depth, port)
    (output_dir / (case.name() + ".command.json")).write_text(
        json.dumps({"argv": cmd}, indent=2), encoding="utf-8")
    log = output_dir / (case.name() + ".server.log")
    print("PROBE", case.name(), flush=True)
    process = None
    try:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                       cwd=str(params.llama_repo) if params.llama_repo else None,
                                       start_new_session=(os.name == "posix"))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{case.name()} exited {process.returncode} at startup:\n"
                                       + bench.read_log_tail(log))
                try:
                    health = bench.request_json("GET", f"http://127.0.0.1:{port}/health", timeout=2)
                    if health.get("status") == "ok":
                        break
                except (HTTPError, URLError, ConnectionError, ValueError, TimeoutError):
                    pass
                time.sleep(0.6)
            else:
                raise RuntimeError(f"{case.name()} model load timed out:\n" + bench.read_log_tail(log))
            url = f"http://127.0.0.1:{port}/completion"
            # Mirror the original benchmark warmup, including fresh-prompt decoding.
            bench.request_json("POST", url, bench.completion_payload(bench.PROMPT_SHORT, 12, params.seed),
                               timeout=timeout)
            payload = bench.completion_payload(bench.PROMPT_SHORT, n_predict, params.seed)
            payload.update({"n_probs": n_probs, "post_sampling_probs": False, "backend_sampling": False})
            response = bench.request_json("POST", url, payload, timeout=timeout)
            (output_dir / (case.name() + ".response.json")).write_text(
                json.dumps({"request": payload, "response": response}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            if not isinstance(response.get("tokens"), list):
                raise RuntimeError(f"{case.name()}: token IDs missing")
            if not isinstance(response.get("completion_probabilities"), list):
                raise RuntimeError(f"{case.name()}: completion_probabilities missing; inspect response")
            print(f"  output tokens {len(response['tokens'])}, "
                  f"top candidates at token 4: {snapshot(response, 3).get('top2_logprob_gap_nats')}",
                  flush=True)
            return response
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, type=Path)
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--mtp", required=True, type=Path)
    p.add_argument("--llama-repo", required=True, type=Path)
    p.add_argument("--primary-device", default="CUDA0")
    p.add_argument("--draft-device", default="CUDA1")
    p.add_argument("--expect-primary", default="5080")
    p.add_argument("--expect-draft", default="3070 Ti")
    p.add_argument("--target-ngl", type=int, default=35)
    p.add_argument("--flash-attn", choices=["on", "off", "auto"], default="on")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-predict", type=int, default=16)
    p.add_argument("--n-probs", type=int, default=8)
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args(argv)
    if args.depth < 1 or args.n_predict < 4 or args.n_probs < 2 or args.target_ngl < 0:
        raise ValueError("depth must be >=1, n-predict >=4, n-probs >=2, target-ngl >=0")
    if args.primary_device == args.draft_device:
        raise ValueError("primary and draft devices must differ")
    configuration = bench.parse_args([
        "--server", str(args.server),
        "--target", str(args.target),
        "--mtp", str(args.mtp),
        "--llama-repo", str(args.llama_repo),
        "--primary-device", args.primary_device, "--draft-device", args.draft_device,
        "--expect-primary", args.expect_primary, "--expect-draft", args.expect_draft,
        "--target-ngl", str(args.target_ngl), "--flash-attn", args.flash_attn,
        "--contexts", "4096", "--depths", str(args.depth),
        "--modes", "A,B,C", "--generations", str(args.n_predict),
    ])
    bench.ensure_models(configuration)
    bench.check_devices(configuration.server, configuration.primary_device,
                        configuration.draft_device,
                        configuration.expect_primary, configuration.expect_draft)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []
    for mode in ("A", "B", "C"):
        try:
            results[mode] = run_case(configuration, mode, args.depth,
                                     args.output_dir, args.n_predict, args.n_probs, args.timeout)
        except Exception as exc:
            failures.append({"mode": mode, "error": str(exc)})
            print(f"FAIL probe {mode}: {exc}", file=sys.stderr)
    summary = {
        "disclaimer": "Diagnostic probabilities may change sampler execution path; not a performance result.",
        "models": {"target": str(args.target), "mtp": str(args.mtp)},
        "settings": {"depth": args.depth, "flash_attn": args.flash_attn,
                     "n_predict": args.n_predict, "n_probs": args.n_probs,
                     "target_ngl": args.target_ngl},
        "comparisons": describe(results),
        "failures": failures,
    }
    path = args.output_dir / "greedy_probe.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["comparisons"], indent=2, ensure_ascii=False))
    print(f"Probe report: {path}")
    return 2 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")
