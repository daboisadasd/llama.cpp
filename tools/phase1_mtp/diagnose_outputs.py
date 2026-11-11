#!/usr/bin/env python3
"""Analyze existing Phase 1 outputs without rerunning or modifying inference.

Compare A (target-only) against B and C, and compare B against C at each
MTP depth. Report first differing token ID and neighboring decoded text.
Exit 0 on a complete analysis even when differences exist; this is diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

CASE_RE = re.compile(r"^(?P<mode>[ABC])_ctx(?P<ctx>\d+)_depth(?P<depth>\d+)\.outputs\.json$")


def first_difference(a, b):
    """Return zero-based first difference, or None when both sequences match."""
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def clip_text(s, at, radius=65):
    if at is None:
        return ""
    start = max(at - radius, 0)
    end = min(at + radius, len(s))
    return ("…" if start else "") + s[start:end] + ("…" if end < len(s) else "")


def load_outputs(root):
    outputs = {}
    case_count = 0
    for path in sorted(root.glob("*.outputs.json")):
        match = CASE_RE.match(path.name)
        if match is None:
            continue
        case_count += 1
        mode, ctx, depth = match.group("mode"), int(match.group("ctx")), int(match.group("depth"))
        if mode == "A" and depth != 0:
            raise ValueError(f"invalid baseline depth in {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} does not contain an output list")
        for row in rows:
            key = (ctx, str(row["prompt_profile"]), int(row["max_tokens"]), int(row.get("repeat", 0)))
            result = row["data"]
            if not isinstance(result, dict):
                raise ValueError(f"{path} contains an invalid output object")
            identity = (mode, depth, key)
            if identity in outputs:
                raise ValueError(f"duplicate output identity {identity} from {path}")
            outputs[identity] = {"data": result, "case": path.name.removesuffix(".outputs.json")}
    if not outputs:
        raise ValueError(f"no Phase 1 .outputs.json files found in {root}")
    return outputs, case_count


def compare(left, right):
    first = left["data"]
    second = right["data"]
    a = first.get("tokens")
    b = second.get("tokens")
    token_available = isinstance(a, list) and isinstance(b, list)
    tok_diff = first_difference(a, b) if token_available else None
    a_text, b_text = str(first.get("content", "")), str(second.get("content", ""))
    char_diff = first_difference(a_text, b_text)
    match = (tok_diff is None and char_diff is None) if token_available else char_diff is None
    result = {
        "reference_case": left["case"],
        "candidate_case": right["case"],
        "match": match,
        "both_have_token_ids": token_available,
        "same_token_ids": tok_diff is None if token_available else None,
        "same_decoded_text": char_diff is None,
        "first_differing_token_index_zero_based": tok_diff,
        "first_differing_character_index_zero_based": char_diff,
        "reference_token_count": len(a) if isinstance(a, list) else None,
        "candidate_token_count": len(b) if isinstance(b, list) else None,
        "reference_text_length": len(a_text),
        "candidate_text_length": len(b_text),
        "reference_stop_type": first.get("stop_type"),
        "candidate_stop_type": second.get("stop_type"),
        "reference_draft_n": (first.get("timings") or {}).get("draft_n"),
        "candidate_draft_n": (second.get("timings") or {}).get("draft_n"),
        "reference_draft_n_accepted": (first.get("timings") or {}).get("draft_n_accepted"),
        "candidate_draft_n_accepted": (second.get("timings") or {}).get("draft_n_accepted"),
    }
    if token_available and tok_diff is not None:
        result["reference_token_window"] = a[max(0,tok_diff - 5):tok_diff + 6]
        result["candidate_token_window"] = b[max(0,tok_diff - 5):tok_diff + 6]
    if char_diff is not None:
        result["reference_text_window"] = clip_text(a_text, char_diff)
        result["candidate_text_window"] = clip_text(b_text, char_diff)
    return result


def diagnose(root):
    outputs, case_count = load_outputs(root)
    controls = {key: entry for (mode, depth, key), entry in outputs.items()
                if mode == "A" and depth == 0}
    if not controls:
        raise ValueError("No A_ctx*_depth0 baseline outputs found")
    comparisons = []
    pair_comparisons = []
    missing_baseline = []
    for (mode, depth, key), entry in sorted(outputs.items()):
        if mode == "A":
            continue
        if key not in controls:
            missing_baseline.append({"case": entry["case"], "conditions": list(key)})
            continue
        row = compare(controls[key], entry)
        row.update({"context": key[0], "prompt": key[1], "max_tokens": key[2],
                    "repeat": key[3], "candidate_mode": mode, "depth": depth})
        comparisons.append(row)
    for (mode, depth, key), entry in sorted(outputs.items()):
        if mode != "B":
            continue
        other = outputs.get(("C", depth, key))
        if other is None:
            continue
        row = compare(entry, other)
        row.update({"context": key[0], "prompt": key[1], "max_tokens": key[2],
                    "repeat": key[3], "depth": depth, "comparison": "B_vs_C"})
        pair_comparisons.append(row)
    summary = {
        "case_files_found": case_count,
        "trial_outputs_found": len(outputs),
        "baseline_comparisons": len(comparisons),
        "baseline_mismatches": sum(not r["match"] for r in comparisons),
        "same_depth_B_vs_C_comparisons": len(pair_comparisons),
        "same_depth_B_vs_C_mismatches": sum(not r["match"] for r in pair_comparisons),
        "missing_baselines": len(missing_baseline),
    }
    return {"summary": summary, "baseline_comparisons": comparisons,
            "same_depth_B_vs_C": pair_comparisons, "missing_baselines": missing_baseline}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, help="default: RESULTS_DIR/output_diagnostics.json")
    args = parser.parse_args(argv)
    report = diagnose(args.results_dir)
    output = args.report_file or args.results_dir / "output_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT DIAGNOSTICS (measured artifacts; no models rerun)")
    print(json.dumps(report["summary"], indent=2))
    for key in ("baseline_comparisons", "same_depth_B_vs_C"):
        print("\n" + ("A versus B/C" if key == "baseline_comparisons" else "B versus C at the same depth"))
        for row in report[key]:
            state = "MATCH" if row["match"] else "DIFF"
            pos = row["first_differing_token_index_zero_based"]
            if pos is None:
                pos = "unavailable" if not row["both_have_token_ids"] else "none"
            print(f"  {row['reference_case']} vs {row['candidate_case']}: {state}, "
                  f"first token index: {pos}; first character index: "
                  f"{row['first_differing_character_index_zero_based']}")
            if not row["match"]:
                if "reference_token_window" in row:
                    print("     token IDs reference:", row["reference_token_window"])
                    print("     token IDs candidate:", row["candidate_token_window"])
                if "reference_text_window" in row:
                    print("     text reference:", repr(row["reference_text_window"]))
                    print("     text candidate:", repr(row["candidate_text_window"]))
    print(f"\nSaved to {output}")
    print("NOTE: greedy differences are unresolved correctness evidence, not proof of invalid sampling.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: {exc}")
