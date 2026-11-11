#!/usr/bin/env python3
"""Audit native llama.cpp DFlash/DSpark seams for a future correct DSpine port.

Does not claim DSpine is implemented. No fork source is modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def audit(root):
    paths = {
        "draft": root / "src/models/dflash.cpp",
        "verify": root / "common/speculative.cpp",
        "convert": root / "conversion/qwen.py",
        "target": root / "src/models/qwen35.cpp",
    }
    files = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    checks = {
        "existing_native_dflash_graph": "llama_model_dflash::graph" in files["draft"],
        "existing_native_dspark_verification": "COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK" in files["verify"],
        "target_feature_extraction_for_block_draft": "llama_set_embeddings_layer_inp_dev" in files["verify"],
        "existing_dflash_gguf_converter": '@ModelBase.register("DFlashDraftModel", "DFlash2DraftModel")' in files["convert"],
        "target_qwen35_model_builder": "llama_model_qwen35" in files["target"],
    }
    # DSpine is NOT interchangeable with draft-dflash or draft-dspark.
    return {
        "fork_source": str(root.resolve()),
        "existing_primitives": checks,
        "source_audit_pass": all(checks.values()),
        "native_dspine_ready": False,
        "missing_runtime_components": [
            "GGUF conversion preserving selected_prefix_bridge tensors and DSpine metadata",
            "GGML/CUDA graph code for learned adjacent-causal writes after each draft layer",
            "DSpine serial-table candidate refinement with target BF16 output head",
            "ReplaySSM handling for hybrid Qwen3.8 target verification",
            "DSpark dynamic verification and sampled-probability correction",
            "worker-side RPC kernel traces and end-to-end native inference validation",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path, required=True)
    a = parser.parse_args(argv)
    report = audit(a.llama_root)
    print(json.dumps(report, indent=2))
    return 0 if report["source_audit_pass"] else 2

if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        sys.exit(f"ERROR: {exc}")
