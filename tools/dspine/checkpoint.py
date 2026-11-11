#!/usr/bin/env python3
"""Inspect Tencent's real Qwen3.8-27B DSpine checkpoint without running SGLang.

This is a strict artifact/architecture preflight for a future native llama.cpp
DSpine backend. It cannot perform DSpine inference and does not substitute
DFlash/MTP. The model remains a safetensors checkpoint until a true converter
and inference graph exist. Requires Python 3.10+; download uses huggingface_hub.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

REPO_ID = "tencent/Qwen3.8-27B-DSpine-b16"
MODEL_FILES = ("config.json", "model.safetensors")
BF16_MTP_FILENAME = "mtp-Qwen3.8-27B-BF16.gguf"
MAX_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
              "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
              "F8_E4M3": 1, "F8_E5M2": 1}


class CheckpointError(ValueError):
    """Invalid/incompatible checkpoint or incomplete download."""


def validate_config(cfg: dict) -> dict:
    """Validate upstream Qwen3.8-27B DSpine schema, not an arbitrary DFlash model."""
    if not isinstance(cfg, dict):
        raise CheckpointError("config.json must contain an object")
    # The upstream implementation inherits DFlashDraftModel. Do not interpret
    # 'DFlashDraftModel' as permission to decode using ordinary draft-dflash.
    if "DFlashDraftModel" not in cfg.get("architectures", []):
        raise CheckpointError("expected upstream DFlashDraftModel base architecture")
    expected_int = {
        "block_size": 16,
        "num_hidden_layers": 5,
        "hidden_size": 5120,
        "vocab_size": 248320,
        "num_target_layers": 64,
        "sliding_window": 8192,
    }
    for name, value in expected_int.items():
        if cfg.get(name) != value:
            raise CheckpointError(f"Qwen3.8 DSpine {name}: expected {value}, got {cfg.get(name)!r}")
    if cfg.get("dtype", cfg.get("torch_dtype")) != "bfloat16":
        raise CheckpointError("expected BF16 DSpine checkpoint")
    if cfg.get("layer_types") != ["sliding_attention"] * 5:
        raise CheckpointError("DSpine draft requires five sliding-attention layers")
    dflash = cfg.get("dflash_config")
    if not isinstance(dflash, dict):
        raise CheckpointError("missing dflash_config")
    expected = {
        "draft_block_causal": True,
        "selected_prefix_terminal": False,
        "selected_prefix_feature_writes": True,
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "target_layer_ids": [1, 16, 31, 46, 61],
    }
    for name, val in expected.items():
        if dflash.get(name) != val:
            raise CheckpointError(f"DSpine dflash_config.{name}: expected {val!r}, got {dflash.get(name)!r}")
    bridge = dflash.get("selected_prefix_code_bridge")
    if not isinstance(bridge, dict) or bridge.get("code_size") != 1024 or bridge.get("rank") != 512:
        raise CheckpointError("expected selected_prefix_code_bridge code_size=1024, rank=512")
    if not isinstance(bridge.get("eps"), (float, int)) or bridge["eps"] <= 0:
        raise CheckpointError("code bridge epsilon must be positive")
    pass2 = cfg.get("code_bridge_second_pass")
    if not isinstance(pass2, dict) or pass2.get("mode") != "serial_table" or pass2.get("first_exact") != 4 or pass2.get("head_topk") != 16:
        raise CheckpointError("missing/unsupported DSpine serial_table second-pass refinement")
    refine = dflash.get("selected_prefix_last_write_refine")
    if not isinstance(refine, dict) or refine.get("kind") != "exact_sender" or refine.get("head_topk") != 16:
        raise CheckpointError("missing/unsupported DSpine last-write exact-sender refinement")
    if dflash.get("selected_prefix_terminal_residual_rank", 0) != 0:
        raise CheckpointError("DSpine residual rank path not yet supported by the reference contract")
    return {
        "draft_layers": 5,
        "draft_block_size": 16,
        "draft_attention_window": 8192,
        "target_layers_checkpoint_indices": [1, 16, 31, 46, 61],
        "gguf_dflash_target_layer_ids": [2, 17, 32, 47, 62],
        "code_size": 1024,
        "code_rank": 512,
        "serial_topk": 16,
        "vocab_size": 248320,
        "requires_target_bf16_output_projection": True,
        "requires_replay_ssm": True,
        "requires_dynamic_verification_for_upstream_parity": True,
    }


def inspect_safetensors(path: Path) -> dict:
    """Validate safetensors layout using the JSON header only; never read 4GB weights."""
    if not path.is_file():
        raise CheckpointError(f"missing safetensors file: {path}")
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        size_raw = fh.read(8)
        if len(size_raw) != 8:
            raise CheckpointError("truncated safetensors header length")
        header_size = struct.unpack("<Q", size_raw)[0]
        if not (2 <= header_size <= MAX_HEADER_BYTES):
            raise CheckpointError(f"invalid safetensors header size: {header_size}")
        if 8 + header_size > file_size:
            raise CheckpointError("safetensors header exceeds file size")
        header_raw = fh.read(header_size)
    try:
        header = json.loads(header_raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CheckpointError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise CheckpointError("safetensors header must be an object")
    data_size = file_size - 8 - header_size
    names = []
    spans = []
    counts = Counter()
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise CheckpointError(f"tensor {name}: invalid index entry")
        shape, offsets, dtype = info.get("shape"), info.get("data_offsets"), info.get("dtype")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise CheckpointError(f"tensor {name}: invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
            raise CheckpointError(f"tensor {name}: invalid data offsets")
        if dtype not in DTYPE_SIZE:
            raise CheckpointError(f"tensor {name}: unsupported dtype {dtype!r}")
        first, last = offsets
        expected = math.prod(shape) * DTYPE_SIZE[dtype]
        if not (0 <= first <= last <= data_size) or last - first != expected:
            raise CheckpointError(f"tensor {name}: invalid byte span {offsets} for {dtype} {shape}")
        names.append(name)
        spans.append((first, last, name))
        counts[dtype] += 1
    spans.sort()
    if not spans:
        raise CheckpointError("no safetensors weights")
    if spans[0][0] != 0 or spans[-1][1] != data_size:
        raise CheckpointError("safetensors data region contains missing leading/trailing bytes")
    for left, right in zip(spans, spans[1:]):
        if left[1] != right[0]:
            raise CheckpointError(f"safetensors tensors overlap or have a gap: {left[2]} -> {right[2]}")
    return {
        "file": str(path.resolve()),
        "bytes": file_size,
        "header_sha256": hashlib.sha256(header_raw).hexdigest(),
        "tensor_count": len(names),
        "dtype_counts": dict(counts),
        "tensors": {name: {"shape": header[name]["shape"], "dtype": header[name]["dtype"]} for name in names},
    }


def validate_tensors(tensor_map: dict, contract: dict) -> dict:
    """Check all learned DSpine bridge inputs, 5 draft layers and version sentinels."""
    def named(key):
        # Some safetensors exporters strip the model. prefix.
        return key if key in tensor_map else "model." + key

    c, h, r, v, layers = (contract["code_size"], 5120, contract["code_rank"],
                         contract["vocab_size"], contract["draft_layers"])
    prefix = "selected_prefix_bridge."
    required = {
        "codebook": [v, c],
        "read_weight": [c, h],
        "early_read_weights": [layers - 1, c, h],
        "feature_write_weights": [layers, h, r],
        "gate.weight": [r, c],
        "gate.bias": [r],
        "predecessor_gate.weight": [r, c],
        "prev.weight": [r, c],
        "out.weight": [c, r],
    }
    matched = []
    for name, shape in required.items():
        key = named(prefix + name)
        if key not in tensor_map:
            raise CheckpointError(f"DSpine code bridge tensor missing: {key}")
        meta = tensor_map[key]
        if meta["shape"] != shape:
            raise CheckpointError(f"DSpine code bridge {key} shape {meta['shape']} != {shape}")
        if meta["dtype"] not in ("BF16", "F32"):
            raise CheckpointError(f"DSpine code bridge {key} unexpected dtype {meta['dtype']}")
        matched.append(key)
    for name in ("independent_write_version", "terminal_disabled_version"):
        key = named(prefix + name)
        if key not in tensor_map or math.prod(tensor_map[key]["shape"]) != 1:
            raise CheckpointError(f"DSpine code bridge version tensor missing: {key}")
        matched.append(key)
    layer_indices = set()
    for key in tensor_map:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))
    if layer_indices != set(range(layers)):
        raise CheckpointError(f"DSpine draft layer indexes {sorted(layer_indices)} != [0,1,2,3,4]")
    return {"bridge_tensors_verified": sorted(matched),
            "draft_layers_seen": sorted(layer_indices),
            "checkpoint_tensors": len(tensor_map)}


def inspect_target_bf16_head(mtp_gguf: Path) -> dict:
    """Reuse the user's already-downloaded BF16 MTP GGUF's output.weight as a source.

    This checks metadata only. It does not prove that the target output tensor
    matches Tencent's SGLang BF16 target weights bitwise.
    """
    # Existing Phase 1 GGUF header inspector ignores model weight payloads.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1_mtp"))
    from inspect_gguf import inspect as inspect_gguf
    data = inspect_gguf(mtp_gguf)
    if data["metadata"].get("general.architecture") != "qwen35":
        raise CheckpointError("BF16 MTP sidecar is not Qwen3.8 architecture")
    selected = data.get("selected_tensors", {})
    head = selected.get("output.weight")
    if not head or head["ggml_type"] != "BF16" or head["shape"] != [5120, 248320]:
        raise CheckpointError(f"BF16 target output head missing/wrong in MTP GGUF: {head}")
    return {"gguf_file": str(mtp_gguf), "tensor": "output.weight", **head,
            "note": "candidate target BF16 lm_head source; equality with reference target not yet established"}


def download(model_dir: Path, config_only=False):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise CheckpointError("Install once: py -3 -m pip install -U huggingface_hub") from exc
    model_dir.mkdir(parents=True, exist_ok=True)
    files = ("config.json",) if config_only else MODEL_FILES
    for name in files:
        print(f"Downloading {REPO_ID}/{name} if not already cached...", flush=True)
        hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(model_dir))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True, help="folder containing config.json/model.safetensors")
    p.add_argument("--download", action="store_true", help="download official config and checkpoint once")
    p.add_argument("--config-only", action="store_true", help="inspect config without downloading 4GB weights")
    p.add_argument("--bf16-mtp-gguf", type=Path, help="optional existing BF16 MTP GGUF containing output.weight")
    p.add_argument("--output", type=Path, help="JSON inspection output")
    a = p.parse_args(argv)
    if a.download:
        download(a.model_dir, a.config_only)
    cfg_file = a.model_dir / "config.json"
    if not cfg_file.is_file():
        raise CheckpointError(f"missing config: {cfg_file} (use --download)")
    config_raw = cfg_file.read_bytes()
    cfg = json.loads(config_raw)
    contract = validate_config(cfg)
    report = {
        "source": REPO_ID,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "contract": contract,
        "status": "CONFIG_VALIDATED_ONLY" if a.config_only else "CHECKPOINT_VALIDATED_NOT_EXECUTABLE",
        "native_llama_cpp_backend_ready": False,
        "warnings": [
            "DSpine is not yet implemented in llama.cpp; do not run draft-dflash or draft-dspark with these weights.",
            "Native port needs inter-layer code bridge, serial refinement and ReplaySSM/DSpark verification.",
            "Tencent's sampler requires an unquantized BF16 target output projection for refinement tables.",
        ],
    }
    if not a.config_only:
        st = inspect_safetensors(a.model_dir / "model.safetensors")
        report["weights"] = {key: value for key, value in st.items() if key != "tensors"}
        report["validated_checkpoint"] = validate_tensors(st["tensors"], contract)
    if a.bf16_mtp_gguf:
        report["candidate_target_head"] = inspect_target_bf16_head(a.bf16_mtp_gguf)
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Saved", a.output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, CheckpointError, ValueError, TypeError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: {exc}")
