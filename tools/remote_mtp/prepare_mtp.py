#!/usr/bin/env python3
"""Download/select a native Qwen3.8-27B MTP sidecar for GGML RPC.

Only the MTP sidecar is changed. The 27B target remains independently
quantized on the host. F16 is converted from the published BF16 MTP GGUF,
not dequantized from Q4_0 (which would not restore lost precision).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1_mtp"))
from inspect_gguf import inspect as inspect_gguf

HF_REPO = "ggml-org/Qwen3.8-27B-GGUF"
PRECISIONS = ("Q4_0", "Q8_0", "BF16", "F16")


def sidecar_filename(precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"unsupported MTP precision {precision!r}; valid: {PRECISIONS}")
    return f"mtp-Qwen3.8-27B-{precision}.gguf"


def validate_sidecar(path: Path, precision: str):
    header = inspect_gguf(path)
    metadata = header.get("metadata", {})
    if metadata.get("general.architecture") not in ("qwen35", "qwen35moe"):
        raise ValueError(f"{path}: unexpected architecture {metadata.get('general.architecture')!r}")
    if not any(key.endswith("nextn_predict_layers") and isinstance(value, int) and value > 0
               for key, value in metadata.items()):
        raise ValueError(f"{path}: no native MTP nextn_predict_layers metadata")
    if header["tensor_count"] <= 0 or header["mtp_tensor_count"] <= 0:
        raise ValueError(f"{path}: missing MTP tensor headers")
    counts = header.get("tensor_type_counts", {})
    if precision in ("BF16", "F16"):
        if counts.get(precision, 0) == 0:
            raise ValueError(f"{path}: no {precision} tensors; tensor types: {counts}")
        if precision == "F16" and counts.get("BF16", 0):
            raise ValueError(f"{path}: still has BF16 tensors after F16 conversion: {counts}")
    return header


def download_source(model_dir: Path, precision: str):
    filename = sidecar_filename(precision)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install downloader once: py -3 -m pip install -U huggingface_hub") from exc
    print(f"Fetching official {precision} MTP sidecar: {HF_REPO}/{filename}", flush=True)
    path = Path(hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=str(model_dir)))
    validate_sidecar(path, precision)
    return path


def convert_bf16_to_f16(source: Path, output: Path, quantizer: Path):
    if output.is_file():
        validate_sidecar(output, "F16")
        print(f"Verified existing F16 MTP GGUF: {output}")
        return output
    if not quantizer.is_file():
        raise FileNotFoundError(f"Missing llama-quantize executable: {quantizer}")
    free_bytes = shutil.disk_usage(output.parent).free
    # Give some extra headroom for GGUF metadata and temp files.
    if free_bytes < source.stat().st_size + 1024**3:
        raise RuntimeError("Not enough free disk space for a 16-bit MTP GGUF plus 1 GiB headroom")
    tmp = output.with_name(output.stem + ".partial.gguf")
    if tmp.exists():
        tmp.unlink()
    print(f"Converting {source.name} -> {output.name} using llama.cpp (BF16 -> F16)", flush=True)
    try:
        subprocess.run([str(quantizer), str(source), str(tmp), "F16"], check=True)
        converted = validate_sidecar(tmp, "F16")
        original = validate_sidecar(source, "BF16")
        if converted["tensor_count"] != original["tensor_count"]:
            raise ValueError("F16 conversion changed tensor count")
        if converted["mtp_tensor_count"] != original["mtp_tensor_count"]:
            raise ValueError("F16 conversion dropped MTP tensors")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"Verified F16 MTP GGUF: {output}", flush=True)
    return output


def prepare(model_dir: Path, precision: str, quantizer: Path | None = None):
    if precision not in PRECISIONS:
        raise ValueError(f"unsupported precision {precision}")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / sidecar_filename(precision)
    if path.is_file():
        validate_sidecar(path, precision)
        print(f"Already present; no download necessary: {path}", flush=True)
        return path
    if precision != "F16":
        return download_source(model_dir, precision)
    source = model_dir / sidecar_filename("BF16")
    if not source.is_file():
        source = download_source(model_dir, "BF16")
    else:
        validate_sidecar(source, "BF16")
    if quantizer is None:
        raise ValueError("Preparing F16 requires --quantizer (llama-quantize.exe)")
    return convert_bf16_to_f16(source, path, quantizer)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", required=True, choices=PRECISIONS)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--quantizer", type=Path, help="llama-quantize executable, required for BF16->F16")
    a = p.parse_args(argv)
    path = prepare(a.model_dir, a.precision, a.quantizer)
    verified = validate_sidecar(path, a.precision)
    print(json.dumps({
        "source": HF_REPO if a.precision != "F16" else "BF16 sidecar converted locally",
        "precision": a.precision,
        "path": str(path.resolve()),
        "bytes": verified["bytes"],
        "tensor_count": verified["tensor_count"],
        "mtp_tensor_count": verified["mtp_tensor_count"],
        "tensor_type_counts": verified["tensor_type_counts"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"ERROR: {exc}")
