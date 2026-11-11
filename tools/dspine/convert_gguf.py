#!/usr/bin/env python3
"""Stream a *complete*, deliberately non-loadable DSpine GGUF from Tencent safetensors.

The artifact uses general.architecture=dspine so existing draft-dflash/dspark
cannot accidentally execute it with the learned code bridge omitted. Weights
retain their upstream names, dtypes, shapes, version scalars and byte contents.
Optionally bundle the target's BF16 output.weight from the existing MTP GGUF.
No large tensors are loaded into RAM. This is a converter, NOT a runtime port.

Requires Python 3.10+ and tools/dspine/checkpoint.py; no third-party imports.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys

from checkpoint import (
    CheckpointError, inspect_safetensors, validate_config, validate_tensors,
)

ALIGN = 32
CHUNK = 8 * 1024 * 1024
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
# GGML tensor types, NOT GGUF metadata value types.
GGML_TYPES = {"F32": 0, "F16": 1, "I8": 24, "I16": 25, "I32": 26,
              "I64": 27, "BF16": 30}
DTYPE_BYTES = {"F32": 4, "F16": 2, "I8": 1, "I16": 2, "I32": 4,
               "I64": 8, "BF16": 2}
# GGUF metadata value tags (not GGML tensor types).
VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
               7: 1, 10: 8, 11: 8, 12: 8}


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # GGUF dimensions: column-first
    dtype: int
    filename: Path
    src_offset: int
    size: int


def u32(n):
    return struct.pack("<I", n)


def u64(n):
    return struct.pack("<Q", n)


def encode_string(s):
    b = s.encode("utf-8")
    return u64(len(b)) + b


def scalar_meta(key, val):
    if isinstance(val, str):
        return encode_string(key) + u32(8) + encode_string(val)
    if type(val) is bool:
        return encode_string(key) + u32(7) + struct.pack("<?", val)
    if type(val) is int and 0 <= val <= 0xffffffff:
        return encode_string(key) + u32(4) + u32(val)
    if isinstance(val, list) and all(type(x) is int and 0 <= x <= 0xffffffff for x in val):
        return encode_string(key) + u32(9) + u32(4) + u64(len(val)) + b"".join(u32(x) for x in val)
    raise ValueError(f"unsupported metadata {key}: {val!r}")


class Reader:
    def __init__(self, file):
        self.file = file
        here = file.tell()
        self.size = file.seek(0, os.SEEK_END)
        file.seek(here)

    def raw(self, n):
        if n < 0 or n > self.size - self.file.tell():
            raise CheckpointError("invalid/truncated GGUF header")
        data = self.file.read(n)
        if len(data) != n:
            raise CheckpointError("truncated GGUF header")
        return data

    def skip(self, n):
        if n < 0 or n > self.size - self.file.tell():
            raise CheckpointError("invalid GGUF metadata element offset")
        self.file.seek(n, os.SEEK_CUR)

    def uint(self, fmt):
        return struct.unpack("<" + fmt, self.raw(struct.calcsize("<" + fmt)))[0]

    def string(self):
        n = self.uint("Q")
        if n > 2**24:
            raise CheckpointError("unreasonable GGUF string length")
        return self.raw(n).decode("utf-8")

    def skip_value(self, typ, depth=0):
        if depth > 4:
            raise CheckpointError("overly nested GGUF metadata array")
        if typ in VALUE_SIZES:
            self.skip(VALUE_SIZES[typ])
        elif typ == 8:
            self.skip(self.uint("Q"))
        elif typ == 9:
            child, count = self.uint("I"), self.uint("Q")
            if count > 1_000_000:
                raise CheckpointError("GGUF metadata array too long")
            if child in VALUE_SIZES:
                self.skip(count * VALUE_SIZES[child])
            elif child in (8, 9):
                for _ in range(count):
                    self.skip_value(child, depth + 1)
            else:
                raise CheckpointError(f"invalid GGUF array element type {child}")
        else:
            raise CheckpointError(f"unsupported GGUF metadata type {typ}")


def parse_source_gguf(path: Path, *, need_head=True):
    """Return verbatim tokenizer metadata bytes and optional BF16 output head span."""
    with path.open("rb") as f:
        r = Reader(f)
        if r.raw(4) != GGUF_MAGIC or r.uint("I") != GGUF_VERSION:
            raise CheckpointError(f"not a GGUF v3 file: {path}")
        n_tensors, n_meta = r.uint("Q"), r.uint("Q")
        if n_tensors > 100_000 or n_meta > 100_000:
            raise CheckpointError("unreasonable GGUF index counts")
        raw_metadata, metadata_names = [], set()
        alignment = ALIGN
        for _ in range(n_meta):
            start = f.tell()
            name = r.string()
            typ = r.uint("I")
            if name in metadata_names:
                raise CheckpointError(f"duplicate GGUF key {name}")
            metadata_names.add(name)
            if name == "general.alignment":
                if typ != 4:
                    raise CheckpointError("unsupported general.alignment type")
                alignment = r.uint("I")
            else:
                r.skip_value(typ)
            end = f.tell()
            if name.startswith("tokenizer."):
                if end - start > 64 * 1024 * 1024:
                    raise CheckpointError("single tokenizer metadata entry exceeds 64 MiB")
                f.seek(start)
                raw_metadata.append(f.read(end - start))
                f.seek(end)
        if alignment < 8 or alignment > 4096 or (alignment & (alignment - 1)):
            raise CheckpointError("invalid GGUF general.alignment")
        if "tokenizer.ggml.tokens" not in metadata_names:
            raise CheckpointError("target GGUF lacks tokenizer.ggml.tokens")
        head = None
        tensors_seen = set()
        for _ in range(n_tensors):
            name = r.string()
            if name in tensors_seen:
                raise CheckpointError(f"duplicate GGUF tensor {name}")
            tensors_seen.add(name)
            nd = r.uint("I")
            if nd < 1 or nd > 8:
                raise CheckpointError("invalid GGUF tensor dimension count")
            dims = [r.uint("Q") for _ in range(nd)]
            dtype, offset = r.uint("I"), r.uint("Q")
            if name == "output.weight":
                head = (tuple(dims), dtype, offset)
        tensor_data_start = align_up(f.tell(), alignment)
        if need_head:
            if head is None or head[0] != (5120, 248320) or head[1] != GGML_TYPES["BF16"]:
                raise CheckpointError("BF16 output.weight [5120,248320] is required in source GGUF")
            size = math.prod(head[0]) * 2
            offset = tensor_data_start + head[2]
            if offset + size > r.size:
                raise CheckpointError("source GGUF BF16 head extends beyond end of file")
            tensor = Tensor("output.weight", head[0], head[1], path, offset, size)
        else:
            tensor = None
        return raw_metadata, tensor


def align_up(n, align=ALIGN):
    return (n + align - 1) // align * align


def checkpoint_tensors(path: Path, config: dict):
    contract = validate_config(config)
    validated = inspect_safetensors(path)
    validate_tensors(validated["tensors"], contract)
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    start = 8 + n
    tensors = []
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in GGML_TYPES:
            raise CheckpointError(f"GGUF conversion of {dtype} tensor {name!r} unsupported (fail closed)")
        size = info["data_offsets"][1] - info["data_offsets"][0]
        dims = tuple(reversed(info["shape"])) or (1,)
        tensors.append(Tensor(name, dims, GGML_TYPES[dtype],
                              path, start + info["data_offsets"][0], size))
    return tensors, validated, contract


def write_gguf(path: Path, *, tensors: list[Tensor], metadata: list[bytes]):
    if len({t.name for t in tensors}) != len(tensors):
        raise CheckpointError("duplicate output tensor names")
    offsets = []
    n_bytes = 0
    for tensor in tensors:
        if tensor.size != math.prod(tensor.shape) * DTYPE_BYTES[next(
                k for k, v in GGML_TYPES.items() if v == tensor.dtype)]:
            raise CheckpointError(f"invalid output tensor size {tensor.name}")
        n_bytes = align_up(n_bytes)
        offsets.append(n_bytes)
        n_bytes += tensor.size

    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    if free < n_bytes + 1024**3:
        raise CheckpointError(f"need {n_bytes + 1024**3} available bytes for GGUF output")
    temp = path.with_name(path.name + ".partial")
    if path.exists() or temp.exists():
        raise CheckpointError("output already exists (will not overwrite original GGUF or partial): " + str(path))
    with temp.open("xb") as output:
        try:
            output.write(GGUF_MAGIC + u32(GGUF_VERSION) + u64(len(tensors)) + u64(len(metadata)))
            for entry in metadata:
                output.write(entry)
            for t, off in zip(tensors, offsets):
                output.write(encode_string(t.name))
                output.write(u32(len(t.shape)))
                for d in t.shape:
                    output.write(u64(d))
                output.write(u32(t.dtype))
                output.write(u64(off))
            output.write(b"\0" * (align_up(output.tell()) - output.tell()))
            base = output.tell()
            for t, off in zip(tensors, offsets):
                output.write(b"\0" * (base + off - output.tell()))
                with t.filename.open("rb") as f:
                    f.seek(t.src_offset)
                    remaining = t.size
                    while remaining:
                        data = f.read(min(CHUNK, remaining))
                        if not data:
                            raise CheckpointError(f"truncated tensor data while copying {t.name}")
                        output.write(data)
                        remaining -= len(data)
            output.flush()
            os.fsync(output.fileno())
        except Exception:
            output.close()
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)
    return {"file": str(path.resolve()), "bytes": path.stat().st_size,
            "tensor_count": len(tensors), "tensor_payload_bytes": n_bytes}


def build(model_dir: Path, target_gguf: Path, output: Path):
    """Convert all Tencent tensors; embed the target output head, and retain tokenizer KV."""
    config_raw = (model_dir / "config.json").read_bytes()
    config = json.loads(config_raw)
    tensors, source, contract = checkpoint_tensors(model_dir / "model.safetensors", config)
    raw_tokenizer, head = parse_source_gguf(target_gguf)
    if head is None or any(t.name == head.name for t in tensors):
        raise CheckpointError("missing or duplicate target BF16 output head")
    tensors.append(head)
    metadata = [
        scalar_meta("general.architecture", "dspine"),
        scalar_meta("general.name", "Qwen3.8-27B-DSpine-b16"),
        scalar_meta("general.file_type", 32),
        scalar_meta("general.alignment", ALIGN),
        scalar_meta("dspine.format_version", 1),
        scalar_meta("dspine.source", "tencent/Qwen3.8-27B-DSpine-b16"),
        scalar_meta("dspine.config_sha256", hashlib.sha256(config_raw).hexdigest()),
        scalar_meta("dspine.safetensors_header_sha256", source["header_sha256"]),
        scalar_meta("dspine.config_json", config_raw.decode("utf-8")),
        scalar_meta("dspine.block_size", contract["draft_block_size"]),
        scalar_meta("dspine.layers", contract["draft_layers"]),
        scalar_meta("dspine.target_layers", contract["gguf_dflash_target_layer_ids"]),
        scalar_meta("dspine.code_size", contract["code_size"]),
        scalar_meta("dspine.code_rank", contract["code_rank"]),
        scalar_meta("dspine.serial_topk", contract["serial_topk"]),
        scalar_meta("dspine.target_bf16_output_included", True),
        scalar_meta("dflash.block_size", contract["draft_block_size"]),
        scalar_meta("dflash.target_layers", contract["gguf_dflash_target_layer_ids"]),
        scalar_meta("dflash.attention.causal", True),
        scalar_meta("dflash.attention.sliding_window", contract["draft_attention_window"]),
    ] + raw_tokenizer
    report = write_gguf(output, tensors=tensors, metadata=metadata)
    report.update({"source_tensor_count": source["tensor_count"],
                   "head_source": str(target_gguf),
                   "metadata_count": len(metadata),
                   "target_head_type": "BF16",
                   "status": "GGUF_CONVERTED_NOT_EXECUTABLE",
                   "warning": "llama.cpp does not yet have a DSpine model loader or speculative runtime; do not select draft-dflash"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--target-bf16-gguf", type=Path, required=True,
                        help="existing native MTP BF16 GGUF containing output.weight and tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = build(args.model_dir, args.target_bf16_gguf, args.output)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, struct.error, CheckpointError) as exc:
        sys.exit(f"ERROR: {exc}")
