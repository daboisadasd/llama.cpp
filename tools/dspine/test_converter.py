#!/usr/bin/env python3
"""Synthetic GGUF round-trip tests; no GPUs, 4GB files, or network required."""
from __future__ import annotations
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import convert_gguf as cv


def string_array_metadata(key, values):
    return (cv.encode_string(key) + cv.u32(9) + cv.u32(8) + cv.u64(len(values))
            + b"".join(cv.encode_string(s) for s in values))


def make_safetensors(path, tensors):
    data = bytearray()
    meta = {}
    for name, shape, dtype, payload in tensors:
        start = len(data)
        data.extend(payload)
        meta[name] = {"shape": shape, "dtype": dtype, "data_offsets": [start, len(data)]}
    header = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    path.write_bytes(cv.u64(len(header)) + header + data)


def read_gguf_data(path):
    with path.open("rb") as f:
        reader = cv.Reader(f)
        assert reader.raw(4) == b"GGUF"
        assert reader.uint("I") == 3
        nt, nkv = reader.uint("Q"), reader.uint("Q")
        for _ in range(nkv):
            reader.string()
            typ = reader.uint("I")
            reader.skip_value(typ)
        tensors = []
        for _ in range(nt):
            name = reader.string()
            ndim = reader.uint("I")
            dims = tuple(reader.uint("Q") for _ in range(ndim))
            kind, offset = reader.uint("I"), reader.uint("Q")
            tensors.append((name, dims, kind, offset))
        base = cv.align_up(f.tell())
        result = {}
        for name, dims, kind, offset in tensors:
            bsize = cv.DTYPE_BYTES[next(k for k, v in cv.GGML_TYPES.items() if v == kind)]
            f.seek(base + offset)
            result[name] = {"shape": dims, "dtype": kind, "data": f.read(bsize * __import__("math").prod(dims))}
        return result


class TestDSpineGGUF(unittest.TestCase):
    def test_round_trip_tensors_and_unicode_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "input.safetensors"
            data = bytes.fromhex("00112233445566778899aabb")
            src.write_bytes(data)
            tensors = [cv.Tensor("model.selected_prefix_bridge.é.weight", (3, 2), 30, src, 0, 12)]
            meta = [cv.scalar_meta("general.architecture", "dspine"),
                    string_array_metadata("tokenizer.ggml.tokens", ["hello", "world"])]
            out = root / "draft.gguf"
            result = cv.write_gguf(out, tensors=tensors, metadata=meta)
            self.assertEqual(result["tensor_count"], 1)
            got = read_gguf_data(out)
            self.assertEqual(got["model.selected_prefix_bridge.é.weight"]["data"], data)
            self.assertEqual(got["model.selected_prefix_bridge.é.weight"]["shape"], (3, 2))
            raw, head = cv.parse_source_gguf(out, need_head=False)
            self.assertIsNone(head)
            self.assertEqual(len(raw), 1)
            self.assertIn("tokenizer.ggml.tokens".encode(), raw[0])

    def test_two_tensors_are_aligned_and_not_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "data.bin"
            src.write_bytes(b"123456789012" + b"F" * 40)
            tensors = [
                cv.Tensor("model.bridge", (3, 2), 30, src, 0, 12),
                cv.Tensor("model.final", (4, 5), 30, src, 12, 40),
            ]
            out = root / "checkpoint.gguf"
            cv.write_gguf(out, tensors=tensors,
                          metadata=[string_array_metadata("tokenizer.ggml.tokens", ["x"])] )
            parsed = read_gguf_data(out)
            self.assertEqual(parsed["model.bridge"]["data"], b"123456789012")
            self.assertEqual(parsed["model.final"]["data"], b"F" * 40)

    def test_checkpoint_safetensors_offsets_and_ggml_dims(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "model.safetensors"
            make_safetensors(src, [
                ("model.selected_prefix_bridge.codebook", [4, 3], "BF16", b"a" * 24),
                ("model.selected_prefix_bridge.independent_write_version", [], "I64", b"b" * 8),
                ("model.layers.0.mlp.down_proj.weight", [2, 3], "F32", b"c" * 24),
            ])
            with patch.object(cv, "validate_config", return_value={}):
                with patch.object(cv, "validate_tensors", return_value={}):
                    tensors, source, _ = cv.checkpoint_tensors(src, {})
            self.assertEqual(source["tensor_count"], 3)
            by_name = {t.name: t for t in tensors}
            self.assertEqual(by_name["model.selected_prefix_bridge.codebook"].shape, (3, 4))
            self.assertEqual(by_name["model.selected_prefix_bridge.independent_write_version"].shape, (1,))
            self.assertEqual(by_name["model.layers.0.mlp.down_proj.weight"].dtype, 0)
            data = src.read_bytes()
            for tensor in tensors:
                self.assertIn(data[tensor.src_offset:tensor.src_offset+tensor.size], (b"a"*24,b"b"*8,b"c"*24))

    def test_cannot_relabel_existing_gguf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "payload.bin"
            src.write_bytes(b"abcd")
            out = root / "do_not_overwrite.gguf"
            out.write_bytes(b"existing")
            with self.assertRaisesRegex(cv.CheckpointError, "already exists"):
                cv.write_gguf(out, tensors=[cv.Tensor("param", (1,), 0, src, 0, 4)],
                              metadata=[string_array_metadata("tokenizer.ggml.tokens", ["x"])])
            self.assertEqual(out.read_bytes(), b"existing")

    def test_reject_invalid_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "payload.bin"
            src.write_bytes(b"x")
            with self.assertRaisesRegex(cv.CheckpointError, "invalid output tensor size"):
                cv.write_gguf(root/"bad.gguf", tensors=[cv.Tensor("weight",(2,),30,src,0,1)],
                              metadata=[string_array_metadata("tokenizer.ggml.tokens", ["x"])])

    def test_skip_large_tokenizer_string_array(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = string_array_metadata("tokenizer.ggml.tokens", [str(i) for i in range(350)])
            out = root / "tok.gguf"
            cv.write_gguf(out, tensors=[], metadata=[raw])
            entries, head = cv.parse_source_gguf(out, need_head=False)
            self.assertEqual(entries, [raw])
            self.assertIsNone(head)


if __name__ == "__main__":
    unittest.main(verbosity=2)
