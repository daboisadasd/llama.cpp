#!/usr/bin/env python3
"""No-GPU tests for Tencent DSpine artifact compatibility and code-bridge math."""
import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import checkpoint as cp
import reference


def config():
    return {
        "architectures": ["DFlashDraftModel"],
        "block_size": 16, "num_hidden_layers": 5, "hidden_size": 5120,
        "vocab_size": 248320, "num_target_layers": 64, "sliding_window": 8192,
        "layer_types": ["sliding_attention"] * 5,
        "dtype": "bfloat16",
        "code_bridge_second_pass": {"mode": "serial_table", "first_exact": 4, "head_topk": 16},
        "dflash_config": {
            "draft_block_causal": True, "selected_prefix_terminal": False,
            "selected_prefix_feature_writes": True,
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_layers": [1,2,3,4,5],
            "target_layer_ids": [1,16,31,46,61],
            "selected_prefix_code_bridge": {"code_size":1024,"rank":512,"eps":1e-6},
            "selected_prefix_last_write_refine": {"kind":"exact_sender","version":1,"head_topk":16},
        }
    }


def create_fake_safetensors(path: Path, tensors):
    names = {}; payload = bytearray()
    for name, (shape, dtype) in tensors.items():
        n = math.prod(shape) * cp.DTYPE_SIZE[dtype]
        begin = len(payload)
        payload.extend(b"\0" * n)
        names[name] = {"dtype":dtype,"shape":shape,"data_offsets":[begin, len(payload)]}
    raw = json.dumps(names).encode()
    path.write_bytes(struct.pack('<Q',len(raw)) + raw + payload)


class TestConfig(unittest.TestCase):
    def test_approved_config(self):
        item = cp.validate_config(config())
        self.assertEqual(item["gguf_dflash_target_layer_ids"], [2,17,32,47,62])
        self.assertEqual(item["code_rank"],512)
        self.assertTrue(item["requires_replay_ssm"])

    def test_wrong_block_fails(self):
        bad=config();bad["block_size"]=8
        with self.assertRaisesRegex(cp.CheckpointError,"block_size"):
            cp.validate_config(bad)

    def test_wrong_bridge_fails(self):
        bad=config();bad["dflash_config"]["selected_prefix_code_bridge"]["rank"]=256
        with self.assertRaisesRegex(cp.CheckpointError,"code_size=1024, rank=512"):
            cp.validate_config(bad)

    def test_no_accidental_dflash_fallback(self):
        bad=config();bad["dflash_config"].pop("selected_prefix_feature_writes")
        with self.assertRaisesRegex(cp.CheckpointError,"feature_writes"):
            cp.validate_config(bad)


class TestSafetensors(unittest.TestCase):
    def test_header_and_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"model.safetensors"
            create_fake_safetensors(path,{"model.layers.0.w":([3,4],"BF16"),"model.layers.1.w":([1],"BF16")})
            o=cp.inspect_safetensors(path)
            self.assertEqual(o["tensor_count"],2)
            self.assertEqual(o["dtype_counts"],{"BF16":2})

    def test_header_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"model.safetensors";path.write_bytes(struct.pack('<Q',100)+b"{}")
            with self.assertRaisesRegex(cp.CheckpointError,"exceeds file size"):
                cp.inspect_safetensors(path)

    def test_invalid_offset(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"model.safetensors"
            create_fake_safetensors(path,{"a":[[5],"BF16"]})
            raw=path.read_bytes();size=struct.unpack('<Q',raw[:8])[0]
            idx=json.loads(raw[8:8+size]);idx["a"]["data_offsets"]=[0,9]
            new=json.dumps(idx).encode()
            path.write_bytes(struct.pack('<Q',len(new))+new+raw[8+size:])
            with self.assertRaisesRegex(cp.CheckpointError,"invalid byte span"):
                cp.inspect_safetensors(path)

    def test_requires_bridge_weights(self):
        contract=cp.validate_config(config())
        with self.assertRaisesRegex(cp.CheckpointError,"code bridge tensor missing"):
            cp.validate_tensors({"model.layers.0.foo":{"shape":[1],"dtype":"F32"}},contract)

    def test_bridge_shapes(self):
        spec=cp.validate_config(config());h=5120;r=512;c=1024;v=248320
        keys={
            "codebook":[v,c], "read_weight":[c,h], "early_read_weights":[4,c,h],
            "feature_write_weights":[5,h,r], "gate.weight":[r,c],"gate.bias":[r],
            "predecessor_gate.weight":[r,c],"prev.weight":[r,c],"out.weight":[c,r],
            "independent_write_version":[], "terminal_disabled_version":[],
        }
        tensors={"model.selected_prefix_bridge."+n:{"shape":sh,"dtype":"BF16"} for n,sh in keys.items()}
        for i in range(5):tensors[f"model.layers.{i}.self_attn.q_proj.weight"]={"shape":[128,128],"dtype":"BF16"}
        o=cp.validate_tensors(tensors,spec)
        self.assertEqual(o["draft_layers_seen"],[0,1,2,3,4])
        tensors["model.selected_prefix_bridge.codebook"]["shape"]= [v,256]
        with self.assertRaisesRegex(cp.CheckpointError,"codebook shape"):
            cp.validate_tensors(tensors,spec)


class TestReference(unittest.TestCase):
    def test_messages_stop_at_block_boundaries(self):
        # R = 1, row = [receiver, sender, value]
        packed=[[0,1,2],[0,3,4],[1,5,8],[2,9,16]]
        got=reference.adjacent_message(packed,[0],2)
        self.assertEqual(got[0],[0.0]);self.assertEqual(got[2],[0.0])
        self.assertAlmostEqual(got[1][0],reference.sigmoid(1)*2)
        self.assertAlmostEqual(got[3][0],reference.sigmoid(7)*8)

    def test_write_and_norm(self):
        raw=[[1,0],[0,1]]; rms=[1,2];message=[[0],[.5]];w=[[2],[4]]
        out,norm=reference.adjacent_write(raw,rms,message,w,2,[1,1])
        self.assertEqual(out[0],[1,0]);self.assertEqual(out[1],[2,5])
        self.assertAlmostEqual(sum(x*x for x in norm[1])/2,1,places=5)

    def test_serial_chain(self):
        # T=3, K=2, table[0] lane0=1; table[1] lane1=0
        idx=[[1,0],[1,0], [0,1],[0,0]]
        candidates=[[5,9],[6,8], [12,13],[20,21]]
        self.assertEqual(reference.serial_chain(idx,candidates,3),[9,6,12,20])


if __name__=='__main__':unittest.main(verbosity=2)
