#!/usr/bin/env python3
"""Offline unit tests for the Phase 1 benchmark harness. No GPU or model required."""
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
import sqlite3
import struct
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import benchmark as b
from inspect_gguf import GGUFError, inspect
from profile_report import summarize as summarize_profile
from spec_log_report import parse_log


def _str(value):
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def fake_gguf(path, arch="qwen35", with_nextn=True):
    values = [("general.architecture", 8, arch), ("qwen35.block_count", 4, 64)]
    if with_nextn:
        values.append(("qwen35.nextn_predict_layers", 4, 1))
    out = io.BytesIO()
    out.write(b"GGUF" + struct.pack("<IQQ", 3, 2, len(values)))
    for key, kind, val in values:
        out.write(_str(key) + struct.pack("<I", kind))
        out.write(_str(val) if kind == 8 else struct.pack("<I", val))
    for name in ("token_embd.weight", "blk.64.nextn.eh_proj.weight"):
        out.write(_str(name))
        out.write(struct.pack("<IQQIQ", 2, 128, 256, 0, 0))
    path.write_bytes(out.getvalue())


class TestGGUF(unittest.TestCase):
    def test_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.gguf"
            fake_gguf(path)
            result = inspect(path)
            self.assertEqual(result["metadata"]["general.architecture"], "qwen35")
            self.assertEqual(result["metadata"]["qwen35.nextn_predict_layers"], 1)
            self.assertEqual(result["tensor_count"], 2)
            self.assertEqual(result["mtp_tensor_count"], 1)

    def test_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "truncated.gguf"
            path.write_bytes(b"GGUF\x03")
            with self.assertRaises(GGUFError):
                inspect(path)

    def test_model_pair_rejects_non_mtp(self):
        with tempfile.TemporaryDirectory() as temp:
            target, mtp = Path(temp) / "target.gguf", Path(temp) / "mtp.gguf"
            fake_gguf(target)
            fake_gguf(mtp, with_nextn=False)
            with self.assertRaisesRegex(RuntimeError, "nextn_predict_layers"):
                b.ensure_models(argparse.Namespace(target=target, mtp=mtp))

    def test_model_pair_rejects_arch_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            target, mtp = Path(temp) / "target.gguf", Path(temp) / "mtp.gguf"
            fake_gguf(target)
            fake_gguf(mtp, arch="llama")
            with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
                b.ensure_models(argparse.Namespace(target=target, mtp=mtp))


class TestBench(unittest.TestCase):
    def cfg(self):
        return b.parse_args(["--server", "/tmp/llama-server", "--target", "/tmp/target.gguf",
                             "--mtp", "/tmp/mtp.gguf", "--dry-run"])

    def test_all_three_modes_share_primary_and_layers(self):
        cfg = self.cfg()
        for mode in "ABC":
            command = b.build_command(cfg, mode, 4096, 3, 8080)
            self.assertEqual(command[command.index("--device") + 1], "CUDA0")
            self.assertEqual(command[command.index("--n-gpu-layers") + 1], "35")
            self.assertEqual(command[command.index("--fit") + 1], "off")
            self.assertNotIn("--rpc", command)
        a = b.build_command(cfg, "A", 4096, 0, 8080)
        primary = b.build_command(cfg, "B", 4096, 3, 8080)
        secondary = b.build_command(cfg, "C", 4096, 3, 8080)
        self.assertNotIn("--spec-draft-model", a)
        self.assertIn("--spec-draft-model", primary)
        self.assertEqual(primary[primary.index("--spec-draft-device") + 1], "CUDA0")
        self.assertEqual(secondary[secondary.index("--spec-draft-device") + 1], "CUDA1")

    def test_cases(self):
        cases = list(b.cases([4096, 16384], [1, 2], ["A", "B", "C"]))
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case.mode == "A" for case in cases), 2)

    def test_metrics(self):
        response = {"content": "same", "tokens": [1, 2, 3], "timings": {
            "predicted_n": 3, "predicted_ms": 100.0, "predicted_per_second": 30,
            "draft_n": 5, "draft_n_accepted": 4, "prompt_n": 100}}
        row, raw = b.flatten_result(response, b.Case("C", 4096, 3), "short", 128, 0, .12)
        self.assertEqual(row["predicted_n"], 3)
        self.assertEqual(row["draft_accept_rate"], .8)
        self.assertEqual(row["predicted_tokens_per_second"], 30)
        self.assertTrue(row["has_token_ids"])
        self.assertEqual(raw["tokens"], [1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, "no draft tokens"):
            b.flatten_result({"content": "x", "timings": {"predicted_n": 1}},
                             b.Case("B", 4096, 1), "short", 8, 0, 1)

    def test_mismatch_fails_closed(self):
        cases = [{"n_ctx": 4096, "prompt_profile": "short", "max_tokens": 64,
                  "repeat": 0, "mode": mode, "case": mode + "_ctx4096_depth0",
                  "output_sha256": digest}
                 for mode, digest in (("A", "same"), ("B", "same"), ("C", "different"))]
        self.assertEqual(len(b.compare_to_baseline(cases)), 1)

    def test_summary_medians(self):
        rows = [{"mode": "A", "n_ctx": 4096, "depth": 0, "prompt_profile": "short",
                 "max_tokens": 128, "predicted_tokens_per_second": tps,
                 "predicted_ms": 128000 / tps, "prompt_ms": 40,
                 "draft_n": 0, "draft_n_accepted": 0, "output_sha256": "same"}
                for tps in [10.0, 20.0, 30.0]]
        result = b.summarize(rows)
        self.assertEqual(result[0]["median_accepted_output_tokens_per_sec"], 20.0)
        self.assertEqual(result[0]["trials"], 3)

    def test_local_http_mock_end_to_end(self):
        # Mock only the HTTP protocol. This does not exercise model inference or GPUs.
        with tempfile.TemporaryDirectory() as root:
            temp = Path(root)
            target, mtp = temp / "target.gguf", temp / "mtp.gguf"
            fake_gguf(target)
            fake_gguf(mtp)
            script = temp / ("fake-llama-server.py" if os.name == "nt" else "fake-llama-server")
            script.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                import json, sys
                from http.server import BaseHTTPRequestHandler, HTTPServer
                if '--version' in sys.argv:
                    print('fake server: synthetic harness unit test ONLY');sys.exit(0)
                if '--list-devices' in sys.argv:
                    print('CUDA0: GeForce RTX 5080 Laptop GPU (mock)')
                    print('CUDA1: GeForce RTX 3070 Ti (mock)')
                    sys.exit(0)
                def get(k):
                    return sys.argv[sys.argv.index(k)+1]
                mode = get('--spec-type')
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self,*a):pass
                    def answer(self,data):
                        body=json.dumps(data).encode()
                        self.send_response(200)
                        self.send_header('Content-Type','application/json')
                        self.send_header('Content-Length',str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    def do_GET(self):
                        self.answer({'status':'ok'})
                    def do_POST(self):
                        count=int(self.headers['Content-Length'])
                        args=json.loads(self.rfile.read(count))
                        self.answer({'content':'xyz','tokens':[4,5,6],'timings':{
                            'predicted_n':3,'predicted_per_second':30.0,
                            'predicted_ms':100.0,'prompt_ms':10.0,'prompt_n':32,
                            'draft_n':0 if mode=='none' else 5,
                            'draft_n_accepted':0 if mode=='none' else 3}})
                HTTPServer(('127.0.0.1',int(get('--port'))),Handler).serve_forever()
            """))
            if os.name == "nt":
                server = temp / "fake-llama-server.cmd"
                server.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
            else:
                server = script
                server.chmod(0o755)
            args = ["--server", str(server), "--target", str(target), "--mtp", str(mtp),
                    "--contexts", "4096", "--depths", "1", "--repeats", "1",
                    "--generations", "16", "--modes", "A,B,C",
                    "--output-dir", str(temp / "results"),
                    "--nvidia-smi", str(temp / "no-smi"), "--load-timeout", "20"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = b.main(args)
            self.assertEqual(rc, 0)
            output = temp / "results"
            self.assertEqual(json.loads((output / "failures.json").read_text()), [])
            self.assertEqual(json.loads((output / "mismatches.json").read_text()), [])
            summaries = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summaries), 3)
            self.assertEqual({row["mode"] for row in summaries}, {"A", "B", "C"})
            self.assertEqual(next(row for row in summaries if row["mode"] == "C")["acceptance_rate"], .6)

    def test_nvidia_smi_fallback_parse(self):
        six = b.GPUProbe.parse("0, GPU-ABC, GeForce RTX 5080, 9000, 75, 20", "trial", "now")
        four = b.GPUProbe.parse("1, GPU-DEF, GeForce RTX 3070 Ti, 2000", "trial", "now")
        self.assertEqual(six[0]["gpu_util_pct"], 75)
        self.assertIsNone(four[0]["gpu_util_pct"])
        self.assertEqual(four[0]["memory_mib"], 2000)

    def test_device_preflight(self):
        class Response:
            returncode = 0
            stdout = ("Available devices:\n"
                      "  CUDA0: NVIDIA GeForce RTX 5080 Laptop GPU (16000 MiB)\n"
                      "  CUDA1: NVIDIA GeForce RTX 3070 Ti (8100 MiB)\n")
            stderr = ""
        with patch.object(b.subprocess, "run", return_value=Response()):
            devices = b.check_devices(Path("/none/llama-server"), "CUDA0", "CUDA1", "5080", "3070 Ti")
            self.assertIn("CUDA1", devices)
            with self.assertRaisesRegex(RuntimeError, "not expected"):
                b.check_devices(Path("/none/llama-server"), "CUDA0", "CUDA1", "3070 Ti", "5080")


class TestProfile(unittest.TestCase):
    def test_draft_log_stats(self):
        logs = ("SPC statistics draft-mtp: #calls(b,g,a) = 1 10 10, dur(b,g,a) = 50.00, 250.00, 8.00 ms\n"
                "target h rows on the device: no\n"
                "SPC statistics draft-mtp: #calls(b,g,a) = 2 30 28, dur(b,g,a) = 70.00, 750.00, 12.00 ms\n")
        result = parse_log(logs)
        self.assertEqual(result["device_hidden_state_path"], "no")
        self.assertEqual(result["after_first_snapshot_delta"]["draft_calls"], 20)
        self.assertEqual(result["mean_draft_ms_per_draft_call"], 25)

    def test_sqlite_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.sqlite"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INT, start INT, end INT)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1,10000000,17000000)")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (deviceId INT, start INT, end INT, bytes INT)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1,0,1000000,4096)")
            db.commit()
            db.close()
            report = summarize_profile(path)
            kernel = report["tables"]["CUPTI_ACTIVITY_KIND_KERNEL"][0]
            self.assertEqual(kernel["device_id"], 1)
            self.assertEqual(kernel["count"], 1)
            self.assertEqual(kernel["cumulative_gpu_ms"], 7.0)
            self.assertEqual(report["tables"]["CUPTI_ACTIVITY_KIND_MEMCPY"][0]["bytes"], 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
