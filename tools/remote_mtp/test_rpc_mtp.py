#!/usr/bin/env python3
"""Offline regression tests for dedicated GGML RPC draft placement.

These tests do NOT run remote GPUs, and do not claim model correctness.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

# Ensure independent test discovery works from a fresh Git checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1_mtp"))

import benchmark as bench
import rpc_benchmark as rpc


def params():
    a = bench.parse_args([
        "--server", "/tmp/llama-server",
        "--target", "/tmp/Qwen3.8-27B-Q4_K_M.gguf",
        "--mtp", "/tmp/mtp-Qwen3.8-27B-Q4_0.gguf",
        "--primary-device", "CUDA0", "--draft-device", "RPC0",
        "--target-ngl", "35", "--contexts", "4096", "--depths", "1,3",
        "--generations", "128", "--modes", "A,B",
    ])
    a.worker = "192.168.1.80:50052"
    a.rpc_draft_device = "RPC0"
    return a


class TestRPCDeviceRouting(unittest.TestCase):
    def test_explicit_target_only_local_draft_only_rpc(self):
        a = params()
        for mode in ("A", "B", "R"):
            command = rpc.build_command(a, mode, 4096, 3, 8080)
            self.assertEqual(command[command.index("--device") + 1], "CUDA0")
            self.assertEqual(command[command.index("--n-gpu-layers") + 1], "35")
            self.assertEqual(command[command.index("--fit") + 1], "off")
        baseline = rpc.build_command(a, "A", 4096, 0, 8080)
        local = rpc.build_command(a, "B", 4096, 3, 8080)
        remote = rpc.build_command(a, "R", 4096, 3, 8080)
        self.assertNotIn("--spec-draft-device", baseline)
        self.assertNotIn("--rpc", baseline)
        self.assertNotIn("--rpc", local)
        self.assertEqual(local[local.index("--spec-draft-device") + 1], "CUDA0")
        self.assertEqual(remote[1:3], ["--rpc", "192.168.1.80:50052"])
        self.assertEqual(remote[remote.index("--spec-draft-device") + 1], "RPC0")
        self.assertEqual(remote[remote.index("--spec-draft-model") + 1], str(a.mtp))

    def test_modes_exclude_egpu(self):
        cases = list(rpc.make_cases([4096, 16384], [1, 3], ["A", "B", "R"]))
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case.mode == "R" for case in cases), 4)
        self.assertFalse(any(case.mode == "C" for case in cases))

    def test_device_listing_exact_names(self):
        devices = rpc.parse_devices("Available devices:\n"
                                    "  CUDA0: NVIDIA RTX 5080 Laptop (16302 MiB)\n"
                                    "  RPC0: 192.168.1.80:50052 (8192 MiB)\n")
        self.assertEqual(devices["CUDA0"].split()[1], "RTX")
        self.assertEqual(devices["RPC0"].split()[0], "192.168.1.80:50052")

    def test_preflight_remote_endpoint_and_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            server = Path(temp) / "llama-server.exe"
            server.write_bytes(b"fake")
            a = SimpleNamespace(server=server, worker="192.168.1.80:50052",
                                primary_device="CUDA0", rpc_draft_device="RPC0",
                                expect_primary="5080")
            with patch.object(rpc, "get_device_listing", side_effect=[
                {"devices": {"CUDA0": "NVIDIA RTX 5080 Laptop"}, "output": "", "cmd": []},
                {"devices": {"CUDA0": "NVIDIA RTX 5080 Laptop",
                             "RPC0": "192.168.1.80:50052"}, "output": "", "cmd": []}
            ]):
                preflight = rpc.preflight(a)
            self.assertFalse(preflight["target_on_rpc"])
            self.assertEqual(preflight["draft_device"], "RPC0")

    def test_preflight_rejects_wrong_rpc(self):
        with tempfile.TemporaryDirectory() as temp:
            server = Path(temp) / "llama-server.exe"
            server.write_bytes(b"fake")
            a = SimpleNamespace(server=server, worker="192.168.1.80:50052",
                                primary_device="CUDA0", rpc_draft_device="RPC0",
                                expect_primary="5080")
            local = {"devices": {"CUDA0": "NVIDIA RTX 5080 Laptop"}, "output": "", "cmd": []}
            remote = {"devices": {"CUDA0": "NVIDIA RTX 5080 Laptop",
                                  "RPC0": "192.168.1.90:50052"}, "output": "", "cmd": []}
            with patch.object(rpc, "get_device_listing", side_effect=[local, remote]):
                with self.assertRaisesRegex(RuntimeError, "not worker"):
                    rpc.preflight(a)

    def test_preflight_rejects_rpc_as_target(self):
        with tempfile.TemporaryDirectory() as temp:
            server = Path(temp) / "llama-server.exe"
            server.write_bytes(b"fake")
            a = SimpleNamespace(server=server, worker="192.168.1.80:50052",
                                primary_device="RPC0", rpc_draft_device="RPC1",
                                expect_primary="5080")
            with self.assertRaisesRegex(ValueError, "local CUDA"):
                rpc.preflight(a)


class TestRepeatedBenchPreset(unittest.TestCase):
    def test_repeat_is_five_samples_at_same_depth(self):
        options = rpc.parse_args([
            "--action", "repeat",
            "--server", "llama-server.exe",
            "--worker", "192.168.1.13:50052",
            "--target", "target.gguf",
            "--mtp", "mtp.gguf",
            "--llama-repo", ".",
            "--output-dir", "results_repeat",
        ])
        self.assertEqual(options.repeats, 5)
        self.assertEqual(options.contexts, [4096])
        self.assertEqual(options.depths, [3])
        self.assertEqual(options.generations, [256])
        self.assertEqual(options.modes, ["A", "B", "R"])

    def test_remote_local_ratio_uses_medians(self):
        make = lambda mode, depth, rate: {
            "mode": mode, "context": 4096, "depth": depth, "prompt": "short",
            "max_tokens": 256, "median_accepted_output_tokens_per_sec": rate,
        }
        cases = [
            make("A", 0, 6.0),
            make("B", 3, 15.0),
            make("R", 3, 14.7),
        ]
        report = rpc.add_local_remote_ratios(cases)
        self.assertAlmostEqual(report[2]["throughput_ratio_R_to_B"], 0.98)
        self.assertIsNone(report[1]["throughput_ratio_R_to_B"])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "summary.csv"
            bench.save_csv(out, report)
            self.assertIn("throughput_ratio_R_to_B", out.read_text())

    def test_remote_ratio_missing_local_comparison(self):
        missing = [{
            "mode": "R", "context": 4096, "depth": 3, "prompt": "short",
            "max_tokens": 256, "median_accepted_output_tokens_per_sec": 10.0,
        }]
        self.assertIsNone(rpc.add_local_remote_ratios(missing)[0]["throughput_ratio_R_to_B"])


class TestPrecisionAndRemoteOnly(unittest.TestCase):
    def test_remote_only_repeat_skips_local_mtp(self):
        opts = rpc.parse_args([
            "--action", "repeat", "--server", "llama-server.exe",
            "--worker", "192.168.1.13:50052",
            "--target", "target.gguf", "--mtp", "mtp-Qwen3.8-27B-BF16.gguf",
            "--llama-repo", ".", "--output-dir", "results_bf16", "--remote-only"
        ])
        self.assertEqual(opts.modes, ["A", "R"])
        self.assertEqual(opts.repeats, 5)

    def test_remote_only_uses_selected_bf16_sidecar(self):
        a = params()
        a.mtp = Path("mtp-Qwen3.8-27B-BF16.gguf")
        command = rpc.build_command(a, "R", 4096, 3, 8090)
        self.assertEqual(command[command.index("--spec-draft-model") + 1],
                         str(a.mtp))
        self.assertEqual(command[command.index("--device") + 1], "CUDA0")
        self.assertEqual(command[command.index("--spec-draft-device") + 1], "RPC0")


class TestRemoteResults(unittest.TestCase):
    def test_remote_matches_local(self):
        make = lambda mode, hash: {
            "mode": mode, "n_ctx": 4096, "depth": 3, "prompt_profile": "short",
            "max_tokens": 128, "repeat": 0, "output_sha256": hash, "output_excerpt": "txt"}
        comparisons, problems = rpc.local_vs_remote([make("B", "same"), make("R", "same")])
        self.assertEqual(comparisons, 1)
        self.assertEqual(problems, [])

    def test_remote_mismatch_is_detected(self):
        make = lambda mode, hash: {
            "mode": mode, "n_ctx": 4096, "depth": 3, "prompt_profile": "short",
            "max_tokens": 128, "repeat": 0, "output_sha256": hash, "output_excerpt": "txt"}
        comparisons, problems = rpc.local_vs_remote([make("B", "first"), make("R", "second")])
        self.assertEqual(comparisons, 1)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["remote_output"], "second")

    def test_target_rpc_buffer_detected_before_draft(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "R.server.log"
            path.write_text("llama_model_load: RPC0 model buffer size = 9000 MiB\n"
                            "common_speculative_init_result: loading draft model 'mtp.gguf'\n")
            self.assertEqual(rpc.check_target_placement(path)["status"], "suspected_target_leak")
            path.write_text("target on CUDA0\n"
                            "common_speculative_init_result: loading draft model 'mtp.gguf'\n"
                            "llama_model_load: RPC0 model buffer size = 1200 MiB\n")
            self.assertEqual(rpc.check_target_placement(path)["status"], "no_target_rpc_buffer_seen")

    def test_trace_count_excludes_inference_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "rpc.log"
            log.write_text("[dispatcher] SET_TENSOR (in: 1 MiB, out: 0 B)\n"
                           "[dispatcher] GRAPH_COMPUTE (in: 1 KiB, out: 0 B)\n"
                           "[dispatcher] GRAPH_COMPUTE done in 2.500 ms\n")
            summary = rpc.rpc_trace_summary([log])
            self.assertEqual(summary["command_counts"]["SET_TENSOR"], 1)
            self.assertEqual(summary["command_counts"]["GRAPH_COMPUTE"], 1)
            self.assertEqual(summary["command_timing_ms"]["GRAPH_COMPUTE"]["sum"], 2.5)
            self.assertIn("whole", summary["scope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
