#!/usr/bin/env python3
"""Fetch only the 27B Q4_K_M target GGUF and Q4_0 native MTP sidecar."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO_ID = "ggml-org/Qwen3.8-27B-GGUF"
FILES = ("Qwen3.8-27B-Q4_K_M.gguf", "mtp-Qwen3.8-27B-Q4_0.gguf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install first: py -3 -m pip install -U huggingface_hub") from exc
    args.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(FILES)} GGUF files from {REPO_ID} to {args.model_dir}", flush=True)
    for filename in FILES:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(args.model_dir))
        if not Path(path).is_file():
            raise RuntimeError(f"Download did not create a file: {path}")
        print(f"Ready: {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
