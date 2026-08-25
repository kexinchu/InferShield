#!/usr/bin/env python3
"""Download the two missing paper backbones into InferShield/Models."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

MODELS = (
    (
        "nvidia/Llama-3.3-70B-Instruct-FP8",
        Path("/workspace/Models/Llama-3.3-70B-Instruct-FP8"),
    ),
    (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        Path("/workspace/Models/DeepSeek-R1-Distill-Qwen-32B"),
    ),
    (
        "casperhansen/llama-3.3-70b-instruct-awq",
        Path("/workspace/Models/Llama-3.3-70B-Instruct-AWQ"),
    ),
)


def main() -> int:
    repo_filter = sys.argv[1] if len(sys.argv) > 1 else None
    for repo_id, dest in MODELS:
        if repo_filter and repo_filter not in repo_id and repo_filter not in str(dest):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOAD_START] repo={repo_id} dest={dest}", flush=True)
        t0 = time.time()
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            resume_download=True,
        )
        print(
            f"[DOWNLOAD_DONE] repo={repo_id} path={path} elapsed_s={time.time() - t0:.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
