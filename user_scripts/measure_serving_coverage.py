#!/usr/bin/env python3
"""Coverage cell via cached_tokens, not TTFT ratios.

Victim inserts PII prefixes; after a settle window, a different principal
probes them. A probe with cached_tokens >= hit_tokens is a failed defense.
Coverage = 100 * (1 - hits/probes).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from ablation_prompts import PII_PROMPTS

N_VICTIMS = 5
N_PROBES = 5
SETTLE_S = 8.0
HIT_TOKENS = 16


def generate(server: str, text: str, user_id: str, max_new: int = 8) -> dict:
    # Keep well under the short Llama-70B context used on 2xA6000.
    text = text[:2400]
    resp = requests.post(
        f"{server}/generate",
        json={
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_new,
                "temperature": 0.0,
                "user_id": user_id,
            },
            "stream": False,
        },
        timeout=180,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.text[:400]}")
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, list) and body:
        body = body[0]
    meta = body.get("meta_info") or {}
    return {
        "cached_tokens": int(meta.get("cached_tokens") or 0),
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ttft-s", type=float, default=None)
    parser.add_argument("--tps", type=float, default=None)
    args = parser.parse_args()

    pii = list(PII_PROMPTS[:N_VICTIMS])
    print(f"[COV] victim insert n={len(pii)}", flush=True)
    for i, text in enumerate(pii):
        generate(args.server, text, f"victim_{i}", max_new=4)
    time.sleep(SETTLE_S)

    hits = 0
    probes = 0
    cached = []
    print(f"[COV] attacker probes={len(pii)*N_PROBES}", flush=True)
    for i, text in enumerate(pii):
        for j in range(N_PROBES):
            rec = generate(args.server, text, f"attacker_{i}_{j}", max_new=4)
            cached.append(rec["cached_tokens"])
            if rec["cached_tokens"] >= HIT_TOKENS:
                hits += 1
            probes += 1

    existing = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except Exception:
            existing = {}
    cell = {
        "model": args.model,
        "B": args.B,
        "ttft_s": args.ttft_s if args.ttft_s is not None else existing.get("ttft_s"),
        "tps": args.tps if args.tps is not None else existing.get("tps"),
        "coverage_pct": (1.0 - hits / probes) * 100.0 if probes else None,
        "attacker_hits": hits,
        "total_probes": probes,
        "probe_cached_tokens": cached,
        "source": "measure_serving_coverage.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cell, indent=2) + "\n")
    print(json.dumps({k: v for k, v in cell.items() if k != "probe_cached_tokens"}, indent=2))


if __name__ == "__main__":
    main()
