#!/usr/bin/env python3
"""Measure one Balanced-B serving cell for the reuse-risk curve.

Must send ``text`` (not input_ids-only). Token-only requests skip the
detector and stay Private.

The shared prefix must be text the conservative detector actually clears;
otherwise nodes never become Budgeted-Shared and other principals cannot hit.
This script polls /get_server_info until a budgeted_shared variant appears
(or the wait expires) and records those visibilities.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# Public-domain prose; avoid "user"/account-like tokens that Tier-2 FPs.
BENIGN_SYSTEM_PROMPT = (
    "Photosynthesis is the process by which green plants convert light energy "
    "into chemical energy. Carbon dioxide and water become sugars and oxygen. "
) * 8

USER_PAYLOADS = [
    "Summarize the steps to brew pour-over coffee.",
    "List three warm-up stretches for desk workers.",
    "Explain what a hash table is in one paragraph.",
    "Give two tips for watering indoor plants.",
    "Describe how a public library card usually works.",
]


def _extract_meta(body) -> Dict:
    if isinstance(body, list) and body:
        body = body[0]
    if not isinstance(body, dict):
        return {}
    return body.get("meta_info") or {}


def generate(server: str, req: Dict, max_new_tokens: int, timeout: float) -> Dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server}/generate",
        json={
            "text": req["text"],
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "user_id": req["user_id"],
            },
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    meta = _extract_meta(resp.json())
    return {
        "user_id": req["user_id"],
        "ttft_ms": (time.perf_counter() - t0) * 1000.0,
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "cached_tokens": int(meta.get("cached_tokens") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
    }


def snapshot_visibilities(server: str) -> Tuple[Counter, List[str]]:
    info = requests.get(f"{server}/get_server_info", timeout=30).json()
    states = info.get("internal_states") or []
    vis = Counter()
    creators = []
    for state in states:
        safekv = state.get("safekv") or {}
        for var in safekv.get("variants") or []:
            vis[str(var.get("visibility"))] += 1
            creators.append(str(var.get("creator_id")))
    return vis, creators


def wait_for_budgeted(server: str, timeout_s: float) -> Tuple[Counter, float]:
    deadline = time.time() + timeout_s
    last = Counter()
    while True:
        last, _ = snapshot_visibilities(server)
        if last.get("budgeted_shared", 0) > 0:
            return last, timeout_s - max(0.0, deadline - time.time())
        if time.time() >= deadline:
            return last, timeout_s
        time.sleep(0.5)


def build_requests(n_users: int) -> List[Dict]:
    reqs = []
    for i in range(n_users):
        payload = USER_PAYLOADS[i % len(USER_PAYLOADS)]
        reqs.append(
            {
                "user_id": f"u{i}",
                "text": BENIGN_SYSTEM_PROMPT + payload,
            }
        )
    return reqs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--admit-wait-s", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reqs = build_requests(args.n_users)
    owner = generate(args.server, reqs[0], args.max_new_tokens, args.timeout)
    same_user = generate(args.server, reqs[0], args.max_new_tokens, args.timeout)
    vis, waited = wait_for_budgeted(args.server, args.admit_wait_s)

    probes = [
        generate(args.server, req, args.max_new_tokens, args.timeout)
        for req in reqs[1:]
    ]
    prompt = sum(r["prompt_tokens"] for r in probes)
    cached = sum(r["cached_tokens"] for r in probes)
    ttfts = [r["ttft_ms"] for r in probes]
    cell = {
        "model": args.model,
        "B": args.B,
        "workload": "benign_system_prompt_no_public",
        "n_users": args.n_users,
        "n_measured": len(probes),
        "admit_wait_s": args.admit_wait_s,
        "admit_waited_s": waited,
        "visibility_counts": dict(vis),
        "owner_cached_tokens": owner["cached_tokens"],
        "same_user_cached_tokens": same_user["cached_tokens"],
        "ttft_s": (sum(ttfts) / len(ttfts) / 1000.0) if ttfts else None,
        "reuse_hit_rate": (cached / prompt) if prompt else None,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "probe_cached": [r["cached_tokens"] for r in probes],
        "admission": "experiment_autoshare",
        "source": "measure_reuse_b_cell.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cell, indent=2) + "\n")
    print(json.dumps(cell, indent=2))


if __name__ == "__main__":
    main()
