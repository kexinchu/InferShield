#!/usr/bin/env python3
"""SafeKV Exp #5 – Strict / Balanced / Public serving performance.

Experiment deliverable:
  "Strict/Balanced/Public performance results"

Measures TTFT (ms) and throughput (tok/s) for each SafeKV policy variant
against the three workload patterns used in the existing frozen results:
  - Single-Request PII   (no cross-tenant prefix overlap)
  - Multi-Turn Chat      (repeated within-session prefixes)
  - System Prompt        (shared system prompt across tenants)

Baselines measured (each requires the correct server config):
  - SafeKV-Strict        (--safekv-mode strict)
  - SafeKV-Balanced      (--safekv-mode balanced)
  - SafeKV-Balanced+Public  (balanced + pre-warmed Verified-Public object)

This script is a measurement harness.  Start the server with the desired
--safekv-mode before running.  Results are appended to a CSV so that all
baseline runs can be aggregated offline.

Usage:
  # Start server with: --safekv-mode strict
  python exp5_serving_perf.py --server http://127.0.0.1:8092 \
      --model phi4 --workload system_prompt --policy strict \
      --n-users 20 --rps 16 --output results/exp5/phi4_strict.csv

  # Start server with: --safekv-mode balanced
  python exp5_serving_perf.py --server http://127.0.0.1:8092 \
      --model phi4 --workload system_prompt --policy balanced \
      --n-users 20 --rps 16 --output results/exp5/phi4_balanced.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from transformers import AutoTokenizer

WORKLOADS = ("single_pii", "multi_turn", "system_prompt")
POLICIES = (
    "strict",
    "balanced",
    "balanced_public",
    "vanilla",
    "cache_partition",
    "shared_system_prompt_emulation",
)


# ── Workload generators ───────────────────────────────────────────────────────

SHARED_SYSTEM_PROMPT_TOKENS = 512  # tokens in the shared system prompt

def _load_texts(dataset: Path, count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    texts = []
    with dataset.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            text = item.get("source_text") or item.get("text", "")
            if len(text.strip()) >= 80:
                texts.append(text.strip())
    rng.shuffle(texts)
    return texts[:count]


def build_requests(
    workload: str,
    texts: List[str],
    tokenizer,
    n_users: int,
    seed: int,
    public_auth: Optional[Dict] = None,
) -> List[Dict]:
    """Return a list of request dicts ready to POST to /generate."""
    rng = random.Random(seed + 1)
    reqs = []

    if workload == "single_pii":
        # Each request is independent; no cross-tenant prefix overlap.
        for i, text in enumerate(texts[:n_users]):
            ids = tokenizer.encode(text, add_special_tokens=True)[:256]
            reqs.append({
                "user_id": f"u{i}",
                "input_ids": ids,
                "auth": None,
            })

    elif workload == "multi_turn":
        # Each user sends 4 turns; turns share a within-session prefix.
        for i in range(n_users):
            session_text = texts[i % len(texts)]
            base_ids = tokenizer.encode(session_text, add_special_tokens=True)[:128]
            for turn in range(4):
                extra = rng.randint(16, 48)
                ids = base_ids + list(rng.randrange(100, 5000) for _ in range(extra))
                reqs.append({
                    "user_id": f"u{i}",
                    "input_ids": ids,
                    "auth": None,
                })

    elif workload == "system_prompt":
        # All users share the same system prompt tokens; payload is user-specific.
        sys_text = texts[0]
        sys_ids = tokenizer.encode(sys_text, add_special_tokens=True)[:SHARED_SYSTEM_PROMPT_TOKENS]
        for i in range(n_users):
            user_text = texts[(i + 1) % len(texts)]
            user_ids = tokenizer.encode(user_text, add_special_tokens=True)[:64]
            ids = sys_ids + user_ids
            reqs.append({
                "user_id": f"u{i}",
                "input_ids": ids,
                "auth": public_auth,  # non-None only for balanced_public
            })

    rng.shuffle(reqs)
    return reqs


# ── Server client ─────────────────────────────────────────────────────────────

class PerfClient:
    def __init__(self, server: str, timeout: float = 300.0):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        try:
            r = requests.post(f"{self.server}/flush_cache", timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"[warn] flush_cache failed ({e}); continuing without flush")
        time.sleep(0.5)

    def generate(self, req: Dict, max_new_tokens: int = 64) -> Tuple[float, int]:
        """POST one request; return (ttft_ms, output_tokens)."""
        params: Dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "user_id": req["user_id"],
        }
        if req.get("auth"):
            params["safekv_public_authorization"] = req["auth"]
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.server}/generate",
            json={"input_ids": req["input_ids"], "sampling_params": params, "stream": True},
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        ttft_ms: Optional[float] = None
        out_tokens = 0
        last_completion_tokens = 0
        saw_error = None
        for raw_line in resp.iter_lines(chunk_size=1):
            if not raw_line:
                continue
            # Some proxies yield str; normalize to bytes for prefix checks.
            if isinstance(raw_line, str):
                raw_line = raw_line.encode("utf-8")
            if not raw_line.startswith(b"data:"):
                continue
            payload = raw_line.split(b"data:", 1)[1].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if isinstance(chunk, dict) and chunk.get("error"):
                saw_error = chunk["error"]
            meta = chunk.get("meta_info") or {}
            if "completion_tokens" in meta:
                last_completion_tokens = int(meta["completion_tokens"] or 0)
            token_ids = chunk.get("token_ids") or chunk.get("output_ids") or []
            text = chunk.get("text")
            # First stream event with progress counts as TTFT. Immediate EOS can
            # yield text="" with finish_reason=stop and completion_tokens>=1.
            if ttft_ms is None and (
                token_ids
                or text not in (None, "")
                or last_completion_tokens > 0
                or meta.get("finish_reason") is not None
            ):
                ttft_ms = (time.perf_counter() - t0) * 1000
            if token_ids:
                out_tokens += len(token_ids)
        if ttft_ms is None:
            detail = f" error={saw_error}" if saw_error else ""
            raise RuntimeError(
                "stream completed without an output token" + detail
            )
        if out_tokens <= 0:
            out_tokens = last_completion_tokens
        return ttft_ms, out_tokens

    def prewarm_public(self, token_ids: List[int], auth: Dict) -> None:
        """Insert a Verified-Public object (operator prewarm)."""
        from sglang.srt.mem_cache.safekv_policy import PublicAuthorization
        params: Dict = {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "user_id": "operator-prewarm",
            "safekv_public_authorization": auth,
        }
        r = requests.post(
            f"{self.server}/generate",
            json={"input_ids": token_ids, "sampling_params": params, "stream": False},
            timeout=self.timeout,
        )
        r.raise_for_status()


# ── Rate-limited throughput driver ────────────────────────────────────────────

def run_workload(
    client: PerfClient,
    reqs: List[Dict],
    rps: float,
    max_new_tokens: int = 64,
    warmup: int = 5,
) -> Tuple[List[float], float]:
    """Send requests at target RPS; return (ttft_list_ms, throughput_tok_s).

    Requests are dispatched in submission order with 1/rps spacing;
    responses are collected asynchronously.  Output tokens from warmup
    requests are excluded from throughput but are counted in elapsed time
    (conservative denominator).
    """
    interval = 1.0 / rps
    results: List[Tuple[float, int]] = [None] * len(reqs)  # type: ignore

    def _worker(idx: int, req: Dict) -> Tuple[int, float, int]:
        ttft, out = client.generate(req, max_new_tokens=max_new_tokens)
        return idx, ttft, out

    t_run_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(reqs), 64)) as pool:
        futures = []
        for i, req in enumerate(reqs):
            if i > 0:
                time.sleep(interval)
            futures.append(pool.submit(_worker, i, req))
        for fut in as_completed(futures):
            idx, ttft, out = fut.result()
            results[idx] = (ttft, out)
    t_run_end = time.perf_counter()

    elapsed = t_run_end - t_run_start
    ttfts: List[float] = []
    total_out_tokens = 0
    for i, (ttft, out) in enumerate(results):
        if i >= warmup:
            ttfts.append(ttft)
            total_out_tokens += out
    throughput = total_out_tokens / elapsed if elapsed > 0 and total_out_tokens > 0 else 0.0
    return ttfts, throughput


# ── Public authorization builder ──────────────────────────────────────────────

def build_public_auth(
    token_ids: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
) -> Dict:
    """Issue a valid Verified-Public authorization for the given system prompt."""
    import time as _time
    from sglang.srt.mem_cache.safekv_policy import PublicRegistry

    key = operator_key.encode("utf-8")
    registry = PublicRegistry(key, policy_epoch=1)
    auth = registry.issue(
        public_object_id="exp5-system-prompt",
        issuer="exp5-ctrl",
        model_id=model_id,
        tokenizer_version=tokenizer_version,
        token_ids=token_ids,
        expires_at=_time.time() + 86400,
    )
    return auth.to_dict()


# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_FIELDS = (
    "model", "policy", "workload", "n_users", "rps",
    "n_requests", "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "throughput_tok_s",
)


def append_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--workload",
        choices=WORKLOADS,
        default="system_prompt",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        required=True,
        help="SafeKV policy the server is configured with",
    )
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--rps", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Output tokens per request (≥64 for meaningful throughput)")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parents[1] / "datasets" / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operator-key",
        default=os.environ.get("SAFEKV_OPERATOR_KEY", "safekv-exp5-operator-key"),
    )
    args = parser.parse_args()

    if (
        args.policy == "shared_system_prompt_emulation"
        and args.workload != "system_prompt"
    ):
        parser.error(
            "shared_system_prompt_emulation is defined only for system_prompt"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    client = PerfClient(args.server)

    server_info = requests.get(
        f"{args.server.rstrip('/')}/get_model_info", timeout=30
    ).json()
    tokenizer_version = server_info["tokenizer_path"]

    texts = _load_texts(args.dataset, args.n_users + 10, args.seed)

    # Every measured cell starts from an empty cache. Public-prefix baselines
    # install their authorized object after this flush so prewarming survives.
    client.flush()

    public_auth = None
    if (
        args.policy in ("balanced_public", "shared_system_prompt_emulation")
        and args.workload == "system_prompt"
    ):
        # Build and install a Verified-Public authorization for the system prompt.
        sys_text = texts[0]
        sys_ids = tokenizer.encode(sys_text, add_special_tokens=True)[:SHARED_SYSTEM_PROMPT_TOKENS]
        public_auth = build_public_auth(sys_ids, args.model, tokenizer_version, args.operator_key)
        print(f"[P5] Prewarming Verified-Public object for system prompt ({len(sys_ids)} tokens)…")
        client.prewarm_public(sys_ids, public_auth)
        print("[P5] Prewarm done.")

    reqs = build_requests(
        args.workload, texts, tokenizer, args.n_users, args.seed, public_auth
    )
    print(
        f"[P5] {args.policy} / {args.workload} / {len(reqs)} requests at {args.rps} RPS"
    )

    ttfts, throughput = run_workload(
        client, reqs, args.rps,
        max_new_tokens=args.max_new_tokens,
        warmup=5,
    )

    n = len(ttfts)
    sorted_ttfts = sorted(ttfts)
    row = {
        "model": args.model,
        "policy": args.policy,
        "workload": args.workload,
        "n_users": args.n_users,
        "rps": args.rps,
        "n_requests": n,
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else 0,
        "p50_ttft_ms": sorted_ttfts[n // 2] if n else 0,
        "p95_ttft_ms": sorted_ttfts[int(n * 0.95)] if n else 0,
        "p99_ttft_ms": sorted_ttfts[int(n * 0.99)] if n else 0,
        "throughput_tok_s": throughput,
    }
    append_row(args.output, row)

    print(f"\n[P5] Results:")
    print(f"  mean TTFT = {row['mean_ttft_ms']:.1f}ms")
    print(f"  p50  TTFT = {row['p50_ttft_ms']:.1f}ms")
    print(f"  p95  TTFT = {row['p95_ttft_ms']:.1f}ms")
    print(f"  throughput = {row['throughput_tok_s']:.1f} tok/s")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
