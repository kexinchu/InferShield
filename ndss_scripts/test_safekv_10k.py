#!/usr/bin/env python3
"""
SafeKV Large-Scale Benchmark
=============================
10k multi-turn queries (>=4 turns per query) x 6 PII-ratio sweeps.
Metrics: TTFT, throughput, KV-cache usage, cross-tenant PII reuse ratio.

Design:
  - Base conversations from ShareGPT (>=4 turns, long prompts)
  - PII snippets injected randomly into user turns at configurable ratio
  - 50 simulated tenants; ~20% prompts duplicated across tenants to probe reuse
  - Server-log parsing for #cached-token and token-usage

Usage:
  python test_safekv_10k.py --port 8092 --model phi4
"""

import argparse
import csv
import json
import os
import random
import re
import string
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "..", "datasets")
SHAREGPT_FILE = os.path.join(DATASETS_DIR, "ShareGPT_V3_unfiltered_cleaned_split.json")
PII_FILE = os.path.join(DATASETS_DIR, "english_pii_43k.jsonl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
HEADERS = {"Content-Type": "application/json"}

# ============================================================
# Constants
# ============================================================
NUM_USERS = 50
MIN_TURNS = 4            # minimum conversation turns per query
MAX_PROMPT_CHARS = 2048  # per-turn character cap
MAX_TOKENS = 64          # generation tokens
CROSS_TENANT_DUP_RATIO = 0.20  # 20% prompts are duplicated across users

# PII snippets pool for injection
PII_SNIPPETS = [
    "My SSN is 321-54-9876.",
    "Contact me at sarah.connor@skynet.io.",
    "My credit card: 4532-8812-3456-7890, exp 11/27, CVV 482.",
    "Call me at +1 (415) 555-0198.",
    "My passport number is X12849305.",
    "Home address: 1425 Maple Drive, Springfield, IL 62704.",
    "My bank account IBAN: DE89370400440532013000.",
    "DOB: March 15, 1988.",
    "Driver license: D400-1234-5678.",
    "IP address of my home server: 192.168.1.105.",
    "MAC address: AA:BB:CC:11:22:33.",
    "My Bitcoin wallet: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.",
    "Login password is Tr0ub4dor&3.",
    "Employee ID: EMP-2024-00491, department: Finance.",
    "Medical record #MR-88214, diagnosis: Type-2 Diabetes.",
    "Vehicle VIN: 1HGCM82633A004352.",
    "My username on the platform is dark_phoenix_99.",
    "Phone IMEI: 35-209900-176148-1.",
    "Ethereum address: 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD73.",
    "PIN code for the vault: 8837.",
]


# ============================================================
# Data loading
# ============================================================
def load_sharegpt_multiturn(path: str, n: int, seed: int = 42) -> List[List[dict]]:
    """Load ShareGPT conversations with >= MIN_TURNS turns.
    Returns list of conversations, each conversation = list of {role, content}."""
    print(f"    Loading ShareGPT from {os.path.basename(path)}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter for multi-turn
    multi = [x for x in data if len(x.get("conversations", [])) >= MIN_TURNS]
    random.seed(seed)
    random.shuffle(multi)
    print(f"    {len(multi)} conversations with >={MIN_TURNS} turns available")

    results = []
    for item in multi:
        conv = []
        for c in item["conversations"]:
            role = "user" if c["from"] == "human" else "assistant"
            text = c["value"].strip()[:MAX_PROMPT_CHARS]
            if text:
                conv.append({"role": role, "content": text})
        # Ensure at least MIN_TURNS messages and starts with user
        if len(conv) >= MIN_TURNS and conv[0]["role"] == "user":
            results.append(conv)
        if len(results) >= n:
            break
    return results


def load_pii_snippets_from_dataset(path: str, n: int = 500) -> List[str]:
    """Load real PII sentences from PII-Masking dataset as injection candidates."""
    snippets = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n * 3:
                break
            item = json.loads(line)
            text = item["source_text"].strip()
            bio = item["mbert_bio_labels"]
            if isinstance(bio, str):
                bio = eval(bio)
            if any(l != "O" for l in bio) and 20 < len(text) < 300:
                snippets.append(text)
            if len(snippets) >= n:
                break
    return snippets


# ============================================================
# PII injection
# ============================================================
def inject_pii_into_conversation(
    conv: List[dict],
    pii_pool: List[str],
    rng: random.Random,
) -> Tuple[List[dict], int]:
    """Inject PII snippets into random user turns of a conversation.
    Returns (modified_conv, num_pii_injected)."""
    modified = []
    n_injected = 0
    for msg in conv:
        if msg["role"] == "user" and rng.random() < 0.5:
            # Insert 1-2 PII snippets
            n_snip = rng.randint(1, 2)
            snippets = rng.sample(pii_pool, min(n_snip, len(pii_pool)))
            # Insert at random positions in the text
            text = msg["content"]
            sentences = text.split(". ")
            if len(sentences) > 1:
                pos = rng.randint(0, len(sentences) - 1)
                for s in snippets:
                    sentences.insert(pos, s)
                    n_injected += 1
                text = ". ".join(sentences)
            else:
                text = " ".join(snippets) + " " + text
                n_injected += len(snippets)
            modified.append({"role": msg["role"], "content": text[:MAX_PROMPT_CHARS]})
        else:
            modified.append(msg)
    return modified, n_injected


# ============================================================
# Build the full prompt from multi-turn conversation
# ============================================================
def build_chat_messages(
    conv: List[dict],
    system_prompt: str = "",
) -> List[dict]:
    """Convert conversation to chat messages for the API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Take first MIN_TURNS messages (or more)
    for msg in conv[:max(MIN_TURNS, len(conv))]:
        messages.append(msg)
    # Ensure last message is from user (API requirement)
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "Please continue."})
    return messages


# ============================================================
# Request result
# ============================================================
@dataclass
class RequestResult:
    query_id: int
    user_id: str
    has_pii: bool
    num_pii_injected: int
    latency_ms: float       # TTFT (time to first token)
    success: bool
    num_prompt_tokens: int   # approximate
    is_duplicate: bool       # cross-tenant duplicate
    status_code: int = 0
    error: str = ""


# ============================================================
# Log-based metrics collector
# ============================================================
class LogMetricsCollector:
    """Parse server log to extract cache metrics during a test run."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.start_offset = 0
        # Record file position before benchmark starts
        if os.path.exists(log_path):
            self.start_offset = os.path.getsize(log_path)

    def collect(self) -> dict:
        """Parse log lines written AFTER the benchmark started."""
        metrics = {
            "prefill_count": 0,
            "total_new_tokens": 0,
            "total_cached_tokens": 0,
            "max_token_usage": 0.0,
            "token_usage_samples": [],
            "throughput_samples": [],
        }
        if not os.path.exists(self.log_path):
            return metrics

        with open(self.log_path, "r") as f:
            f.seek(self.start_offset)
            for line in f:
                if "Prefill batch" not in line:
                    continue
                m_new = re.search(r"#new-token:\s*(\d+)", line)
                m_cached = re.search(r"#cached-token:\s*(\d+)", line)
                m_usage = re.search(r"token usage:\s*([\d.]+)", line)
                m_tp = re.search(r"input throughput \(token/s\):\s*([\d.]+)", line)

                if m_new:
                    metrics["total_new_tokens"] += int(m_new.group(1))
                if m_cached:
                    metrics["total_cached_tokens"] += int(m_cached.group(1))
                if m_usage:
                    usage = float(m_usage.group(1))
                    metrics["token_usage_samples"].append(usage)
                    metrics["max_token_usage"] = max(metrics["max_token_usage"], usage)
                if m_tp:
                    metrics["throughput_samples"].append(float(m_tp.group(1)))
                metrics["prefill_count"] += 1

        return metrics


# ============================================================
# Worker
# ============================================================
def send_one(
    query_id: int,
    messages: List[dict],
    user_id: str,
    has_pii: bool,
    num_pii: int,
    is_dup: bool,
    port: int,
    model: str,
) -> RequestResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    approx_tokens = sum(len(m["content"]) // 4 for m in messages)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "user_id": user_id,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=180)
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(
            query_id=query_id, user_id=user_id, has_pii=has_pii,
            num_pii_injected=num_pii, latency_ms=latency,
            success=(resp.status_code == 200), num_prompt_tokens=approx_tokens,
            is_duplicate=is_dup, status_code=resp.status_code,
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(
            query_id=query_id, user_id=user_id, has_pii=has_pii,
            num_pii_injected=num_pii, latency_ms=latency,
            success=False, num_prompt_tokens=approx_tokens,
            is_duplicate=is_dup, error=str(e)[:120],
        )


# ============================================================
# Build workload for a given PII ratio
# ============================================================
def build_workload(
    conversations: List[List[dict]],
    pii_pool: List[str],
    pii_ratio: float,
    num_queries: int,
    seed: int,
    system_prompt: str = "",
) -> List[Tuple[List[dict], str, bool, int, bool]]:
    """
    Returns list of (messages, user_id, has_pii, num_pii_injected, is_duplicate).
    """
    rng = random.Random(seed)
    users = [f"user_{i:03d}" for i in range(NUM_USERS)]

    n_pii = int(num_queries * pii_ratio)
    n_clean = num_queries - n_pii
    n_dup = int(num_queries * CROSS_TENANT_DUP_RATIO)
    n_unique = num_queries - n_dup

    workload = []

    # --- Unique queries ---
    idx = 0
    for i in range(min(n_unique, len(conversations))):
        conv = conversations[idx % len(conversations)]
        idx += 1
        user = rng.choice(users)

        if i < int(n_unique * pii_ratio):
            conv_mod, n_inj = inject_pii_into_conversation(conv, pii_pool, rng)
            msgs = build_chat_messages(conv_mod, system_prompt)
            workload.append((msgs, user, True, n_inj, False))
        else:
            msgs = build_chat_messages(conv, system_prompt)
            workload.append((msgs, user, False, 0, False))

    # --- Cross-tenant duplicates (same conversation, different users) ---
    # Pick a pool of conversations to duplicate
    dup_pool_size = max(1, n_dup // 4)  # ~4 users per duplicated conv
    dup_convs = conversations[:dup_pool_size]
    for i in range(n_dup):
        conv = dup_convs[i % dup_pool_size]
        user = users[i % NUM_USERS]

        if rng.random() < pii_ratio:
            conv_mod, n_inj = inject_pii_into_conversation(conv, pii_pool, rng)
            msgs = build_chat_messages(conv_mod, system_prompt)
            workload.append((msgs, user, True, n_inj, True))
        else:
            msgs = build_chat_messages(conv, system_prompt)
            workload.append((msgs, user, False, 0, True))

    # Trim to exact num_queries
    workload = workload[:num_queries]
    rng.shuffle(workload)
    return workload


# ============================================================
# Progress tracker
# ============================================================
class ProgressTracker:
    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.label = label
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, success: bool):
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1
            if self.completed % 500 == 0 or self.completed == self.total:
                elapsed = time.time() - self.start_time
                qps = self.completed / elapsed if elapsed > 0 else 0
                print(f"    [{self.label}] {self.completed}/{self.total} "
                      f"({self.failed} fail) | {qps:.1f} qps | {elapsed:.0f}s")


# ============================================================
# Run one PII-ratio experiment
# ============================================================
def run_experiment(
    pii_ratio: float,
    conversations: List[List[dict]],
    pii_pool: List[str],
    num_queries: int,
    workers: int,
    port: int,
    model: str,
    seed: int,
    log_path: str,
    system_prompt: str,
) -> Tuple[List[RequestResult], dict]:
    label = f"PII={int(pii_ratio*100):3d}%"
    print(f"\n{'='*70}")
    print(f"  Experiment: {label} | {num_queries} queries | {workers} workers")
    print(f"{'='*70}")

    # Build workload
    workload = build_workload(
        conversations, pii_pool, pii_ratio, num_queries, seed, system_prompt
    )
    n_pii = sum(1 for _, _, hp, _, _ in workload if hp)
    n_dup = sum(1 for _, _, _, _, dup in workload if dup)
    print(f"  Workload: {len(workload)} total, {n_pii} PII, {n_dup} cross-tenant dups")

    # Flush cache before each experiment
    try:
        requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=10)
        time.sleep(2)
        print(f"  Cache flushed.")
    except:
        print(f"  WARNING: Could not flush cache, results may be affected.")

    # Start log collector
    log_collector = LogMetricsCollector(log_path)
    tracker = ProgressTracker(len(workload), label)

    results: List[RequestResult] = [None] * len(workload)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, (msgs, uid, has_pii, n_inj, is_dup) in enumerate(workload):
            f = executor.submit(
                send_one, i, msgs, uid, has_pii, n_inj, is_dup, port, model
            )
            futures[f] = i

        for f in as_completed(futures):
            idx = futures[f]
            result = f.result()
            results[idx] = result
            tracker.update(result.success)

    elapsed = time.time() - start_time
    results = [r for r in results if r is not None]

    # Collect log metrics
    time.sleep(1)  # let log flush
    log_metrics = log_collector.collect()

    print(f"  Completed in {elapsed:.1f}s ({len(results)/elapsed:.1f} qps)")
    return results, log_metrics


# ============================================================
# Analyze and report
# ============================================================
def analyze_experiment(
    pii_ratio: float,
    results: List[RequestResult],
    log_metrics: dict,
) -> dict:
    ok = [r for r in results if r.success]
    if not ok:
        return {"pii_ratio": pii_ratio, "error": "no successful requests"}

    lats = [r.latency_ms for r in ok]
    pii_lats = [r.latency_ms for r in ok if r.has_pii]
    clean_lats = [r.latency_ms for r in ok if not r.has_pii]
    dup_lats = [r.latency_ms for r in ok if r.is_duplicate]
    dup_pii_lats = [r.latency_ms for r in ok if r.is_duplicate and r.has_pii]
    dup_clean_lats = [r.latency_ms for r in ok if r.is_duplicate and not r.has_pii]

    total_tokens = sum(r.num_prompt_tokens for r in ok)
    total_cached = log_metrics.get("total_cached_tokens", 0)
    total_new = log_metrics.get("total_new_tokens", 0)
    cache_hit_rate = total_cached / (total_cached + total_new) if (total_cached + total_new) > 0 else 0

    usage_samples = log_metrics.get("token_usage_samples", [])
    max_usage = log_metrics.get("max_token_usage", 0)
    avg_usage = np.mean(usage_samples) if usage_samples else 0

    # Cross-tenant PII reuse: compare dup-PII latency vs dup-clean latency
    # If PII duplicates get cache hits like clean duplicates, privacy is leaking
    pii_reuse_ratio = 0.0
    if dup_pii_lats and dup_clean_lats:
        # If PII dups are as fast as clean dups, that suggests cross-tenant reuse
        avg_dup_pii = np.mean(dup_pii_lats)
        avg_dup_clean = np.mean(dup_clean_lats)
        # Ratio close to 1.0 = PII leaking (same cache behavior)
        # Ratio >> 1.0 = PII isolated (slower because no cross-tenant cache)
        pii_reuse_ratio = avg_dup_clean / avg_dup_pii if avg_dup_pii > 0 else 0

    elapsed_s = (max(r.latency_ms for r in ok) - min(r.latency_ms for r in ok)) / 1000
    throughput_qps = len(ok) / elapsed_s if elapsed_s > 0 else 0

    stats = {
        "pii_ratio": pii_ratio,
        "total_queries": len(results),
        "success": len(ok),
        "failed": len(results) - len(ok),
        # Latency
        "ttft_mean_ms": np.mean(lats),
        "ttft_p50_ms": np.percentile(lats, 50),
        "ttft_p95_ms": np.percentile(lats, 95),
        "ttft_p99_ms": np.percentile(lats, 99),
        # PII vs clean latency
        "ttft_pii_mean_ms": np.mean(pii_lats) if pii_lats else 0,
        "ttft_clean_mean_ms": np.mean(clean_lats) if clean_lats else 0,
        # KV-cache
        "kv_cache_hit_rate": cache_hit_rate,
        "kv_total_cached_tokens": total_cached,
        "kv_total_new_tokens": total_new,
        "kv_max_usage_frac": max_usage,
        "kv_avg_usage_frac": avg_usage,
        # Throughput
        "throughput_qps": throughput_qps,
        "total_prompt_tokens": total_tokens,
        # Cross-tenant PII reuse
        "dup_pii_count": len(dup_pii_lats),
        "dup_clean_count": len(dup_clean_lats),
        "dup_pii_mean_ms": np.mean(dup_pii_lats) if dup_pii_lats else 0,
        "dup_clean_mean_ms": np.mean(dup_clean_lats) if dup_clean_lats else 0,
        "pii_cross_tenant_reuse_ratio": pii_reuse_ratio,
    }
    return stats


def print_stats(stats: dict):
    r = stats["pii_ratio"]
    print(f"\n  --- PII={int(r*100)}% Results ---")
    print(f"  Queries:        {stats['success']}/{stats['total_queries']} success")
    print(f"  TTFT mean:      {stats['ttft_mean_ms']:.1f} ms")
    print(f"  TTFT P50/P95:   {stats['ttft_p50_ms']:.1f} / {stats['ttft_p95_ms']:.1f} ms")
    print(f"  TTFT PII mean:  {stats['ttft_pii_mean_ms']:.1f} ms")
    print(f"  TTFT clean:     {stats['ttft_clean_mean_ms']:.1f} ms")
    print(f"  KV cache hit:   {stats['kv_cache_hit_rate']*100:.2f}%")
    print(f"  KV cached tok:  {stats['kv_total_cached_tokens']}")
    print(f"  KV new tok:     {stats['kv_total_new_tokens']}")
    print(f"  KV max usage:   {stats['kv_max_usage_frac']*100:.2f}%")
    print(f"  Throughput:     {stats['throughput_qps']:.1f} qps")
    print(f"  Cross-tenant PII reuse ratio: {stats['pii_cross_tenant_reuse_ratio']:.4f}")
    print(f"    (dup_pii={stats['dup_pii_count']}, "
          f"dup_pii_lat={stats['dup_pii_mean_ms']:.1f}ms, "
          f"dup_clean={stats['dup_clean_count']}, "
          f"dup_clean_lat={stats['dup_clean_mean_ms']:.1f}ms)")


# ============================================================
# Save results
# ============================================================
def save_all_results(
    all_stats: List[dict],
    all_results: Dict[float, List[RequestResult]],
    output_dir: str,
    tag: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Summary CSV: one row per PII ratio
    summary_path = os.path.join(output_dir, f"safekv_{tag}_summary.csv")
    if all_stats:
        keys = list(all_stats[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for s in all_stats:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in s.items()})
    print(f"\n  Summary CSV: {summary_path}")

    # Detail CSV: one row per request, across all experiments
    detail_path = os.path.join(output_dir, f"safekv_{tag}_detail.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pii_ratio", "query_id", "user_id", "has_pii", "num_pii_injected",
            "latency_ms", "success", "num_prompt_tokens", "is_duplicate",
        ])
        for ratio, results in sorted(all_results.items()):
            for r in results:
                writer.writerow([
                    ratio, r.query_id, r.user_id, r.has_pii, r.num_pii_injected,
                    f"{r.latency_ms:.2f}", r.success, r.num_prompt_tokens, r.is_duplicate,
                ])
    print(f"  Detail CSV:  {detail_path}")

    # Human-readable report
    report_path = os.path.join(output_dir, f"safekv_{tag}_report.txt")
    with open(report_path, "w") as f:
        f.write("SafeKV 10k Benchmark Report\n")
        f.write(f"{'='*70}\n\n")

        header = (f"{'PII%':>5} | {'TTFT':>8} {'P50':>8} {'P95':>8} | "
                  f"{'Cache%':>7} {'MaxUsg':>7} | {'QPS':>6} | {'PIIReuse':>9}")
        sep = "-" * len(header)
        f.write(header + "\n")
        f.write(sep + "\n")

        for s in all_stats:
            line = (f"{int(s['pii_ratio']*100):>4}% | "
                    f"{s['ttft_mean_ms']:>7.1f} {s['ttft_p50_ms']:>7.1f} {s['ttft_p95_ms']:>7.1f} | "
                    f"{s['kv_cache_hit_rate']*100:>6.2f}% {s['kv_max_usage_frac']*100:>6.2f}% | "
                    f"{s['throughput_qps']:>5.1f} | "
                    f"{s['pii_cross_tenant_reuse_ratio']:>8.4f}")
            f.write(line + "\n")

        f.write(f"\n{'='*70}\n")
        f.write("Column description:\n")
        f.write("  PII%      : fraction of queries with injected PII\n")
        f.write("  TTFT      : mean time-to-first-token (ms)\n")
        f.write("  P50/P95   : latency percentiles (ms)\n")
        f.write("  Cache%    : KV-cache hit rate (cached_tokens / total_tokens)\n")
        f.write("  MaxUsg    : peak KV-cache memory usage fraction\n")
        f.write("  QPS       : queries per second\n")
        f.write("  PIIReuse  : cross-tenant PII reuse ratio\n")
        f.write("              (clean_dup_lat / pii_dup_lat; ~1.0 = PII leaking,\n")
        f.write("               <1.0 or >>1.0 = PII isolated)\n")
    print(f"  Report:      {report_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SafeKV 10k Benchmark")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--model", type=str, default="phi4")
    parser.add_argument("--num-queries", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--pii-ratios", type=str, default="0,10,25,50,75,100",
                        help="Comma-separated PII percentage values")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful assistant. Answer concisely.",
                        help="Shared system prompt across users")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for output filenames")
    args = parser.parse_args()

    pii_ratios = [int(x.strip()) / 100.0 for x in args.pii_ratios.split(",")]
    if not args.tag:
        args.tag = time.strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(SCRIPT_DIR, "logs", "phi4.log")

    print("=" * 70)
    print("SafeKV 10k Multi-Turn Benchmark")
    print(f"  Server:       http://127.0.0.1:{args.port}")
    print(f"  Model:        {args.model}")
    print(f"  Queries/exp:  {args.num_queries}")
    print(f"  Workers:      {args.workers}")
    print(f"  PII ratios:   {[f'{r*100:.0f}%' for r in pii_ratios]}")
    print(f"  Users:        {NUM_USERS}")
    print(f"  Min turns:    {MIN_TURNS}")
    print(f"  Max tokens:   {MAX_TOKENS}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 70)

    # Health check
    try:
        resp = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=5)
        assert resp.status_code == 200, f"status={resp.status_code}"
        print("Server health: OK")
    except Exception as e:
        print(f"ERROR: Server unreachable: {e}")
        sys.exit(1)

    # Load base data
    print("\nLoading data...")
    conversations = load_sharegpt_multiturn(SHAREGPT_FILE, args.num_queries * 2, args.seed)
    pii_pool = PII_SNIPPETS + load_pii_snippets_from_dataset(PII_FILE, 200)
    print(f"  {len(conversations)} multi-turn conversations loaded")
    print(f"  {len(pii_pool)} PII snippets in injection pool")

    # Run experiments
    all_stats = []
    all_results = {}

    for ratio in pii_ratios:
        results, log_metrics = run_experiment(
            pii_ratio=ratio,
            conversations=conversations,
            pii_pool=pii_pool,
            num_queries=args.num_queries,
            workers=args.workers,
            port=args.port,
            model=args.model,
            seed=args.seed + int(ratio * 100),
            log_path=log_path,
            system_prompt=args.system_prompt,
        )
        stats = analyze_experiment(ratio, results, log_metrics)
        print_stats(stats)
        all_stats.append(stats)
        all_results[ratio] = results

    # Final comparison table
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS PII RATIOS")
    print(f"{'='*70}")
    header = (f"{'PII%':>5} | {'TTFT':>8} {'P50':>8} {'P95':>8} | "
              f"{'Cache%':>7} {'MaxUsg':>7} | {'QPS':>6} | {'PIIReuse':>9}")
    print(header)
    print("-" * len(header))
    for s in all_stats:
        line = (f"{int(s['pii_ratio']*100):>4}% | "
                f"{s['ttft_mean_ms']:>7.1f} {s['ttft_p50_ms']:>7.1f} {s['ttft_p95_ms']:>7.1f} | "
                f"{s['kv_cache_hit_rate']*100:>6.2f}% {s['kv_max_usage_frac']*100:>6.2f}% | "
                f"{s['throughput_qps']:>5.1f} | "
                f"{s['pii_cross_tenant_reuse_ratio']:>8.4f}")
        print(line)

    # Save
    save_all_results(all_stats, all_results, args.output_dir, args.tag)
    print(f"\nAll experiments completed.")


if __name__ == "__main__":
    main()
