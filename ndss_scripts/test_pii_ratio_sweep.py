#!/usr/bin/env python3
"""
PII Ratio Sweep Benchmark: SGLang vs Cache-Partition vs SafeKV
==============================================================
Measures throughput (tok/s) and TTFT (ms) at PII ratios R = 0,5,10,20,50,100%

Workload design
---------------
  N_TYPES   distinct prompt templates, each = SHARED_PREFIX (2048t, non-PII,
            identical for ALL users) + TYPE_SUFFIX (2048t, PII or non-PII).
  R%        of the N_TYPES templates contain PII.

  For each template, N_WARMUP users warm up the cache (triggers K>=2 promotion
  in SafeKV), then N_MEASURE different users are issued concurrently.

Steady-state behaviour
----------------------
  SGLang          : ALL templates cached after warmup → cache hits for every
                    measure user → max throughput regardless of R.
  Cache-Partition : NOTHING shared across users → every user full-prefills →
                    constant low throughput regardless of R.
  SafeKV          : non-PII templates promoted to public after warmup →
                    cache hits like SGLang; PII templates stay private →
                    isolated like Cache-Partition.
                    ⇒ graceful degradation from SGLang (R=0) → C-P (R=100).

Usage
-----
  python test_pii_ratio_sweep.py --model qwen32b [--port 8090]
  # Run all three models sequentially:
  for m in qwen32b phi4 qwen30b; do
    python test_pii_ratio_sweep.py --model $m
  done
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
LOG_DIR     = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PYTHON      = "/home/kec23008/.venv/bin/python3"
PII_DATASET = Path("/home/kec23008/InferShield/datasets/english_pii_43k.jsonl")
SHAREGPT    = Path("/home/kec23008/InferShield/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")

# ── model configs ──────────────────────────────────────────────────────────
MODEL_CONFIGS: Dict[str, dict] = {
    "qwen32b": {
        "path": "/home/kec23008/Models/Qwen3-32B",
        "tp": 2, "dp": 1, "maxlen": 32768,
        "mem_frac": "0.85", "cuda_devices": "0,1",
        "max_workers": 8,
    },
    "qwen30b": {
        "path": "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
        "tp": 2, "dp": 1, "maxlen": 32768,
        "mem_frac": "0.90", "cuda_devices": "0,1",
        "max_workers": 16,
    },
    "phi4": {
        "path": "/home/kec23008/Models/Phi-4",
        "tp": 1, "dp": 2, "maxlen": 16384,
        "mem_frac": "0.90", "cuda_devices": "0,1",
        "max_workers": 24,
    },
}

# ── systems under test ─────────────────────────────────────────────────────
SYSTEMS: Dict[str, dict] = {
    "sglang": {
        "label": "SGLang",
        "safekv_args": [],
        "use_uid": False,   # no per-user isolation
    },
    "cache_partition": {
        "label": "Cache-Partition",
        "safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
            "--safekv-private-only",        # all content always private
        ],
        "use_uid": True,
    },
    "safekv": {
        "label": "SafeKV",
        "safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",  # K=2
        ],
        "use_uid": True,
    },
    "cache_solidarity": {
        "label": "CacheSolidarity",
        "safekv_args": [
            "--safekv-cache-solidarity",
        ],
        "use_uid": True,
        # warmup with uid=None bypasses solidarity path → only seeds shared namespace
        # (avoids double-copy pool exhaustion: shared 43K + solidarity 43K > 76K for phi4)
        "warmup_use_uid": False,
        # restart server between R values to prevent cascade pool exhaustion
        "restart_per_ratio": True,
        # limit concurrency so in-flight solidarity copies don't exhaust the pool
        # phi4 pool ~76K: shared 43K + 4×4K copies = ~59K < 76K (headroom for LRU)
        # qwen30b pool ~365K (MoE, small KV): 43K + 12×4K = ~91K << 365K
        # qwen32b pool ~137K: 43K + 6×4K = ~67K < 137K
        "max_workers_override": {"phi4": 4, "qwen30b": 12, "qwen32b": 6},
        # longer timeout: solidarity requests may queue behind pool-pressure-slowed requests
        "request_timeout": 360,
    },
}

PII_RATIOS   = [0, 5, 10, 20, 50, 100]

# ── workload parameters ────────────────────────────────────────────────────
N_TYPES              = 20    # distinct prompt templates
N_WARMUP_PER_TYPE    = 4     # warmup users per template (>= K=2 to trigger promotion)
N_MEASURE_PER_TYPE   = 10    # measurement users per template (concurrent)
SHARED_PREFIX_TOKENS = 2048
SUFFIX_TOKENS        = 2048
WARMUP_WAIT_S        = 50    # wait for async PII detection after warmup
MAX_TOKENS           = 128
COMPLETION_CONFIG    = {"max_tokens": MAX_TOKENS, "temperature": 0,
                        "enable_thinking": False}


# ── data classes ───────────────────────────────────────────────────────────
@dataclass
class ReqResult:
    user_id:  Optional[str]
    ttft_s:   Optional[float]   # time-to-first-token in seconds
    latency_s: Optional[float]  # total latency
    n_out_tokens: int = 0
    error:    Optional[str] = None


@dataclass
class RatioResult:
    system:   str
    r_pct:    int
    ttft_mean_ms: float
    ttft_p50_ms:  float
    ttft_p95_ms:  float
    throughput_tps: float       # output tokens / elapsed wall-clock seconds
    n_requests: int
    n_errors:   int


# ── server helpers ─────────────────────────────────────────────────────────
def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"],
                                      text=True, stderr=subprocess.DEVNULL)
        for pid in out.strip().split():
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)
        time.sleep(4)
    except subprocess.CalledProcessError:
        pass


def _start_server(model: str, port: int,
                  safekv_args: List[str], tag: str) -> subprocess.Popen:
    cfg = MODEL_CONFIGS[model]
    log_path = LOG_DIR / f"pii_ratio_{model}_{tag}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    env["LD_LIBRARY_PATH"] = (
        "/home/kec23008/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path",          cfg["path"],
        "--host",                "127.0.0.1",
        "--port",                str(port),
        "--dtype",               "float16",
        "--trust-remote-code",
        "--tp-size",             str(cfg["tp"]),
        "--dp-size",             str(cfg["dp"]),
        "--context-length",      str(cfg["maxlen"]),
        "--served-model-name",   model,
        "--attention-backend",   "torch_native",
        "--disable-cuda-graph",
        "--mem-fraction-static", cfg["mem_frac"],
        "--enable-metrics",
    ] + cfg.get("extra_args", []) + safekv_args

    print(f"  [server] Starting {model} / {tag} → {log_path.name}", flush=True)
    fh = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)


def _wait_ready(port: int, timeout: int = 480) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    spin = ["|", "/", "-", "\\"]
    i = 0
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"\n  [server] :{port} ready!", flush=True)
                return True
        except Exception:
            pass
        print(f"\r  [server] {spin[i%4]} waiting :{port}…", end="", flush=True)
        i += 1
        time.sleep(5)
    print(f"\n  [ERROR] server :{port} did not start in {timeout}s", flush=True)
    return False


# ── request helper ─────────────────────────────────────────────────────────
def _send_one(port: int, model: str,
              user_id: Optional[str], prompt: str,
              timeout: int = 180) -> ReqResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        **COMPLETION_CONFIG,
    }
    if user_id is not None:
        payload["user_id"] = user_id

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    n_out = 0

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=timeout)
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = (chunk.get("choices", [{}])[0]
                               .get("delta", {}).get("content", ""))
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_out += 1
            except json.JSONDecodeError:
                pass
        elapsed = time.perf_counter() - t0
        return ReqResult(user_id=user_id, ttft_s=ttft,
                         latency_s=elapsed, n_out_tokens=n_out)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return ReqResult(user_id=user_id, ttft_s=None,
                         latency_s=elapsed, n_out_tokens=0,
                         error=str(exc)[:120])


# ── dataset builder ────────────────────────────────────────────────────────
def build_corpus() -> Tuple[str, List[str], List[str]]:
    """Load shared prefix, PII entries, and non-PII entries once."""
    print("  [corpus] Loading datasets…", flush=True)

    # shared prefix: first ~SHARED_PREFIX_TOKENS tokens from ShareGPT human turns
    sharegpt_turns: List[str] = []
    with open(SHAREGPT, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                t = conv.get("value", "").strip()
                if t:
                    sharegpt_turns.append(t)
    random.shuffle(sharegpt_turns)

    # Build shared prefix by concatenating turns until we hit the token target
    prefix_parts: List[str] = []
    prefix_tokens = 0
    for turn in sharegpt_turns:
        t = count_tokens(turn)
        if prefix_tokens + t <= SHARED_PREFIX_TOKENS * 1.05:
            prefix_parts.append(turn)
            prefix_tokens += t
        if prefix_tokens >= SHARED_PREFIX_TOKENS:
            break
    shared_prefix = "\n\n".join(prefix_parts)
    print(f"  [corpus] Shared prefix: ~{count_tokens(shared_prefix)} tokens",
          flush=True)

    # PII entries
    pii_entries: List[str] = []
    with open(PII_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("source_text", "")
                if txt:
                    pii_entries.append(txt)
            except json.JSONDecodeError:
                pass
    random.shuffle(pii_entries)
    print(f"  [corpus] PII entries: {len(pii_entries)}", flush=True)

    # Non-PII entries: remaining ShareGPT turns (exclude those used in prefix)
    nonpii_entries = sharegpt_turns[len(prefix_parts):]
    print(f"  [corpus] Non-PII entries: {len(nonpii_entries)}", flush=True)

    return shared_prefix, pii_entries, nonpii_entries


def build_dataset(
    r_pct: int,
    shared_prefix: str,
    pii_entries: List[str],
    nonpii_entries: List[str],
    n_types: int = N_TYPES,
    seed: int = 42,
) -> List[Tuple[str, bool]]:   # (prompt_text, is_pii)
    """
    Build n_types distinct prompt templates.
    r_pct% of templates have a PII suffix; rest have a non-PII suffix.
    Returns list of (prompt, is_pii) — one entry per TYPE (not per user).
    """
    rng = random.Random(seed)
    n_pii    = max(0, min(n_types, round(n_types * r_pct / 100)))
    n_nonpii = n_types - n_pii

    def _build_suffix(entries: List[str], start: int) -> Tuple[str, int]:
        parts, tok, idx = [], 0, start
        while tok < SUFFIX_TOKENS and idx < len(entries):
            txt = entries[idx]
            t   = count_tokens(txt)
            parts.append(txt)
            tok += t
            idx += 1
            if tok >= SUFFIX_TOKENS:
                break
        return "\n\n".join(parts), idx

    pii_idx, nonpii_idx = 0, 0
    pii_pool   = list(pii_entries)
    npii_pool  = list(nonpii_entries)
    rng.shuffle(pii_pool)
    rng.shuffle(npii_pool)

    templates: List[Tuple[str, bool]] = []

    for i in range(n_pii):
        suffix, pii_idx = _build_suffix(pii_pool, pii_idx)
        prompt = (f"Background context:\n{shared_prefix}\n\n"
                  f"User data record {i}:\n{suffix}\n\n"
                  f"Please summarize the above information.")
        templates.append((prompt, True))

    for i in range(n_nonpii):
        suffix, nonpii_idx = _build_suffix(npii_pool, nonpii_idx)
        prompt = (f"Background context:\n{shared_prefix}\n\n"
                  f"General information {i}:\n{suffix}\n\n"
                  f"Please summarize the above information.")
        templates.append((prompt, False))

    rng.shuffle(templates)
    tok_lens = [count_tokens(p) for p, _ in templates[:3]]
    print(f"  [dataset] R={r_pct}%  pii={n_pii}  non-pii={n_nonpii}  "
          f"sample token lengths: {tok_lens}", flush=True)
    return templates


# ── warmup ─────────────────────────────────────────────────────────────────
def warmup(port: int, model: str, use_uid: bool,
           templates: List[Tuple[str, bool]],
           n_warmup: int = N_WARMUP_PER_TYPE,
           max_workers: int = 16,
           request_timeout: int = 180) -> None:
    """Send n_warmup users per template to seed the cache.

    use_uid=False (used for CacheSolidarity): all warmup requests use uid=None,
    bypassing the solidarity copy path and seeding only the shared namespace.
    This prevents warmup from double-using the pool (shared + solidarity copies).
    """
    print(f"  [warmup] Sending {n_warmup} users × {len(templates)} templates "
          f"({n_warmup * len(templates)} requests)…", flush=True)

    tasks: List[Tuple[Optional[str], str]] = []
    for t_idx, (prompt, _) in enumerate(templates):
        for u_idx in range(n_warmup):
            if not use_uid:
                uid = None
            else:
                uid = f"warmup_t{t_idx}_u{u_idx}_{uuid.uuid4().hex[:6]}"
            tasks.append((uid, prompt))

    random.shuffle(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_send_one, port, model, uid, prompt,
                          request_timeout)
                for uid, prompt in tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if done % 20 == 0:
                print(f"    warmup {done}/{len(tasks)}…", flush=True)
    print(f"  [warmup] Done in {time.time()-t0:.1f}s. "
          f"Waiting {WARMUP_WAIT_S}s for PII detection…", flush=True)
    time.sleep(WARMUP_WAIT_S)


# ── measurement ────────────────────────────────────────────────────────────
def measure(port: int, model: str, use_uid: bool,
            templates: List[Tuple[str, bool]],
            n_measure: int = N_MEASURE_PER_TYPE,
            max_workers: int = 8,
            request_timeout: int = 180) -> RatioResult:
    """Send n_measure users per template concurrently; record TTFT + throughput."""
    tasks: List[Tuple[Optional[str], str]] = []
    for t_idx, (prompt, _) in enumerate(templates):
        for u_idx in range(n_measure):
            uid = (f"measure_t{t_idx}_u{u_idx}_{uuid.uuid4().hex[:6]}"
                   if use_uid else None)
            tasks.append((uid, prompt))

    random.shuffle(tasks)
    results: List[ReqResult] = []

    t_wall_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_send_one, port, model, uid, prompt,
                          request_timeout)
                for uid, prompt in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
    t_wall = time.time() - t_wall_start

    good   = [r for r in results if r.ttft_s is not None and r.error is None]
    ttfts  = sorted(r.ttft_s for r in good)
    n_err  = len(results) - len(good)
    total_out_tokens = sum(r.n_out_tokens for r in results)

    if not ttfts:
        return RatioResult(system="?", r_pct=-1,
                           ttft_mean_ms=float("nan"), ttft_p50_ms=float("nan"),
                           ttft_p95_ms=float("nan"), throughput_tps=0.0,
                           n_requests=len(tasks), n_errors=n_err)

    def pct(arr, p):
        idx = max(0, min(len(arr)-1, int(len(arr)*p/100)))
        return arr[idx]

    return RatioResult(
        system="?", r_pct=-1,
        ttft_mean_ms = float(np.mean(ttfts)) * 1000,
        ttft_p50_ms  = pct(ttfts, 50) * 1000,
        ttft_p95_ms  = pct(ttfts, 95) * 1000,
        throughput_tps = total_out_tokens / t_wall if t_wall > 0 else 0.0,
        n_requests = len(tasks),
        n_errors   = n_err,
    )


# ── main ───────────────────────────────────────────────────────────────────
def run_model(model: str, port: int, corpus: Tuple,
              ts: str, args=None) -> List[RatioResult]:
    shared_prefix, pii_entries, nonpii_entries = corpus
    cfg = MODEL_CONFIGS[model]
    mw  = cfg["max_workers"]
    all_results: List[RatioResult] = []

    systems_to_run = {k: v for k, v in SYSTEMS.items() if k in (args.systems or SYSTEMS)}
    for sys_name, sys_cfg in systems_to_run.items():
        print(f"\n{'='*65}", flush=True)
        print(f"  Model={model}  System={sys_cfg['label']}", flush=True)
        print(f"{'='*65}", flush=True)

        restart_per_ratio = sys_cfg.get("restart_per_ratio", False)
        mw_overrides      = sys_cfg.get("max_workers_override", {})
        eff_mw            = mw_overrides.get(model, mw) if mw_overrides else mw
        warmup_use_uid    = sys_cfg.get("warmup_use_uid", sys_cfg["use_uid"])
        req_timeout       = sys_cfg.get("request_timeout", 180)

        proc = None

        def _start_and_wait(tag_suffix=""):
            nonlocal proc
            _kill_port(port)
            time.sleep(2)
            proc = _start_server(model, port, sys_cfg["safekv_args"],
                                 f"{sys_name}{tag_suffix}")
            if not _wait_ready(port, timeout=480):
                print(f"  [ERROR] Server not ready", flush=True)
                proc.terminate()
                return False
            time.sleep(3)
            return True

        def _stop_server():
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            time.sleep(3)

        # Start server once if we're NOT restarting per ratio
        if not restart_per_ratio:
            if not _start_and_wait():
                continue

        try:
            for r_pct in PII_RATIOS:
                print(f"\n  ── R={r_pct}% ──────────────────────────", flush=True)

                # Restart server for each R value (CacheSolidarity: avoid cascade pool exhaustion)
                if restart_per_ratio:
                    if not _start_and_wait(f"_r{r_pct}"):
                        continue

                # Build fresh dataset for this R value
                templates = build_dataset(
                    r_pct, shared_prefix, pii_entries, nonpii_entries,
                    seed=r_pct * 100 + 7
                )

                # Warmup
                if sys_name != "cache_partition":
                    warmup(port, model, warmup_use_uid, templates,
                           n_warmup=N_WARMUP_PER_TYPE, max_workers=eff_mw,
                           request_timeout=req_timeout)
                else:
                    print(f"  [warmup] cache_partition: no prefix sharing, "
                          f"warmup skipped (1 pass for server load)", flush=True)
                    tasks = [(None if not sys_cfg["use_uid"]
                              else f"cp_warm_{i}", prompt)
                             for i, (prompt, _) in enumerate(templates)]
                    with ThreadPoolExecutor(max_workers=eff_mw) as ex:
                        list(as_completed([ex.submit(_send_one, port, model,
                                                     uid, prompt)
                                           for uid, prompt in tasks[:5]]))

                # Measure
                res = measure(port, model, sys_cfg["use_uid"], templates,
                              n_measure=N_MEASURE_PER_TYPE, max_workers=eff_mw,
                              request_timeout=req_timeout)
                res.system = sys_name
                res.r_pct  = r_pct
                all_results.append(res)

                print(f"  ✓ TTFT_mean={res.ttft_mean_ms:.0f}ms  "
                      f"P50={res.ttft_p50_ms:.0f}ms  "
                      f"P95={res.ttft_p95_ms:.0f}ms  "
                      f"TPS={res.throughput_tps:.1f}  "
                      f"errors={res.n_errors}/{res.n_requests}",
                      flush=True)

                # Stop server after each R value if restarting per ratio
                if restart_per_ratio:
                    print(f"  [teardown] Stopping server after R={r_pct}%…", flush=True)
                    _stop_server()

        except KeyboardInterrupt:
            print("\n  [INTERRUPT]", flush=True)
            _stop_server()
            raise

        except Exception:
            print(f"\n  [ERROR] system {sys_name}:", flush=True)
            traceback.print_exc()
            _stop_server()

        finally:
            if not restart_per_ratio:
                print(f"\n  [teardown] Stopping {sys_name} server…", flush=True)
                _stop_server()

    return all_results


def save_csv(results: List[RatioResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "r_pct", "ttft_mean_ms", "ttft_p50_ms",
                    "ttft_p95_ms", "throughput_tps",
                    "n_requests", "n_errors"])
        for r in results:
            w.writerow([r.system, r.r_pct,
                        f"{r.ttft_mean_ms:.1f}", f"{r.ttft_p50_ms:.1f}",
                        f"{r.ttft_p95_ms:.1f}", f"{r.throughput_tps:.2f}",
                        r.n_requests, r.n_errors])
    print(f"  [CSV] {path}", flush=True)


def plot_results(results: List[RatioResult], path: Path, model: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"PII Ratio Sweep — {model}", fontsize=13, fontweight="bold")

    colors = {"sglang": "#2ca02c", "cache_partition": "#d62728",
              "safekv": "#1f77b4", "cache_solidarity": "#ff7f0e"}
    labels = {"sglang": "SGLang", "cache_partition": "Cache-Partition",
              "safekv": "SafeKV", "cache_solidarity": "CacheSolidarity"}
    markers = {"sglang": "o", "cache_partition": "s", "safekv": "^",
               "cache_solidarity": "D"}

    for sys_name in ["sglang", "cache_partition", "safekv", "cache_solidarity"]:
        rows = sorted([r for r in results if r.system == sys_name],
                      key=lambda x: x.r_pct)
        if not rows:
            continue
        xs   = [r.r_pct for r in rows]
        tpss = [r.throughput_tps for r in rows]
        ttfs = [r.ttft_mean_ms for r in rows]

        ax1.plot(xs, tpss, color=colors[sys_name], marker=markers[sys_name],
                 lw=2, ms=7, label=labels[sys_name])
        ax2.plot(xs, ttfs, color=colors[sys_name], marker=markers[sys_name],
                 lw=2, ms=7, label=labels[sys_name])

    for ax in (ax1, ax2):
        ax.set_xlabel("PII Ratio R (%)", fontsize=11)
        ax.set_xticks(PII_RATIOS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax2.set_ylabel("TTFT mean (ms)", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {path}", flush=True)


def print_table(results: List[RatioResult], model: str) -> None:
    print(f"\n{'='*80}")
    print(f"  PII Ratio Sweep — {model}")
    print(f"  R(%) | {'System':<18} | {'TPS':>8} | {'TTFT_mean':>10} | "
          f"{'TTFT_P50':>9} | {'TTFT_P95':>9} | Err")
    print(f"{'─'*80}")
    for r_pct in PII_RATIOS:
        for sys_name in ["sglang", "cache_partition", "safekv", "cache_solidarity"]:
            row = next((r for r in results
                        if r.system == sys_name and r.r_pct == r_pct), None)
            if row is None:
                continue
            label = {"sglang": "SGLang", "cache_partition": "Cache-Partition",
                     "safekv": "SafeKV", "cache_solidarity": "CacheSolidarity"}[sys_name]
            print(f"  {r_pct:>4} | {label:<18} | {row.throughput_tps:>8.2f} | "
                  f"{row.ttft_mean_ms:>9.0f}ms | "
                  f"{row.ttft_p50_ms:>8.0f}ms | "
                  f"{row.ttft_p95_ms:>8.0f}ms | "
                  f"{row.n_errors}/{row.n_requests}")
        print(f"{'─'*80}")


def main():
    parser = argparse.ArgumentParser(
        description="PII ratio sweep: SGLang vs Cache-Partition vs SafeKV")
    parser.add_argument("--model", default="qwen32b",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS.keys()),
                        default=None, help="Run only these systems (default: all)")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"pii_ratio_sweep_{args.model}_{ts}.csv"
    png_path = LOG_DIR / f"pii_ratio_sweep_{args.model}_{ts}.png"

    print(f"\n{'='*65}", flush=True)
    print(f"  PII Ratio Sweep Benchmark", flush=True)
    print(f"  Model : {args.model}   Port: {args.port}", flush=True)
    print(f"  Ratios: {PII_RATIOS}", flush=True)
    print(f"  Types : {N_TYPES}  Warmup/type: {N_WARMUP_PER_TYPE}  "
          f"Measure/type: {N_MEASURE_PER_TYPE}", flush=True)
    print(f"  Shared prefix: {SHARED_PREFIX_TOKENS}t  Suffix: {SUFFIX_TOKENS}t", flush=True)
    print(f"{'='*65}\n", flush=True)

    corpus = build_corpus()
    results = run_model(args.model, args.port, corpus, ts, args=args)

    if results:
        print_table(results, args.model)
        save_csv(results, csv_path)
        plot_results(results, png_path, args.model)

    print(f"\nDone. Results in {LOG_DIR}", flush=True)


if __name__ == "__main__":
    main()
