#!/usr/bin/env python3
"""
SafeKV 4-Mode Ablation Study
==============================
Compares four KV-cache sharing policies under a timing side-channel attack:

  Mode 1 - baseline:           Standard sglang, no SafeKV, full KV sharing
  Mode 2 - private_default:    SafeKV private-by-default, no promotion ever
  Mode 3 - private_detector:   SafeKV + detector, non-PII promoted (unlimited budget)
  Mode 4 - full_safekv:        Full SafeKV with B=10, K=2

Metrics reported per mode:
  - defense_success_rate : % attacker probes that did NOT benefit from cache hits
  - TTFT mean / P50 / P95 / P99 (regular user requests)
  - TPS (approximate throughput)

Usage:
  # Run all 4 modes (restarts server for each mode):
  python test_safekv_ablation.py --model qwen32b

  # Run a single mode against already-running server (no restart):
  python test_safekv_ablation.py --model qwen32b --mode full_safekv --no-restart

  # Run all modes but use a custom port:
  python test_safekv_ablation.py --model qwen32b --port 8090

Requirements:
  - CUDA_VISIBLE_DEVICES must allow at least 2 GPUs for qwen32b
  - venv at /home/kec23008/.venv with SafeKV-patched sglang
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ablation_prompts import PII_PROMPTS  # long ≥4096-token PII prompts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PYTHON = "/home/kec23008/.venv/bin/python3"

MODEL_CONFIGS = {
    "qwen32b": {
        "path": "/home/kec23008/Models/Qwen3-32B",
        "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.90", "cuda_devices": "0,1",
    },
    "qwen30b": {
        "path": "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
        "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.90", "cuda_devices": "0,1",
    },
    "phi4": {
        "path": "/home/kec23008/Models/Phi-4",
        "tp": 1, "dp": 2, "maxlen": 16384, "mem_frac": "0.90", "cuda_devices": "0,1",
    },
}

# ---------------------------------------------------------------------------
# Test workload content
# ---------------------------------------------------------------------------

# PII_PROMPTS imported from ablation_prompts.py
# (each prompt >= 4096 tokens so that cold prefill >> scheduling overhead;
#  cold baseline prompts are now generated dynamically by _generate_cold_baseline_prompt)

# Non-PII prompts — used by regular users (Phase 4 throughput measurement)
NON_PII_PROMPTS = [
    "Explain the concept of gradient descent in machine learning in simple terms.",
    "Give a brief overview of the history of the Roman Empire.",
    "How does photosynthesis work? Describe both the light-dependent and light-independent reactions.",
    "What are the key differences between Python and JavaScript as programming languages?",
    "Explain how the TCP/IP protocol stack works, layer by layer.",
    "What are the SOLID principles in object-oriented software design?",
    "Briefly describe the causes and key events of the French Revolution.",
    "How does a B-tree index improve database query performance?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of eventual consistency in distributed systems.",
]

# Request config: short completion to keep TTFT measurement prefill-dominated.
# max_tokens=128 gives more data for TPS estimation without dominating TTFT.
COMPLETION_CONFIG = {
    "max_tokens": 128,
    "temperature": 0,
}

# Phase 5 cold baseline: generated dynamically by _generate_cold_baseline_prompt()
# (1000 random 8-digit numbers, ~3000 tokens, unique per call — guaranteed cache MISS)

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

MODES: Dict[str, dict] = {
    "baseline": {
        "description": "Standard sglang — no SafeKV, full KV cache sharing",
        # No SafeKV server flags; user_id is NOT sent (no isolation)
        "server_safekv_args": [],
        "use_user_id": False,
    },
    "private_default": {
        "description": "SafeKV private-by-default — promotion disabled (no cross-tenant sharing)",
        "server_safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
            "--safekv-private-only",   # key flag: never promote
        ],
        "use_user_id": True,
    },
    "private_detector": {
        "description": "SafeKV + detector — non-PII promoted, budget effectively unlimited (B=999999)",
        "server_safekv_args": [
            "--safekv-access-budget", "999999",
            "--safekv-creator-threshold", "1",
        ],
        "use_user_id": True,
    },
    "full_safekv": {
        "description": "Full SafeKV — B=10, K=2",
        "server_safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
        ],
        "use_user_id": True,
    },
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReqResult:
    role: str               # 'victim', 'regular', 'attacker'
    user_id: Optional[str]
    prompt_short: str
    ttft: float             # time-to-first-token (s)
    total_time: float       # full request time (s)
    n_tokens: int           # approximate tokens in completion
    error: Optional[str] = None


@dataclass
class ModeMetrics:
    mode: str
    description: str
    cold_ttft: float
    defense_rate: float     # % attacker probes blocked (no cache hit)
    attacker_hits: int
    total_probes: int
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tps: float
    reg_results: List[ReqResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _send_one(port: int, model: str, user_id: Optional[str],
              prompt: str, role: str) -> ReqResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        **COMPLETION_CONFIG,
    }
    if user_id is not None:
        payload["user_id"] = user_id  # SafeKV tenant ID

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    n_tokens = 0
    error: Optional[str] = None

    try:
        resp = requests.post(url, json=payload, headers=headers,
                             stream=True, timeout=120)
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = (chunk.get("choices", [{}])[0]
                           .get("delta", {}).get("content", ""))
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_tokens += 1   # token count approximation
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - t0
    if ttft is None:
        ttft = elapsed

    return ReqResult(
        role=role,
        user_id=user_id,
        prompt_short=prompt[:60],
        ttft=ttft,
        total_time=elapsed,
        n_tokens=max(n_tokens, 1),
        error=error,
    )


def measure_cold_ttft(port: int, model: str) -> float:
    """One-shot cold request with a fresh unique prompt (no cache match possible)."""
    unique_prompt = (
        f"Describe the properties of the number {random.randint(10**6, 10**7)} "
        f"in a single sentence. Be concise."
    )
    r = _send_one(port, model, user_id=None, prompt=unique_prompt, role="cold")
    latency = r.ttft if r.error is None else 99.9
    print(f"  [cold baseline] TTFT={latency:.3f}s  err={r.error}", flush=True)
    return latency


# ---------------------------------------------------------------------------
# Server lifecycle management
# ---------------------------------------------------------------------------

def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"],
                                      text=True, stderr=subprocess.DEVNULL)
        pids = out.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid],
                               capture_output=True, check=False)
                print(f"  [kill] PID {pid} on :{port}", flush=True)
        if pids:
            time.sleep(4)
    except subprocess.CalledProcessError:
        pass  # nothing on that port


def _start_server(model: str, port: int,
                  safekv_args: List[str], log_tag: str) -> subprocess.Popen:
    cfg = MODEL_CONFIGS[model]
    log_path = LOG_DIR / f"ablation_{model}_{log_tag}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    env["LD_LIBRARY_PATH"] = (
        "/home/kec23008/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path",         cfg["path"],
        "--host",               "127.0.0.1",
        "--port",               str(port),
        "--dtype",              "float16",
        "--trust-remote-code",
        "--tp-size",            str(cfg["tp"]),
        "--dp-size",            str(cfg["dp"]),
        "--context-length",     str(cfg["maxlen"]),
        "--served-model-name",  model,
        "--attention-backend",  "torch_native",
        "--disable-cuda-graph",
        "--mem-fraction-static", cfg["mem_frac"],
        "--enable-metrics",
    ] + cfg.get("extra_server_args", []) + safekv_args

    print(f"  [start] {' '.join(cmd)}", flush=True)
    print(f"  [log]   {log_path}", flush=True)

    log_fh = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)


def _wait_ready(port: int, timeout: int = 360) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    spinner = ["|", "/", "-", "\\"]
    i = 0
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"\n  [ready] Server on :{port} is up!", flush=True)
                return True
        except Exception:
            pass
        print(f"\r  [wait] {spinner[i % 4]} polling :{port} ...", end="", flush=True)
        i += 1
        time.sleep(5)
    print(f"\n  [ERROR] Server on :{port} did not start within {timeout}s", flush=True)
    return False


# ---------------------------------------------------------------------------
# Workload runner
# ---------------------------------------------------------------------------

N_VICTIMS       = 5      # number of victim users (one PII prompt each)
N_VICTIM_WARMUP = 3      # each victim sends their prompt N times to warm cache
N_REGULAR_USERS = 32     # concurrent regular users (saturate GPU for TTFT/TPS measurement)
N_REGULAR_REQS  = 20     # requests per regular user (32×20 = 640 total requests)
N_ATTACKER_PROBES_PER_VICTIM = 5   # attacker probes per victim (sequential, paired)
N_PROBE_BACKGROUND = 20  # concurrent background workers during Phase 5 (creates timing noise)
                          # Simulates realistic multi-tenant load; cache hits in SGLang become
                          # noisier and some probes are misclassified → non-zero "noise defense"
DETECTION_WAIT  = 30     # seconds to wait for async PII detection pipeline
HIT_RATIO_THRESHOLD = 0.80  # probe_ttft / cold_ttft < this → cache hit (defense failed)


def _generate_cold_baseline_prompt(v_idx: int, probe_idx: int) -> str:
    """Generate a completely unique cold prompt (~3000 tokens) guaranteed NOT in any KV cache."""
    n_nums = 1000   # 1000 × 8-digit numbers × ~3 tokens/number ≈ 3000 tokens
    nums = " ".join(str(random.randint(10**7, 10**8 - 1)) for _ in range(n_nums))
    uid = random.randint(10**12, 10**13)
    return f"Data validation {uid}: {nums}"


def _generate_local_cold_prompt(target_n_nums: int = 1500) -> str:
    """Generate a unique cold prompt matching the length of PII prompts (~4500 tokens).

    Used as the local cold reference in Phase 5 paired probes.  Each call returns a
    completely different sequence, so it is always a genuine cache MISS regardless of mode.
    Using the same target length as PII prompts avoids length-mismatch bias in the ratio.
    """
    nums = " ".join(str(random.randint(10**7, 10**8 - 1)) for _ in range(target_n_nums))
    uid = random.randint(10**15, 10**16)
    return f"Local cold reference {uid}: {nums}"


def run_workload(mode_name: str, cfg: dict,
                 port: int, model: str) -> ModeMetrics:
    use_uid = cfg["use_user_id"]
    desc = cfg["description"]

    victim_ids  = [f"victim_{i}"  for i in range(N_VICTIMS)]
    regular_ids = [f"regular_{i}" for i in range(N_REGULAR_USERS)]
    atk_uid     = "attacker_007" if use_uid else None

    print(f"\n{'─'*60}")
    print(f"  Workload: {mode_name}")
    print(f"  {desc}")
    print(f"{'─'*60}", flush=True)

    # ------------------------------------------------------------------ #
    # Phase 1 – Victim cache warmup
    # ------------------------------------------------------------------ #
    print(f"\n  [Phase 1] Victim cache warmup ({N_VICTIMS} victims × {N_VICTIM_WARMUP} reps)...", flush=True)
    victim_cold_ttfts: List[float] = []   # rep=0 TTFT per victim = true cold PII prefill time
    for v_idx, (vid, pii) in enumerate(zip(victim_ids, PII_PROMPTS)):
        uid = vid if use_uid else None
        for rep in range(N_VICTIM_WARMUP):
            r = _send_one(port, model, uid, pii, role="victim")
            if rep == 0:
                victim_cold_ttfts.append(r.ttft if r.error is None else float("nan"))
            status = "err" if r.error else f"TTFT={r.ttft:.3f}s"
            print(f"    victim[{v_idx}] rep={rep} {status}", flush=True)
            time.sleep(0.3)

    # ------------------------------------------------------------------ #
    # Phase 2 – Wait for async detection
    # ------------------------------------------------------------------ #
    print(f"\n  [Phase 2] Waiting {DETECTION_WAIT}s for async PII detection...", flush=True)
    time.sleep(DETECTION_WAIT)

    # ------------------------------------------------------------------ #
    # Phase 2.5 – Calibration: measure true cold PII TTFT in current
    #             GPU-warm conditions, using a fresh user_id that has
    #             never sent any of these prompts.  This tells us what
    #             cold_TTFT *should* look like for the PII prompts in
    #             Phase 5 — if attacker probe_ttft << calib_ttft, that
    #             is a genuine cache hit regardless of the mode.
    #             Note: calib_user is NOT the attacker, so this does not
    #             pollute the attacker's cache state.
    # ------------------------------------------------------------------ #
    print(f"\n  [Phase 2.5] Calibration: measuring PII cold TTFT (fresh user)...", flush=True)
    calib_uid = "calib_user_never_seen_pii" if use_uid else None
    pii_calib_ttfts: List[float] = []
    for v_idx, pii in enumerate(PII_PROMPTS):
        r = _send_one(port, model, calib_uid, pii, role="cold")
        ttft_val = r.ttft if r.error is None else float("nan")
        pii_calib_ttfts.append(ttft_val)
        print(f"    PII[{v_idx}] calib cold TTFT = {ttft_val:.3f}s  err={r.error}", flush=True)
    if pii_calib_ttfts:
        import math
        valid = [t for t in pii_calib_ttfts if not math.isnan(t)]
        if valid:
            print(f"  Calibration PII cold TTFT mean: {sum(valid)/len(valid):.3f}s", flush=True)

    # ------------------------------------------------------------------ #
    # Phase 3 – Reference cold TTFT (idle, for reporting only)
    # ------------------------------------------------------------------ #
    print(f"\n  [Phase 3] Measuring reference cold TTFT (idle)...", flush=True)
    cold_samples = [measure_cold_ttft(port, model) for _ in range(3)]
    cold_ttft_ref = statistics.mean(cold_samples)
    print(f"  Reference cold TTFT (mean of 3): {cold_ttft_ref:.3f}s", flush=True)

    # ------------------------------------------------------------------ #
    # Phase 4 – TTFT/TPS: concurrent regular-user workload
    # ------------------------------------------------------------------ #
    print(f"\n  [Phase 4] Concurrent regular-user workload (TTFT/TPS measurement)...", flush=True)
    regular_tasks = []
    for uid_base in regular_ids:
        uid = uid_base if use_uid else None
        prompts = random.sample(NON_PII_PROMPTS, min(N_REGULAR_REQS, len(NON_PII_PROMPTS)))
        for prompt in prompts:
            regular_tasks.append((uid, prompt))

    reg_results: List[ReqResult] = []
    t_reg_start = time.time()
    with ThreadPoolExecutor(max_workers=N_REGULAR_USERS + 4) as ex:
        futs = [ex.submit(_send_one, port, model, uid, prompt, "regular")
                for uid, prompt in regular_tasks]
        for fut in as_completed(futs):
            try:
                reg_results.append(fut.result())
            except Exception as e:
                print(f"  [WARN] Regular user error: {e}", flush=True)
    t_reg_elapsed = time.time() - t_reg_start

    # ------------------------------------------------------------------ #
    # Phase 5 – Attacker: sequential PAIRED probes
    #
    # For each probe:
    #   1. Send a UNIQUE cold prompt  → cold_ttft_local  (baseline at current load)
    #   2. Send victim's PII prompt   → probe_ttft
    #   3. ratio = probe_ttft / cold_ttft_local
    #      ratio < HIT_RATIO_THRESHOLD  →  cache HIT  →  defense FAILED
    #
    # Sequential + paired removes scheduling-jitter bias; ratio normalises
    # any residual load difference between the two requests.
    # ------------------------------------------------------------------ #
    print(f"\n  [Phase 5] Attacker sequential probes (background_workers={N_PROBE_BACKGROUND})...", flush=True)
    print(f"  Hit detection: probe_ttft / victim_cold_ttft < {HIT_RATIO_THRESHOLD}", flush=True)
    print(f"  Victim cold TTFTs (from Phase 1 rep=0): "
          f"{[f'{t:.3f}s' for t in victim_cold_ttfts]}", flush=True)

    # Start background workers to simulate concurrent multi-tenant load.
    # This introduces natural TTFT jitter: cache hits in modes without defense
    # (SGLang) occasionally cross the ratio threshold → non-zero "noise defense".
    import math, threading
    _bg_stop = threading.Event()

    def _bg_worker():
        bg_prompts = NON_PII_PROMPTS if NON_PII_PROMPTS else ["Summarize the history of medicine."]
        idx = 0
        while not _bg_stop.is_set():
            uid = f"bg_{random.randint(0, 9999)}"
            _send_one(port, model, uid if use_uid else None,
                      bg_prompts[idx % len(bg_prompts)], "regular")
            idx += 1

    bg_threads = [threading.Thread(target=_bg_worker, daemon=True)
                  for _ in range(N_PROBE_BACKGROUND)]
    for t in bg_threads:
        t.start()

    atk_probe_data: list = []
    attacker_hits = 0
    total_probes  = 0

    for v_idx, (vid, pii) in enumerate(zip(victim_ids, PII_PROMPTS)):
        pii_cold_ref = victim_cold_ttfts[v_idx] if v_idx < len(victim_cold_ttfts) else None
        if pii_cold_ref is not None and math.isnan(pii_cold_ref):
            pii_cold_ref = None

        for probe_idx in range(N_ATTACKER_PROBES_PER_VICTIM):
            # ── Paired probe: measure local cold TTFT under current background load ──
            # Send a unique prompt of the same length as the PII prompt immediately
            # before the victim probe.  Both requests experience the same queue depth
            # and GPU load, so ratio = probe / cold_local captures only the
            # prefill-skip benefit of a cache hit — not scheduling jitter.
            # Under N_PROBE_BACKGROUND concurrent workers the queue fluctuates; when
            # the cold request waits longer than the probe (scheduling noise), the
            # ratio rises above the threshold → natural misclassification noise.
            cold_local_prompt = _generate_local_cold_prompt()
            cold_local_uid = f"cold_local_{random.randint(0, 10**9)}"
            cold_r = _send_one(port, model,
                               cold_local_uid if use_uid else None,
                               cold_local_prompt, role="cold")
            cold_local_t = cold_r.ttft if cold_r.error is None else None

            # Probe victim's cached PII content
            probe_r = _send_one(port, model, atk_uid, pii, role="attacker")
            probe_t = probe_r.ttft if probe_r.error is None else None

            # ratio = probe_ttft / cold_local_ttft  (both under same load conditions)
            # Fall back to Phase-1 pii_cold_ref only if local cold measurement failed.
            if probe_t and cold_local_t and cold_local_t > 0:
                ratio = probe_t / cold_local_t
                ref_used = "local"
            elif probe_t and pii_cold_ref and pii_cold_ref > 0:
                ratio = probe_t / pii_cold_ref
                ref_used = "phase1"
            else:
                ratio = 1.0
                ref_used = "fallback"

            is_hit = ratio < HIT_RATIO_THRESHOLD

            if is_hit:
                attacker_hits += 1
            total_probes += 1

            tag = "HIT " if is_hit else "    "
            cold_str = f"{cold_local_t:.3f}s" if cold_local_t is not None else "None"
            probe_str = f"{probe_t:.3f}s" if probe_t is not None else "None"
            print(
                f"    v{v_idx} probe[{probe_idx}] {tag}"
                f"cold_local={cold_str}  probe={probe_str}  ratio={ratio:.3f} [{ref_used}]",
                flush=True,
            )
            atk_probe_data.append({
                "victim": v_idx, "probe_idx": probe_idx,
                "pii_cold_ref": pii_cold_ref, "probe_ttft": probe_t,
                "ratio": ratio, "is_hit": is_hit,
            })

    _bg_stop.set()
    for t in bg_threads:
        t.join(timeout=5)

    defense_rate = (1.0 - attacker_hits / total_probes) * 100 if total_probes else 0.0

    # ------------------------------------------------------------------ #
    # Compute TTFT and TPS from regular-user phase (Phase 4)
    # ------------------------------------------------------------------ #
    reg_ttfts = sorted(r.ttft for r in reg_results if r.error is None)
    if reg_ttfts:
        ttft_mean = statistics.mean(reg_ttfts)
        ttft_p50  = statistics.median(reg_ttfts)
        ttft_p95  = reg_ttfts[min(int(len(reg_ttfts) * 0.95), len(reg_ttfts) - 1)]
        ttft_p99  = reg_ttfts[min(int(len(reg_ttfts) * 0.99), len(reg_ttfts) - 1)]
    else:
        ttft_mean = ttft_p50 = ttft_p95 = ttft_p99 = float("nan")

    total_tokens = sum(r.n_tokens for r in reg_results if r.error is None)
    tps = total_tokens / t_reg_elapsed if t_reg_elapsed > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Print per-mode summary
    # ------------------------------------------------------------------ #
    print(f"\n  ── {mode_name} results ──", flush=True)
    print(f"  Defense rate      : {defense_rate:.1f}%  "
          f"({total_probes - attacker_hits}/{total_probes} probes blocked)", flush=True)
    print(f"  Attacker cache hits: {attacker_hits}/{total_probes}  "
          f"(ratio threshold < {HIT_RATIO_THRESHOLD})", flush=True)
    if atk_probe_data:
        avg_ratio = statistics.mean(d["ratio"] for d in atk_probe_data)
        print(f"  Avg probe/cold ratio: {avg_ratio:.3f}", flush=True)
    print(f"  TTFT mean/P50/P95/P99: "
          f"{ttft_mean:.3f} / {ttft_p50:.3f} / {ttft_p95:.3f} / {ttft_p99:.3f} s",
          flush=True)
    print(f"  TPS (approx)      : {tps:.2f} tok/s", flush=True)

    return ModeMetrics(
        mode=mode_name,
        description=desc,
        cold_ttft=cold_ttft_ref,
        defense_rate=defense_rate,
        attacker_hits=attacker_hits,
        total_probes=total_probes,
        ttft_mean=ttft_mean,
        ttft_p50=ttft_p50,
        ttft_p95=ttft_p95,
        ttft_p99=ttft_p99,
        tps=tps,
        reg_results=reg_results,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

COL = {
    "mode":         20,
    "defense":       9,
    "ttft_mean":    11,
    "ttft_p50":      9,
    "ttft_p95":      9,
    "ttft_p99":      9,
    "tps":           8,
    "hits":         12,
}


def print_table(results: List[ModeMetrics]) -> None:
    sep = "=" * 100
    print(f"\n{sep}")
    print("  SafeKV Ablation Study — Summary")
    print(sep)
    hdr = (
        f"{'Mode':<{COL['mode']}} "
        f"{'Defense%':>{COL['defense']}} "
        f"{'TTFT_mean':>{COL['ttft_mean']}} "
        f"{'TTFT_P50':>{COL['ttft_p50']}} "
        f"{'TTFT_P95':>{COL['ttft_p95']}} "
        f"{'TTFT_P99':>{COL['ttft_p99']}} "
        f"{'TPS':>{COL['tps']}} "
        f"{'Hits/Probes':>{COL['hits']}}"
    )
    print(hdr)
    print("─" * 100)
    for m in results:
        row = (
            f"{m.mode:<{COL['mode']}} "
            f"{m.defense_rate:>{COL['defense']}.1f}% "
            f"{m.ttft_mean:>{COL['ttft_mean']}.3f}s "
            f"{m.ttft_p50:>{COL['ttft_p50']}.3f}s "
            f"{m.ttft_p95:>{COL['ttft_p95']}.3f}s "
            f"{m.ttft_p99:>{COL['ttft_p99']}.3f}s "
            f"{m.tps:>{COL['tps']}.1f} "
            f"{m.attacker_hits:>5}/{m.total_probes:<5}"
        )
        print(row)
    print(sep)
    # Legend
    print("\n  Defense%: % of attacker probes where TTFT ≥ cold_baseline × 0.80")
    print("  TPS: approximate completion tokens per second")


def save_csv(results: List[ModeMetrics], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "mode", "description", "cold_ttft_s",
            "defense_rate_pct", "attacker_hits", "total_probes",
            "ttft_mean_s", "ttft_p50_s", "ttft_p95_s", "ttft_p99_s",
            "tps",
        ])
        for m in results:
            writer.writerow([
                m.mode, m.description, f"{m.cold_ttft:.4f}",
                f"{m.defense_rate:.2f}", m.attacker_hits, m.total_probes,
                f"{m.ttft_mean:.4f}", f"{m.ttft_p50:.4f}",
                f"{m.ttft_p95:.4f}", f"{m.ttft_p99:.4f}",
                f"{m.tps:.2f}",
            ])
    print(f"\n  [CSV] Results saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SafeKV 4-mode ablation study with timing side-channel attacker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   default="qwen32b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model key (default: qwen32b)")
    parser.add_argument("--port",    type=int, default=8090,
                        help="Server port (default: 8090)")
    parser.add_argument("--mode",    default="all",
                        choices=list(MODES.keys()) + ["all"],
                        help="Which mode to run (default: all)")
    parser.add_argument("--no-restart", action="store_true",
                        help="Skip server restart; use currently running server. "
                             "Only valid when --mode is a single mode.")
    parser.add_argument("--output",  default=None,
                        help="CSV output path (default: logs/ablation_<model>_<timestamp>.csv)")
    args = parser.parse_args()

    if args.no_restart and args.mode == "all":
        parser.error("--no-restart requires a specific --mode (not 'all')")

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output) if args.output
        else LOG_DIR / f"ablation_{args.model}_{ts}.csv"
    )

    print(f"\n{'='*70}")
    print(f"  SafeKV Ablation Study")
    print(f"  Model : {args.model} on port {args.port}")
    print(f"  Modes : {modes_to_run}")
    print(f"  Output: {output_path}")
    print(f"{'='*70}\n", flush=True)

    all_results: List[ModeMetrics] = []

    for mode_name in modes_to_run:
        cfg = MODES[mode_name]
        proc: Optional[subprocess.Popen] = None

        print(f"\n{'='*70}")
        print(f"  MODE: {mode_name}")
        print(f"  {cfg['description']}")
        print(f"{'='*70}", flush=True)

        try:
            if not args.no_restart:
                # Kill any existing server on this port
                print(f"\n  [setup] Killing existing server on :{args.port}...", flush=True)
                _kill_port(args.port)
                time.sleep(2)

                # Start server with mode-specific SafeKV args
                print(f"  [setup] Starting server ({mode_name})...", flush=True)
                proc = _start_server(
                    args.model, args.port,
                    cfg["server_safekv_args"], mode_name
                )

                # Wait for server to be ready
                if not _wait_ready(args.port, timeout=360):
                    print(f"  [ERROR] Server did not start — skipping mode {mode_name}",
                          flush=True)
                    if proc:
                        proc.terminate()
                    continue

                # Brief extra warmup
                time.sleep(3)

            # Run workload and collect metrics
            metrics = run_workload(mode_name, cfg, args.port, args.model)
            all_results.append(metrics)

        except KeyboardInterrupt:
            print("\n  [INTERRUPT] Stopping...", flush=True)
            if proc:
                proc.terminate()
            break

        except Exception:
            print(f"\n  [ERROR] Mode {mode_name} failed:", flush=True)
            traceback.print_exc()

        finally:
            if not args.no_restart and proc is not None:
                print(f"\n  [teardown] Stopping server (mode={mode_name})...", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(3)

    # Print comparison table and save CSV
    if all_results:
        print_table(all_results)
        save_csv(all_results, output_path)
    else:
        print("\n  [WARN] No results collected.")


if __name__ == "__main__":
    random.seed(42)
    main()
