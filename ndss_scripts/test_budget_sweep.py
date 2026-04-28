#!/usr/bin/env python3
"""
SafeKV Budget (B) × Threshold (K) Sweep
=========================================
Measures defense_rate, TPS, and TTFT across all combinations of
access budget B and creator threshold K.

Default:
  B values : 50, 100, 200
  K values : 2, 4, 8
  Models   : phi4, qwen30b, qwen32b  (sequential)

Usage:
  python test_budget_sweep.py --model all
  python test_budget_sweep.py --model phi4 --budgets 50 100 200 --thresholds 2 4 8
  python test_budget_sweep.py --model phi4 --no-restart --budget-single 100 --threshold-single 2
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from ablation_prompts import PII_PROMPTS

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

MODEL_PORTS = {"qwen32b": 8090, "qwen30b": 8094, "phi4": 8092}

DEFAULT_B_VALUES = [50, 100, 200]
DEFAULT_K_VALUES = [2, 4, 8]

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

COMPLETION_CONFIG = {"max_tokens": 128, "temperature": 0}

N_VICTIMS                    = 5
N_VICTIM_WARMUP              = 3
N_REGULAR_USERS              = 32
N_REGULAR_REQS               = 20
N_ATTACKER_PROBES_PER_VICTIM = 5
N_PROBE_BACKGROUND           = 20
DETECTION_WAIT               = 30
HIT_RATIO_THRESHOLD          = 0.80


# ---------------------------------------------------------------------------
@dataclass
class ReqResult:
    role: str
    user_id: Optional[str]
    ttft: float
    total_time: float
    n_tokens: int
    error: Optional[str] = None


@dataclass
class SweepResult:
    model: str
    budget_B: int
    threshold_K: int
    cold_ttft: float
    defense_rate: float
    attacker_hits: int
    total_probes: int
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tps: float


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _send_one(port: int, model: str, user_id: Optional[str],
              prompt: str, role: str) -> ReqResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        **COMPLETION_CONFIG,
    }
    if user_id is not None:
        payload["user_id"] = user_id

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    n_tokens = 0
    error: Optional[str] = None

    try:
        resp = requests.post(url, json=payload,
                             headers={"Content-Type": "application/json"},
                             stream=True, timeout=30)
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
                    n_tokens += 1
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - t0
    if ttft is None:
        ttft = elapsed
    return ReqResult(role=role, user_id=user_id, ttft=ttft,
                     total_time=elapsed, n_tokens=max(n_tokens, 1), error=error)


def measure_cold_ttft(port: int, model: str) -> float:
    unique_prompt = (
        f"Describe the properties of the number {random.randint(10**6, 10**7)} "
        f"in a single sentence. Be concise."
    )
    r = _send_one(port, model, user_id=None, prompt=unique_prompt, role="cold")
    latency = r.ttft if r.error is None else 99.9
    print(f"  [cold baseline] TTFT={latency:.3f}s  err={r.error}", flush=True)
    return latency


def _generate_local_cold_prompt(target_n_nums: int = 1500) -> str:
    nums = " ".join(str(random.randint(10**7, 10**8 - 1)) for _ in range(target_n_nums))
    uid = random.randint(10**15, 10**16)
    return f"Local cold reference {uid}: {nums}"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"],
                                      text=True, stderr=subprocess.DEVNULL)
        pids = out.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True, check=False)
                print(f"  [kill] PID {pid} on :{port}", flush=True)
        if pids:
            time.sleep(4)
    except subprocess.CalledProcessError:
        pass


def _start_server(model: str, port: int, budget_B: int, threshold_K: int,
                  log_tag: str) -> subprocess.Popen:
    cfg = MODEL_CONFIGS[model]
    log_path = LOG_DIR / f"sweep_{model}_{log_tag}.log"

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
        "--safekv-access-budget",      str(budget_B),
        "--safekv-creator-threshold",  str(threshold_K),
    ]

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
# Workload
# ---------------------------------------------------------------------------

def run_workload(model: str, budget_B: int, threshold_K: int,
                 port: int) -> SweepResult:
    print(f"\n{'─'*60}", flush=True)
    print(f"  Model={model}  B={budget_B}  K={threshold_K}", flush=True)
    print(f"{'─'*60}", flush=True)

    victim_ids  = [f"victim_{i}"  for i in range(N_VICTIMS)]
    regular_ids = [f"regular_{i}" for i in range(N_REGULAR_USERS)]
    atk_uid     = "attacker_007"

    # Phase 1
    print(f"\n  [Phase 1] Victim warmup ({N_VICTIMS} × {N_VICTIM_WARMUP} reps)...", flush=True)
    victim_cold_ttfts: List[float] = []
    for v_idx, (vid, pii) in enumerate(zip(victim_ids, PII_PROMPTS)):
        for rep in range(N_VICTIM_WARMUP):
            r = _send_one(port, model, vid, pii, role="victim")
            if rep == 0:
                victim_cold_ttfts.append(r.ttft if r.error is None else float("nan"))
            status = "err" if r.error else f"TTFT={r.ttft:.3f}s"
            print(f"    victim[{v_idx}] rep={rep} {status}", flush=True)
            time.sleep(0.3)

    # Phase 2
    print(f"\n  [Phase 2] Waiting {DETECTION_WAIT}s for async PII detection...", flush=True)
    time.sleep(DETECTION_WAIT)

    # Phase 2.5
    print(f"\n  [Phase 2.5] PII cold TTFT calibration...", flush=True)
    calib_uid = "calib_user_never_seen_pii"
    for v_idx, pii in enumerate(PII_PROMPTS):
        r = _send_one(port, model, calib_uid, pii, role="cold")
        ttft_val = r.ttft if r.error is None else float("nan")
        print(f"    PII[{v_idx}] calib cold TTFT = {ttft_val:.3f}s", flush=True)

    # Phase 3
    print(f"\n  [Phase 3] Reference cold TTFT...", flush=True)
    cold_samples = [measure_cold_ttft(port, model) for _ in range(3)]
    cold_ttft_ref = statistics.mean(cold_samples)
    print(f"  Reference cold TTFT: {cold_ttft_ref:.3f}s", flush=True)

    # Phase 4
    print(f"\n  [Phase 4] Concurrent regular-user workload...", flush=True)
    regular_tasks = []
    for uid_base in regular_ids:
        prompts = random.sample(NON_PII_PROMPTS, min(N_REGULAR_REQS, len(NON_PII_PROMPTS)))
        for prompt in prompts:
            regular_tasks.append((uid_base, prompt))

    reg_results: List[ReqResult] = []
    t_reg_start = time.time()
    with ThreadPoolExecutor(max_workers=N_REGULAR_USERS + 4) as ex:
        futs = [ex.submit(_send_one, port, model, uid, prompt, "regular")
                for uid, prompt in regular_tasks]
        for fut in as_completed(futs):
            try:
                reg_results.append(fut.result())
            except Exception as e:
                print(f"  [WARN] {e}", flush=True)
    t_reg_elapsed = time.time() - t_reg_start

    # Phase 5
    import math
    print(f"\n  [Phase 5] Attacker probes (background={N_PROBE_BACKGROUND})...", flush=True)

    _bg_stop = threading.Event()

    def _bg_worker():
        idx = 0
        while not _bg_stop.is_set():
            uid = f"bg_{random.randint(0, 9999)}"
            _send_one(port, model, uid, NON_PII_PROMPTS[idx % len(NON_PII_PROMPTS)], "regular")
            idx += 1

    bg_threads = [threading.Thread(target=_bg_worker, daemon=True)
                  for _ in range(N_PROBE_BACKGROUND)]
    for t in bg_threads:
        t.start()

    attacker_hits = 0
    total_probes  = 0

    for v_idx, (vid, pii) in enumerate(zip(victim_ids, PII_PROMPTS)):
        pii_cold_ref = victim_cold_ttfts[v_idx] if v_idx < len(victim_cold_ttfts) else None
        if pii_cold_ref is not None and math.isnan(pii_cold_ref):
            pii_cold_ref = None

        for probe_idx in range(N_ATTACKER_PROBES_PER_VICTIM):
            cold_local_prompt = _generate_local_cold_prompt()
            cold_local_uid = f"cold_local_{random.randint(0, 10**9)}"
            cold_r = _send_one(port, model, cold_local_uid, cold_local_prompt, "cold")
            cold_local_t = cold_r.ttft if cold_r.error is None else None

            probe_r = _send_one(port, model, atk_uid, pii, "attacker")
            probe_t = probe_r.ttft if probe_r.error is None else None

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
            cold_str = f"{cold_local_t:.3f}s" if cold_local_t is not None else "err"
            probe_str = f"{probe_t:.3f}s" if probe_t is not None else "err"
            print(f"    v{v_idx} probe[{probe_idx}] {tag}"
                  f"cold={cold_str}  probe={probe_str}  ratio={ratio:.3f} [{ref_used}]",
                  flush=True)

    _bg_stop.set()
    for t in bg_threads:
        t.join(timeout=5)

    defense_rate = (1.0 - attacker_hits / total_probes) * 100 if total_probes else 0.0

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

    print(f"\n  ── B={budget_B} K={threshold_K} results ──", flush=True)
    print(f"  Defense rate : {defense_rate:.1f}%  ({total_probes-attacker_hits}/{total_probes} blocked)", flush=True)
    print(f"  TTFT mean/P50/P95/P99: {ttft_mean:.3f}/{ttft_p50:.3f}/{ttft_p95:.3f}/{ttft_p99:.3f} s", flush=True)
    print(f"  TPS          : {tps:.2f} tok/s", flush=True)

    return SweepResult(
        model=model, budget_B=budget_B, threshold_K=threshold_K,
        cold_ttft=cold_ttft_ref, defense_rate=defense_rate,
        attacker_hits=attacker_hits, total_probes=total_probes,
        ttft_mean=ttft_mean, ttft_p50=ttft_p50,
        ttft_p95=ttft_p95, ttft_p99=ttft_p99, tps=tps,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: List[SweepResult]) -> None:
    sep = "=" * 112
    print(f"\n{sep}")
    print("  SafeKV B × K Sweep — Summary")
    print(sep)
    hdr = (
        f"{'Model':<10} {'B':>5} {'K':>4} {'Defense%':>9} "
        f"{'TTFT_mean':>10} {'TTFT_P50':>9} {'TTFT_P95':>9} {'TTFT_P99':>9} "
        f"{'TPS':>8} {'Hits/Total':>11}"
    )
    print(hdr)
    print("─" * 112)
    cur_model = None
    for m in results:
        if cur_model and cur_model != m.model:
            print("─" * 112)
        cur_model = m.model
        row = (
            f"{m.model:<10} {m.budget_B:>5} {m.threshold_K:>4} {m.defense_rate:>9.1f}% "
            f"{m.ttft_mean:>10.3f}s {m.ttft_p50:>9.3f}s "
            f"{m.ttft_p95:>9.3f}s {m.ttft_p99:>9.3f}s "
            f"{m.tps:>8.1f} "
            f"{m.attacker_hits:>5}/{m.total_probes:<5}"
        )
        print(row)
    print(sep)
    print("\n  Defense%: % of attacker probes where TTFT ≥ cold_baseline × 0.80")
    print("  TPS: approximate completion tokens per second (Phase 4 regular users)")


def save_csv(results: List[SweepResult], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model", "budget_B", "threshold_K",
            "cold_ttft_s", "defense_rate_pct", "attacker_hits", "total_probes",
            "ttft_mean_s", "ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "tps",
        ])
        for m in results:
            writer.writerow([
                m.model, m.budget_B, m.threshold_K,
                f"{m.cold_ttft:.4f}", f"{m.defense_rate:.2f}",
                m.attacker_hits, m.total_probes,
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
        description="SafeKV B × K sweep — defense_rate / TPS / TTFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_B_VALUES,
                        metavar="B", help="Budget B values (default: 50 100 200)")
    parser.add_argument("--thresholds", nargs="+", type=int, default=DEFAULT_K_VALUES,
                        metavar="K", help="Creator threshold K values (default: 2 4 8)")
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--budget-single", type=int, default=None)
    parser.add_argument("--threshold-single", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.no_restart:
        if args.model == "all":
            parser.error("--no-restart requires a specific --model")
        if args.budget_single is None or args.threshold_single is None:
            parser.error("--no-restart requires --budget-single and --threshold-single")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    # Build (B, K) combinations
    if args.no_restart:
        combos: List[Tuple[int, int]] = [(args.budget_single, args.threshold_single)]
    else:
        combos = [(b, k) for b in args.budgets for k in args.thresholds]

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output) if args.output
        else LOG_DIR / f"bk_sweep_{ts}.csv"
    )

    total_runs = len(models_to_run) * len(combos)
    print(f"\n{'='*70}")
    print(f"  SafeKV B × K Sweep")
    print(f"  Models     : {models_to_run}")
    print(f"  B values   : {args.budgets}")
    print(f"  K values   : {args.thresholds}")
    print(f"  Combinations: {len(combos)}  (B×K)")
    print(f"  Total runs : {total_runs}")
    print(f"  Output     : {output_path}")
    print(f"{'='*70}\n", flush=True)

    all_results: List[SweepResult] = []
    run_idx = 0

    for model in models_to_run:
        port = MODEL_PORTS[model]

        for budget_B, threshold_K in combos:
            run_idx += 1
            proc = None
            print(f"\n{'='*70}")
            print(f"  [{run_idx}/{total_runs}] MODEL={model}  B={budget_B}  K={threshold_K}  port={port}")
            print(f"{'='*70}", flush=True)

            try:
                if not args.no_restart:
                    print(f"  [setup] Killing port :{port}...", flush=True)
                    _kill_port(port)
                    time.sleep(2)

                    proc = _start_server(model, port, budget_B, threshold_K,
                                         log_tag=f"B{budget_B}_K{threshold_K}")

                    if not _wait_ready(port, timeout=360):
                        print(f"  [ERROR] Server did not start — skipping", flush=True)
                        if proc:
                            proc.terminate()
                        continue
                    time.sleep(3)

                result = run_workload(model, budget_B, threshold_K, port)
                all_results.append(result)

            except KeyboardInterrupt:
                print("\n  [INTERRUPT] Stopping...", flush=True)
                if proc:
                    proc.terminate()
                break

            except Exception:
                print(f"\n  [ERROR] model={model} B={budget_B} K={threshold_K} failed:", flush=True)
                traceback.print_exc()

            finally:
                if not args.no_restart and proc is not None:
                    print(f"  [teardown] Stopping server...", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    time.sleep(3)

    if all_results:
        print_table(all_results)
        save_csv(all_results, output_path)
    else:
        print("\n  [WARN] No results collected.")


if __name__ == "__main__":
    random.seed(42)
    main()
