#!/usr/bin/env python3
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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3

try:
    from sklearn.metrics import roc_auc_score as _sklearn_auc
    def roc_auc_score(labels, scores):
        return _sklearn_auc(labels, scores)
except ImportError:
    def roc_auc_score(labels, scores):
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        if not pos or not neg:
            return None
        n_correct = sum(p > n for p in pos for n in neg)
        n_tied    = sum(p == n for p in pos for n in neg)
        total = len(pos) * len(neg)
        return (n_correct + 0.5 * n_tied) / total

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PYTHON = "/home/kec23008/.venv/bin/python3"

MODEL_CONFIGS = {
    "qwen32b": {"path": "/home/kec23008/Models/Qwen3-32B", "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.90", "cuda_devices": "0,1"},
    "qwen30b":  {"path": "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507", "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.90", "cuda_devices": "0,1"},
    "phi4":     {"path": "/home/kec23008/Models/Phi-4", "tp": 1, "dp": 2, "maxlen": 16384, "mem_frac": "0.85", "cuda_devices": "0,1"},
}

MODES = {
    "baseline": {
        "description": "No-SafeGuard: standard sglang, full KV cache sharing",
        "server_safekv_args": [],
        "use_user_id": False,
    },
    "full_safekv": {
        "description": "SafeGuard: async PII detector + access budget (no private-only)",
        "server_safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
        ],
        "use_user_id": True,
    },
    "private_default": {
        "description": "SafeGuard: private-by-default, immediate protection",
        "server_safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
            "--safekv-private-only",
        ],
        "use_user_id": True,
    },
}

N_VICTIMS = 30
N_COLD_POOL = 200               # cold probes — same PII distribution, never warmed up
TARGET_TOKENS_MIN = 1500
TARGET_TOKENS_MAX = 2000
EXPERIMENT_DURATION = 300
# Low concurrency keeps the server queue empty so TTFT reflects actual
# prefill time (cache-hit ~0.1s vs cache-miss ~1.5s), not queue wait.
REQUESTS_PER_SECOND = 2         # 1 victim + 1 cold per tick
CACHED_PER_SECOND   = 1
WINDOW_SECONDS      = 30        # sliding window width for ROC-AUC
RECORD_INTERVAL     = 5         # record AUC every N seconds

COMPLETION_CONFIG = {"max_tokens": 1, "temperature": 0}

PII_DATASET = Path("/home/kec23008/InferShield/datasets/english_pii_43k.jsonl")


@dataclass
class ProbePoint:
    t: float
    probe_ttft: float
    label: int


def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL)
        pids = out.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True, check=False)
                print(f"  [kill] PID {pid} on :{port}", flush=True)
        if pids:
            time.sleep(4)
    except subprocess.CalledProcessError:
        pass


def _start_server(model: str, port: int, safekv_args: List[str], log_tag: str) -> subprocess.Popen:
    cfg = MODEL_CONFIGS[model]
    log_path = LOG_DIR / f"ttp_{model}_{log_tag}.log"

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
    ] + safekv_args

    print(f"  [start] cmd: {' '.join(cmd[:6])} ...", flush=True)
    print(f"  [log]   {log_path}", flush=True)

    log_fh = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)


def _wait_ready(port: int, timeout: int = 420) -> bool:
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


def _send_one(port: int, model: str, user_id: Optional[str], prompt: str) -> float:
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

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
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
                content = (chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""))
                if content and ttft is None:
                    ttft = time.perf_counter() - t0
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        print(f"  [WARN] request error: {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    return ttft if ttft is not None else elapsed


def build_prompts(n_victims: int, n_cold: int) -> Tuple[List[str], List[str]]:
    """Load PII entries; build victim prompts then cold prompts of same length.
    Cold prompts use distinct PII entries never submitted during warmup."""
    print(f"  [build] Loading PII dataset from {PII_DATASET} ...", flush=True)
    entries = []
    with open(PII_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("source_text", "")
                if text:
                    entries.append(text)
            except json.JSONDecodeError:
                pass

    print(f"  [build] Loaded {len(entries)} PII entries", flush=True)
    random.shuffle(entries)

    def _build_batch(label: str, count: int, start_idx: int) -> Tuple[List[str], int]:
        prompts = []
        idx = start_idx
        for i in range(count):
            parts = []
            total_tokens = 0
            prefix = f"{label} {i} private record:\n\n"
            total_tokens += count_tokens(prefix)
            while total_tokens < TARGET_TOKENS_MIN and idx < len(entries):
                text = entries[idx]
                t = count_tokens(text)
                parts.append(text)
                total_tokens += t
                idx += 1
                if total_tokens >= TARGET_TOKENS_MIN:
                    break
            prompt = prefix + "\n\n".join(parts)
            prompts.append(prompt)
            if label == "Victim":
                print(f"    victim[{i}] tokens={count_tokens(prompt)}", flush=True)
        return prompts, idx

    victim_prompts, idx_after_victims = _build_batch("Victim", n_victims, 0)
    cold_prompts, _ = _build_batch("Cold", n_cold, idx_after_victims)

    print(f"  [build] Built {len(victim_prompts)} victim prompts, {len(cold_prompts)} cold prompts", flush=True)
    return victim_prompts, cold_prompts


def compute_auc(points: List[ProbePoint], t_now: Optional[float] = None) -> Optional[float]:
    if not points:
        return None
    if t_now is None:
        t_now = points[-1].t
    cutoff = t_now - WINDOW_SECONDS
    recent = [p for p in points if p.t >= cutoff]
    if len(recent) < 4:
        return None
    labels = [p.label for p in recent]
    scores = [-p.probe_ttft for p in recent]
    if len(set(labels)) < 2:
        return None
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return None


def run_experiment(
    mode_name: str, mode_cfg: dict, port: int, model: str,
    victim_prompts: List[str], cold_prompts: List[str],
) -> Tuple[List[float], List[Optional[float]]]:
    use_uid = mode_cfg["use_user_id"]
    print(f"\n{'='*60}", flush=True)
    print(f"  Running mode: {mode_name}", flush=True)
    print(f"  {mode_cfg['description']}", flush=True)
    print(f"{'='*60}", flush=True)

    victim_ids = [f"victim_{i}" for i in range(len(victim_prompts))]

    # Use a fresh UUID per probe to prevent attacker from building its own KV cache.
    # This ensures AUC measures ONLY victim-cache visibility, not attacker self-caching.
    def make_atk_uid() -> Optional[str]:
        return f"atk_{uuid.uuid4().hex[:12]}" if use_uid else None

    print(f"\n  [Phase 0] Warmup: all victims send their PII prompts ...", flush=True)
    def warmup_one(i):
        uid = victim_ids[i] if use_uid else None
        ttft = _send_one(port, model, uid, victim_prompts[i])
        return i, ttft

    warmup_workers = 8 if model == "phi4" else min(len(victim_prompts), 16)
    with ThreadPoolExecutor(max_workers=warmup_workers) as ex:
        futs = [ex.submit(warmup_one, i) for i in range(len(victim_prompts))]
        for fut in as_completed(futs):
            try:
                i, ttft = fut.result()
                print(f"    warmup victim[{i}] TTFT={ttft:.3f}s", flush=True)
            except Exception as e:
                print(f"    [WARN] warmup error: {e}", flush=True)

    print(f"\n  [Attack phase] Running for {EXPERIMENT_DURATION}s ...", flush=True)

    all_points: List[ProbePoint] = []
    times_recorded: List[float] = []
    aucs_recorded: List[Optional[float]] = []
    next_record_t = 0.0

    t_start = time.time()
    t_end = t_start + EXPERIMENT_DURATION

    rps = REQUESTS_PER_SECOND
    cached_rps = CACHED_PER_SECOND
    COLD_PER_SECOND = rps - cached_rps

    while time.time() < t_end:
        tick_start = time.time()
        if tick_start >= t_end:
            break

        cached_victims = random.choices(range(len(victim_prompts)), k=cached_rps)

        # Build jobs list: equal mix of cached (label=1) and cold (label=0).
        # SEQUENTIAL execution: each probe runs alone so the server is idle
        # when it arrives — TTFT reflects true prefill time, not queue wait.
        # Concurrent submission causes victim+cold to land in the same prefill
        # batch (equal TTFT → AUC=0.5 regardless of caching).
        all_jobs = [(True, v) for v in cached_victims] + [(False, 0)] * COLD_PER_SECOND
        random.shuffle(all_jobs)

        for is_cached, v_idx in all_jobs:
            try:
                if is_cached:
                    ttft = _send_one(port, model, make_atk_uid(), victim_prompts[v_idx])
                    all_points.append(ProbePoint(t=time.time() - t_start, probe_ttft=ttft, label=1))
                else:
                    cold_prompt = random.choice(cold_prompts)
                    ttft = _send_one(port, model, make_atk_uid(), cold_prompt)
                    all_points.append(ProbePoint(t=time.time() - t_start, probe_ttft=ttft, label=0))
            except Exception as exc:
                print(f"  [WARN] probe error: {exc}", flush=True)

        elapsed_now = time.time() - t_start
        if elapsed_now >= next_record_t:
            auc = compute_auc(all_points, t_now=elapsed_now)
            times_recorded.append(elapsed_now)
            aucs_recorded.append(auc)
            print(f"    t={elapsed_now:.1f}s  n_points={len(all_points)}  AUC={auc}", flush=True)
            next_record_t = elapsed_now + RECORD_INTERVAL

        # pace to ~1 tick per second
        sleep_remaining = 1.0 - (time.time() - tick_start)
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)

    final_auc = compute_auc(all_points)
    if not times_recorded or times_recorded[-1] < EXPERIMENT_DURATION - 1:
        times_recorded.append(time.time() - t_start)
        aucs_recorded.append(final_auc)

    return times_recorded, aucs_recorded


def main():
    parser = argparse.ArgumentParser(description="SafeKV Time-to-Protection experiment")
    parser.add_argument("--model", default="qwen32b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"time_to_protection_{args.model}_{ts}.csv"
    png_path = LOG_DIR / f"time_to_protection_{args.model}_{ts}.png"

    print(f"\n{'='*70}", flush=True)
    print(f"  Time-to-Protection Experiment", flush=True)
    print(f"  Model: {args.model}  Port: {args.port}", flush=True)
    print(f"  N_VICTIMS={N_VICTIMS}  Duration={EXPERIMENT_DURATION}s  Window={WINDOW_SECONDS}s  RPS={REQUESTS_PER_SECOND}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Build prompts once, reuse across all modes (each mode re-warms up victims)
    print(f"\n  [build] Building victim and cold prompt pools ...", flush=True)
    victim_prompts, cold_prompts = build_prompts(N_VICTIMS, N_COLD_POOL)
    print(f"  [build] victim={len(victim_prompts)}, cold_pool={len(cold_prompts)}", flush=True)

    modes_to_run = list(MODES.keys())
    results = {}

    for mode_name in modes_to_run:
        mode_cfg = MODES[mode_name]
        proc = None

        try:
            if not args.no_restart:
                print(f"\n  [setup] Killing existing server on :{args.port}...", flush=True)
                _kill_port(args.port)
                time.sleep(2)

                print(f"  [setup] Starting server ({mode_name})...", flush=True)
                proc = _start_server(args.model, args.port, mode_cfg["server_safekv_args"], mode_name)

                if not _wait_ready(args.port, timeout=420):
                    print(f"  [ERROR] Server not ready — skipping {mode_name}", flush=True)
                    if proc:
                        proc.terminate()
                    continue

                time.sleep(3)

            times_rec, aucs_rec = run_experiment(mode_name, mode_cfg, args.port, args.model,
                                                  victim_prompts, cold_prompts)
            results[mode_name] = (times_rec, aucs_rec)

        except KeyboardInterrupt:
            print("\n  [INTERRUPT]", flush=True)
            if proc:
                proc.terminate()
            break

        except Exception:
            print(f"\n  [ERROR] Mode {mode_name} failed:", flush=True)
            traceback.print_exc()

        finally:
            if not args.no_restart and proc is not None:
                print(f"\n  [teardown] Stopping server ({mode_name})...", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(3)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "time_s", "roc_auc"])
        for mode_name, (times_rec, aucs_rec) in results.items():
            for t, auc in zip(times_rec, aucs_rec):
                writer.writerow([mode_name, f"{t:.2f}", f"{auc:.4f}" if auc is not None else "nan"])
    print(f"\n  [CSV] Saved: {csv_path}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"baseline": "#d62728", "private_default": "#2ca02c"}
    labels_map = {"baseline": "No-SafeGuard", "private_default": "SafeGuard (private_default)"}

    for mode_name, (times_rec, aucs_rec) in results.items():
        valid = [(t, a) for t, a in zip(times_rec, aucs_rec) if a is not None]
        if not valid:
            continue
        ts_plot, auc_plot = zip(*valid)
        ax.plot(ts_plot, auc_plot,
                label=labels_map.get(mode_name, mode_name),
                color=colors.get(mode_name, None),
                linewidth=2, marker="o", markersize=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROC-AUC (sliding window)")
    ax.set_title(f"Time to Protection — {args.model}\n(window={WINDOW_SECONDS}s, {REQUESTS_PER_SECOND} req/s attacker, TTFT-based cache-hit attack)")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  [PNG] Saved: {png_path}", flush=True)

    print(f"\n  Done. Results in {LOG_DIR}", flush=True)


if __name__ == "__main__":
    random.seed(42)
    main()
