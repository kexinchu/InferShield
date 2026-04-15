#!/usr/bin/env python3
"""
Benchmark client — collects per-query metrics via SSE streaming:
  TTFT, TPOT, Total_Time, Input_Tokens, Output_Tokens, TPS
Also samples GPU memory & power via nvidia-smi.

Usage:
  python3 bench_metrics.py --server 127.0.0.1:8090 [--num-queries 50] [--max-tokens 128] [--concurrency 1]
  python3 bench_metrics.py --server 127.0.0.1:8090 --dataset /path/to/data.jsonl
"""

import argparse
import json
import time
import sys
import os
import subprocess
import threading
import csv
import io
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import requests

# ------------------------------------------------------------------ #
#  GPU Monitor (background thread)
# ------------------------------------------------------------------ #

class GPUMonitor:
    """Samples GPU memory/power in a background thread via nvidia-smi."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: List[Dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,memory.used,memory.total,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5
                )
                ts = time.time()
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        self.samples.append({
                            "ts": ts,
                            "gpu": int(parts[0]),
                            "mem_used_mib": float(parts[1]),
                            "mem_total_mib": float(parts[2]),
                            "power_w": float(parts[3]),
                        })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> Dict:
        if not self.samples:
            return {}
        mem_used = [s["mem_used_mib"] for s in self.samples]
        power = [s["power_w"] for s in self.samples]
        return {
            "gpu_mem_used_avg_mib": np.mean(mem_used),
            "gpu_mem_used_max_mib": np.max(mem_used),
            "gpu_power_avg_w": np.mean(power),
            "gpu_power_max_w": np.max(power),
            "gpu_samples": len(self.samples),
        }


# ------------------------------------------------------------------ #
#  Streaming request — collects TTFT & per-token times
# ------------------------------------------------------------------ #

def send_streaming_request(
    server_url: str,
    prompt,  # str or List[Dict] (messages array)
    model_name: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> Dict:
    """Send a single streaming chat-completion request and collect timing."""
    url = f"http://{server_url}/v1/chat/completions"
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    result = {
        "ttft_ms": None,        # time to first token
        "tpot_ms": None,        # avg time per output token (excluding first)
        "total_time_ms": None,
        "input_tokens": None,
        "output_tokens": 0,
        "tps": None,            # output tokens / total_time
        "token_times_ms": [],   # per-chunk arrival times
        "content": "",
        "error": None,
    }

    try:
        t_start = time.perf_counter()
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True,
            timeout=120,
        )
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        t_first_token = None
        prev_ts = t_start
        token_intervals = []

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check for usage in final chunk (sent with stream_options)
            usage = chunk.get("usage")
            if usage:
                result["input_tokens"] = usage.get("prompt_tokens")

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            text = delta.get("content", "")
            if not text:
                continue

            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
                result["ttft_ms"] = (t_first_token - t_start) * 1000
            else:
                token_intervals.append((now - prev_ts) * 1000)

            prev_ts = now
            result["output_tokens"] += 1
            result["content"] += text

        t_end = time.perf_counter()
        result["total_time_ms"] = (t_end - t_start) * 1000
        result["token_times_ms"] = token_intervals

        if token_intervals:
            result["tpot_ms"] = np.mean(token_intervals)

        if result["output_tokens"] > 0 and result["total_time_ms"] > 0:
            result["tps"] = result["output_tokens"] / (result["total_time_ms"] / 1000)

        # If usage wasn't returned in stream, try to estimate
        if result["input_tokens"] is None:
            result["input_tokens"] = -1  # unknown

    except Exception as e:
        result["error"] = str(e)

    return result


# ------------------------------------------------------------------ #
#  Non-streaming fallback (also collects usage from server)
# ------------------------------------------------------------------ #

def send_request(
    server_url: str,
    prompt,  # str or List[Dict] (messages array)
    model_name: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> Dict:
    """Non-streaming request — no TTFT/TPOT, but usage is accurate."""
    url = f"http://{server_url}/v1/chat/completions"
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    result = {
        "ttft_ms": None,
        "tpot_ms": None,
        "total_time_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "tps": None,
        "content": "",
        "error": None,
    }
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,
        )
        t1 = time.perf_counter()
        result["total_time_ms"] = (t1 - t0) * 1000

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result

        body = resp.json()
        result["content"] = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        result["input_tokens"] = usage.get("prompt_tokens", -1)
        result["output_tokens"] = usage.get("completion_tokens", 0)
        if result["output_tokens"] > 0:
            result["tps"] = result["output_tokens"] / (result["total_time_ms"] / 1000)
    except Exception as e:
        result["error"] = str(e)
    return result


# ------------------------------------------------------------------ #
#  Prompt sources
# ------------------------------------------------------------------ #

DEFAULT_PROMPTS = [
    "Explain the concept of KV-cache in transformer inference in 3 sentences.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the main differences between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet in 100 words.",
    "List 5 best practices for securing a REST API.",
    "What is the time complexity of quicksort and why?",
    "Explain how attention mechanism works in transformers.",
    "Write a bash one-liner to find all .py files modified in the last 24 hours.",
    "What is differential privacy? Explain in simple terms.",
    "Compare and contrast BERT and GPT architectures.",
]


def load_prompts_from_jsonl(path: str, sample_n: int = 50):
    """Load prompts from a JSONL file.

    Returns list of prompts. Each prompt is either:
      - str: single-turn prompt
      - List[Dict]: multi-turn messages array (from prepare_benchmark_data.py)
    """
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Multi-turn format from prepare_benchmark_data.py
            if "messages" in item:
                prompts.append(item["messages"])
            elif "source_text" in item:
                prompts.append(item["source_text"])
            elif "prompt" in item:
                prompts.append(item["prompt"])
            elif "rewritten" in item:
                prompts.append(item["rewritten"])
            elif "text" in item:
                prompts.append(item["text"])
            if len(prompts) >= sample_n:
                break
    return prompts


# ------------------------------------------------------------------ #
#  Wait for server ready
# ------------------------------------------------------------------ #

def detect_model_name(server_url: str) -> str:
    """Auto-detect the model name from the server."""
    try:
        r = requests.get(f"http://{server_url}/v1/models", timeout=5)
        if r.status_code == 200:
            models = r.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return "default"


def wait_for_server(server_url: str, timeout: int = 300):
    """Poll the health endpoint until the server is ready."""
    url = f"http://{server_url}/health"
    print(f"[INFO] Waiting for server at {server_url} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"[INFO] Server ready ({time.time()-t0:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"[ERROR] Server not ready after {timeout}s")
    return False


# ------------------------------------------------------------------ #
#  Main benchmark loop
# ------------------------------------------------------------------ #

def run_benchmark(args):
    server_url = args.server
    model_name = args.model_name
    max_tokens = args.max_tokens
    num_queries = args.num_queries
    concurrency = args.concurrency
    use_stream = not args.no_stream

    # Load prompts
    if args.dataset and os.path.exists(args.dataset):
        prompts = load_prompts_from_jsonl(args.dataset, sample_n=num_queries)
        print(f"[INFO] Loaded {len(prompts)} prompts from {args.dataset}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"[INFO] Using {len(prompts)} built-in prompts")

    # Expand prompts to num_queries by cycling
    while len(prompts) < num_queries:
        prompts = prompts + prompts
    prompts = prompts[:num_queries]

    # Wait for server
    if not wait_for_server(server_url):
        sys.exit(1)

    # Auto-detect model name if default
    if model_name == "default":
        model_name = detect_model_name(server_url)
        print(f"[INFO] Auto-detected model: {model_name}")

    # Warmup
    print(f"[INFO] Warmup: sending 3 requests ...")
    for i in range(min(3, len(prompts))):
        send_request(server_url, prompts[i], model_name, max_tokens=16)

    # Start GPU monitor
    gpu_mon = GPUMonitor(interval=1.0)
    gpu_mon.start()

    # Run benchmark
    send_fn = send_streaming_request if use_stream else send_request
    results = []
    errors = 0

    print(f"\n{'='*60}")
    print(f" Benchmark: {model_name}")
    print(f" Queries: {num_queries}  Concurrency: {concurrency}  Streaming: {use_stream}")
    print(f" Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    bench_start = time.perf_counter()

    if concurrency == 1:
        for i, prompt in enumerate(prompts):
            r = send_fn(server_url, prompt, model_name, max_tokens)
            if r["error"]:
                errors += 1
                print(f"  [{i+1}/{num_queries}] ERROR: {r['error']}")
            else:
                results.append(r)
                print(f"  [{i+1}/{num_queries}] TTFT={r['ttft_ms']:.1f}ms  "
                      f"Output={r['output_tokens']}tok  "
                      f"Total={r['total_time_ms']:.1f}ms  "
                      f"TPS={r['tps']:.1f}" if r['tps'] else
                      f"  [{i+1}/{num_queries}] Total={r['total_time_ms']:.1f}ms")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(send_fn, server_url, p, model_name, max_tokens): idx
                for idx, p in enumerate(prompts)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                r = fut.result()
                if r["error"]:
                    errors += 1
                    print(f"  [{idx+1}/{num_queries}] ERROR: {r['error']}")
                else:
                    results.append(r)

    bench_end = time.perf_counter()
    bench_time = bench_end - bench_start

    gpu_mon.stop()

    # ---- Report ----
    if not results:
        print("\n[ERROR] No successful requests. Cannot generate report.")
        return

    ttft_list = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    tpot_list = [r["tpot_ms"] for r in results if r["tpot_ms"] is not None]
    total_list = [r["total_time_ms"] for r in results]
    input_tok_list = [r["input_tokens"] for r in results if r["input_tokens"] and r["input_tokens"] > 0]
    output_tok_list = [r["output_tokens"] for r in results]
    tps_list = [r["tps"] for r in results if r["tps"] is not None]

    def pstats(arr, name):
        if not arr:
            return f"  {name:20s}: N/A"
        a = np.array(arr)
        return (f"  {name:20s}: "
                f"avg={np.mean(a):8.2f}  "
                f"p50={np.percentile(a,50):8.2f}  "
                f"p95={np.percentile(a,95):8.2f}  "
                f"p99={np.percentile(a,99):8.2f}  "
                f"min={np.min(a):8.2f}  "
                f"max={np.max(a):8.2f}")

    gpu_info = gpu_mon.summary()

    report = []
    report.append("")
    report.append("=" * 70)
    report.append(f"  BENCHMARK RESULTS — {model_name}")
    report.append(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append(f"  Successful / Total:  {len(results)} / {num_queries}  (errors={errors})")
    report.append(f"  Wall clock time:     {bench_time:.2f}s")
    report.append(f"  Concurrency:         {concurrency}")
    report.append("")
    report.append("--- Latency (ms) ---")
    report.append(pstats(ttft_list, "TTFT"))
    report.append(pstats(tpot_list, "TPOT"))
    report.append(pstats(total_list, "Total_Time"))
    report.append("")
    report.append("--- Tokens ---")
    report.append(pstats(input_tok_list, "Input_Tokens"))
    report.append(pstats(output_tok_list, "Output_Tokens"))
    report.append("")
    report.append("--- Throughput ---")
    report.append(pstats(tps_list, "TPS (tok/s)"))
    total_output_tok = sum(output_tok_list)
    report.append(f"  {'Aggregate TPS':20s}: {total_output_tok / bench_time:.2f} tok/s")
    report.append("")

    if gpu_info:
        report.append("--- GPU ---")
        report.append(f"  GPU Mem Used (avg):  {gpu_info['gpu_mem_used_avg_mib']:.0f} MiB")
        report.append(f"  GPU Mem Used (max):  {gpu_info['gpu_mem_used_max_mib']:.0f} MiB")
        report.append(f"  GPU Power (avg):     {gpu_info['gpu_power_avg_w']:.1f} W")
        report.append(f"  GPU Power (max):     {gpu_info['gpu_power_max_w']:.1f} W")
        report.append(f"  GPU Samples:         {gpu_info['gpu_samples']}")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    # ---- Save results ----
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use short model name for file naming (strip path)
    short_name = model_name.rstrip("/").split("/")[-1]
    base_name = f"{short_name}_{ts}"

    # Save summary
    summary_path = os.path.join(out_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(report_text)
    print(f"\n[INFO] Summary saved to {summary_path}")

    # Save per-query CSV
    csv_path = os.path.join(out_dir, f"{base_name}_detail.csv")
    fieldnames = ["query_id", "ttft_ms", "tpot_ms", "total_time_ms",
                  "input_tokens", "output_tokens", "tps"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({
                "query_id": i,
                "ttft_ms": f"{r['ttft_ms']:.2f}" if r["ttft_ms"] else "",
                "tpot_ms": f"{r['tpot_ms']:.2f}" if r["tpot_ms"] else "",
                "total_time_ms": f"{r['total_time_ms']:.2f}",
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "tps": f"{r['tps']:.2f}" if r["tps"] else "",
            })
    print(f"[INFO] Detail CSV saved to {csv_path}")

    # Save GPU samples
    if gpu_mon.samples:
        gpu_path = os.path.join(out_dir, f"{base_name}_gpu.csv")
        with open(gpu_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ts", "gpu", "mem_used_mib", "mem_total_mib", "power_w"])
            writer.writeheader()
            writer.writerows(gpu_mon.samples)
        print(f"[INFO] GPU data saved to {gpu_path}")


# ------------------------------------------------------------------ #
#  Outlier detection: IQR-based (8 × Q3 rule from the spec)
# ------------------------------------------------------------------ #

def detect_outliers(results: List[Dict], field: str = "total_time_ms"):
    """Flag values exceeding 8 × Q3 as per the benchmark spec."""
    values = [r[field] for r in results if r.get(field) is not None]
    if not values:
        return []
    q3 = np.percentile(values, 75)
    threshold = 8 * q3
    outliers = [(i, v) for i, v in enumerate(values) if v > threshold]
    if outliers:
        print(f"\n[WARN] {len(outliers)} outlier(s) detected for '{field}' (threshold = 8×Q3 = {threshold:.2f}):")
        for idx, val in outliers:
            print(f"  query {idx}: {val:.2f}")
    return outliers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang Benchmark Client")
    parser.add_argument("--server", default="127.0.0.1:8090",
                        help="Server address host:port")
    parser.add_argument("--model-name", default="default",
                        help="Model name for API (arbitrary string)")
    parser.add_argument("--num-queries", type=int, default=50,
                        help="Number of queries to send")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens per query")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to JSONL dataset for prompts")
    parser.add_argument("--no-stream", action="store_true",
                        help="Use non-streaming mode (no TTFT/TPOT)")

    args = parser.parse_args()
    run_benchmark(args)

    # Outlier detection on results (re-read from CSV)
    # Integrated into the report above, but can also be run standalone
