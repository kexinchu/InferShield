#!/usr/bin/env python3
"""SafeKV Exp #10 – membership robustness across serving conditions.

The protocol matches corrected P3: balanced challenge bits, equal-length
victim control requests, independent calibration prefixes, and an attacker-only
query budget.  Trial-level output permits threshold-free AUC recomputation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


CONDITIONS = (
    "local",
    "jitter_10ms",
    "jitter_30ms",
    "mixed_lengths",
    "continuous_batching",
    "background_load",
)

CSV_FIELDS = (
    "model",
    "policy",
    "condition",
    "trial_id",
    "challenge_bit",
    "predicted_bit",
    "ttft_ms",
    "threshold_ms",
    "target_length",
    "attacker_queries_used",
    "victim_setup_used",
    "background_requests",
)


class Client:
    def __init__(self, server: str, timeout: float = 180.0):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        response = requests.post(
            f"{self.server}/flush_cache", timeout=30
        )
        if response.status_code not in (200, 400):
            response.raise_for_status()
        # Some live-server builds reject explicit flushing while KV nodes are
        # referenced.  Challenge and calibration prefixes are disjoint across
        # every trial and condition, so residual unrelated entries cannot turn
        # a target miss into a hit.
        time.sleep(0.3)

    def generate(self, token_ids: List[int], user_id: str) -> float:
        params = {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "user_id": user_id,
        }
        t0 = time.perf_counter()
        response = requests.post(
            f"{self.server}/generate",
            json={
                "input_ids": token_ids,
                "sampling_params": params,
                "stream": False,
            },
            timeout=self.timeout,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        response.raise_for_status()
        return elapsed


def load_prefixes(
    dataset: Path, model_path: str, count: int, seed: int
) -> List[List[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    texts = []
    with dataset.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            text = item.get("source_text") or item.get("text", "")
            if len(text.strip()) >= 80:
                texts.append(text.strip())
    random.Random(seed).shuffle(texts)
    prefixes = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)[:128]
        if len(ids) >= 48:
            prefixes.append(ids)
        if len(prefixes) == count:
            break
    if len(prefixes) != count:
        raise RuntimeError(f"needed {count} prefixes, found {len(prefixes)}")
    return prefixes


def calibrate(
    client: Client,
    prefixes: List[List[int]],
    n_miss: int = 8,
    n_hit: int = 8,
) -> float:
    miss = []
    hit = []
    for index, prefix in enumerate(prefixes[:n_miss]):
        client.flush()
        miss.append(client.generate(prefix, f"calib-miss-{index}"))
    for index, prefix in enumerate(
        prefixes[n_miss : n_miss + n_hit]
    ):
        client.flush()
        client.generate(prefix, f"calib-victim-{index}")
        hit.append(client.generate(prefix, f"calib-attacker-{index}"))
    threshold = (statistics.mean(miss) + statistics.mean(hit)) / 2
    print(
        f"CALIB miss={statistics.mean(miss):.2f} "
        f"hit={statistics.mean(hit):.2f} threshold={threshold:.2f}"
    )
    return threshold


def start_background(
    client: Client,
    prefixes: List[List[int]],
    workers: int,
    delay_s: float,
    prompt_length: int,
) -> Tuple[threading.Event, List[threading.Thread], List[int]]:
    stop = threading.Event()
    counts = [0] * workers

    def worker(worker_id: int) -> None:
        index = worker_id
        while not stop.is_set():
            prefix = prefixes[index % len(prefixes)][:prompt_length]
            try:
                client.generate(prefix, f"background-{worker_id}")
                counts[worker_id] += 1
            except requests.RequestException:
                pass
            index += workers
            if delay_s:
                time.sleep(delay_s)

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(workers)
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    return stop, threads, counts


def stop_background(
    stop: threading.Event,
    threads: List[threading.Thread],
    counts: List[int],
) -> int:
    stop.set()
    for thread in threads:
        thread.join(timeout=10)
    return sum(counts)


def roc_auc(labels: List[int], values: List[float]) -> float:
    positive = [v for y, v in zip(labels, values) if y == 1]
    negative = [v for y, v in zip(labels, values) if y == 0]
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos < neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def adv_mi(labels: List[int], predictions: List[int]) -> float:
    positive = [p for y, p in zip(labels, predictions) if y == 1]
    negative = [p for y, p in zip(labels, predictions) if y == 0]
    tpr = sum(positive) / len(positive)
    fpr = sum(negative) / len(negative)
    return abs(tpr - fpr)


def bootstrap_adv_ci(
    labels: List[int],
    predictions: List[int],
    seed: int,
    n_boot: int = 2000,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    indices_by_class = {
        bit: [i for i, label in enumerate(labels) if label == bit]
        for bit in (0, 1)
    }
    samples = []
    for _ in range(n_boot):
        indices = []
        for bit in (0, 1):
            pool = indices_by_class[bit]
            indices.extend(rng.choice(pool) for _ in pool)
        samples.append(
            adv_mi(
                [labels[i] for i in indices],
                [predictions[i] for i in indices],
            )
        )
    samples.sort()
    return samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]


def run_condition(
    client: Client,
    condition: str,
    challenge_prefixes: List[List[int]],
    background_prefixes: List[List[int]],
    bits: List[int],
    threshold_ms: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rng = random.Random(seed)
    rows = []
    attacker_queries = len(bits)

    for trial_id, bit in enumerate(bits):
        target_source = challenge_prefixes[2 * trial_id]
        control_source = challenge_prefixes[2 * trial_id + 1]
        max_length = min(len(target_source), len(control_source), 96)
        if condition == "mixed_lengths":
            length = rng.randint(32, max_length)
        else:
            length = max_length
        target = target_source[:length]
        control = control_source[:length]

        client.flush()
        background = None
        if condition == "continuous_batching":
            background = start_background(
                client, background_prefixes, 2, 0.10, 32
            )
        elif condition == "background_load":
            background = start_background(
                client, background_prefixes, 4, 0.02, 96
            )

        victim_prefix = target if bit == 1 else control
        client.generate(victim_prefix, f"victim-{condition}-{trial_id}")
        ttft = client.generate(
            target, f"attacker-{trial_id % 2}"
        )
        attacker_queries -= 1

        background_requests = 0
        if background is not None:
            background_requests = stop_background(*background)

        if condition == "jitter_10ms":
            ttft = max(0.0, ttft + rng.uniform(-10.0, 10.0))
        elif condition == "jitter_30ms":
            ttft = max(0.0, ttft + rng.uniform(-30.0, 30.0))

        predicted = int(ttft < threshold_ms)
        rows.append(
            {
                "condition": condition,
                "trial_id": trial_id,
                "challenge_bit": bit,
                "predicted_bit": predicted,
                "ttft_ms": f"{ttft:.4f}",
                "threshold_ms": f"{threshold_ms:.4f}",
                "target_length": length,
                "attacker_queries_used": 1,
                "victim_setup_used": 1,
                "background_requests": background_requests,
            }
        )

    if attacker_queries != 0:
        raise AssertionError("attacker query budget accounting mismatch")

    labels = [int(row["challenge_bit"]) for row in rows]
    predictions = [int(row["predicted_bit"]) for row in rows]
    timings = [float(row["ttft_ms"]) for row in rows]
    positives = [p for y, p in zip(labels, predictions) if y == 1]
    negatives = [p for y, p in zip(labels, predictions) if y == 0]
    tpr = sum(positives) / len(positives)
    fpr = sum(negatives) / len(negatives)
    ci_lo, ci_hi = bootstrap_adv_ci(
        labels, predictions, seed=seed
    )
    hit_times = [v for y, v in zip(labels, timings) if y == 1]
    miss_times = [v for y, v in zip(labels, timings) if y == 0]
    summary = {
        "condition": condition,
        "n_challenges": len(rows),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "attacker_queries": len(rows),
        "background_requests": sum(
            int(row["background_requests"]) for row in rows
        ),
        "tpr": tpr,
        "fpr": fpr,
        "adv_mi": abs(tpr - fpr),
        "adv_ci_lo": ci_lo,
        "adv_ci_hi": ci_hi,
        "auc": roc_auc(labels, timings),
        "delta_mean_ms": statistics.mean(miss_times)
        - statistics.mean(hit_times),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy", choices=("strict", "balanced"), required=True)
    parser.add_argument("--n-challenges", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parents[1]
        / "datasets"
        / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n_challenges % 2:
        raise ValueError("--n-challenges must be even")

    calibration_count = 16
    background_count = 32
    challenge_count = 2 * args.n_challenges * len(CONDITIONS)
    required = calibration_count + challenge_count + background_count
    prefixes = load_prefixes(
        args.dataset, args.model_path, required, args.seed
    )
    calibration = prefixes[:calibration_count]
    challenges = prefixes[
        calibration_count : calibration_count + challenge_count
    ]
    background = prefixes[-background_count:]

    client = Client(args.server)
    threshold = calibrate(client, calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    all_summaries: Dict[str, object] = {}
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for condition_index, condition in enumerate(CONDITIONS):
            bits = [0] * (args.n_challenges // 2) + [1] * (
                args.n_challenges // 2
            )
            random.Random(args.seed + condition_index).shuffle(bits)
            condition_start = 2 * args.n_challenges * condition_index
            condition_prefixes = challenges[
                condition_start : condition_start + 2 * args.n_challenges
            ]
            rows, summary = run_condition(
                client,
                condition,
                condition_prefixes,
                background,
                bits,
                threshold,
                seed=args.seed + condition_index,
            )
            for row in rows:
                writer.writerow(
                    {
                        "model": args.model,
                        "policy": args.policy,
                        **row,
                    }
                )
            fh.flush()
            all_summaries[condition] = summary
            print(
                f"P10 {args.model}/{args.policy}/{condition}: "
                f"AUC={summary['auc']:.3f} "
                f"Adv={summary['adv_mi']:.3f} "
                f"CI=[{summary['adv_ci_lo']:.3f},"
                f"{summary['adv_ci_hi']:.3f}]",
                flush=True,
            )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "policy": args.policy,
                "Q": args.n_challenges,
                "conditions": all_summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"P10_DONE {args.model} {args.policy}")


if __name__ == "__main__":
    main()
