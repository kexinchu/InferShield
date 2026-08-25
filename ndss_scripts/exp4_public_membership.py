#!/usr/bin/env python3
"""SafeKV Exp #4 – Public usage-membership independence.

Experiment deliverable:
  "Public usage-membership results"

For each registered Public prefix we prewarm and pin the object before
sampling the challenge bit.  The two worlds differ only in whether the victim
subsequently uses that prefix.

Variants tested:
  - prewarmed_pinned_public   (control-plane schedule, victim use irrelevant)
  - reactive_materialized     (Public object appears only when victim uses it)
  - unregistered_private      (no Public entry at all — Private namespace only)

Metrics:
  - hit_rate  in victim-use world vs no-use world (should be equal for prewarmed_pinned)
  - membership_accuracy across variants
  - adv_mi = 2*|accuracy - 0.5|

Usage:
  python exp4_public_membership.py \
      --server http://127.0.0.1:8092 \
      --model phi4 --model-path /path/to/phi4 \
      --n-challenges 40 \
      --output results/exp4/phi4.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from transformers import AutoTokenizer

from sglang.srt.mem_cache.safekv_policy import PublicRegistry

CSV_FIELDS = (
    "model", "variant", "trial_id",
    "challenge_bit", "predicted_bit", "correct",
    "ttft_use_ms", "ttft_nouse_ms", "threshold_ms",
    "victim_used", "hit_in_use_world", "hit_in_nouse_world",
)


class Client:
    def __init__(self, server: str, timeout: float = 120.0):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        requests.post(f"{self.server}/flush_cache", timeout=30).raise_for_status()
        time.sleep(0.3)

    def generate(
        self,
        token_ids: List[int],
        user_id: str,
        auth: Optional[Dict] = None,
    ) -> Tuple[float, int]:
        params: Dict = {"max_new_tokens": 1, "temperature": 0.0, "user_id": user_id}
        if auth:
            params["safekv_public_authorization"] = auth
        t0 = time.perf_counter()
        r = requests.post(
            f"{self.server}/generate",
            json={"input_ids": token_ids, "sampling_params": params, "stream": False},
            timeout=self.timeout,
        )
        ttft_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return ttft_ms, len(r.json().get("token_ids", []))


def issue_auth(
    token_ids: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    object_id: str,
) -> Dict:
    key = operator_key.encode("utf-8")
    reg = PublicRegistry(key, policy_epoch=1)
    auth = reg.issue(
        public_object_id=object_id,
        issuer="exp4-ctrl",
        model_id=model_id,
        tokenizer_version=tokenizer_version,
        token_ids=token_ids[:-1],
        expires_at=time.time() + 86400,
    )
    return auth.to_dict()


def run_challenge(
    client: Client,
    token_ids: List[int],
    trial_id: int,
    variant: str,
    challenge_bit: int,
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    threshold_ms: float,
) -> Dict:
    """Run one usage-membership challenge for a specific variant."""
    auth = None
    object_id = f"exp4-pub-{trial_id}"
    victim = f"victim-{trial_id}"
    attacker = f"attacker-{trial_id}"

    client.flush()

    if variant == "prewarmed_pinned_public":
        # Prewarm and pin BEFORE sampling challenge bit.
        auth = issue_auth(token_ids, model_id, tokenizer_version, operator_key, object_id)
        client.generate(token_ids, "operator-prewarm", auth)  # prewarm

        if challenge_bit == 1:
            client.generate(token_ids, victim)  # victim uses

        # Measure attacker hit rate from Public (should be same regardless of challenge_bit).
        ttft_use, _ = client.generate(token_ids, attacker)
        client.flush()
        # Measure no-use world.
        auth2 = issue_auth(token_ids, model_id, tokenizer_version, operator_key, f"exp4-pub-{trial_id}-b")
        client.generate(token_ids, "operator-prewarm", auth2)
        ttft_nouse, _ = client.generate(token_ids, attacker)

    elif variant == "reactive_materialized":
        # No prewarm. Public object appears only when victim uses it.
        auth = issue_auth(token_ids, model_id, tokenizer_version, operator_key, object_id)
        if challenge_bit == 1:
            client.generate(token_ids, victim, auth)  # reactive materialization

        ttft_use, _ = client.generate(token_ids, attacker)
        client.flush()
        ttft_nouse, _ = client.generate(token_ids, attacker)

    elif variant == "unregistered_private":
        # No Public entry. Attacker can only hit via Private namespace (strict isolation).
        if challenge_bit == 1:
            client.generate(token_ids, victim)

        ttft_use, _ = client.generate(token_ids, attacker)
        client.flush()
        ttft_nouse, _ = client.generate(token_ids, attacker)

    else:
        raise ValueError(f"unknown variant: {variant}")

    hit_use = int(ttft_use < threshold_ms)
    hit_nouse = int(ttft_nouse < threshold_ms)
    # Attacker predicts based on the use-world measurement.
    predicted = 1 if ttft_use < threshold_ms else 0

    return {
        "model": model_id,
        "variant": variant,
        "trial_id": trial_id,
        "challenge_bit": challenge_bit,
        "predicted_bit": predicted,
        "correct": int(predicted == challenge_bit),
        "ttft_use_ms": f"{ttft_use:.2f}",
        "ttft_nouse_ms": f"{ttft_nouse:.2f}",
        "threshold_ms": f"{threshold_ms:.2f}",
        "victim_used": challenge_bit,
        "hit_in_use_world": hit_use,
        "hit_in_nouse_world": hit_nouse,
    }


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


def load_prefixes(dataset: Path, tokenizer, count: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    texts = []
    with dataset.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            text = item.get("source_text") or item.get("text", "")
            if len(text.strip()) >= 80:
                texts.append(text.strip())
    rng.shuffle(texts)
    prefixes = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)[:128]
        if len(ids) >= 32:
            prefixes.append(ids)
        if len(prefixes) >= count:
            break
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-challenges", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parents[1] / "datasets" / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operator-key",
        default=os.environ.get("SAFEKV_OPERATOR_KEY", "safekv-exp4-operator-key"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    client = Client(args.server)
    server_info = requests.get(f"{args.server.rstrip('/')}/get_model_info", timeout=30).json()
    tokenizer_version = server_info["tokenizer_path"]

    prefixes = load_prefixes(args.dataset, tokenizer, args.n_challenges * 3 + 10, args.seed)

    # Estimate threshold from the first prefix.
    print("[P4] Estimating TTFT threshold…")
    calib = prefixes[0]
    client.flush()
    miss_times = [client.generate(calib, "calib-attacker")[0] for _ in range(5)]
    client.flush()
    client.generate(calib, "calib-victim")
    hit_times = [client.generate(calib, "calib-attacker")[0] for _ in range(5)]
    threshold_ms = (statistics.mean(miss_times) + statistics.mean(hit_times)) / 2
    print(f"[P4] Threshold: {threshold_ms:.1f}ms (miss={statistics.mean(miss_times):.0f}ms hit={statistics.mean(hit_times):.0f}ms)")

    rng = random.Random(args.seed)
    VARIANTS = ("prewarmed_pinned_public", "reactive_materialized", "unregistered_private")

    total = len(VARIANTS) * args.n_challenges
    done = 0

    for i, variant in enumerate(VARIANTS):
        print(f"\n[P4] Variant: {variant}")
        for j in range(args.n_challenges):
            prefix = prefixes[(i * args.n_challenges + j) % len(prefixes)]
            challenge_bit = rng.randint(0, 1)
            row = run_challenge(
                client, prefix, i * args.n_challenges + j,
                variant, challenge_bit,
                args.model, tokenizer_version, args.operator_key,
                threshold_ms,
            )
            append_row(args.output, row)
            done += 1
            print(
                f"  [{done}/{total}] bit={challenge_bit} pred={row['predicted_bit']} "
                f"correct={row['correct']} ttft_use={row['ttft_use_ms']}ms",
                flush=True,
            )

    # Summarize.
    print("\n=== Summary by variant ===")
    rows = []
    with args.output.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for variant in VARIANTS:
        vrows = [r for r in rows if r["variant"] == variant]
        correct = sum(int(r["correct"]) for r in vrows)
        n = len(vrows)
        acc = correct / n if n else 0
        adv = 2 * abs(acc - 0.5)
        print(f"  {variant:<35} acc={acc:.3f} adv_mi={adv:.3f} n={n}")

    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
