#!/usr/bin/env python3
"""SafeKV Exp #3 – End-to-end membership inference and token recovery.

Protocol (aligned with 03_threat_model.tex)
--------------------------------------------
* **Membership inference**: for each trial the oracle selects a challenge
  bit b.  The adversary observes the TTFT transcript and outputs b_hat.

  b=1: victim (separate principal) submits the TARGET prefix.
  b=0: victim submits an equal-length CONTROL prefix (different tokens).
  In both worlds the victim request count and prompt-length distribution
  are matched, so any TTFT difference attributable to queuing / batching
  is equalized.  Only the attacker's probes count toward budget Q.

* **Challenge balance**: exactly n//2 trials have b=1 and n//2 have b=0,
  shuffled by RNG.  This ensures class-balanced metrics and avoids bias
  in the |TPR−FPR| estimator.

* **Threshold calibration**: calibration uses prefixes that are DISTINCT
  from any challenge or recovery prefix; calibration requests are NOT
  counted toward Q.

* **Token recovery**: the legacy uniform mode retains the original weak
  candidate protocol.  Stronger modes either guarantee the true token is among
  shuffled distractors or construct a model-plausible secret from the model's
  top-k next-token distribution before victim insertion.  The latter models an
  attacker with an offline language-model prior; setup-prior queries are
  reported separately and do not consume the remote attack budget Q.

Metrics
-------
  accuracy    – fraction of correctly classified challenge bits
  TPR         – Pr[b_hat=1 | b=1]
  FPR         – Pr[b_hat=1 | b=0]
  adv_mi      – |TPR − FPR|  (paper definition in §2 / §3)
  roc_auc     – area under TPR-FPR curve (computed without sklearn via
                pairwise Wilcoxon / Mann-Whitney estimator)
  rec_frac    – fraction of tokens exactly recovered (recovery experiment)
  95% Wilson CI reported on accuracy, TPR, FPR; stratified bootstrap CIs on
  adv_mi and ROC-AUC.

Policies
--------
  strict | balanced | legacy | vanilla | sglan | cache_partition
  (each requires a separate server run in the appropriate mode)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ── Server client ─────────────────────────────────────────────────────────────

class AttackClient:
    def __init__(self, server: str, timeout: float = 120.0):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        r = requests.post(f"{self.server}/flush_cache", timeout=30)
        # SafeKV returns 400 when protected cache entries make a flush a no-op.
        # Prefixes are disjoint across trials, so this response is safe to
        # tolerate and is recorded by the server.
        if r.status_code != 400:
            r.raise_for_status()
        time.sleep(0.4)          # brief settle after flush

    def generate(self, token_ids: List[int], user_id: str) -> Tuple[float, List[int]]:
        """Return (ttft_ms, output_token_ids)."""
        params = {"max_new_tokens": 1, "temperature": 0.0, "user_id": user_id}
        t0 = time.perf_counter()
        r = requests.post(
            f"{self.server}/generate",
            json={"input_ids": token_ids, "sampling_params": params, "stream": False},
            timeout=self.timeout,
        )
        ttft_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        return ttft_ms, body.get("token_ids", body.get("output_ids", []))

    def top_candidates(
        self, token_ids: List[int], user_id: str, k: int
    ) -> List[int]:
        """Return top-k next-token IDs for experiment setup."""
        params = {"max_new_tokens": 1, "temperature": 0.0, "user_id": user_id}
        r = requests.post(
            f"{self.server}/generate",
            json={
                "input_ids": token_ids,
                "sampling_params": params,
                "stream": False,
                "return_logprob": True,
                "top_logprobs_num": k,
                "logprob_start_len": -1,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        meta = body.get("meta_info", {}) or {}
        top = meta.get("output_top_logprobs", [])
        ids: List[int] = []

        def _absorb(token_id: object) -> None:
            if isinstance(token_id, int) and token_id not in ids:
                ids.append(token_id)

        if top and isinstance(top, list):
            first = top[0]
            # Common SGLang shape: List[List[logprob, token_id, text]]
            if isinstance(first, list):
                for item in first:
                    if isinstance(item, (list, tuple)):
                        if len(item) >= 2 and isinstance(item[1], int):
                            _absorb(item[1])
                        elif len(item) >= 3 and isinstance(item[2], int):
                            _absorb(item[2])
                    elif isinstance(item, dict):
                        _absorb(item.get("token_id") or item.get("id"))
            elif isinstance(first, dict):
                for item in top:
                    if isinstance(item, dict):
                        _absorb(item.get("token_id") or item.get("id"))

        # Fallback: ranked output token logprobs, then decoded token_ids.
        if not ids:
            for item in meta.get("output_token_logprobs", []) or []:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 2 and isinstance(item[1], int):
                        _absorb(item[1])
                    elif len(item) >= 3 and isinstance(item[2], int):
                        _absorb(item[2])
        if not ids:
            for token_id in body.get("token_ids") or body.get("output_ids") or []:
                _absorb(token_id)

        if not ids:
            raise RuntimeError(
                "output_top_logprobs contained no token IDs; "
                f"meta_keys={sorted(meta.keys())} top_type={type(top).__name__}"
            )
        return ids[:k]


# ── Prefix generation ─────────────────────────────────────────────────────────

def load_prefixes_tokenized(
    dataset: Path, model_path: str, count: int, seed: int
) -> List[List[int]]:
    """Load & tokenize PII snippets. Returns list of token-id lists (len 32-128)."""
    import json as _json
    from transformers import AutoTokenizer
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    texts: List[str] = []
    with dataset.open(encoding="utf-8") as fh:
        for line in fh:
            item = _json.loads(line)
            text = item.get("source_text") or item.get("text", "")
            if len(text.strip()) >= 80:
                texts.append(text.strip())
    rng.shuffle(texts)
    prefixes: List[List[int]] = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=True)[:128]
        if len(ids) >= 32:
            prefixes.append(ids)
        if len(prefixes) >= count:
            break
    return prefixes


def make_synthetic_prefixes(count: int, length: int, seed: int) -> List[List[int]]:
    """Deterministic synthetic prefixes as fallback if dataset unavailable."""
    rng = random.Random(seed)
    return [
        [rng.randint(100, 50000) for _ in range(length)]
        for _ in range(count)
    ]


# ── Calibration (uses SEPARATE prefixes, not counted toward Q) ────────────────

def estimate_threshold(
    client: AttackClient,
    calib_prefixes: List[List[int]],
    victim_id: str,
    attacker_id: str,
    n_miss: int = 8,
    n_hit: int = 8,
    post_victim_settle_ms: float = 0.0,
) -> float:
    """Estimate hit/miss TTFT midpoint using dedicated calibration prefixes.

    Uses the FIRST n_miss prefixes for miss-world and the NEXT n_hit for hit-world
    to avoid any ordering confound.  Neither set overlaps with challenge prefixes.
    """
    # Miss world: attacker probes cold cache.
    client.flush()
    miss_times: List[float] = []
    for pfx in calib_prefixes[:n_miss]:
        ttft, _ = client.generate(pfx, attacker_id)
        miss_times.append(ttft)

    # Hit world: victim inserts, then attacker probes the same prefix.
    # (This calibrates what a BALANCED-mode cross-tenant hit looks like.)
    hit_times: List[float] = []
    for pfx in calib_prefixes[n_miss: n_miss + n_hit]:
        client.flush()
        client.generate(pfx, victim_id)          # victim inserts
        if post_victim_settle_ms > 0:
            time.sleep(post_victim_settle_ms / 1000.0)
        ttft, _ = client.generate(pfx, attacker_id)
        hit_times.append(ttft)

    miss_mean = statistics.mean(miss_times) if miss_times else 300.0
    hit_mean  = statistics.mean(hit_times)  if hit_times  else 50.0
    threshold = (miss_mean + hit_mean) / 2
    print(
        f"[calib] miss_mean={miss_mean:.1f}ms  hit_mean={hit_mean:.1f}ms"
        f"  threshold={threshold:.1f}ms"
    )
    return threshold


# ── Membership inference trial ────────────────────────────────────────────────

def membership_trial(
    client: AttackClient,
    target_prefix: List[int],
    control_prefix: List[int],   # equal-length, different tokens
    challenge_bit: int,
    victim_id: str,
    attacker_accounts: List[str],
    threshold_ms: float,
    q_attacker: List[int],       # mutable; counts ONLY attacker probes
    v_setup: List[int],          # mutable; counts victim setup requests (reported, not bounded)
    post_victim_settle_ms: float = 0.0,
) -> Tuple[int, float, int]:
    """Execute one membership challenge.

    Returns (predicted_bit, attacker_ttft_ms, attacker_queries_used).
    """
    client.flush()

    # Victim action: always sends exactly one request of the same length.
    if challenge_bit == 1:
        client.generate(target_prefix, victim_id)   # target in cache
    else:
        client.generate(control_prefix, victim_id)  # control in cache; target absent

    v_setup[0] += 1
    if post_victim_settle_ms > 0:
        # Allow async non-sensitive promotion into Budgeted-Shared before probe.
        time.sleep(post_victim_settle_ms / 1000.0)

    if q_attacker[0] <= 0:
        return -1, 0.0, 0

    # Attacker probes target; distributes across accounts.
    acct = attacker_accounts[q_attacker[0] % len(attacker_accounts)]
    ttft, _ = client.generate(target_prefix, acct)
    q_attacker[0] -= 1
    predicted = 1 if ttft < threshold_ms else 0
    return predicted, ttft, 1


# ── Token recovery trial ──────────────────────────────────────────────────────

def token_recovery_trial(
    client: AttackClient,
    public_prefix: List[int],
    victim_id: str,
    attacker_id: str,
    n_tokens: int,
    q_attacker: List[int],        # only attacker probes counted
    v_setup: List[int],
    threshold_ms: float,
    vocab_sample: int = 20,
    seed: int = 42,
    recovery_mode: str = "uniform",
    post_victim_settle_ms: float = 0.0,
) -> Tuple[int, int, List[bool], int, int]:
    """Adaptive token-by-token recovery.

    uniform: legacy sampled candidates; true token is not guaranteed.
    guaranteed: true token plus k-1 shuffled uniform distractors.
    logit_topk: construct a model-plausible secret from top-k next-token
      candidates before victim insertion, then shuffle those candidates.

    Returns (tokens_recovered, tokens_attempted, per_position_correct,
    attacker_queries_used, setup_prior_queries).
    """
    rng = random.Random(seed)
    setup_prior_queries = 0
    candidate_sets: List[List[int]] = []

    if recovery_mode == "logit_topk":
        secret: List[int] = []
        setup_known = list(public_prefix)
        for pos in range(n_tokens):
            candidates = client.top_candidates(
                setup_known, user_id=f"setup-prior-{seed}-{pos}", k=vocab_sample
            )
            setup_prior_queries += 1
            true_tok = candidates[0]
            secret.append(true_tok)
            # Fill short server top-k responses with unique neutral distractors.
            while len(candidates) < vocab_sample:
                cand = rng.randint(100, 50000)
                if cand not in candidates:
                    candidates.append(cand)
            rng.shuffle(candidates)
            candidate_sets.append(candidates)
            setup_known.append(true_tok)
        # Setup-prior requests must not leave cache state visible to the attack.
        client.flush()
    else:
        secret = [rng.randint(100, 50000) for _ in range(n_tokens)]
        for true_tok in secret:
            if recovery_mode == "guaranteed":
                candidates = [true_tok]
                while len(candidates) < vocab_sample:
                    cand = rng.randint(100, 50000)
                    if cand not in candidates:
                        candidates.append(cand)
                rng.shuffle(candidates)
            elif recovery_mode == "uniform":
                candidates = [
                    rng.randint(100, 50000) for _ in range(vocab_sample)
                ]
            else:
                raise ValueError(f"unknown recovery mode: {recovery_mode}")
            candidate_sets.append(candidates)

    full_victim = public_prefix + secret

    # Victim caches the extended prefix.
    client.flush()
    client.generate(full_victim, victim_id)
    v_setup[0] += 1
    if post_victim_settle_ms > 0:
        time.sleep(post_victim_settle_ms / 1000.0)

    recovered = 0
    attacker_queries_used = 0
    correct_per_pos: List[bool] = []
    known = list(public_prefix)

    for pos in range(n_tokens):
        true_tok = secret[pos]
        candidates = candidate_sets[pos]
        committed: Optional[int] = None
        for cand in candidates:
            if q_attacker[0] <= 0:
                break
            probe = known + [cand]
            ttft, _ = client.generate(probe, attacker_id)
            q_attacker[0] -= 1
            attacker_queries_used += 1
            if ttft < threshold_ms:
                committed = cand
                break
        hit_correct = (committed == true_tok)
        correct_per_pos.append(hit_correct)
        if committed is not None and hit_correct:
            recovered += 1
            known = known + [true_tok]
        else:
            # Adversary failed at this position; adaptive chain breaks.
            break

    return (
        recovered,
        n_tokens,
        correct_per_pos,
        attacker_queries_used,
        setup_prior_queries,
    )


# ── Statistics helpers ────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def roc_auc_manual(y_true: List[int], y_score: List[float]) -> float:
    """Pairwise Mann-Whitney AUC estimator (no sklearn dependency).

    Lower TTFT means 'more likely hit', so we pass negated scores so that
    higher score = more likely b=1.
    """
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    n_concordant = sum(1 for p in pos for q in neg if p < q)    # hit faster
    n_tie       = sum(1 for p in pos for q in neg if p == q)
    total = len(pos) * len(neg)
    return (n_concordant + 0.5 * n_tie) / total


def bootstrap_ci(
    y_true: List[int], y_pred: List[int], n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    """Bootstrap 95% CI on |TPR − FPR|."""
    rng = random.Random(seed)
    n = len(y_true)
    advs: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        pos = [p for t, p in zip(yt, yp) if t == 1]
        neg = [p for t, p in zip(yt, yp) if t == 0]
        tpr = (sum(pos) / len(pos)) if pos else 0.0
        fpr = (sum(neg) / len(neg)) if neg else 0.0
        advs.append(abs(tpr - fpr))
    advs.sort()
    lo = advs[int(0.025 * n_boot)]
    hi = advs[int(0.975 * n_boot)]
    return lo, hi


def bootstrap_auc_ci(
    y_true: List[int], y_score: List[float], n_boot: int = 2000, seed: int = 1
) -> Tuple[float, float]:
    """Stratified bootstrap 95% CI for ROC-AUC."""
    rng = random.Random(seed)
    pos = [score for label, score in zip(y_true, y_score) if label == 1]
    neg = [score for label, score in zip(y_true, y_score) if label == 0]
    if not pos or not neg:
        return (float("nan"), float("nan"))
    aucs: List[float] = []
    for _ in range(n_boot):
        pos_sample = [pos[rng.randrange(len(pos))] for _ in pos]
        neg_sample = [neg[rng.randrange(len(neg))] for _ in neg]
        aucs.append(
            roc_auc_manual(
                [1] * len(pos_sample) + [0] * len(neg_sample),
                pos_sample + neg_sample,
            )
        )
    aucs.sort()
    return aucs[int(0.025 * n_boot)], aucs[int(0.975 * n_boot)]


# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_FIELDS = (
    "model", "policy", "access_budget_B", "budget_Q", "n_attacker_accounts",
    "trial_id", "experiment_type",
    "challenge_bit", "predicted_bit", "ttft_ms", "threshold_ms",
    "correct", "attacker_queries_used", "victim_setup_used",
    "tokens_attempted", "tokens_recovered", "recovery_mode",
    "setup_prior_queries",
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


def summarize(path: Path) -> Dict:
    """Compute aligned metrics: accuracy, TPR, FPR, AdvMI=|TPR-FPR|, AUC, rec_frac."""
    if not path.exists():
        return {}
    rows: List[Dict] = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    mi_rows = [r for r in rows if r["experiment_type"] == "membership"
               and r["predicted_bit"] not in ("", "-1")]
    rec_rows = [r for r in rows if r["experiment_type"] == "recovery"]

    result: Dict = {}

    if mi_rows:
        y_true = [int(r["challenge_bit"]) for r in mi_rows]
        y_pred = [int(r["predicted_bit"])  for r in mi_rows]
        ttfts  = [float(r["ttft_ms"])      for r in mi_rows]

        n = len(y_true)
        n_pos = sum(y_true)
        n_neg = n - n_pos

        n_correct  = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)

        acc = n_correct / n
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        adv = abs(tpr - fpr)

        acc_ci  = wilson_ci(n_correct, n)
        tpr_ci  = wilson_ci(tp, n_pos) if n_pos else (0.0, 1.0)
        fpr_ci  = wilson_ci(fp, n_neg) if n_neg else (0.0, 1.0)
        adv_ci  = bootstrap_ci(y_true, y_pred)

        auc = roc_auc_manual(y_true, ttfts)
        auc_ci = bootstrap_auc_ci(y_true, ttfts)

        result.update({
            "n_mi_trials": n, "n_pos": n_pos, "n_neg": n_neg,
            "accuracy": round(acc, 4),
            "accuracy_ci_lo": round(acc_ci[0], 4),
            "accuracy_ci_hi": round(acc_ci[1], 4),
            "tpr": round(tpr, 4),
            "tpr_ci_lo": round(tpr_ci[0], 4),
            "tpr_ci_hi": round(tpr_ci[1], 4),
            "fpr": round(fpr, 4),
            "fpr_ci_lo": round(fpr_ci[0], 4),
            "fpr_ci_hi": round(fpr_ci[1], 4),
            "adv_mi": round(adv, 4),
            "adv_mi_ci_lo": round(adv_ci[0], 4),
            "adv_mi_ci_hi": round(adv_ci[1], 4),
            "roc_auc": round(auc, 4) if not math.isnan(auc) else None,
            "roc_auc_ci_lo": (
                round(auc_ci[0], 4) if not math.isnan(auc_ci[0]) else None
            ),
            "roc_auc_ci_hi": (
                round(auc_ci[1], 4) if not math.isnan(auc_ci[1]) else None
            ),
        })

    if rec_rows:
        attempted = sum(int(r["tokens_attempted"]) for r in rec_rows if r["tokens_attempted"])
        recovered = sum(int(r["tokens_recovered"]) for r in rec_rows if r["tokens_recovered"])
        result["token_recovery_fraction"] = round(recovered / attempted, 4) if attempted else 0.0
        result["tokens_attempted"] = attempted
        result["tokens_recovered"] = recovered
        result["n_rec_trials"] = len(rec_rows)
        result["full_secret_recovery_fraction"] = round(
            sum(
                int(r["tokens_recovered"]) == int(r["tokens_attempted"])
                for r in rec_rows
                if r["tokens_attempted"]
            )
            / len(rec_rows),
            4,
        )
        result["attacker_queries_used_recovery"] = sum(
            int(r["attacker_queries_used"]) for r in rec_rows
        )
        result["setup_prior_queries"] = sum(
            int(r.get("setup_prior_queries") or 0) for r in rec_rows
        )

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True, help="Short model name for CSV")
    parser.add_argument("--model-path", required=True, help="HF model path for tokenizer")
    parser.add_argument("--policy", required=True,
                        help="strict|balanced|legacy|vanilla|sglan|cache_partition")
    parser.add_argument("--access-budget-B", type=int, default=-1,
                        help="Effective cross-tenant budget; -1 when not applicable")
    parser.add_argument("--budget-Q", type=int, default=200,
                        help="Attacker probe budget (victim setup NOT counted)")
    parser.add_argument("--n-challenges", type=int, default=100,
                        help="Total membership challenges (must be even for 50/50 balance)")
    parser.add_argument("--n-attacker-accounts", type=int, default=2)
    parser.add_argument("--n-recovery-trials", type=int, default=10)
    parser.add_argument("--n-tokens-to-recover", type=int, default=5)
    parser.add_argument("--vocab-sample", type=int, default=20,
                        help="Candidate tokens per recovery position")
    parser.add_argument(
        "--recovery-mode",
        choices=("uniform", "guaranteed", "logit_topk"),
        default="uniform",
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--membership-only", action="store_true")
    phase.add_argument("--recovery-only", action="store_true")
    parser.add_argument(
        "--post-victim-settle-ms",
        type=float,
        default=0.0,
        help="Wait after victim insert so async Balanced promotion can complete",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).parents[1] / "datasets" / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic token IDs instead of tokenizing from dataset")
    args = parser.parse_args()

    if args.n_challenges % 2 != 0:
        parser.error("--n-challenges must be even (required for 50/50 challenge balance)")

    rng = random.Random(args.seed)
    client = AttackClient(args.server)

    run_membership = not args.recovery_only
    run_recovery = not args.membership_only

    # Allocate disjoint prefixes for each enabled phase.
    N_CALIB_MISS = 8
    N_CALIB_HIT  = 8
    n_total = (
        N_CALIB_MISS
        + N_CALIB_HIT
        + (args.n_challenges * 2 if run_membership else 0)
        + (args.n_recovery_trials if run_recovery else 0)
    )

    print(f"[P3] Loading {n_total} prefixes…")
    if args.synthetic or not args.dataset.exists():
        all_prefixes = make_synthetic_prefixes(n_total, length=64, seed=args.seed)
    else:
        all_prefixes = load_prefixes_tokenized(
            args.dataset, args.model_path, n_total, args.seed
        )
    if len(all_prefixes) < n_total:
        print(f"  WARNING: only {len(all_prefixes)} prefixes available; padding with synthetic")
        extra = make_synthetic_prefixes(n_total - len(all_prefixes), 64, args.seed + 99)
        all_prefixes += extra

    idx = 0
    calib_miss_pfx = all_prefixes[idx: idx + N_CALIB_MISS]; idx += N_CALIB_MISS
    calib_hit_pfx  = all_prefixes[idx: idx + N_CALIB_HIT];  idx += N_CALIB_HIT
    targets = (
        all_prefixes[idx: idx + args.n_challenges] if run_membership else []
    )
    idx += len(targets)
    controls = (
        all_prefixes[idx: idx + args.n_challenges] if run_membership else []
    )
    idx += len(controls)
    rec_prefixes = (
        all_prefixes[idx: idx + args.n_recovery_trials] if run_recovery else []
    )

    # Build exactly-balanced challenge sequence.
    challenge_bits = (
        [1] * (args.n_challenges // 2) + [0] * (args.n_challenges // 2)
        if run_membership
        else []
    )
    rng.shuffle(challenge_bits)

    attacker_accounts = [f"atk-{k}" for k in range(args.n_attacker_accounts)]

    # Calibration (NOT counted toward Q).
    print("[P3] Calibrating threshold…")
    threshold_ms = estimate_threshold(
        client, calib_miss_pfx + calib_hit_pfx,
        victim_id="calib-victim", attacker_id="calib-attacker",
        n_miss=N_CALIB_MISS, n_hit=N_CALIB_HIT,
        post_victim_settle_ms=args.post_victim_settle_ms,
    )

    q_attacker = [args.budget_Q]   # only attacker probes
    v_setup    = [0]               # victim setup (tracked, not bounded)

    # ── Membership inference ──────────────────────────────────────────────────
    if run_membership:
        print(f"\n[P3] Membership: {args.n_challenges} trials (50/50), Q={args.budget_Q}")
    for i, cb in enumerate(challenge_bits):
        if q_attacker[0] <= 0:
            print(f"[P3] Attacker budget exhausted at trial {i}")
            break
        predicted, ttft, used = membership_trial(
            client,
            targets[i], controls[i], cb,
            victim_id=f"mi-victim-{i}",
            attacker_accounts=attacker_accounts,
            threshold_ms=threshold_ms,
            q_attacker=q_attacker,
            v_setup=v_setup,
            post_victim_settle_ms=args.post_victim_settle_ms,
        )
        correct = int(predicted == cb) if predicted != -1 else ""
        append_row(args.output, {
            "model": args.model,
            "policy": args.policy,
            "access_budget_B": args.access_budget_B,
            "budget_Q": args.budget_Q,
            "n_attacker_accounts": args.n_attacker_accounts,
            "trial_id": i,
            "experiment_type": "membership",
            "challenge_bit": cb,
            "predicted_bit": predicted,
            "ttft_ms": f"{ttft:.2f}",
            "threshold_ms": f"{threshold_ms:.2f}",
            "correct": correct,
            "attacker_queries_used": used,
            "victim_setup_used": v_setup[0],
            "tokens_attempted": "",
            "tokens_recovered": "",
            "recovery_mode": "",
            "setup_prior_queries": "",
        })
        print(
            f"  [{i+1:3d}/{args.n_challenges}] b={cb} b_hat={predicted:2d}"
            f"  ttft={ttft:6.0f}ms  Q_left={q_attacker[0]}",
            flush=True,
        )

    # ── Token recovery ────────────────────────────────────────────────────────
    if run_recovery:
        print(
            f"\n[P3] Recovery: {args.n_recovery_trials} trials, "
            f"{args.n_tokens_to_recover} tokens each, mode={args.recovery_mode}"
        )
    for i in range(args.n_recovery_trials if run_recovery else 0):
        if q_attacker[0] <= 0:
            print(f"[P3] Budget insufficient for recovery trial {i}")
            break
        recovered, attempted, per_pos, queries_used, setup_prior_queries = token_recovery_trial(
            client,
            rec_prefixes[i],
            victim_id=f"rec-victim-{i}",
            attacker_id=f"rec-attacker-{i % args.n_attacker_accounts}",
            n_tokens=args.n_tokens_to_recover,
            q_attacker=q_attacker,
            v_setup=v_setup,
            threshold_ms=threshold_ms,
            vocab_sample=args.vocab_sample,
            seed=args.seed + i,
            recovery_mode=args.recovery_mode,
            post_victim_settle_ms=args.post_victim_settle_ms,
        )
        append_row(args.output, {
            "model": args.model,
            "policy": args.policy,
            "access_budget_B": args.access_budget_B,
            "budget_Q": args.budget_Q,
            "n_attacker_accounts": args.n_attacker_accounts,
            "trial_id": args.n_challenges + i,
            "experiment_type": "recovery",
            "challenge_bit": "",
            "predicted_bit": "",
            "ttft_ms": "",
            "threshold_ms": f"{threshold_ms:.2f}",
            "correct": "",
            "attacker_queries_used": queries_used,
            "victim_setup_used": v_setup[0],
            "tokens_attempted": attempted,
            "tokens_recovered": recovered,
            "recovery_mode": args.recovery_mode,
            "setup_prior_queries": setup_prior_queries,
        })
        print(
            f"  [{i+1}/{args.n_recovery_trials}] recovered={recovered}/{attempted}"
            f"  probes={queries_used} prior={setup_prior_queries}"
            f"  Q_left={q_attacker[0]}",
            flush=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    summ = summarize(args.output)
    summ_path = args.output.with_suffix(".summary.json")
    summ_path.write_text(json.dumps(summ, indent=2))
    print(f"\n[P3] Summary written → {summ_path}")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
