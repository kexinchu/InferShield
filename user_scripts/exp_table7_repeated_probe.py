#!/usr/bin/env python3
"""Table-7 repeated-probe membership / reuse cell.

One victim prefix (PII text). No flush between attacker probes. B is
exercised only when the prefix is Budgeted-Shared and the same fingerprint
is observed many times.

Y-axis for the figure is NOT single-probe AUC. Primary metrics:
  * cumulative TTFT hits (side-channel success) vs probe index
  * cumulative cached_tokens hits (actual KV reuse) vs probe index
  * hits in the first k probes vs B

AUC is optional and only defined on the hit-count score across n games.

    # Detector holds the PII prefix (attack should fail at every B):
    python user_scripts/exp_table7_repeated_probe.py --server http://127.0.0.1:8092 \\
        --B 10 --admission detector --output out.json

    # Forced FN / residual B path (needs --safekv-experiment-autoshare on the server):
    python user_scripts/exp_table7_repeated_probe.py --server http://127.0.0.1:8092 \\
        --B 10 --admission autoshare --output out.json

k must exceed the largest B you care about (default 160 > 150).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import requests

from ablation_prompts import DUMMY_PROMPTS_BASE, PII_PROMPTS

HIT_TOKENS = 16
TEXT_CAP = 2400


def _text(src: str) -> str:
    return src[:TEXT_CAP]


def generate(server: str, text: str, user_id: str, timeout: float) -> Dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server}/generate",
        json={
            "text": text,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "user_id": user_id,
            },
            "stream": False,
        },
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.text[:400]}")
    body = resp.json()
    if isinstance(body, list) and body:
        body = body[0]
    meta = body.get("meta_info") or {}
    return {
        "ttft_ms": (time.perf_counter() - t0) * 1000.0,
        "cached_tokens": int(meta.get("cached_tokens") or 0),
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
    }


def flush(server: str) -> None:
    r = requests.post(f"{server}/flush_cache", timeout=30)
    if r.status_code != 400:
        r.raise_for_status()
    time.sleep(0.4)


def calibrate(server: str, pii: str, dummy: str, settle_s: float, timeout: float) -> float:
    miss: List[float] = []
    flush(server)
    for i in range(8):
        miss.append(generate(server, dummy, f"calib-miss-{i}", timeout)["ttft_ms"])
    hit: List[float] = []
    for i in range(8):
        flush(server)
        generate(server, pii, "calib-victim", timeout)
        generate(server, pii, "calib-victim", timeout)
        time.sleep(settle_s)
        hit.append(generate(server, pii, "calib-attacker", timeout)["ttft_ms"])
    miss_mean = statistics.mean(miss)
    hit_mean = statistics.mean(hit)
    threshold = (miss_mean + hit_mean) / 2.0
    print(
        f"[calib] miss_mean={miss_mean:.1f}ms  hit_mean={hit_mean:.1f}ms  "
        f"threshold={threshold:.1f}ms",
        flush=True,
    )
    return threshold


def one_game(
    server: str,
    *,
    member: bool,
    pii: str,
    dummy: str,
    k: int,
    settle_s: float,
    timeout: float,
    threshold_ms: float,
    game_id: int,
) -> Dict:
    flush(server)
    victim = f"rp-victim-{game_id}"
    generate(server, pii if member else dummy, victim, timeout)
    generate(server, pii if member else dummy, victim, timeout)
    time.sleep(settle_s)

    probes = []
    ttft_hits = 0
    cache_hits = 0
    for t in range(k):
        # Fresh principal each probe. Reusing one attacker_id would hit
        # that attacker's own Private cache and look like a B-independent hit.
        attacker = f"rp-attacker-{game_id}-{t}"
        row = generate(server, pii, attacker, timeout)
        cache_hit = int(row["cached_tokens"] >= HIT_TOKENS)
        ttft_hit = int(row["ttft_ms"] < threshold_ms)
        cache_hits += cache_hit
        ttft_hits += ttft_hit
        probes.append(
            {
                "t": t,
                "ttft_ms": row["ttft_ms"],
                "cached_tokens": row["cached_tokens"],
                "cache_hit": cache_hit,
                "ttft_hit": ttft_hit,
                "cum_cache_hits": cache_hits,
                "cum_ttft_hits": ttft_hits,
            }
        )
    return {
        "member": member,
        "ttft_hits": ttft_hits,
        "cache_hits": cache_hits,
        "probes": probes,
    }


def summarize(games: List[Dict], k: int) -> Dict:
    members = [g for g in games if g["member"]]
    controls = [g for g in games if not g["member"]]

    def mean_cum(group: List[Dict], key: str) -> List[float]:
        if not group:
            return [0.0] * k
        out = []
        for t in range(k):
            out.append(statistics.mean(g["probes"][t][key] for g in group))
        return out

    def mean_total(group: List[Dict], key: str) -> float:
        if not group:
            return 0.0
        return statistics.mean(g[key] for g in group)

    scores = [g["ttft_hits"] for g in games]
    labels = [1 if g["member"] else 0 for g in games]
    auc = None
    if len(set(labels)) == 2:
        auc = mann_whitney_auc(labels, scores)

    return {
        "n_member": len(members),
        "n_control": len(controls),
        "mean_ttft_hits_member": mean_total(members, "ttft_hits"),
        "mean_ttft_hits_control": mean_total(controls, "ttft_hits"),
        "mean_cache_hits_member": mean_total(members, "cache_hits"),
        "mean_cache_hits_control": mean_total(controls, "cache_hits"),
        "cum_ttft_hits_member": mean_cum(members, "cum_ttft_hits"),
        "cum_ttft_hits_control": mean_cum(controls, "cum_ttft_hits"),
        "cum_cache_hits_member": mean_cum(members, "cum_cache_hits"),
        "cum_cache_hits_control": mean_cum(controls, "cum_cache_hits"),
        "hitcount_auc": auc,
    }


def mann_whitney_auc(y: List[int], scores: List[float]) -> float:
    pos = [s for s, lab in zip(scores, y) if lab == 1]
    neg = [s for s, lab in zip(scores, y) if lab == 0]
    if not pos or not neg:
        return 0.5
    gt = sum(1.0 for p in pos for n in neg if p > n)
    eq = sum(1.0 for p in pos for n in neg if p == n)
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8092")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--k", type=int, default=160, help="probes on the same prefix")
    parser.add_argument("--n-games", type=int, default=12, help="even; half member / half control")
    parser.add_argument("--admission", choices=("detector", "autoshare"), default="detector")
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--victim-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.n_games % 2:
        raise SystemExit("--n-games must be even")
    pii = _text(PII_PROMPTS[args.victim_index])
    dummy = _text(DUMMY_PROMPTS_BASE[args.victim_index])
    print(
        f"[T7] B={args.B} k={args.k} n={args.n_games} admission={args.admission} "
        f"pii_chars={len(pii)}"
    )
    threshold = calibrate(args.server, pii, dummy, args.settle_s, args.timeout)

    bits = [1] * (args.n_games // 2) + [0] * (args.n_games // 2)
    games = []
    for i, bit in enumerate(bits):
        game = one_game(
            args.server,
            member=bool(bit),
            pii=pii,
            dummy=dummy,
            k=args.k,
            settle_s=args.settle_s,
            timeout=args.timeout,
            threshold_ms=threshold,
            game_id=i,
        )
        games.append(game)
        print(
            f"  [{i+1}/{args.n_games}] member={bit}  "
            f"ttft_hits={game['ttft_hits']}  cache_hits={game['cache_hits']}"
        )

    summary = summarize(games, args.k)
    payload = {
        "model": args.model,
        "B": args.B,
        "k": args.k,
        "n_games": args.n_games,
        "admission": args.admission,
        "threshold_ms": threshold,
        "hit_tokens": HIT_TOKENS,
        "y_axis": "cumulative_hits_not_single_probe_auc",
        "summary": summary,
        "games": [
            {
                "member": g["member"],
                "ttft_hits": g["ttft_hits"],
                "cache_hits": g["cache_hits"],
                "probes": g["probes"],
            }
            for g in games
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[T7] member ttft/cache hits={summary['mean_ttft_hits_member']:.2f}/"
        f"{summary['mean_cache_hits_member']:.2f}  "
        f"control={summary['mean_ttft_hits_control']:.2f}/"
        f"{summary['mean_cache_hits_control']:.2f}  "
        f"hitcount_auc={summary['hitcount_auc']}  -> {args.output}"
    )


if __name__ == "__main__":
    main()
