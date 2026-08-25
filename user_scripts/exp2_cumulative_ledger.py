#!/usr/bin/env python3
"""SafeKV Exp #2 – Cumulative-ledger enforcement under churn, recovery, and races.

Experiment deliverable:
  "Cumulative-ledger enforcement results"

  Plot: x-axis = eviction/reinsertion or restart count, y-axis = cumulative
  cross-tenant hits.  Strict=0, Balanced≤B (plateau after B total hits, never
  resets on eviction), Legacy per-residency (resets on each eviction → linear
  growth).

This script runs as a CPU-only unit test against the RadixCache directly
(mock memory pool, no GPU needed).  The invariant being proven is
deterministic: it does not depend on timing or statistical sampling.

Sections
--------
1. Eviction/reinsertion sweep  – verify cumulative hits ≤ B after N evictions
2. Concurrent probe race       – N threads race for the last budget slot; overshoot must be 0
3. Restart simulation          – rebuild cache from DurableLedger; budget not reset
4. Legacy (per-residency) comparison – show that the old policy gives B*N hits
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
import tempfile
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# ── Import SafeKV policy modules (CPU-only) ──────────────────────────────────
# Use the same minimal stub pattern as test_safekv_radix_invariants.py.
import sys
import types

_pool_stub = types.ModuleType("sglang.srt.mem_cache.memory_pool")
_pool_stub.ReqToTokenPool = object  # type: ignore[attr-defined]
_pool_stub.TokenToKVPoolAllocator = object  # type: ignore[attr-defined]
sys.modules["sglang.srt.mem_cache.memory_pool"] = _pool_stub

import torch  # noqa: E402

from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: E402
from sglang.srt.mem_cache.safekv_policy import (  # noqa: E402
    DurableLedger,
    PublicRegistry,
    Visibility,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

OPERATOR_KEY = b"safekv-exp2-test-key"
MODEL_ID = "phi-4-14b"
TOKENIZER = "phi-4-14b-tokenizer"
BUDGET_B = 10


def _make_cache(
    mode: str = "balanced",
    ledger_path: Optional[str] = None,
) -> RadixCache:
    from unittest.mock import MagicMock
    mock_alloc = MagicMock()
    mock_alloc.device = torch.device("cpu")
    mock_judge = MagicMock()
    return RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=mock_alloc,
        page_size=1,
        private_judge_client=mock_judge,
        disable=False,
        access_budget_B=BUDGET_B,
        safekv_mode=mode,
        operator_key=OPERATOR_KEY.decode(),
        policy_epoch=1,
        model_id=MODEL_ID,
        tokenizer_version=TOKENIZER,
        ledger_path=ledger_path,
    )


def _insert_budgeted(cache: RadixCache, user_id: str, tokens: List[int]) -> str:
    """Insert tokens for user_id then promote to BUDGETED_SHARED.  Return fingerprint.

    Uses RadixCache._node_fingerprint for ledger key consistency across cycles.
    """
    # Insert as int64 tensors so torch.cat works in match_prefix.
    value = torch.tensor(list(range(len(tokens))), dtype=torch.int64)
    cache.insert(key=tokens, value=value, user_id=user_id)

    root = cache._private_root(user_id, create=False)
    if root is None:
        return ""

    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        if node.visibility == Visibility.PRIVATE.value and not node.permanently_private:
            # Use the same fingerprint formula as _match_prefix_helper.
            fp = cache._node_fingerprint(node.key)
            if not cache.ledger.is_operational:
                node.visibility = Visibility.EXHAUSTED_PRIVATE.value
                node.permanently_private = True
            else:
                charged = cache.ledger.charged_hits(fp)
                node.access_budget = charged
                if charged >= cache.access_budget_B:
                    node.visibility = Visibility.EXHAUSTED_PRIVATE.value
                    node.permanently_private = True
                else:
                    node.visibility = Visibility.BUDGETED_SHARED.value
        stack.extend(node.children.values())
    return cache._node_fingerprint(tokens)


def _simulate_cross_tenant_hits(
    cache: RadixCache,
    owner_id: str,
    requester_id: str,
    tokens: List[int],
    n: int,
) -> int:
    """Attempt n cross-tenant lookups; return actual hits served."""
    hits = 0
    for _ in range(n):
        result, _ = cache.match_prefix(key=tokens, user_id=requester_id)
        if len(result) > 0:
            hits += 1
    return hits


def _evict_all(cache: RadixCache) -> None:
    """Simulated eviction: set all node values to None and remove children."""
    for root in list(cache.private_roots.values()):
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            node.value = None
    cache.private_roots.clear()
    cache.evictable_size_ = 0


# ── Section 1: Eviction/reinsertion sweep ────────────────────────────────────

def run_eviction_sweep(
    n_cycles: int,
    tokens: List[int],
    output_dir: Path,
) -> List[Dict[str, object]]:
    """Sweep N eviction/reinsertion cycles; record cumulative hits per policy."""
    rows = []
    policies = {
        "strict": "strict",
        "balanced": "balanced",
        "legacy": "legacy",  # balanced but resets budget on each reinsertion
    }

    for policy_name, mode in policies.items():
        ledger_file = output_dir / f"exp2_ledger_{policy_name}.json"
        if ledger_file.exists():
            ledger_file.unlink()

        cache = _make_cache(mode=mode if mode != "legacy" else "balanced",
                            ledger_path=str(ledger_file))
        cumulative_hits = 0
        owner = "victim-owner"
        attacker = "attacker-1"

        for cycle in range(n_cycles):
            # Insert / promote.
            fingerprint = _insert_budgeted(cache, owner, tokens)

            # Attempt BUDGET_B cross-tenant hits.
            hits = _simulate_cross_tenant_hits(
                cache, owner, attacker, tokens, BUDGET_B + 2
            )
            cumulative_hits += hits

            # Balanced persists reservations across eviction. Legacy explicitly
            # discards the per-residency counter after eviction.
            cache.flush_budgets_to_ledger()
            _evict_all(cache)
            if policy_name == "legacy":
                cache.ledger.reset()

            if policy_name == "strict":
                expected = 0
            elif policy_name == "legacy":
                expected = (cycle + 1) * BUDGET_B
            else:
                expected = min(BUDGET_B, (cycle + 1) * BUDGET_B)
            within = cumulative_hits <= BUDGET_B
            rows.append({
                "policy": policy_name,
                "cycle": cycle,
                "hits_this_cycle": hits,
                "cumulative_hits": cumulative_hits,
                "budget_B": BUDGET_B,
                "within_budget": int(within),
                "verdict": "pass" if cumulative_hits == expected else "fail",
            })
            print(
                f"  [{policy_name}] cycle={cycle} hits={hits} "
                f"cumulative={cumulative_hits} "
                f"{'PASS' if cumulative_hits == expected else 'FAIL'}",
                flush=True,
            )

    return rows


# ── Section 2: Concurrent probe race ─────────────────────────────────────────

def run_concurrent_race(
    n_workers: List[int],
    tokens: List[int],
    output_dir: Path,
    repetitions: int = 100,
) -> List[Dict[str, object]]:
    """N threads race for the final budget slot; verify overshoot = 0."""
    rows = []
    for n in n_workers:
        max_overshoot = 0
        violating_trials = 0
        race_hits = 0
        for trial in range(repetitions):
            cache = _make_cache(mode="balanced")
            owner = "race-victim"
            _insert_budgeted(cache, owner, tokens)

            # Pre-exhaust budget to B-1 so only one more hit is allowed.
            pre_hits = _simulate_cross_tenant_hits(
                cache, owner, "pre-attacker", tokens, BUDGET_B - 1
            )
            barrier = threading.Barrier(n)

            def worker(idx: int) -> int:
                barrier.wait(timeout=10)
                val, _ = cache.match_prefix(
                    key=tokens, user_id=f"racer-{trial}-{idx}"
                )
                return 1 if len(val) > 0 else 0

            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(worker, range(n)))

            accepted = sum(results)
            overshoot = max(0, pre_hits + accepted - BUDGET_B)
            race_hits += accepted
            max_overshoot = max(max_overshoot, overshoot)
            violating_trials += int(overshoot > 0)
        rows.append({
            "n_concurrent": n,
            "repetitions": repetitions,
            "racing_requests": n * repetitions,
            "race_hits": race_hits,
            "budget_B": BUDGET_B,
            "max_overshoot": max_overshoot,
            "violating_trials": violating_trials,
            "verdict": "pass" if max_overshoot == 0 else "fail",
        })
        print(
            f"  race N={n}: repetitions={repetitions} "
            f"max_overshoot={max_overshoot} "
            f"{'PASS' if max_overshoot == 0 else 'FAIL'}",
            flush=True,
        )

    return rows


# ── Section 3: Restart simulation ────────────────────────────────────────────

def run_restart_simulation(
    tokens: List[int],
    output_dir: Path,
    recovery_repetitions: int = 100,
) -> List[Dict[str, object]]:
    """Verify that a restart respects the durable ledger budget."""
    rows = []
    ledger_file = output_dir / "exp2_restart_ledger.json"
    if ledger_file.exists():
        ledger_file.unlink()

    owner = "restart-victim"
    attacker = "restart-attacker"

    # Phase 1: serve B hits, exhaust budget, flush, simulate restart.
    cache1 = _make_cache(mode="balanced", ledger_path=str(ledger_file))
    _insert_budgeted(cache1, owner, tokens)
    hits1 = _simulate_cross_tenant_hits(cache1, owner, attacker, tokens, BUDGET_B)
    cache1.ledger.flush()
    del cache1

    # Phase 2: new cache instance loading same ledger (simulates process restart).
    # After restart, ledger budget must prevent further cross-tenant hits.
    recovery_samples_us = []
    for _ in range(recovery_repetitions):
        t0 = time.perf_counter()
        recovered = DurableLedger(path=str(ledger_file))
        recovery_samples_us.append((time.perf_counter() - t0) * 1e6)
        if not recovered.is_operational:
            raise RuntimeError("ledger recovery unexpectedly failed")

    cache2 = _make_cache(mode="balanced", ledger_path=str(ledger_file))
    _insert_budgeted(cache2, owner, tokens)  # reinsertion after restart — restores from ledger
    hits2 = _simulate_cross_tenant_hits(cache2, owner, attacker, tokens, BUDGET_B + 5)
    cache2.ledger.flush()

    total = hits1 + hits2
    within = total <= BUDGET_B
    rows.append({
        "phase1_hits": hits1,
        "phase2_hits": hits2,
        "total_hits": total,
        "budget_B": BUDGET_B,
        "within_budget": int(within),
        "recovery_repetitions": recovery_repetitions,
        "recovery_p50_us": statistics.median(recovery_samples_us),
        "recovery_p95_us": sorted(recovery_samples_us)[
            int(0.95 * (len(recovery_samples_us) - 1))
        ],
        "recovery_p99_us": sorted(recovery_samples_us)[
            int(0.99 * (len(recovery_samples_us) - 1))
        ],
        "verdict": "pass" if within else "fail",
    })
    print(
        f"  restart: phase1={hits1} phase2={hits2} total={total} "
        f"budget={BUDGET_B} {'PASS' if within else 'FAIL'}",
        flush=True,
    )
    return rows


# ── Section 4: Replicas and fail-closed recovery ─────────────────────────────

def run_replica_races(
    tokens: List[int],
    replica_counts: List[int],
    n_threads: int = 128,
    repetitions: int = 20,
) -> List[Dict[str, object]]:
    """Verify that replicas sharing a disk ledger draw from one budget."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n_replicas in replica_counts:
            max_overshoot = 0
            violating_trials = 0
            for trial in range(repetitions):
                ledger_path = os.path.join(
                    tmp, f"replicas-{n_replicas}-{trial}.json"
                )
                caches = [
                    _make_cache(mode="balanced", ledger_path=ledger_path)
                    for _ in range(n_replicas)
                ]
                for cache in caches:
                    _insert_budgeted(cache, "replica-victim", tokens)

                barrier = threading.Barrier(n_threads)

                def worker(idx: int) -> int:
                    barrier.wait(timeout=10)
                    cache = caches[idx % n_replicas]
                    value, _ = cache.match_prefix(
                        key=tokens, user_id=f"replica-attacker-{idx}"
                    )
                    return int(len(value) > 0)

                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    accepted = sum(pool.map(worker, range(n_threads)))
                charged = sum(
                    DurableLedger(path=ledger_path).snapshot().values()
                )
                overshoot = max(0, accepted - BUDGET_B)
                max_overshoot = max(max_overshoot, overshoot)
                violating_trials += int(
                    overshoot > 0 or charged != BUDGET_B
                )

            rows.append({
                "n_replicas": n_replicas,
                "n_concurrent": n_threads,
                "repetitions": repetitions,
                "max_overshoot": max_overshoot,
                "violating_trials": violating_trials,
                "verdict": (
                    "pass"
                    if max_overshoot == 0 and violating_trials == 0
                    else "fail"
                ),
            })
            print(
                f"  replicas={n_replicas}: race={n_threads} "
                f"repetitions={repetitions} max_overshoot={max_overshoot} "
                f"{rows[-1]['verdict'].upper()}",
                flush=True,
            )
    return rows


def run_fail_closed_tests(
    tokens: List[int], output_dir: Path
) -> List[Dict[str, object]]:
    """Exercise unavailable, corrupt, and reserve-before-response states."""
    rows = []

    unavailable = _make_cache(mode="balanced")
    _insert_budgeted(unavailable, "unavailable-victim", tokens)
    unavailable.ledger.set_available(False)
    unavailable_hits = _simulate_cross_tenant_hits(
        unavailable, "unavailable-victim", "attacker", tokens, BUDGET_B + 2
    )
    rows.append({
        "scenario": "ledger_unavailable",
        "attempts": BUDGET_B + 2,
        "served_hits": unavailable_hits,
        "charged_hits": 0,
        "pre_restore_hits": unavailable_hits,
        "conservative_charges": 0,
        "verdict": "pass" if unavailable_hits == 0 else "fail",
    })

    corrupt_path = output_dir / "exp2_corrupt_ledger.json"
    corrupt_path.write_text("{not-json", encoding="utf-8")
    corrupt = _make_cache(mode="balanced", ledger_path=str(corrupt_path))
    _insert_budgeted(corrupt, "corrupt-victim", tokens)
    corrupt_hits = _simulate_cross_tenant_hits(
        corrupt, "corrupt-victim", "attacker", tokens, BUDGET_B + 2
    )
    rows.append({
        "scenario": "recovery_incomplete",
        "attempts": BUDGET_B + 2,
        "served_hits": corrupt_hits,
        "charged_hits": 0,
        "pre_restore_hits": corrupt_hits,
        "conservative_charges": 0,
        "verdict": (
            "pass"
            if corrupt_hits == 0 and not corrupt.ledger.is_operational
            else "fail"
        ),
    })

    crash_path = output_dir / "exp2_after_reserve_ledger.json"
    if crash_path.exists():
        crash_path.unlink()
    before_crash = _make_cache(mode="balanced", ledger_path=str(crash_path))
    fingerprint = _insert_budgeted(
        before_crash, "after-reserve-victim", tokens
    )
    accepted, charged, _ = before_crash.ledger.reserve_hit(
        fingerprint, BUDGET_B
    )
    del before_crash  # crash after durable reservation, before KV response

    recovered = _make_cache(mode="balanced", ledger_path=str(crash_path))
    _insert_budgeted(recovered, "after-reserve-victim", tokens)
    recovered_hits = _simulate_cross_tenant_hits(
        recovered,
        "after-reserve-victim",
        "attacker",
        tokens,
        BUDGET_B + 2,
    )
    final_charged = recovered.ledger.charged_hits(fingerprint)
    conservative = final_charged - recovered_hits
    rows.append({
        "scenario": "crash_after_reserve",
        "attempts": BUDGET_B + 2,
        "served_hits": recovered_hits,
        "charged_hits": final_charged,
        "pre_restore_hits": 0,
        "conservative_charges": conservative,
        "verdict": (
            "pass"
            if accepted
            and charged == 1
            and recovered_hits == BUDGET_B - 1
            and final_charged == BUDGET_B
            and conservative == 1
            else "fail"
        ),
    })

    for row in rows:
        print(
            f"  {row['scenario']}: served={row['served_hits']} "
            f"charged={row['charged_hits']} {row['verdict'].upper()}",
            flush=True,
        )
    return rows


# ── Section 5: Strict Mode ────────────────────────────────────────────────────

def run_strict_mode(
    tokens: List[int],
    n_cycles: int,
) -> List[Dict[str, object]]:
    """Strict Mode must serve zero cross-tenant hits."""
    rows = []
    cache = _make_cache(mode="strict")
    owner = "strict-victim"
    attacker = "strict-attacker"

    cumulative = 0
    for cycle in range(n_cycles):
        cache.insert(key=tokens, value=torch.tensor(list(range(len(tokens))), dtype=torch.int64), user_id=owner)
        hits = _simulate_cross_tenant_hits(cache, owner, attacker, tokens, BUDGET_B + 2)
        cumulative += hits
        _evict_all(cache)
        rows.append({
            "policy": "strict",
            "cycle": cycle,
            "hits_this_cycle": hits,
            "cumulative_hits": cumulative,
            "verdict": "pass" if cumulative == 0 else "fail",
        })
    print(f"  strict: {n_cycles} cycles, cumulative hits = {cumulative} "
          f"{'PASS' if cumulative == 0 else 'FAIL'}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "exp2_ledger",
    )
    parser.add_argument("--n-cycles", type=int, default=20,
                        help="Eviction/reinsertion cycles per policy")
    parser.add_argument(
        "--race-workers",
        nargs="+",
        type=int,
        default=[2, 8, 32, 128],
        help="Concurrent attacker counts to test in the race",
    )
    parser.add_argument("--race-repetitions", type=int, default=100)
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--replica-race-workers", type=int, default=128)
    parser.add_argument("--replica-repetitions", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use a synthetic sensitive prefix (fixed for reproducibility).
    tokens = list(range(100, 164))  # 64 token IDs

    print(f"\n{'='*60}")
    print(f" SafeKV Exp #2 – Cumulative Ledger Enforcement")
    print(f" budget B={BUDGET_B}  cycles={args.n_cycles}  race={args.race_workers}")
    print(f"{'='*60}\n")

    # Section 1.
    print("── Section 1: Eviction/reinsertion sweep ──")
    eviction_rows = run_eviction_sweep(args.n_cycles, tokens, args.output_dir)
    _write_csv(args.output_dir / "exp2_eviction_sweep.csv", eviction_rows)

    # Section 1b: Strict Mode zero-hit verification.
    print("\n── Section 1b: Strict Mode ──")
    strict_rows = run_strict_mode(tokens, args.n_cycles)
    _write_csv(args.output_dir / "exp2_strict_mode.csv", strict_rows)

    # Section 2.
    print("\n── Section 2: Concurrent probe race ──")
    race_rows = run_concurrent_race(
        args.race_workers,
        tokens,
        args.output_dir,
        repetitions=args.race_repetitions,
    )
    _write_csv(args.output_dir / "exp2_race.csv", race_rows)

    # Section 3.
    print("\n── Section 3: Restart simulation ──")
    restart_rows = run_restart_simulation(tokens, args.output_dir)
    _write_csv(args.output_dir / "exp2_restart.csv", restart_rows)

    # Section 4.
    print("\n── Section 4: Shared-ledger replica races ──")
    replica_rows = run_replica_races(
        tokens,
        args.replicas,
        n_threads=args.replica_race_workers,
        repetitions=args.replica_repetitions,
    )
    _write_csv(args.output_dir / "exp2_replicas.csv", replica_rows)

    # Section 5.
    print("\n── Section 5: Fail-closed and crash points ──")
    fail_closed_rows = run_fail_closed_tests(tokens, args.output_dir)
    _write_csv(args.output_dir / "exp2_fail_closed.csv", fail_closed_rows)

    # ── Final verdict ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    all_pass = True
    for label, rows in [
        ("Eviction sweep (balanced)", [r for r in eviction_rows if r["policy"] == "balanced"]),
        ("Strict mode",               strict_rows),
        ("Concurrent race",           race_rows),
        ("Restart simulation",        restart_rows),
        ("Replica race",              replica_rows),
        ("Fail-closed recovery",      fail_closed_rows),
    ]:
        section_pass = all(r["verdict"] == "pass" for r in rows)
        all_pass = all_pass and section_pass
        print(f"  {label:<35} {'PASS' if section_pass else 'FAIL'}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"{'='*60}\n")

    # Generate paper result sentences.
    balanced_total = sum(
        r["cumulative_hits"]
        for r in eviction_rows
        if r["policy"] == "balanced" and r["cycle"] == args.n_cycles - 1
    )
    strict_total = sum(r["cumulative_hits"] for r in strict_rows
                       if r["cycle"] == args.n_cycles - 1)
    legacy_final = next(
        (r["cumulative_hits"] for r in eviction_rows
         if r["policy"] == "legacy" and r["cycle"] == args.n_cycles - 1),
        "N/A"
    )
    summary = {
        "budget_B": BUDGET_B,
        "n_cycles": args.n_cycles,
        "balanced_total_hits": balanced_total,
        "strict_total_hits": strict_total,
        "legacy_total_hits": legacy_final,
        "race_workers": args.race_workers,
        "race_max_overshoots": [r["max_overshoot"] for r in race_rows],
        "race_repetitions": args.race_repetitions,
        "restart_total_hits": restart_rows[0]["total_hits"] if restart_rows else None,
        "recovery_p50_us": restart_rows[0]["recovery_p50_us"],
        "recovery_p95_us": restart_rows[0]["recovery_p95_us"],
        "recovery_p99_us": restart_rows[0]["recovery_p99_us"],
        "replica_counts": args.replicas,
        "replica_max_overshoots": [
            r["max_overshoot"] for r in replica_rows
        ],
        "fail_closed": {
            r["scenario"]: r["verdict"] for r in fail_closed_rows
        },
        "all_pass": all_pass,
    }
    (args.output_dir / "exp2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results in {args.output_dir}")

    if not all_pass:
        raise SystemExit(1)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {path}")


if __name__ == "__main__":
    main()
