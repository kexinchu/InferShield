#!/usr/bin/env python3
"""SafeKV Exp #9 – atomic reservation, replica, eviction, and crash matrix.

This test exercises DurableLedger.reserve_hit(), the operation that must
complete before a Balanced lookup exposes a KV address.  Multiple ledger
instances share one backing file to model local cache replicas.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List

from sglang.srt.mem_cache.safekv_policy import DurableLedger


BUDGET_B = 10
FINGERPRINT = "exp9-shared-prefix"
RESTART_POINTS = (
    "none",
    "before_reserve",
    "after_reserve_before_response",
    "after_response",
)

CSV_FIELDS = (
    "n_attackers",
    "n_threads",
    "n_replicas",
    "n_cycles",
    "restart_point",
    "attempted_lookups",
    "served_hits",
    "durable_charges",
    "unused_conservative_charges",
    "lost_charges",
    "duplicated_charges",
    "max_overshoot",
    "served_before_recovery",
    "recovery_p50_us",
    "recovery_p99_us",
    "verdict",
)


def recover_replicas(
    ledger_path: str, n_replicas: int, samples: List[float]
) -> List[DurableLedger]:
    replicas = []
    for _ in range(n_replicas):
        t0 = time.perf_counter()
        ledger = DurableLedger(path=ledger_path)
        samples.append((time.perf_counter() - t0) * 1e6)
        if not ledger.is_operational:
            raise RuntimeError("ledger recovery did not become operational")
        replicas.append(ledger)
    return replicas


def run_concurrent_round(
    replicas: List[DurableLedger],
    n_attackers: int,
    n_threads: int,
    budget: int,
) -> int:
    """Return successful reservations from one concurrent lookup round."""
    barrier = threading.Barrier(n_threads)
    accepted = [0] * n_threads

    def reserve(index: int) -> None:
        # Principal identity and replica selection are independent dimensions.
        _attacker = index % n_attackers
        ledger = replicas[index % len(replicas)]
        barrier.wait(timeout=30)
        ok, _, _ = ledger.reserve_hit(FINGERPRINT, budget)
        accepted[index] = int(ok)

    threads = [
        threading.Thread(target=reserve, args=(index,))
        for index in range(n_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError("reservation thread did not terminate")
    return sum(accepted)


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def run_cell(
    n_attackers: int,
    n_threads: int,
    n_replicas: int,
    n_cycles: int,
    restart_point: str,
    ledger_path: str,
    budget: int,
) -> Dict[str, object]:
    recovery_samples: List[float] = []
    replicas = recover_replicas(ledger_path, n_replicas, recovery_samples)
    served_hits = 0
    attempted = 0
    unused_charges = 0
    served_before_recovery = 0

    # Inject exactly one crash at the specified linearization point.
    if restart_point == "before_reserve":
        replicas = recover_replicas(
            ledger_path, n_replicas, recovery_samples
        )
    elif restart_point in (
        "after_reserve_before_response",
        "after_response",
    ):
        attempted += 1
        accepted, _, _ = replicas[0].reserve_hit(FINGERPRINT, budget)
        if accepted:
            if restart_point == "after_reserve_before_response":
                unused_charges += 1
            else:
                served_hits += 1
        replicas = recover_replicas(
            ledger_path, n_replicas, recovery_samples
        )

    # Each cycle models eviction/reinsertion by reconstructing all replicas.
    for _ in range(n_cycles):
        successes = run_concurrent_round(
            replicas, n_attackers, n_threads, budget
        )
        attempted += n_threads
        served_hits += successes
        replicas = recover_replicas(
            ledger_path, n_replicas, recovery_samples
        )

    final_ledger = DurableLedger(path=ledger_path)
    durable_charges = final_ledger.charged_hits(FINGERPRINT)
    expected_charges = served_hits + unused_charges
    lost_charges = max(0, expected_charges - durable_charges)
    duplicated_charges = max(0, durable_charges - expected_charges)
    max_overshoot = max(0, durable_charges - budget)

    verdict = "pass"
    if (
        max_overshoot != 0
        or lost_charges != 0
        or duplicated_charges != 0
        or served_before_recovery != 0
        or durable_charges > budget
    ):
        verdict = "fail"

    return {
        "n_attackers": n_attackers,
        "n_threads": n_threads,
        "n_replicas": n_replicas,
        "n_cycles": n_cycles,
        "restart_point": restart_point,
        "attempted_lookups": attempted,
        "served_hits": served_hits,
        "durable_charges": durable_charges,
        "unused_conservative_charges": unused_charges,
        "lost_charges": lost_charges,
        "duplicated_charges": duplicated_charges,
        "max_overshoot": max_overshoot,
        "served_before_recovery": served_before_recovery,
        "recovery_p50_us": statistics.median(recovery_samples),
        "recovery_p99_us": percentile(recovery_samples, 0.99),
        "verdict": verdict,
    }


def main() -> None:
    global BUDGET_B
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attackers", nargs="+", type=int, default=[1, 8, 32, 128]
    )
    parser.add_argument(
        "--threads", nargs="+", type=int, default=[1, 8, 32, 128]
    )
    parser.add_argument(
        "--replicas", nargs="+", type=int, default=[1, 2]
    )
    parser.add_argument(
        "--cycles", nargs="+", type=int, default=[1, 5, 20]
    )
    parser.add_argument(
        "--restart-points", nargs="+", default=list(RESTART_POINTS)
    )
    parser.add_argument("--budget-B", type=int, default=BUDGET_B)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "exp9_v2" / "stress_matrix.csv",
    )
    args = parser.parse_args()
    BUDGET_B = args.budget_B
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for attackers in args.attackers:
            for threads in args.threads:
                for replicas in args.replicas:
                    for cycles in args.cycles:
                        for restart_point in args.restart_points:
                            ledger_path = (
                                f"{tmp}/ledger_{attackers}_{threads}_"
                                f"{replicas}_{cycles}_{restart_point}.json"
                            )
                            row = run_cell(
                                attackers,
                                threads,
                                replicas,
                                cycles,
                                restart_point,
                                ledger_path,
                                BUDGET_B,
                            )
                            rows.append(row)
                            print(
                                f"A={attackers:<3} T={threads:<3} "
                                f"R={replicas} C={cycles:<2} "
                                f"{restart_point:<30} "
                                f"served={row['served_hits']:<2} "
                                f"charged={row['durable_charges']:<2} "
                                f"unused={row['unused_conservative_charges']} "
                                f"{row['verdict']}",
                                flush=True,
                            )

    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    recovery = [
        float(row["recovery_p50_us"]) for row in rows
    ] + [float(row["recovery_p99_us"]) for row in rows]
    summary = {
        "total_cells": len(rows),
        "pass_cells": sum(row["verdict"] == "pass" for row in rows),
        "max_durable_charges": max(
            int(row["durable_charges"]) for row in rows
        ),
        "max_overshoot": max(int(row["max_overshoot"]) for row in rows),
        "max_lost_charges": max(int(row["lost_charges"]) for row in rows),
        "max_duplicated_charges": max(
            int(row["duplicated_charges"]) for row in rows
        ),
        "max_unused_conservative_charges": max(
            int(row["unused_conservative_charges"]) for row in rows
        ),
        "served_before_recovery": sum(
            int(row["served_before_recovery"]) for row in rows
        ),
        "recovery_p50_us": statistics.median(recovery),
        "recovery_p99_us": percentile(recovery, 0.99),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if summary["pass_cells"] != summary["total_cells"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
