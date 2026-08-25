#!/usr/bin/env python3
"""Validate and aggregate SafeKV Exp #1 raw trial CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


VIOLATIONS = (
    "unauth_public_promotions",
    "victim_node_relabels",
    "private_address_aliases",
    "cross_tenant_private_hits",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if any(row["error"] for row in rows):
        raise SystemExit("raw data contains failed trials")

    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row["model"], int(row["A"]), row["auth"], row["schedule"])
        ].append(row)

    expected_groups = len({row["model"] for row in rows}) * 40
    if len(grouped) != expected_groups:
        raise SystemExit(
            f"expected {expected_groups} groups, found {len(grouped)}"
        )

    summary = []
    for key, trials in sorted(grouped.items()):
        if len(trials) != args.trials:
            raise SystemExit(f"{key}: expected {args.trials}, found {len(trials)}")
        model, attackers, auth, schedule = key
        aggregate = {
            "model": model,
            "A": attackers,
            "auth": auth,
            "schedule": schedule,
            "trials": len(trials),
        }
        for field in VIOLATIONS:
            aggregate[field] = sum(int(row[field]) for row in trials)
        aggregate["all_public_created"] = int(
            all(int(row["public_object_created"]) == 1 for row in trials)
        )
        aggregate["any_public_created"] = int(
            any(int(row["public_object_created"]) > 0 for row in trials)
        )
        aggregate["all_victim_private_owned"] = int(
            all(int(row["victim_private_still_owned"]) == 1 for row in trials)
        )
        aggregate["any_public_private_alias"] = int(
            any(int(row["public_reuses_victim_kv"]) for row in trials)
        )
        aggregate["pass"] = int(all(int(row["pass"]) for row in trials))
        summary.append(aggregate)

    if not all(row["pass"] for row in summary):
        raise SystemExit("one or more Exp #1 configurations failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "exp1_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    latex_path = args.output_dir / "exp1_table_rows.tex"
    with latex_path.open("w") as handle:
        for row in summary:
            separate = "yes" if row["auth"] == "valid" else "no"
            handle.write(
                f"{row['model']} & {row['A']} & {row['auth']} & "
                f"{row['schedule']} & "
                + " & ".join(str(row[field]) for field in VIOLATIONS)
                + f" & {separate} \\\\\n"
            )

    print(
        f"validated {len(rows)} trials across {len(summary)} configurations; "
        f"wrote {summary_path} and {latex_path}"
    )


if __name__ == "__main__":
    main()
