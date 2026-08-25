#!/usr/bin/env python3
"""Aggregate Experiment 3 repetitions without treating aliases as measurements."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = (
    SCRIPT_DIR
    / "results"
    / "submission_gap_experiments"
    / "e3_serving_repeated"
)
RUN_DIR = RESULT_ROOT / "runs"
OUTPUT_DIR = RESULT_ROOT / "aggregated"
SEEDS = (20260821, 20260822, 20260823)
MODELS = ("phi4", "qwen30b", "qwen32b")
WORKLOADS = ("single_pii", "multi_turn", "system_prompt")
MEASURED_POLICIES = ("vanilla", "strict", "balanced", "balanced_public")
EMULATED_POLICY = "shared_system_prompt_emulation"
T_975_DF2 = 4.302652729911275

Cell = Tuple[str, str, str]


def expected_cells() -> List[Cell]:
    cells = [
        (model, policy, workload)
        for model in MODELS
        for policy in MEASURED_POLICIES
        for workload in WORKLOADS
    ]
    cells.extend(
        (model, EMULATED_POLICY, "system_prompt") for model in MODELS
    )
    return cells


def load_repetitions() -> Dict[Cell, List[Dict[str, object]]]:
    grouped: Dict[Cell, List[Dict[str, object]]] = defaultdict(list)
    for sentinel_path in sorted(RUN_DIR.glob("*/*.complete.json")):
        metadata = json.loads(sentinel_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            continue
        csv_path = sentinel_path.with_name(
            sentinel_path.name.removesuffix(".complete.json") + ".csv"
        )
        if not csv_path.is_file():
            raise ValueError(f"sentinel has no result CSV: {sentinel_path}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"expected one row in {csv_path}, found {len(rows)}")
        row = rows[0]
        cell = (row["model"], row["policy"], row["workload"])
        for field, index in (("model", 0), ("policy", 1), ("workload", 2)):
            if metadata[field] != cell[index]:
                raise ValueError(f"metadata mismatch for {field}: {sentinel_path}")
        seed = int(metadata["seed"])
        repetition = int(metadata["repetition"])
        grouped[cell].append(
            {
                "seed": seed,
                "repetition": repetition,
                "mean_ttft_ms": float(row["mean_ttft_ms"]),
                "throughput_tok_s": float(row["throughput_tok_s"]),
                "source": str(csv_path.relative_to(RESULT_ROOT)),
            }
        )
    return grouped


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) != 3:
        raise ValueError(f"expected 3 repetitions, found {len(values)}")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    margin = T_975_DF2 * sample_sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_standard_deviation": sample_sd,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def main() -> None:
    grouped = load_repetitions()
    expected = set(expected_cells())
    unexpected = set(grouped) - expected
    if unexpected:
        raise ValueError(f"unexpected measured cells: {sorted(unexpected)}")

    summaries: List[Dict[str, object]] = []
    for cell in expected_cells():
        reps = grouped.get(cell, [])
        observed_seeds = sorted(int(rep["seed"]) for rep in reps)
        observed_reps = sorted(int(rep["repetition"]) for rep in reps)
        if observed_seeds != list(SEEDS) or observed_reps != [1, 2, 3]:
            raise ValueError(
                f"incomplete cell {cell}: repetitions={observed_reps}, "
                f"seeds={observed_seeds}"
            )
        ttft = summarize([float(rep["mean_ttft_ms"]) for rep in reps])
        throughput = summarize(
            [float(rep["throughput_tok_s"]) for rep in reps]
        )
        summaries.append(
            {
                "model": cell[0],
                "policy": cell[1],
                "workload": cell[2],
                "evidence_type": (
                    "explicit_emulation"
                    if cell[1] == EMULATED_POLICY
                    else "measured"
                ),
                "n_repetitions": 3,
                "seeds": list(SEEDS),
                "ttft_ms": ttft,
                "throughput_tok_s": throughput,
                "sources": [rep["source"] for rep in sorted(
                    reps, key=lambda item: int(item["repetition"])
                )],
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "e3_serving_summary.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(
            {
                "confidence_interval": {
                    "level": 0.95,
                    "method": "two-sided Student t",
                    "degrees_of_freedom": 2,
                    "critical_value": T_975_DF2,
                },
                "alias_policy": {
                    "cache_partition": {
                        "alias_of": "strict",
                        "numeric_rows_emitted": False,
                    }
                },
                "unavailable_baselines": {
                    "CacheSolidarity": {
                        "status": "unavailable",
                        "numeric_rows_emitted": False,
                    }
                },
                "cells": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    json_tmp.replace(json_path)

    csv_path = OUTPUT_DIR / "e3_serving_summary.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    fields = (
        "model",
        "policy",
        "workload",
        "evidence_type",
        "n_repetitions",
        "ttft_ms_mean",
        "ttft_ms_sample_sd",
        "ttft_ms_ci95_lower",
        "ttft_ms_ci95_upper",
        "throughput_tok_s_mean",
        "throughput_tok_s_sample_sd",
        "throughput_tok_s_ci95_lower",
        "throughput_tok_s_ci95_upper",
    )
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "model": item["model"],
                    "policy": item["policy"],
                    "workload": item["workload"],
                    "evidence_type": item["evidence_type"],
                    "n_repetitions": item["n_repetitions"],
                    "ttft_ms_mean": item["ttft_ms"]["mean"],
                    "ttft_ms_sample_sd": item["ttft_ms"][
                        "sample_standard_deviation"
                    ],
                    "ttft_ms_ci95_lower": item["ttft_ms"]["ci95_lower"],
                    "ttft_ms_ci95_upper": item["ttft_ms"]["ci95_upper"],
                    "throughput_tok_s_mean": item["throughput_tok_s"]["mean"],
                    "throughput_tok_s_sample_sd": item["throughput_tok_s"][
                        "sample_standard_deviation"
                    ],
                    "throughput_tok_s_ci95_lower": item[
                        "throughput_tok_s"
                    ]["ci95_lower"],
                    "throughput_tok_s_ci95_upper": item[
                        "throughput_tok_s"
                    ]["ci95_upper"],
                }
            )
    csv_tmp.replace(csv_path)
    print(f"[E3_AGGREGATION_COMPLETE] cells={len(summaries)} output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
