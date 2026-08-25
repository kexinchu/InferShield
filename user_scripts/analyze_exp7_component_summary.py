#!/usr/bin/env python3
"""Synthesize revised P7 ablation evidence from P3--P7 raw results.

The deterministic mechanism deltas come from exp7_ablation_invariants.csv.
Attack AUCs are recomputed from P3/P4 request-level CSVs, and serving ranges
are derived from P5.  The output is a compact JSON artifact used to populate
the paper's component-analysis table without treating unlike protocols as one
monolithic experiment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ("phi4", "qwen30b", "qwen32b")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def auc_lower_is_positive(labels: Sequence[int], values: Sequence[float]) -> float:
    """Pairwise Mann--Whitney AUC with lower latency as the positive score."""
    positive = [v for y, v in zip(labels, values) if y == 1]
    negative = [v for y, v in zip(labels, values) if y == 0]
    if not positive or not negative:
        raise ValueError("AUC requires both classes")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos < neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def value_range(values: Iterable[float], digits: int = 3) -> list[float]:
    vals = list(values)
    return [round(min(vals), digits), round(max(vals), digits)]


def p3_auc(policy: str) -> list[float]:
    aucs = []
    for model in MODELS:
        rows = [
            row
            for row in read_csv(RESULTS / "exp3" / f"{model}_{policy}_v2.csv")
            if row["experiment_type"] == "membership"
        ]
        aucs.append(
            auc_lower_is_positive(
                [int(row["challenge_bit"]) for row in rows],
                [float(row["ttft_ms"]) for row in rows],
            )
        )
    return value_range(aucs)


def p4_variant(variant: str) -> dict[str, list[float]]:
    aucs = []
    deltas = []
    for model in MODELS:
        rows = [
            row
            for row in read_csv(RESULTS / "exp4" / f"{model}.csv")
            if row["variant"] == variant
        ]
        labels = [int(row["challenge_bit"]) for row in rows]
        values = [float(row["ttft_use_ms"]) for row in rows]
        aucs.append(auc_lower_is_positive(labels, values))
        use = [v for y, v in zip(labels, values) if y == 1]
        no_use = [v for y, v in zip(labels, values) if y == 0]
        deltas.append(sum(no_use) / len(no_use) - sum(use) / len(use))
    return {
        "auc_range": value_range(aucs),
        "delta_ttft_ms_range": value_range(deltas, digits=1),
    }


def p5_policy(policy: str) -> dict[str, list[float]]:
    system_overheads = []
    throughput_ratios = []
    for model in MODELS:
        vanilla_system = read_csv(
            RESULTS / "exp5_v2" / f"{model}_vanilla_system_prompt.csv"
        )[-1]
        policy_system = read_csv(
            RESULTS / "exp5_v2" / f"{model}_{policy}_system_prompt.csv"
        )[-1]
        system_overheads.append(
            100
            * (
                float(policy_system["mean_ttft_ms"])
                / float(vanilla_system["mean_ttft_ms"])
                - 1
            )
        )

        vanilla_mt = read_csv(
            RESULTS / "exp5_v2" / f"{model}_vanilla_multi_turn.csv"
        )[-1]
        policy_mt = read_csv(
            RESULTS / "exp5_v2" / f"{model}_{policy}_multi_turn.csv"
        )[-1]
        throughput_ratios.append(
            float(policy_mt["throughput_tok_s"])
            / float(vanilla_mt["throughput_tok_s"])
        )
    return {
        "system_prompt_overhead_pct_range": value_range(system_overheads, digits=0),
        "multi_turn_throughput_ratio_range": value_range(
            throughput_ratios, digits=2
        ),
    }


def main() -> None:
    invariants = read_csv(RESULTS / "exp7_ablation_invariants.csv")
    microbench = json.loads(
        (RESULTS / "exp6" / "registry_microbench.csv").read_text()
    )
    metadata = microbench["metadata_sizes"]

    summary = {
        "deterministic_ablation": invariants,
        "empirical_evidence": {
            "private_namespace": {
                "membership_auc_range": p3_auc("strict"),
                **p5_policy("strict"),
            },
            "budgeted_reuse": {
                "membership_auc_range": p3_auc("balanced"),
                **p5_policy("balanced"),
            },
            "reactive_public": p4_variant("reactive_materialized"),
            "prewarmed_public": {
                **p4_variant("prewarmed_pinned_public"),
                **p5_policy("balanced_public"),
            },
        },
        "serialized_metadata": {
            "public_authorization_bytes": metadata[
                "PublicAuthorization_json_bytes"
            ],
            "registry_bytes_per_entry_at_100": round(
                metadata["PublicRegistry_100_entry_json_bytes"] / 100, 1
            ),
            "ledger_bytes_per_entry_at_1000": round(
                metadata["DurableLedger_1000_entry_json_bytes"] / 1000, 1
            ),
        },
    }

    output = RESULTS / "exp7_component_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["empirical_evidence"], indent=2))
    print(f"Results -> {output}")


if __name__ == "__main__":
    main()
