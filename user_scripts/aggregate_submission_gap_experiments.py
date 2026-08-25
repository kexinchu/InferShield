#!/usr/bin/env python3
"""Aggregate E1/E2/E4/E5 (and E3 if present) into independent summary files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent / "results" / "submission_gap_experiments"
OUT = ROOT / "aggregated"
MODELS = ("phi4", "qwen30b", "qwen32b")
BUDGETS = (0, 1, 10, 50, 100)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_e1() -> Dict:
    cells = []
    for model in MODELS:
        summ = _load_json(
            ROOT / "e1_vanilla_attack" / f"{model}_vanilla_membership.summary.json"
        )
        if not summ:
            continue
        cells.append(
            {
                "model": model,
                "policy": "vanilla",
                "roc_auc": summ.get("roc_auc"),
                "roc_auc_ci_lo": summ.get("roc_auc_ci_lo"),
                "roc_auc_ci_hi": summ.get("roc_auc_ci_hi"),
                "adv_mi": summ.get("adv_mi"),
                "adv_mi_ci_lo": summ.get("adv_mi_ci_lo"),
                "adv_mi_ci_hi": summ.get("adv_mi_ci_hi"),
                "tpr": summ.get("tpr"),
                "fpr": summ.get("fpr"),
                "n_mi_trials": summ.get("n_mi_trials"),
            }
        )
    return {"experiment": "e1_vanilla_attack", "cells": cells}


def aggregate_e2() -> Dict:
    cells = []
    for model in MODELS:
        for B in BUDGETS:
            mi = _load_json(
                ROOT / "e2_budget_sweep" / f"{model}_B{B}_membership.summary.json"
            )
            meta = _load_json(
                ROOT / "e2_budget_sweep" / f"{model}_B{B}_system_prompt.meta.json"
            )
            row = meta.get("row", {}) if meta else {}
            if not mi and not row:
                continue
            cells.append(
                {
                    "model": model,
                    "access_budget_B": B,
                    "safekv_mode": "strict" if B == 0 else "balanced",
                    "roc_auc": mi.get("roc_auc"),
                    "roc_auc_ci_lo": mi.get("roc_auc_ci_lo"),
                    "roc_auc_ci_hi": mi.get("roc_auc_ci_hi"),
                    "adv_mi": mi.get("adv_mi"),
                    "adv_mi_ci_lo": mi.get("adv_mi_ci_lo"),
                    "adv_mi_ci_hi": mi.get("adv_mi_ci_hi"),
                    "system_prompt_mean_ttft_ms": (
                        float(row["mean_ttft_ms"]) if row.get("mean_ttft_ms") else None
                    ),
                    "system_prompt_throughput_tok_s": (
                        float(row["throughput_tok_s"])
                        if row.get("throughput_tok_s")
                        else None
                    ),
                }
            )
    return {"experiment": "e2_budget_sweep", "cells": cells}


def aggregate_e4() -> Dict:
    path = ROOT / "e4_principal_binding" / "manifest.json"
    return {
        "experiment": "e4_principal_binding",
        "manifest": _load_json(path),
        "complete": path.exists(),
    }


def aggregate_e5() -> Dict:
    cells = []
    for model in MODELS:
        for mode in ("guaranteed", "logit_topk"):
            summ = _load_json(
                ROOT
                / "e5_strong_recovery"
                / f"{model}_balanced_B100_{mode}.summary.json"
            )
            if not summ:
                continue
            cells.append(
                {
                    "model": model,
                    "recovery_mode": mode,
                    "access_budget_B": 100,
                    "token_recovery_fraction": summ.get("token_recovery_fraction"),
                    "full_secret_recovery_fraction": summ.get(
                        "full_secret_recovery_fraction"
                    ),
                    "tokens_attempted": summ.get("tokens_attempted"),
                    "tokens_recovered": summ.get("tokens_recovered"),
                    "n_rec_trials": summ.get("n_rec_trials"),
                    "attacker_queries_used_recovery": summ.get(
                        "attacker_queries_used_recovery"
                    ),
                    "setup_prior_queries": summ.get("setup_prior_queries"),
                }
            )
    return {"experiment": "e5_strong_recovery", "cells": cells}


def aggregate_e3() -> Dict:
    path = (
        ROOT
        / "e3_serving_repeated"
        / "aggregated"
        / "e3_serving_summary.json"
    )
    return {
        "experiment": "e3_serving_repeated",
        "summary": _load_json(path),
        "complete": path.exists(),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e1 = aggregate_e1()
    e2 = aggregate_e2()
    e4 = aggregate_e4()
    e5 = aggregate_e5()
    e3 = aggregate_e3()

    (OUT / "e1_vanilla_attack_summary.json").write_text(
        json.dumps(e1, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "e2_budget_sweep_summary.json").write_text(
        json.dumps(e2, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "e4_principal_binding_summary.json").write_text(
        json.dumps(e4, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "e5_strong_recovery_summary.json").write_text(
        json.dumps(e5, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "e3_serving_repeated_summary.json").write_text(
        json.dumps(e3, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUT / "e1_vanilla_attack_summary.csv", e1["cells"])
    write_csv(OUT / "e2_budget_sweep_summary.csv", e2["cells"])
    write_csv(OUT / "e5_strong_recovery_summary.csv", e5["cells"])

    status = {
        "e1_cells": len(e1["cells"]),
        "e2_cells": len(e2["cells"]),
        "e3_complete": e3["complete"],
        "e4_complete": e4["complete"],
        "e5_cells": len(e5["cells"]),
        "output_dir": str(OUT),
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    print("[AGG_ALL_DONE]")


if __name__ == "__main__":
    main()
