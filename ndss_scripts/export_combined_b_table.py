#!/usr/bin/env python3
"""Export the combined AUC / Adv / TTFT / TPS / Coverage grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
COMBINED = ROOT / "combined_b_table"
E2 = ROOT / "submission_gap_experiments" / "e2_budget_sweep"
BUDGETS = (0, 10, 25, 50, 75, 100, 150)
MODELS = (
    ("phi4", "Phi-4-14B"),
    ("qwen30b", "Qwen3-30B-A3B"),
    ("qwen32b", "Qwen3-32B"),
    ("ds_r1_qwen32b", "DeepSeek-R1-Distill-Qwen-32B"),
    ("llama70b_awq", "Llama-3.3-70B-AWQ"),
)


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def row_for(model: str, B: int) -> dict:
    mi = _load(COMBINED / f"{model}_B{B}_membership.summary.json")
    if mi is None:
        mi = _load(E2 / f"{model}_B{B}_membership.summary.json")
    srv = _load(COMBINED / f"{model}_B{B}_serving.json")
    return {
        "model": model,
        "B": B,
        "auc": None if mi is None else mi.get("roc_auc"),
        "auc_ci_lo": None if mi is None else mi.get("roc_auc_ci_lo"),
        "auc_ci_hi": None if mi is None else mi.get("roc_auc_ci_hi"),
        "adv_mi": None if mi is None else mi.get("adv_mi"),
        "ttft_s": None if srv is None else srv.get("ttft_s"),
        "tps": None if srv is None else srv.get("tps"),
        "coverage_pct": None if srv is None else srv.get("coverage_pct"),
        "membership_source": "missing" if mi is None else ("combined" if (COMBINED / f"{model}_B{B}_membership.summary.json").exists() else "e2"),
        "serving_source": "missing" if srv is None else "combined",
    }


def fmt(v, nd=3):
    if v is None:
        return "   —  "
    return f"{v:6.{nd}f}"


def main() -> None:
    rows = [row_for(m, B) for m, _ in MODELS for B in BUDGETS]
    COMBINED.mkdir(parents=True, exist_ok=True)
    csv_path = COMBINED / "combined_b_table.csv"
    json_path = COMBINED / "combined_b_table.json"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps({"budgets": BUDGETS, "rows": rows}, indent=2) + "\n")

    print("Model                          B    AUC            Adv     TTFT(s)    TPS    Coverage")
    print("-" * 92)
    for (key, label), group_start in zip(MODELS, range(0, len(rows), len(BUDGETS))):
        for i, B in enumerate(BUDGETS):
            r = rows[group_start + i]
            name = label if i == 0 else ""
            auc = "   —  "
            if r["auc"] is not None:
                lo = r["auc_ci_lo"]
                hi = r["auc_ci_hi"]
                if lo is not None and hi is not None:
                    auc = f"{r['auc']:.3f} [{lo:.2f},{hi:.2f}]"
                else:
                    auc = f"{r['auc']:.3f}"
            print(
                f"{name:<28} {B:>4}  {auc:<16} {fmt(r['adv_mi'])}  "
                f"{fmt(r['ttft_s'])}  {fmt(r['tps'],1)}  {fmt(r['coverage_pct'],1)}"
            )
        print()
    print(f"CSV  {csv_path}")
    print(f"JSON {json_path}")


if __name__ == "__main__":
    main()
