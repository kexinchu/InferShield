#!/usr/bin/env python3
"""Merge measured reuse cells into reuse_risk_curve.json and replot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO / "user_scripts" / "results" / "reuse_risk_curve" / "reuse_risk_curve.json"
)
CELL_DIR = REPO / "user_scripts" / "results" / "reuse_risk_curve" / "cells"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--cells", type=Path, default=CELL_DIR)
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    serving = {(row["model"], int(row["B"])): dict(row) for row in data["serving"]}

    merged = 0
    for path in sorted(args.cells.glob("*.json")):
        cell = json.loads(path.read_text())
        if cell.get("workload") != "benign_system_prompt_no_public":
            print(f"[skip] {path.name}: not a benign-prefix cell")
            continue
        key = (cell["model"], int(cell["B"]))
        row = serving.get(key, {"model": cell["model"], "B": cell["B"]})
        if cell.get("reuse_hit_rate") is not None:
            row["reuse_hit_rate"] = cell["reuse_hit_rate"]
        serving[key] = row
        merged += 1

    data["serving"] = [serving[k] for k in sorted(serving, key=lambda x: (x[0], x[1]))]
    if merged and any(row.get("reuse_hit_rate") is not None for row in data["serving"]):
        data["status"] = "reuse_partial" if merged < 18 else "reuse_measured"
    args.data.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[merged] {merged} cells -> {args.data}")


if __name__ == "__main__":
    main()
