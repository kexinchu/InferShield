#!/usr/bin/env python3
"""Render the corrected P10 robustness matrix as a dense paper figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("phi4", "qwen30b", "qwen32b")
POLICIES = ("strict", "balanced")
CONDITIONS = (
    "local",
    "jitter_10ms",
    "jitter_30ms",
    "mixed_lengths",
    "continuous_batching",
    "background_load",
)
CONDITION_LABELS = (
    "Local",
    r"$\pm$10 ms",
    r"$\pm$30 ms",
    "Mixed len.",
    "Cont. batch",
    "Bg. load",
)
MODEL_LABELS = {
    "phi4": "Phi-4",
    "qwen30b": "Qwen30B",
    "qwen32b": "Qwen32B",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "results" / "exp10_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1]
        / "user_scripts"
        / "figures"
        / "attack_robustness_heatmap.pdf",
    )
    args = parser.parse_args()

    rows = [(model, policy) for model in MODELS for policy in POLICIES]
    auc = np.zeros((len(rows), len(CONDITIONS)))
    annotations = []
    for row_index, (model, policy) in enumerate(rows):
        data = json.loads(
            (args.results / f"{model}_{policy}.summary.json").read_text()
        )
        annotation_row = []
        for column, condition in enumerate(CONDITIONS):
            cell = data["conditions"][condition]
            auc[row_index, column] = cell["auc"]
            annotation_row.append(
                f"{cell['auc']:.2f}\n"
                f"{cell['adv_mi']:.2f} "
                f"[{cell['adv_ci_lo']:.2f},{cell['adv_ci_hi']:.2f}]"
            )
        annotations.append(annotation_row)

    figure, axis = plt.subplots(figsize=(13.5, 4.2))
    image = axis.imshow(
        auc,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    for row in range(auc.shape[0]):
        for column in range(auc.shape[1]):
            axis.text(
                column,
                row,
                annotations[row][column],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(auc[row, column] - 0.5) > 0.28 else "black",
            )

    axis.set_xticks(range(len(CONDITIONS)), CONDITION_LABELS)
    axis.set_yticks(
        range(len(rows)),
        [
            f"{MODEL_LABELS[model]} / {policy.capitalize()}"
            for model, policy in rows
        ],
    )
    axis.set_xlabel("Cell: AUC (top); AdvMI [95% bootstrap CI] (bottom)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("ROC-AUC (0.5 = chance)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
