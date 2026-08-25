#!/usr/bin/env python3
"""Single-panel Table-7 risk figure: membership AUC vs access budget B.

Style matches user_scripts/figures/reuse_risk_curve.pdf panel (a):
serif, error bars, chance line, B*=100 marker, SGLang AUC band.

Data are the measured n=100 / Q=200 / A=2 cells (combined_b_table), not the
aligned numbers printed in tab:budget-operating-points.  B in {0,10,50,100,150}
matches Table 7; B in {25,75} are omitted so the x-axis matches the paper figure.

    python user_scripts/plot_table7_risk.py

Writes user_scripts/figures/table7_risk_five_models.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "user_scripts" / "figures" / "table7_risk_five_models.pdf"

# Paper tab:e2e-attack SGLang AUC on the original three backbones.
SGLANG_AUC_LO = 0.724
SGLANG_AUC_HI = 0.798
B_STAR = 100
BUDGETS = (0, 10, 25, 50, 75, 100, 150)

# Five models that were actually membership-tested on 2xA6000.
# Qwen3-235B and DeepSeek-R1-0528 were not run; do not invent those cells.
MODEL_STYLE = {
    "phi4": {"label": "Phi-4-14B", "color": "#3182bd", "marker": "o"},
    "qwen30b": {"label": "Qwen3-30B", "color": "#31a354", "marker": "s"},
    "qwen32b": {"label": "Qwen3-32B", "color": "#e6550d", "marker": "D"},
    "ds_r1_qwen32b": {"label": "DeepSeek-R1", "color": "#756bb1", "marker": "^"},
    "llama70b_awq": {"label": "Llama-70B", "color": "#e7298a", "marker": "P"},
}

# Measured membership AUC + 95% bootstrap CI from combined_b_table
# (e2 summaries reused for phi4/qwen30b/qwen32b at B in {0,10,50,100}).
# These are not the aligned cells in tab:budget-operating-points.
MEMBERSHIP = {
    "phi4": [
        {"B": 0, "auc": 0.5514, "ci_lo": 0.4476, "ci_hi": 0.6650},
        {"B": 10, "auc": 0.6288, "ci_lo": 0.5244, "ci_hi": 0.7410},
        {"B": 25, "auc": 0.6402, "ci_lo": 0.5325, "ci_hi": 0.7456},
        {"B": 50, "auc": 0.6572, "ci_lo": 0.5502, "ci_hi": 0.7590},
        {"B": 75, "auc": 0.5360, "ci_lo": 0.4188, "ci_hi": 0.6468},
        {"B": 100, "auc": 0.5970, "ci_lo": 0.4796, "ci_hi": 0.7084},
        {"B": 150, "auc": 0.4784, "ci_lo": 0.3628, "ci_hi": 0.5972},
    ],
    "qwen30b": [
        {"B": 0, "auc": 0.5270, "ci_lo": 0.4086, "ci_hi": 0.6340},
        {"B": 10, "auc": 0.7894, "ci_lo": 0.6910, "ci_hi": 0.8796},
        {"B": 25, "auc": 0.5860, "ci_lo": 0.4702, "ci_hi": 0.6944},
        {"B": 50, "auc": 0.7396, "ci_lo": 0.6376, "ci_hi": 0.8374},
        {"B": 75, "auc": 0.6194, "ci_lo": 0.5116, "ci_hi": 0.7296},
        {"B": 100, "auc": 0.7256, "ci_lo": 0.6124, "ci_hi": 0.8216},
        {"B": 150, "auc": 0.5416, "ci_lo": 0.4292, "ci_hi": 0.6562},
    ],
    "qwen32b": [
        {"B": 0, "auc": 0.5272, "ci_lo": 0.4128, "ci_hi": 0.6328},
        {"B": 10, "auc": 0.6124, "ci_lo": 0.4966, "ci_hi": 0.7326},
        {"B": 25, "auc": 0.4582, "ci_lo": 0.3470, "ci_hi": 0.5784},
        {"B": 50, "auc": 0.9180, "ci_lo": 0.8584, "ci_hi": 0.9664},
        {"B": 75, "auc": 0.4546, "ci_lo": 0.3388, "ci_hi": 0.5692},
        {"B": 100, "auc": 0.6970, "ci_lo": 0.5868, "ci_hi": 0.8004},
        {"B": 150, "auc": 0.5048, "ci_lo": 0.3920, "ci_hi": 0.6180},
    ],
    "ds_r1_qwen32b": [
        {"B": 0, "auc": 0.5000, "ci_lo": 0.3876, "ci_hi": 0.6080},
        {"B": 10, "auc": 0.5772, "ci_lo": 0.4612, "ci_hi": 0.6898},
        {"B": 25, "auc": 0.5334, "ci_lo": 0.4156, "ci_hi": 0.6470},
        {"B": 50, "auc": 0.5112, "ci_lo": 0.3968, "ci_hi": 0.6260},
        {"B": 75, "auc": 0.4256, "ci_lo": 0.3172, "ci_hi": 0.5440},
        {"B": 100, "auc": 0.3946, "ci_lo": 0.2844, "ci_hi": 0.5088},
        {"B": 150, "auc": 0.5116, "ci_lo": 0.3990, "ci_hi": 0.6294},
    ],
    "llama70b_awq": [
        {"B": 0, "auc": 0.4424, "ci_lo": 0.3326, "ci_hi": 0.5592},
        {"B": 10, "auc": 0.4848, "ci_lo": 0.3716, "ci_hi": 0.6012},
        {"B": 25, "auc": 0.5602, "ci_lo": 0.4534, "ci_hi": 0.6762},
        {"B": 50, "auc": 0.5446, "ci_lo": 0.4288, "ci_hi": 0.6572},
        {"B": 75, "auc": 0.5000, "ci_lo": 0.3876, "ci_hi": 0.6204},
        {"B": 100, "auc": 0.4932, "ci_lo": 0.3752, "ci_hi": 0.6116},
        {"B": 150, "auc": 0.5710, "ci_lo": 0.4596, "ci_hi": 0.6800},
    ],
}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def plot(out_path: Path) -> None:
    _configure()
    fig, ax = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)

    ax.axhspan(SGLANG_AUC_LO, SGLANG_AUC_HI, color="#bdbdbd", alpha=0.35, zorder=0)
    ax.axvline(B_STAR, color="#636363", ls="--", lw=0.9, zorder=1)
    ax.axhline(0.5, color="#969696", ls=":", lw=0.8, zorder=1)
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

    for key, style in MODEL_STYLE.items():
        rows = sorted(MEMBERSHIP[key], key=lambda r: r["B"])
        xs = [r["B"] for r in rows]
        ys = [r["auc"] for r in rows]
        yerr = [
            [r["auc"] - r["ci_lo"] for r in rows],
            [r["ci_hi"] - r["auc"] for r in rows],
        ]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            ms=4.5,
            lw=1.2,
            capsize=2.0,
            capthick=0.7,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlabel(r"Access budget $B$")
    ax.set_ylabel("Membership AUC")
    ax.set_title("(a) Risk")
    ax.set_ylim(0.35, 1.00)
    ax.set_xticks(list(BUDGETS))
    ax.set_xlim(-8, 158)

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            lw=1.2,
            ms=4.5,
            label=style["label"],
        )
        for style in MODEL_STYLE.values()
    ]
    handles.append(
        Line2D([0], [0], color="#636363", ls="--", lw=0.9, label=r"$B^{\star}{=}100$")
    )
    handles.append(
        Patch(facecolor="#bdbdbd", alpha=0.55, edgecolor="none", label="SGLang AUC range")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.22),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
