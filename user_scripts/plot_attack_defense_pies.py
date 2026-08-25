#!/usr/bin/env python3
"""Redraw the six attack_defense_pie_*.pdf figures.

Edit DATA (and optionally LABELS) below, then:

    python user_scripts/plot_attack_defense_pies.py

Overwrites the six pies under user_scripts/figures/.  Pass --legend
to also refresh attack_defense_pie_legend.pdf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "user_scripts" / "figures"

# ColorBrewer Blues — sampled from the original PDFs.
C_T1 = "#6baed6"  # light  (Tier-1)
C_T2 = "#3182bd"  # mid    (Tier-2)
C_T3 = "#08519c"  # dark   (budget exhaustion / old Tier-3)
COLORS = (C_T1, C_T2, C_T3)
# Third slice is residual detector FN absorbed by the B-cap, not a third
# classifier.  Paper caption: "budget exhaustion".
LABELS = ("Tier-1", "Tier-2", "Budget exhaustion")

# Chat-template Detector-Window rerun, n=500, seed=42
# (user_scripts/results/detector_window_rerun/detector_window_20260819_002302.csv).
# Old paper pies, for reference:
#   phi_4      37.8 / 55.1 / 7.2
#   qwen_30b   40.0 / 58.3 / 1.8
#   qwen_32b   39.9 / 58.2 / 1.9
#   qwen_235b  39.8 / 58.1 / 2.1
#   llama_70b  39.9 / 58.1 / 2.0
#   deepseek   39.1 / 56.9 / 4.0
DATA = {
    "phi_4": (45.2, 54.8, 0.0),
    "qwen_30b": (45.2, 51.2, 3.6),
    "qwen_32b": (45.2, 52.9, 1.9),
    "qwen_235b": (45.2, 52.7, 2.1),
    "llama_70b": (45.2, 40.4, 14.4),
    "deepseek": (45.2, 52.4, 2.4),
}

FILES = {
    "phi_4": "attack_defense_pie_phi_4.pdf",
    "qwen_30b": "attack_defense_pie_qwen_30b.pdf",
    "qwen_32b": "attack_defense_pie_qwen_32b.pdf",
    "qwen_235b": "attack_defense_pie_qwen_235b.pdf",
    "llama_70b": "attack_defense_pie_llama_70b.pdf",
    "deepseek": "attack_defense_pie_deepseek.pdf",
}

# Original cropped page is ~156 x 111 pt.  Keep the pie circular and let
# tight bbox + a small pad reproduce that landscape frame.
FIGSIZE = (2.40, 2.40)
FONTSIZE = 11


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 3,
            "ps.fonttype": 3,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def _autopct(value: float) -> str:
    return f"{value:.1f}%" if value > 0 else ""


def draw_pie(path: Path, sizes: tuple[float, float, float]) -> None:
    kept_sizes = []
    kept_colors = []
    for size, color in zip(sizes, COLORS):
        if size > 0:
            kept_sizes.append(size)
            kept_colors.append(color)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.pie(
        kept_sizes,
        colors=kept_colors,
        startangle=90,
        counterclock=True,
        autopct=_autopct,
        pctdistance=0.62,
        wedgeprops={"edgecolor": "white", "linewidth": 1.6},
        textprops={
            "color": "white",
            "weight": "bold",
            "fontsize": FONTSIZE,
            "fontfamily": "DejaVu Sans",
        },
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_legend(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 0.22))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = (0.02, 0.30, 0.58)
    for x, color, label in zip(xs, COLORS, LABELS):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.28),
                0.055,
                0.44,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            x + 0.07,
            0.50,
            label,
            va="center",
            ha="left",
            fontsize=11,
            fontfamily="DejaVu Sans",
            color="black",
        )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIG_DIR,
        help="Figure directory (default: user_scripts/figures)",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Also overwrite attack_defense_pie_legend.pdf",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _configure()

    for key, fname in FILES.items():
        dest = args.out_dir / fname
        draw_pie(dest, DATA[key])
        print(f"[pie] {dest.name}  {DATA[key]}")

    if args.legend:
        dest = args.out_dir / "attack_defense_pie_legend.pdf"
        draw_legend(dest)
        print(f"[legend] {dest.name}  {LABELS}")


if __name__ == "__main__":
    main()
