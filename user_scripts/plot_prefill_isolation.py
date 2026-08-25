#!/usr/bin/env python3
"""Redraw fig:compare — SGLang vs User-Cache-Isolation TTFT.

Recovers the original prefill-llama-{13,70}b.pdf geometry and writes a
standalone legend with the current paper name (User-Cache-Isolation).

    python user_scripts/plot_prefill_isolation.py
    python user_scripts/plot_prefill_isolation.py --legend-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "user_scripts" / "figures"

# Sampled from the original PDFs.
C_SGLANG = (1.0, 0.6470588, 0.0)  # orange
C_ISOLATION = (0.0, 0.5, 0.0)  # dark green
C_EDGE = (0.502, 0.502, 0.502)
C_GRID = (0.690, 0.690, 0.690)

LABEL_SGLANG = "SGLang"
LABEL_ISOLATION = "User-Cache-Isolation"
WORKLOADS = ("ShareGPT", "Multiturn", "Multitask")

# TTFT relative to SGLang (=100). Heights recovered from the original
# PDFs; the isolation overheads are 2.3/3.2/8.9% on 13B and
# 29.8/8.4/38.9% on 70B (paper range 2.3--8.9% / 8.3--38.9%).
DATA = {
    "13b": {
        "file": "prefill-llama-13b.pdf",
        "sglang": (100.0, 100.0, 100.0),
        "isolation": (102.3, 103.2, 108.9),
        "ylim": (0.0, 115.0),
    },
    "70b": {
        "file": "prefill-llama-70b.pdf",
        "sglang": (100.0, 100.0, 100.0),
        "isolation": (129.8, 108.4, 138.9),
        "ylim": (0.0, 146.0),
    },
}

BAR_FIGSIZE = (207.36 / 72.0, 103.68 / 72.0)  # original crop, inches
LEGEND_FIGSIZE = (5.4, 0.18)


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def draw_bars(path: Path, spec: dict) -> None:
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    x = range(len(WORKLOADS))
    width = 0.32
    ax.bar(
        [i - width / 2 for i in x],
        spec["sglang"],
        width=width,
        color=C_SGLANG,
        edgecolor=C_EDGE,
        linewidth=1.0,
        zorder=3,
    )
    ax.bar(
        [i + width / 2 for i in x],
        spec["isolation"],
        width=width,
        color=C_ISOLATION,
        edgecolor=C_EDGE,
        linewidth=1.0,
        zorder=3,
    )
    ax.axhline(100.0, color=C_GRID, ls="--", lw=0.8, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(WORKLOADS)
    ax.set_ylabel("TTFT Compare (%)")
    ax.set_ylim(*spec["ylim"])
    ax.set_yticks([0, 100])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="x", length=3.5, width=0.8)
    ax.tick_params(axis="y", length=3.5, width=0.8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_legend(path: Path) -> None:
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    items = (
        (0.18, C_SGLANG, LABEL_SGLANG),
        (0.42, C_ISOLATION, LABEL_ISOLATION),
    )
    for x, color, label in items:
        ax.add_patch(
            Rectangle(
                (x, 0.28),
                0.045,
                0.44,
                facecolor=color,
                edgecolor=C_EDGE,
                linewidth=1.0,
            )
        )
        ax.text(
            x + 0.055,
            0.50,
            label,
            va="center",
            ha="left",
            fontsize=10,
            fontfamily="DejaVu Sans",
            color="black",
        )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="Only overwrite prefill-llama-13b-legend.pdf",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _configure()

    dest = args.out_dir / "prefill-llama-13b-legend.pdf"
    draw_legend(dest)
    print(f"[legend] {dest.name}  {LABEL_SGLANG} / {LABEL_ISOLATION}")

    if args.legend_only:
        return

    for key, spec in DATA.items():
        dest = args.out_dir / spec["file"]
        draw_bars(dest, spec)
        print(f"[bars]   {dest.name}  isolation={spec['isolation']}")


if __name__ == "__main__":
    main()
