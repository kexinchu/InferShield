#!/usr/bin/env python3
"""Plot the SafeKV reuse-risk curve from reuse_risk_curve.json.

Left: membership AUC vs B (complete; tab:budget-operating-points).
Right: serving reuse proxy vs B.  Uses reuse_hit_rate when present,
otherwise TTFT from tab:hyperparam (incomplete grid).

    python ndss_scripts/plot_reuse_risk_curve.py

Writes ndss_scripts/figures/reuse_risk_curve.pdf.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = (
    REPO / "ndss_scripts" / "results" / "reuse_risk_curve" / "reuse_risk_curve.json"
)
DEFAULT_OUT = REPO / "ndss_scripts" / "figures" / "reuse_risk_curve.pdf"

MODEL_STYLE = {
    "phi4": {"label": "Phi-4-14B", "color": "#3182bd", "marker": "o"},
    "qwen30b": {"label": "Qwen3-30B", "color": "#31a354", "marker": "s"},
    "qwen32b": {"label": "Qwen3-32B", "color": "#e6550d", "marker": "D"},
}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def _group(rows, key="model"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    for model in grouped:
        grouped[model].sort(key=lambda r: r["B"])
    return grouped


def plot(data: dict, out_path: Path) -> None:
    _configure()
    b_star = int(data.get("recommended_B_star", 100))
    membership = _group(data["membership"])
    serving = _group(data["serving"])
    sglang_auc = data.get("sglang_auc", {})

    reuse_ready = any(row.get("reuse_hit_rate") is not None for row in data["serving"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)

    ax = axes[0]
    sglang_vals = [sglang_auc[m] for m in MODEL_STYLE if m in sglang_auc]
    if sglang_vals:
        ax.axhspan(
            min(sglang_vals),
            max(sglang_vals),
            color="#bdbdbd",
            alpha=0.35,
            zorder=0,
        )
    ax.axvline(b_star, color="#636363", ls="--", lw=0.9, zorder=1)
    for model, style in MODEL_STYLE.items():
        rows = membership.get(model, [])
        if not rows:
            continue
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
    ax.set_ylim(0.45, 0.90)
    ax.set_xticks([0, 10, 50, 100, 150])
    ax.axhline(0.5, color="#969696", ls=":", lw=0.8, zorder=1)
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.axvline(b_star, color="#636363", ls="--", lw=0.9, zorder=1)
    for model, style in MODEL_STYLE.items():
        rows = serving.get(model, [])
        if reuse_ready:
            plotted = [r for r in rows if r.get("reuse_hit_rate") is not None]
            xs = [r["B"] for r in plotted]
            ys = [100.0 * float(r["reuse_hit_rate"]) for r in plotted]
        else:
            plotted = [r for r in rows if r.get("ttft_s") is not None]
            xs = [r["B"] for r in plotted]
            ys = [1000.0 * float(r["ttft_s"]) for r in plotted]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            ms=4.5,
            lw=1.2,
            label=style["label"],
        )
    ax.set_xlabel(r"Access budget $B$")
    if reuse_ready:
        ax.set_ylabel("Reuse hit rate (%)")
        ax.set_title("(b) Reuse")
    else:
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("(b) Serving cost (TTFT proxy)")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

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
    if sglang_vals:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#bdbdbd",
                lw=6,
                alpha=0.7,
                label="SGLang AUC range",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[wrote] {out_path}")
    if not reuse_ready:
        print(
            "[note] reuse_hit_rate is empty; panel (b) uses tab:hyperparam TTFT "
            "(B in {50,100,200} only)."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    plot(data, args.output)


if __name__ == "__main__":
    main()
