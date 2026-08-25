#!/usr/bin/env python3
"""Plot the repeated-probe Table-7 figure.

Left: mean cumulative TTFT hits vs probe index (one line per B).
Right: mean hits in k probes vs B (member vs control).

    python ndss_scripts/plot_table7_repeated_probe.py \
        --indir ndss_scripts/results/table7_repeated_probe \
        --output ndss_scripts/figures/table7_repeated_probe.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "ndss_scripts" / "results" / "table7_repeated_probe"
DEFAULT_OUT = REPO / "ndss_scripts" / "figures" / "table7_repeated_probe.pdf"

B_COLORS = {
    0: "#636363",
    10: "#3182bd",
    25: "#31a354",
    50: "#e6550d",
    75: "#756bb1",
    100: "#e7298a",
    150: "#8c6d31",
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


def load_cells(indir: Path) -> list:
    rows = []
    for path in sorted(indir.glob("*_B*_repeated.json")):
        data = json.loads(path.read_text())
        rows.append(data)
    rows.sort(key=lambda d: int(d["B"]))
    return rows


def plot(cells: list, out_path: Path) -> None:
    if not cells:
        raise SystemExit(f"no *_B*_repeated.json cells")
    _configure()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)

    ax = axes[0]
    k = int(cells[0]["k"])
    xs = list(range(k))
    for cell in cells:
        B = int(cell["B"])
        ys = cell["summary"]["cum_ttft_hits_member"]
        ax.plot(
            xs,
            ys,
            color=B_COLORS.get(B, "#333333"),
            lw=1.2,
            label=rf"$B={B}$",
        )
        ax.axhline(B if B > 0 else 0, color=B_COLORS.get(B, "#333333"), ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("Probe index on the same prefix")
    ax.set_ylabel("Cumulative TTFT hits")
    ax.set_title("(a) Side-channel hits")
    ax.set_xlim(0, k - 1)
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

    ax = axes[1]
    Bs = [int(c["B"]) for c in cells]
    member = [c["summary"]["mean_ttft_hits_member"] for c in cells]
    control = [c["summary"]["mean_ttft_hits_control"] for c in cells]
    cache = [c["summary"]["mean_cache_hits_member"] for c in cells]
    ax.plot(Bs, member, color="#3182bd", marker="o", ms=4.5, lw=1.2, label="TTFT hits (member)")
    ax.plot(Bs, control, color="#969696", marker="s", ms=4.5, lw=1.2, label="TTFT hits (control)")
    ax.plot(Bs, cache, color="#31a354", marker="D", ms=4.5, lw=1.2, label="KV hits (member)")
    ax.plot(Bs, Bs, color="#bdbdbd", ls="--", lw=0.9, label=r"ledger cap $B$")
    ax.set_xlabel(r"Access budget $B$")
    ax.set_ylabel(rf"Hits in $k={k}$ probes")
    ax.set_title("(b) Hits vs $B$")
    ax.set_xticks(Bs)
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], color=B_COLORS[int(c["B"])], lw=1.2, label=rf"$B={c['B']}$")
        for c in cells
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(7, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.14),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(load_cells(args.indir), args.output)


if __name__ == "__main__":
    main()
