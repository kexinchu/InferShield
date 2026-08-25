#!/usr/bin/env python3
"""PromptPeek-aligned token-recovery ASR.

ASR_tok = recovered tokens / (trials × 5), TTFT-only (no cached_tokens).
PromptPeek reports 99 / 98 / 95% at 148–306 req/token.

    python user_scripts/plot_table7_asr.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "user_scripts" / "results" / "asr_recovery"
DEFAULT_OUT = REPO / "user_scripts" / "figures" / "table7_asr.pdf"
PROMPTPEEK = 0.95
B_STAR = 100

MODEL_STYLE = {
    "phi4": {"label": "Phi-4-14B", "color": "#3182bd", "marker": "o"},
    "qwen30b": {"label": "Qwen3-30B", "color": "#31a354", "marker": "s"},
    "qwen32b": {"label": "Qwen3-32B", "color": "#e6550d", "marker": "D"},
    "llama70b_awq": {"label": "Llama-3.3-70B", "color": "#756bb1", "marker": "^"},
    "ds_r1_qwen32b": {"label": "DeepSeek-R1-32B", "color": "#636363", "marker": "v"},
}
POLICY_ORDER = ("vanilla", "strict", "autoshare")
POLICY_LABEL = {
    "vanilla": "SGLang",
    "strict": "Strict",
    "autoshare": r"Balanced $B{=}100$",
}
BUDGETS = (0, 10, 50, 100, 150)


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


def load_summaries(indir: Path) -> list[dict]:
    rows = []
    for path in sorted(indir.glob("*_ttft.summary.json")):
        data = json.loads(path.read_text())
        if int(data.get("n_rec_trials") or 0) < 20:
            continue
        rows.append(data)
    return rows


def plot(rows: list[dict], out_path: Path) -> None:
    _configure()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)

    ax = axes[0]
    models = [k for k in MODEL_STYLE if any(r.get("model") == k for r in rows)]
    width = 0.22
    x0 = list(range(len(models)))
    for i, policy in enumerate(POLICY_ORDER):
        xs = [x + (i - 1) * width for x in x0]
        ys = []
        for model in models:
            hit = [
                r
                for r in rows
                if r.get("model") == model
                and r.get("policy") == policy
                and (
                    policy != "autoshare"
                    or int(r.get("access_budget_B") or -1) in (100, -1)
                )
            ]
            ys.append(hit[-1]["asr_tok"] if hit else None)
        ax.bar(
            xs,
            [0 if y is None else y for y in ys],
            width=width,
            color=["#9ecae1", "#a1d99b", "#fdae6b"][i],
            edgecolor="white",
            label=POLICY_LABEL[policy],
        )
    ax.axhline(PROMPTPEEK, color="#636363", ls="--", lw=0.9)
    ax.axhline(1.0 / 6.0, color="#969696", ls=":", lw=0.8)
    ax.set_xticks(x0)
    ax.set_xticklabels([MODEL_STYLE[m]["label"] for m in models] or ["(pending)"])
    ax.set_ylabel(r"Token ASR")
    ax.set_title("(a) Recovery vs policy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.axhline(PROMPTPEEK, color="#636363", ls="--", lw=0.9, zorder=1)
    ax.axhline(1.0 / 6.0, color="#969696", ls=":", lw=0.8, zorder=1)
    ax.axvline(B_STAR, color="#636363", ls="--", lw=0.9, zorder=1)
    plotted = False
    for key, style in MODEL_STYLE.items():
        cells = {}
        for r in rows:
            if r.get("model") != key:
                continue
            pol = r.get("policy")
            if pol == "strict":
                cells[0] = r["asr_tok"]
            elif pol == "vanilla":
                cells["sg"] = r["asr_tok"]
            elif pol in ("autoshare", "balanced"):
                cells[int(r.get("access_budget_B") or 100)] = r["asr_tok"]
        xs = [B for B in BUDGETS if B in cells]
        if not xs:
            continue
        plotted = True
        ax.plot(
            xs,
            [cells[B] for B in xs],
            color=style["color"],
            marker=style["marker"],
            ms=4.5,
            lw=1.2,
            label=style["label"],
        )
    ax.set_xlabel(r"Access budget $B$")
    ax.set_ylabel(r"Token ASR")
    ax.set_title(r"(b) Residual $B$ window")
    ax.set_xticks(list(BUDGETS))
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", ls=":", lw=0.5, color="#d9d9d9")
    ax.set_axisbelow(True)
    if not plotted:
        ax.text(0.5, 0.5, "B-sweep pending", ha="center", va="center", transform=ax.transAxes)

    handles = [
        Patch(facecolor=c, edgecolor="none", label=lab)
        for c, lab in zip(
            ["#9ecae1", "#a1d99b", "#fdae6b"],
            [POLICY_LABEL[p] for p in POLICY_ORDER],
        )
    ]
    handles.append(Line2D([0], [0], color="#636363", ls="--", lw=0.9, label="PromptPeek 95%"))
    handles.append(Line2D([0], [0], color="#969696", ls=":", lw=0.8, label=r"Chance $1/k$"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.16),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[wrote] {out_path}  cells={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(load_summaries(args.indir), args.output)


if __name__ == "__main__":
    main()
