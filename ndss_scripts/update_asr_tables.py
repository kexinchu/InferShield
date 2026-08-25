#!/usr/bin/env python3
"""Dump Table 5 / Table 7 ASR cells from recovery summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDIR = REPO / "ndss_scripts" / "results" / "asr_recovery"

TABLE5_MODELS = ("phi4", "qwen30b", "qwen32b")
TABLE7_MODELS = ("phi4", "qwen30b", "qwen32b", "llama70b_awq", "ds_r1_qwen32b")
TEX_NAME = {
    "phi4": "Phi-4-14B",
    "qwen30b": "Qwen3-30B",
    "qwen32b": "Qwen3-32B",
    "llama70b_awq": "Llama-3.3-70B",
    "ds_r1_qwen32b": "DeepSeek-R1-32B",
}
BUDGETS = (0, 10, 50, 100, 150)


def load_cells() -> dict[tuple, dict]:
    cells: dict[tuple, dict] = {}
    for path in INDIR.glob("*_ttft.summary.json"):
        data = json.loads(path.read_text())
        if int(data.get("n_rec_trials") or 0) < 20:
            continue
        key = (
            data.get("model"),
            data.get("policy"),
            int(data.get("access_budget_B") or -1),
        )
        cells[key] = data
    return cells


def pick(cells: dict, model: str, policy: str, b: int = -1) -> dict | None:
    return cells.get((model, policy, b))


def fmt_tok(row: dict | None) -> str:
    if not row:
        return "---"
    return f"{row['tokens_recovered']}/{row['tokens_attempted']}"


def fmt_sec(row: dict | None) -> str:
    if not row:
        return "---"
    return f"{row['full_secrets']}/{row['n_rec_trials']}"


def dump(cells: dict) -> str:
    lines = ["=== Table 5  (3 models × SGLang / Strict / Balanced B=100) ==="]
    lines.append(f"{'Model':<18} {'Policy':<12} {'Tok ASR':>12} {'Sec ASR':>10}")
    for model in TABLE5_MODELS:
        for policy, b in (("vanilla", -1), ("strict", -1), ("autoshare", 100)):
            row = pick(cells, model, policy, b)
            label = {"vanilla": "SGLang", "strict": "Strict", "autoshare": "Balanced"}[
                policy
            ]
            extra = ""
            if row:
                extra = f"  ({100*row['asr_tok']:.1f}% tok, {100*row['asr_sec']:.1f}% sec)"
            lines.append(
                f"{TEX_NAME[model]:<18} {label:<12} {fmt_tok(row):>12} {fmt_sec(row):>10}{extra}"
            )
        lines.append("")
    lines.append("=== Table 7  (5 models × B) Token ASR / Secret ASR ===")
    header = f"{'Model':<18}" + "".join(f"{'B='+str(b):>14}" for b in BUDGETS)
    lines.append(header)
    for model in TABLE7_MODELS:
        bits = [f"{TEX_NAME[model]:<18}"]
        for b in BUDGETS:
            row = pick(cells, model, "autoshare", b)
            if b == 0 and row is None:
                row = pick(cells, model, "strict", -1)
            bits.append(f"{fmt_tok(row):>14}" if row else f"{'---':>14}")
        lines.append("".join(bits))
        bits = [f"{'':<18}"]
        for b in BUDGETS:
            row = pick(cells, model, "autoshare", b)
            if b == 0 and row is None:
                row = pick(cells, model, "strict", -1)
            bits.append(f"{fmt_sec(row):>14}" if row else f"{'---':>14}")
        lines.append("".join(bits) + "   (secret)")
    done5 = sum(
        1
        for m in TABLE5_MODELS
        for p, b in (("vanilla", -1), ("strict", -1), ("autoshare", 100))
        if pick(cells, m, p, b)
    )
    done7 = 0
    for m in TABLE7_MODELS:
        for b in BUDGETS:
            row = pick(cells, m, "autoshare", b)
            if b == 0 and row is None:
                row = pick(cells, m, "strict", -1)
            if row:
                done7 += 1
    lines.append(f"\nProgress: Table5 {done5}/9   Table7 {done7}/25")
    return "\n".join(lines)


def patch_table5(tex: str, cells: dict) -> str:
    """Replace Tok/Sec ASR placeholders for completed Table-5 cells."""
    # Rows are already written as 237/250 & 40/50 or ---. Patch per-model blocks.
    mapping = {
        ("qwen32b", "vanilla"): pick(cells, "qwen32b", "vanilla", -1),
        ("qwen32b", "strict"): pick(cells, "qwen32b", "strict", -1),
        ("qwen30b", "vanilla"): pick(cells, "qwen30b", "vanilla", -1),
        ("qwen30b", "strict"): pick(cells, "qwen30b", "strict", -1),
        ("phi4", "autoshare"): pick(cells, "phi4", "autoshare", 100),
        ("qwen32b", "autoshare"): pick(cells, "qwen32b", "autoshare", 100),
        ("qwen30b", "autoshare"): pick(cells, "qwen30b", "autoshare", 100),
    }
    # Qwen3-32B SGLang line
    row = mapping[("qwen32b", "vanilla")]
    if row:
        tex = re.sub(
            r"(Qwen3-32B\n & SGLang & 0\.74 & 0\.35 & 0\.39 & 0\.766\\ \[\.67, \.86\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    row = mapping[("qwen32b", "strict")]
    if row:
        tex = re.sub(
            r"(Qwen3-32B\n(?:.*\n){1} & Strict & 0\.51 & 0\.50 & 0\.01 & 0\.510\\ \[\.41, \.61\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    row = mapping[("qwen30b", "vanilla")]
    if row:
        tex = re.sub(
            r"(Qwen3-30B\n & SGLang & 0\.70 & 0\.38 & 0\.32 & 0\.724\\ \[\.63, \.82\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    row = mapping[("qwen30b", "strict")]
    if row:
        tex = re.sub(
            r"(Qwen3-30B\n(?:.*\n){1} & Strict & 0\.50 & 0\.50 & 0\.00 & 0\.505\\ \[\.40, \.61\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    row = mapping[("qwen30b", "autoshare")]
    if row:
        tex = re.sub(
            r"(Qwen3-30B\n(?:.*\n){2} & Balanced & 0\.52 & 0\.49 & 0\.03 & 0\.545\\ \[\.44, \.65\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    row = mapping[("qwen32b", "autoshare")]
    if row:
        tex = re.sub(
            r"(Qwen3-32B\n(?:.*\n){2} & Balanced & 0\.56 & 0\.49 & 0\.04 & 0\.552\\ \[\.45, \.65\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    row = mapping[("phi4", "autoshare")]
    if row:
        tex = re.sub(
            r"(Phi-4-14B\n(?:.*\n){2} & Balanced & 0\.54 & 0\.48 & 0\.06 & 0\.560\\ \[\.46, \.66\] &) --- & ---",
            rf"\1 {fmt_tok(row)} & {fmt_sec(row)}",
            tex,
            count=1,
        )
    return tex


def patch_table7(tex: str, cells: dict) -> str:
    """Rewrite tab:budget-operating-points as ASR vs B when enough cells exist."""
    rows_tex = []
    any_real = False
    for model in TABLE7_MODELS:
        first = True
        for b in BUDGETS:
            row = pick(cells, model, "autoshare", b)
            if b == 0 and row is None:
                row = pick(cells, model, "strict", -1)
            if row:
                any_real = True
            name = TEX_NAME[model] if first else ""
            first = False
            rows_tex.append(
                f"{name} & {b} & {fmt_tok(row)} & {fmt_sec(row)} \\\\"
            )
        rows_tex.append("\\midrule")
    if rows_tex:
        rows_tex.pop()  # last midrule
    if not any_real:
        return tex
    block = "\n".join(rows_tex)
    new_table = f"""\\begin{{tabular}}{{llrr}}
\\toprule
\\textbf{{Model}} & $\\boldsymbol{{B}}$ &
  \\textbf{{Tok.\\ ASR}} & \\textbf{{Sec.\\ ASR}} \\\\
\\midrule
{block}
\\bottomrule
\\end{{tabular}}"""
    tex, n = re.subn(
        r"\\begin\{tabular\}\{llrr\}\n\\toprule\n\\textbf\{Model\} & \$\\boldsymbol\{B\}\$.*?\\end\{tabular\}",
        new_table.replace("\\", r"\\"),
        tex,
        count=1,
        flags=re.S,
    )
    # re.subn with replace('\\','\\\\') is error-prone; do a simpler splice
    return tex if n else tex


def main() -> None:
    cells = load_cells()
    print(dump(cells))


if __name__ == "__main__":
    main()
