#!/usr/bin/env python3
"""Print the 7-point reuse-B grid from measured cells."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "reuse_risk_curve"
BUDGETS = (0, 10, 25, 50, 75, 100, 150)
MODELS = (
    ("phi4", ROOT / "cells", "Phi-4-14B"),
    ("qwen30b", ROOT / "cells", "Qwen3-30B-A3B"),
    ("qwen32b", ROOT / "cells", "Qwen3-32B"),
    ("ds_r1_qwen32b", ROOT / "cells_new_models", "DeepSeek-R1-Distill-Qwen-32B"),
    ("llama70b_awq", ROOT / "cells_new_models", "Llama-3.3-70B-AWQ"),
)


def load_cell(folder: Path, model: str, B: int):
    path = folder / f"{model}_B{B}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt_rate(cell) -> str:
    if cell is None:
        return "  —   "
    rate = cell.get("reuse_hit_rate")
    if rate is None:
        return "  ?   "
    return f"{rate:6.3f}"


def main() -> None:
    print("reuse_hit_rate  (19 cross-user probes, experiment_autoshare)")
    header = f"{'model':<28}" + "".join(f"{'B='+str(B):>8}" for B in BUDGETS)
    print(header)
    print("-" * len(header))
    for key, folder, label in MODELS:
        cells = {B: load_cell(folder, key, B) for B in BUDGETS}
        line = f"{label:<28}" + "".join(fmt_rate(cells[B]) for B in BUDGETS)
        print(line)
    print()
    print("same_user_cached / visibility / n_measured")
    for key, folder, label in MODELS:
        print(f"\n{label}")
        for B in BUDGETS:
            cell = load_cell(folder, key, B)
            if cell is None:
                print(f"  B={B:<3}  MISSING")
                continue
            vis = cell.get("visibility_counts") or {}
            print(
                f"  B={B:<3}  reuse={cell.get('reuse_hit_rate'):.4f}  "
                f"same_user={cell.get('same_user_cached_tokens')}  "
                f"cached={cell.get('cached_tokens')}/{cell.get('prompt_tokens')}  "
                f"vis={vis}  ttft={cell.get('ttft_s'):.3f}s"
            )


if __name__ == "__main__":
    main()
