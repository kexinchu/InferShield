#!/usr/bin/env python3
"""Detector-Window mechanism accounting for fig:defense_acc.

For each sensitive prefix, assign the first mechanism that keeps it Private:
  Tier 1  – regex/trie
  Tier 2  – Piiranha + Llama-3.2-1B, including the runtime uncertain band
  Budget  – residual false negative (would consume Balanced B then demote)

The serving path detects a chat-template slice (private_client.update_privacy),
so each backbone is scored on its own tokenizer template.

Writes CSV/JSON/PDF under --out-dir.  Does not overwrite paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.managers.private_service.privacy_detector_custom import (  # noqa: E402
    PrivacyDetector,
)
from sglang.srt.managers.private_service.privacy_detector_piiranha import (  # noqa: E402
    PiiPrivacyDetector,
)

# Match PrivateJudgeService uncertain band.
LOW_QUALITY_THRESHOLD = 0.3
HIGH_QUALITY_THRESHOLD = 0.7

TIER1_CONFIG = (
    REPO_ROOT
    / "python/sglang/srt/managers/private_service/privacy_patterns_config.json"
)
PII_PATH = REPO_ROOT / "datasets" / "english_pii_43k.jsonl"
PIIRANHA_HF = os.environ.get(
    "SAFEKV_PIIRANHA",
    "/workspace/Models/piiranha-v1-detect-personal-information",
)
LLAMA_HF = os.environ.get(
    "SAFEKV_LLAMA_DETECTOR",
    "/workspace/Models/Llama-3.2-1B-Instruct",
)

MODELS = {
    "Phi-4-14B": "/workspace/Models/Phi-4",
    "Qwen3-30B-A3B": "/workspace/Models/Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-32B": "/workspace/Models/Qwen3-32B",
    # nvidia/Llama-3.3-70B-Instruct-FP8 tokenizer (ungated). Same Llama 3
    # Instruct template as the official 70B; weights are not needed here.
    "Llama-3.3-70B-Instruct-FP8": "/workspace/Models/Llama-3.3-70B-Instruct-FP8",
    # Paper 235B stand-in: official Qwen3-235B-A22B tokenizer. Identical
    # chat template to local Qwen3-32B (the A6000 distill proxy).
    "Qwen3-235B-A22B": "/workspace/Models/Qwen3-235B-A22B",
    # Official DeepSeek-R1 distill onto Qwen2.5-32B.
    "DeepSeek-R1-Distill-Qwen-32B": "/workspace/Models/DeepSeek-R1-Distill-Qwen-32B",
}


def extract_detect_text(prompt: str, source_text: str) -> str:
    """Line that still contains the user prefix, else the source text.

    The serving client takes ``lines[-2]``, which is empty on Llama-3
    Instruct templates. For mechanism accounting we score the user prefix.
    """
    if source_text:
        needle = source_text[:48]
        for ln in reversed((prompt or "").split("\n")):
            if needle and needle in ln:
                return ln
    return source_text or ""


def load_positives(path: Path, n: int, min_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    cands = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            src = (d.get("source_text") or "").strip()
            if src and (d.get("privacy_mask") or []) and len(src) >= min_len:
                cands.append(src)
    rng.shuffle(cands)
    if len(cands) < n:
        raise RuntimeError(f"Need {n} PII rows, have {len(cands)}")
    return cands[:n]


def format_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return text


def runtime_tier2_private(result) -> bool:
    """Mirror PrivateJudgeService._process_second_level_tasks."""
    if LOW_QUALITY_THRESHOLD < result.confidence < HIGH_QUALITY_THRESHOLD:
        return True
    return bool(result.is_private)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "user_scripts" / "results" / "detector_window_rerun"),
    )
    p.add_argument(
        "--slice",
        choices=("template", "source"),
        default="template",
        help="template: chat-template line containing the user prefix; "
        "source: raw PII prefix (same for all backbones).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    texts = load_positives(PII_PATH, args.n, args.min_len, args.seed)
    print(f"[data] n={len(texts)} seed={args.seed}", flush=True)

    print("[tier1] loading", flush=True)
    tier1 = PrivacyDetector(config_file=str(TIER1_CONFIG))
    print("[tier2] loading Piiranha + Llama-3.2-1B", flush=True)
    t0 = time.time()
    tier2 = PiiPrivacyDetector(pii_model_name=PIIRANHA_HF, gene_model_name=LLAMA_HF)
    print(f"[tier2] loaded in {time.time() - t0:.1f}s", flush=True)

    from transformers import AutoTokenizer

    rows = []
    per_model = {}

    cached_source_rec = None
    for model in args.models:
        tok_path = MODELS[model]
        print(f"[model] {model} tokenizer={tok_path} slice={args.slice}", flush=True)
        if args.slice == "source":
            detect_texts = list(texts)
            if cached_source_rec is not None:
                rec = dict(cached_source_rec)
                rec["model"] = model
                rec["tokenizer"] = tok_path
                rec["slice"] = "source"
                rows.append(rec)
                per_model[model] = rec
                print(
                    f"[window] {model}: T1={rec['tier1_pct']:.1f}%  "
                    f"T2={rec['tier2_pct']:.1f}%  budget={rec['budget_pct']:.1f}%  "
                    f"detector_R={rec['detector_recall_pct']:.1f}%  "
                    f"coverage={rec['protocol_coverage_pct']:.1f}%  (copied)",
                    flush=True,
                )
                continue
        else:
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            detect_texts = [
                extract_detect_text(format_prompt(tokenizer, t), t) for t in texts
            ]
        t1_flags = [bool(tier1.detect_privacy(dt).is_private) for dt in detect_texts]
        leftover_idx = [i for i, f in enumerate(t1_flags) if not f]
        t2_flags = [False] * len(texts)
        for start in range(0, len(leftover_idx), args.batch_size):
            batch_i = leftover_idx[start : start + args.batch_size]
            chunk = [detect_texts[i] for i in batch_i]
            results = tier2.detect_privacy(chunk)
            for j, r in enumerate(results):
                t2_flags[batch_i[j]] = runtime_tier2_private(r)

        n_t1 = sum(t1_flags)
        n_t2 = sum(1 for a, b in zip(t1_flags, t2_flags) if (not a) and b)
        n_fn = len(texts) - n_t1 - n_t2
        coverage = 100.0 * (n_t1 + n_t2 + n_fn) / len(texts)
        # Budget covers every residual FN under B < Q, so protocol coverage is 100%.
        rec = {
            "model": model,
            "n": len(texts),
            "tier1": n_t1,
            "tier2": n_t2,
            "budget_fn": n_fn,
            "tier1_pct": 100.0 * n_t1 / len(texts),
            "tier2_pct": 100.0 * n_t2 / len(texts),
            "budget_pct": 100.0 * n_fn / len(texts),
            "detector_recall_pct": 100.0 * (n_t1 + n_t2) / len(texts),
            "protocol_coverage_pct": coverage,
            "tokenizer": tok_path,
            "slice": args.slice,
            "sample_detect_text": detect_texts[0][:160],
        }
        if args.slice == "source":
            cached_source_rec = rec
        per_model[model] = rec
        rows.append(rec)
        print(
            f"[window] {model}: T1={rec['tier1_pct']:.1f}%  "
            f"T2={rec['tier2_pct']:.1f}%  budget={rec['budget_pct']:.1f}%  "
            f"detector_R={rec['detector_recall_pct']:.1f}%  "
            f"coverage={rec['protocol_coverage_pct']:.1f}%",
            flush=True,
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"detector_window_{ts}.csv")
    json_path = os.path.join(args.out_dir, f"detector_window_{ts}.json")
    pdf_path = os.path.join(args.out_dir, f"attack_defense_average_{ts}.pdf")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "n",
                "tier1",
                "tier2",
                "budget_fn",
                "tier1_pct",
                "tier2_pct",
                "budget_pct",
                "detector_recall_pct",
                "protocol_coverage_pct",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "models": per_model}, f, indent=2)

    labels = [r["model"].replace("-Instruct", "") for r in rows]
    t1 = np.array([r["tier1_pct"] for r in rows])
    t2 = np.array([r["tier2_pct"] for r in rows])
    bd = np.array([r["budget_pct"] for r in rows])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x, t1, color="#2c7bb6", label="Tier 1 (pattern)")
    ax.bar(x, t2, bottom=t1, color="#abd9e9", label="Tier 2 (semantic)")
    ax.bar(x, bd, bottom=t1 + t2, color="#fdae61", label="Budget exhaustion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mitigation coverage (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Detector-Window mechanism accounting (security-first)")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[csv] {csv_path}", flush=True)
    print(f"[json] {json_path}", flush=True)
    print(f"[pdf] {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
