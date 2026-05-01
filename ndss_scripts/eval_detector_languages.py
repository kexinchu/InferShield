#!/usr/bin/env python3
"""
Per-language detector evaluation for W4 reviewer table.

For each of {en, fr, de, it}:
  - 500 positives = source_text (has at least one PII span)
  - 500 negatives = source_text with all privacy_mask spans excised (default)
                    OR human turns from ShareGPT (--sharegpt-neg, English only)
Both Tier 1 (Pattern) and Tier 2 (Piiranha + Llama-3.2-1B) are run on every
sample. Reported metrics: Recall, Precision, FPR.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# Make the in-repo sglang importable so we get our patched detectors,
# not the .venv copy.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.managers.private_service.privacy_detector_custom import PrivacyDetector
from sglang.srt.managers.private_service.privacy_detector_piiranha import PiiPrivacyDetector


DATASETS = {
    "en": REPO_ROOT / "datasets" / "english_pii_43k.jsonl",
    "fr": REPO_ROOT / "datasets" / "french_pii_62k.jsonl",
    "de": REPO_ROOT / "datasets" / "german_pii_52k.jsonl",
    "it": REPO_ROOT / "datasets" / "italian_pii_50k.jsonl",
}

TIER1_CONFIG = REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "private_service" / "privacy_patterns_config.json"

PIIRANHA_HF = "/home/kec23008/Models/piiranha-v1"
LLAMA_HF = "/home/kec23008/Models/Llama-3.2-1B-Instruct"

SHAREGPT_PATH = REPO_ROOT / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json"

NEG_SOURCES = ("excised", "sharegpt", "agnews", "wikitext")


def load_sharegpt_negatives(json_path: Path, n: int, min_len: int, seed: int) -> list:
    """Extract human turns from ShareGPT as natural non-PII negatives."""
    with open(json_path) as f:
        data = json.load(f)
    texts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human":
                t = turn.get("value", "").strip()
                if len(t) >= min_len:
                    texts.append(t)
    rng = random.Random(seed)
    rng.shuffle(texts)
    if len(texts) < n:
        raise RuntimeError(f"Not enough ShareGPT turns: have {len(texts)}, need {n}")
    return texts[:n]


def load_hf_negatives(hf_name: str, hf_config: str, split: str,
                      text_field: str, n: int, min_len: int, seed: int) -> list:
    """Load negatives from a HuggingFace dataset (AG News / Wikitext etc.)."""
    from datasets import load_dataset
    ds = load_dataset(hf_name, hf_config, split=split, trust_remote_code=True)
    texts = [ex[text_field].strip() for ex in ds if len(ex[text_field].strip()) >= min_len]
    rng = random.Random(seed)
    rng.shuffle(texts)
    if len(texts) < n:
        raise RuntimeError(f"Not enough {hf_name} samples: have {len(texts)}, need {n}")
    return texts[:n]


def make_negative(source_text: str, privacy_mask: list) -> str:
    """Excise all PII spans from source_text. Returns the residual (non-PII) text."""
    spans = sorted(
        [(m["start"], m["end"]) for m in privacy_mask if m.get("start") is not None],
        reverse=True,
    )
    s = source_text
    for start, end in spans:
        s = s[:start] + s[end:]
    # Collapse whitespace artifacts left behind
    s = " ".join(s.split())
    return s


def load_positives(jsonl_path: Path, n: int, min_text_len: int, seed: int) -> list:
    """Load N positive samples (texts with PII) from a PII-masked JSONL file."""
    rng = random.Random(seed)
    candidates = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            src = d.get("source_text", "")
            mask = d.get("privacy_mask") or []
            if src and mask and len(src) >= min_text_len:
                candidates.append(src)
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise RuntimeError(
            f"Not enough usable rows in {jsonl_path}: have {len(candidates)}, need {n}"
        )
    return candidates[:n]


def load_samples(jsonl_path: Path, n_pos: int, n_neg: int, min_text_len: int, seed: int):
    """Load N positives and N negatives (excised) from a PII-masked JSONL file."""
    rng = random.Random(seed)
    candidates = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            src = d.get("source_text", "")
            mask = d.get("privacy_mask") or []
            if not src or not mask:
                continue
            neg_text = make_negative(src, mask)
            if len(src) < min_text_len or len(neg_text) < min_text_len:
                continue
            candidates.append((src, neg_text))

    rng.shuffle(candidates)
    if len(candidates) < max(n_pos, n_neg):
        raise RuntimeError(
            f"Not enough usable rows in {jsonl_path}: have {len(candidates)}, "
            f"need >= {max(n_pos, n_neg)}"
        )
    pos = [c[0] for c in candidates[:n_pos]]
    neg = [c[1] for c in candidates[n_pos : n_pos + n_neg]]
    return pos, neg


def confusion(preds: list, labels: list) -> dict:
    tp = fp = tn = fn = 0
    for p, y in zip(preds, labels):
        if y and p:
            tp += 1
        elif y and not p:
            fn += 1
        elif (not y) and p:
            fp += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "precision": precision, "fpr": fpr,
    }


def run_tier1(detector: PrivacyDetector, texts: list) -> list:
    out = []
    for t in texts:
        r = detector.detect_privacy(t)
        out.append(bool(r.is_private))
    return out


def run_tier2(detector: PiiPrivacyDetector, texts: list, batch_size: int,
              tier1_preds: list = None) -> list:
    """Run Piiranha+Llama. If `tier1_preds` is given, skip samples Tier 1 already
    flagged (cascade mode) — the cascade prediction for those is True regardless.
    Returns a list of booleans aligned with `texts`."""
    n = len(texts)
    out = [False] * n

    if tier1_preds is None:
        # Standalone: run Piiranha+Llama on every sample
        idxs_to_run = list(range(n))
    else:
        # Cascade: skip samples that Tier 1 already flagged → cascade pred is True for them
        for i, t1 in enumerate(tier1_preds):
            if t1:
                out[i] = True
        idxs_to_run = [i for i, t1 in enumerate(tier1_preds) if not t1]

    for batch_start in range(0, len(idxs_to_run), batch_size):
        batch_idxs = idxs_to_run[batch_start : batch_start + batch_size]
        chunk = [texts[i] for i in batch_idxs]
        results = detector.detect_privacy(chunk)
        for j, r in enumerate(results):
            out[batch_idxs[j]] = bool(r.is_private)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pos", type=int, default=500)
    p.add_argument("--n-neg", type=int, default=500)
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--langs", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--skip-tier1", action="store_true")
    p.add_argument("--skip-tier2", action="store_true")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "ndss_scripts" / "logs"))
    p.add_argument("--neg-source", choices=NEG_SOURCES, default="excised",
                   help="Where to draw negative samples from. "
                        "Non-excised sources are English-only.")
    p.add_argument("--sharegpt-path", default=str(SHAREGPT_PATH))
    args = p.parse_args()

    # Non-English-language datasets only make sense with excised negatives.
    if args.neg_source != "excised" and set(args.langs) - {"en"}:
        print(f"[warn] --neg-source={args.neg_source} forces --langs en", flush=True)
        args.langs = ["en"]

    print(f"[config] n_pos={args.n_pos} n_neg={args.n_neg} langs={args.langs} "
          f"neg_source={args.neg_source}", flush=True)

    # ---- Build negative pool (once, reused for all langs when not excised) ----
    external_neg = None
    if args.neg_source == "sharegpt":
        external_neg = load_sharegpt_negatives(
            Path(args.sharegpt_path), args.n_neg, args.min_len, args.seed
        )
        print(f"[data] neg pool: {len(external_neg)} texts from ShareGPT", flush=True)
    elif args.neg_source == "agnews":
        external_neg = load_hf_negatives(
            "ag_news", None, "train", "text", args.n_neg, args.min_len, args.seed
        )
        print(f"[data] neg pool: {len(external_neg)} texts from AG News", flush=True)
    elif args.neg_source == "wikitext":
        external_neg = load_hf_negatives(
            "wikitext", "wikitext-103-raw-v1", "train", "text",
            args.n_neg, args.min_len, args.seed
        )
        print(f"[data] neg pool: {len(external_neg)} texts from Wikitext-103", flush=True)

    # ---- Pre-sample everything before loading models ----
    samples = {}  # lang -> {"texts": [...], "labels": [1/0...]}
    if external_neg is not None:
        pos = load_positives(DATASETS["en"], args.n_pos, args.min_len, args.seed)
        texts = pos + external_neg
        labels = [1] * len(pos) + [0] * len(external_neg)
        samples["en"] = {"texts": texts, "labels": labels}
        print(f"[data] en: pos={len(pos)} neg={len(external_neg)}", flush=True)
    else:
        for lang in args.langs:
            path = DATASETS[lang]
            pos, neg = load_samples(path, args.n_pos, args.n_neg, args.min_len, args.seed)
            texts = pos + neg
            labels = [1] * len(pos) + [0] * len(neg)
            samples[lang] = {"texts": texts, "labels": labels}
            print(f"[data] {lang}: pos={len(pos)} neg={len(neg)}", flush=True)

    # ---- Tier 1 ----
    tier1_preds_cache = {}  # lang -> per-sample bool list (used by cascade)
    tier1_results = {}      # lang -> conf dict
    if not args.skip_tier1:
        print("[tier1] loading PrivacyDetector …", flush=True)
        t0 = time.time()
        tier1 = PrivacyDetector(config_file=str(TIER1_CONFIG))
        print(f"[tier1] loaded in {time.time()-t0:.1f}s", flush=True)
        for lang in args.langs:
            t0 = time.time()
            preds = run_tier1(tier1, samples[lang]["texts"])
            elapsed = time.time() - t0
            tier1_preds_cache[lang] = preds
            conf = confusion(preds, samples[lang]["labels"])
            tier1_results[lang] = conf
            print(
                f"[tier1] {lang}: R={conf['recall']*100:.2f}%  "
                f"P={conf['precision']*100:.2f}%  FPR={conf['fpr']*100:.2f}%  "
                f"({elapsed:.1f}s, {len(preds)} samples)",
                flush=True,
            )

    # ---- Tier 2 (cascade: Tier 1 ∨ Piiranha+Llama on Tier 1 misses) ----
    tier2_results = {}
    if not args.skip_tier2:
        print("[tier2] loading PiiPrivacyDetector (Piiranha + Llama-3.2-1B) …", flush=True)
        t0 = time.time()
        tier2 = PiiPrivacyDetector(
            pii_model_name=PIIRANHA_HF,
            gene_model_name=LLAMA_HF,
        )
        print(f"[tier2] loaded in {time.time()-t0:.1f}s", flush=True)
        for lang in args.langs:
            t0 = time.time()
            t1_preds = tier1_preds_cache.get(lang)  # None if --skip-tier1
            preds = run_tier2(tier2, samples[lang]["texts"], args.batch_size,
                              tier1_preds=t1_preds)
            elapsed = time.time() - t0
            conf = confusion(preds, samples[lang]["labels"])
            tier2_results[lang] = conf
            n_skipped = sum(1 for p in (t1_preds or [])) - sum(1 for p in (t1_preds or []) if not p)
            print(
                f"[tier2] {lang}: R={conf['recall']*100:.2f}%  "
                f"P={conf['precision']*100:.2f}%  FPR={conf['fpr']*100:.2f}%  "
                f"({elapsed:.1f}s, {len(preds)} samples, {n_skipped} skipped via tier1)",
                flush=True,
            )

    # ---- Write CSV ----
    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = "" if args.neg_source == "excised" else f"_{args.neg_source}_neg"
    out_path = os.path.join(args.out_dir, f"detector_lang_eval{suffix}_{ts}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "lang", "n_pos", "n_neg", "tp", "fp", "tn", "fn",
                    "recall", "precision", "fpr"])
        for lang in args.langs:
            if lang in tier1_results:
                c = tier1_results[lang]
                w.writerow(["Tier 1 (Pattern)", lang, args.n_pos, args.n_neg,
                            c["tp"], c["fp"], c["tn"], c["fn"],
                            f"{c['recall']:.4f}", f"{c['precision']:.4f}", f"{c['fpr']:.4f}"])
            if lang in tier2_results:
                c = tier2_results[lang]
                w.writerow(["Tier 2 (Tier 1 + Piiranha + Llama-3.2-1B)", lang, args.n_pos, args.n_neg,
                            c["tp"], c["fp"], c["tn"], c["fn"],
                            f"{c['recall']:.4f}", f"{c['precision']:.4f}", f"{c['fpr']:.4f}"])
    print(f"[csv] {out_path}", flush=True)


if __name__ == "__main__":
    main()
