#!/usr/bin/env bash
# Table 10 protocol: 500 PII positives + 500 OOD negatives per corpus.
# Writes new CSVs only; does not touch paper artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SAFEKV_PYTHON:-python3}"
OUT="${ROOT}/ndss_scripts/results/detector_ood_rerun"
LOGDIR="${ROOT}/ndss_scripts/logs/detector_ood_rerun"
N_POS="${N_POS:-500}"
N_NEG="${N_NEG:-500}"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
export SAFEKV_PIIRANHA="${SAFEKV_PIIRANHA:-/workspace/Models/piiranha-v1-detect-personal-information}"
export SAFEKV_LLAMA_DETECTOR="${SAFEKV_LLAMA_DETECTOR:-/workspace/Models/Llama-3.2-1B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT}" "${LOGDIR}"
cd "${ROOT}"

for src in sharegpt agnews wikitext; do
  echo "===== ${src} n_pos=${N_POS} n_neg=${N_NEG} ====="
  "${PY}" ndss_scripts/eval_detector_languages.py \
    --langs en \
    --n-pos "${N_POS}" \
    --n-neg "${N_NEG}" \
    --neg-source "${src}" \
    --out-dir "${OUT}" \
    2>&1 | tee "${LOGDIR}/${src}_$(date +%Y%m%d_%H%M%S).log"
done

echo "Results in ${OUT}"
