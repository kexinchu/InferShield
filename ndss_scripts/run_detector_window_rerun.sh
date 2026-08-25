#!/usr/bin/env bash
# Detector-Window rerun for fig:defense_acc. New results only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SAFEKV_PYTHON:-python3}"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SAFEKV_PIIRANHA="${SAFEKV_PIIRANHA:-/workspace/Models/piiranha-v1-detect-personal-information}"
export SAFEKV_LLAMA_DETECTOR="${SAFEKV_LLAMA_DETECTOR:-/workspace/Models/Llama-3.2-1B-Instruct}"
OUT="${ROOT}/ndss_scripts/results/detector_window_rerun"
LOG="${ROOT}/ndss_scripts/logs/detector_window_rerun"
mkdir -p "${OUT}" "${LOG}"
"${PY}" "${ROOT}/ndss_scripts/eval_detector_window.py" \
  --n 500 --seed 42 --out-dir "${OUT}" \
  2>&1 | tee "${LOG}/run_$(date +%Y%m%d_%H%M%S).log"
