#!/bin/bash
# ============================================================
# SafeKV Full Ablation: Qwen3-32B → Qwen3-30B-A3B → Phi-4
# 顺序执行所有三个模型的 4-mode ablation test
# Usage: bash run_all_ablation.sh
# ============================================================

set -euo pipefail

export LD_LIBRARY_PATH="/home/kec23008/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PYTHON="/home/kec23008/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/all_models_ablation_${TS}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MASTER_LOG}"; }

log "======================================================"
log "SafeKV Full Ablation Study — All Models"
log "Start time: $(date)"
log "======================================================"

MODELS=("qwen32b" "qwen30b" "phi4")

for MODEL in "${MODELS[@]}"; do
    log ""
    log "======================================================"
    log ">>> Starting model: ${MODEL}"
    log "======================================================"

    OUT_CSV="${LOG_DIR}/ablation_${MODEL}_${TS}.csv"

    cd "${SCRIPT_DIR}"
    if "${PYTHON}" test_safekv_ablation.py \
        --model "${MODEL}" \
        --output "${OUT_CSV}" \
        2>&1 | tee -a "${MASTER_LOG}"; then
        log "<<< DONE: ${MODEL}  (CSV: ${OUT_CSV})"
    else
        log "!!! ERROR: ${MODEL} ablation FAILED (exit code $?)"
        log "    Continuing with next model..."
    fi

    # GPU 冷却间隔
    log "Cooling down 60s before next model..."
    sleep 60
done

log ""
log "======================================================"
log "All models complete. Summary CSVs:"
for MODEL in "${MODELS[@]}"; do
    CSV="${LOG_DIR}/ablation_${MODEL}_${TS}.csv"
    if [[ -f "${CSV}" ]]; then
        log "  ${MODEL}: ${CSV}"
        cat "${CSV}" | tee -a "${MASTER_LOG}"
    fi
done
log "Master log: ${MASTER_LOG}"
log "End time: $(date)"
log "======================================================"
