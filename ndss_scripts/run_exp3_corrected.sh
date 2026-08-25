#!/usr/bin/env bash
# run_exp3_corrected.sh  –  Corrected P3 end-to-end membership & recovery.
#
# Protocol fixes vs. old script:
#   * b=0 world: victim sends equal-length CONTROL prefix (not absent)
#   * Q counts only attacker probes; victim setup tracked separately
#   * Challenge bits: exactly n//2 positive, n//2 negative, shuffled
#   * Calibration uses SEPARATE prefixes distinct from challenge set
#   * AdvMI = |TPR – FPR| (not 2|acc–0.5|)
#   * ROC-AUC via pairwise Mann-Whitney (no sklearn)
#   * 95% Wilson CI on TPR/FPR; bootstrap CI on AdvMI
#
# Usage:
#   SAFEKV_MODE=strict ./run_exp3_corrected.sh phi4
#   SAFEKV_MODE=balanced ./run_exp3_corrected.sh phi4

set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"

MODEL_KEY="${1:-phi4}"
POLICY="${SAFEKV_MODE:-strict}"
SERVER="http://127.0.0.1:8092"
PORT=8092
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
RESULTS_DIR="${SCRIPT_DIR}/results/exp3"
mkdir -p "${RESULTS_DIR}"

declare -A MODEL_PATHS=(
    ["phi4"]="/workspace/Models/Phi-4"
    ["qwen30b"]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    ["qwen32b"]="/workspace/Models/Qwen3-32B"
)
MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"

OUT="${RESULTS_DIR}/${MODEL_KEY}_${POLICY}_v2.csv"
echo "============================================================"
echo " Model:   ${MODEL_KEY}  Policy: ${POLICY}"
echo " Output:  ${OUT}"
echo "============================================================"

# Wait for server to be ready (caller must start the server first).
echo "[run_exp3] Waiting for server on port ${PORT}..."
for i in $(seq 1 60); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1 || \
       curl -sf "${SERVER}/v1/models" >/dev/null 2>&1; then
        echo "[run_exp3] Server ready."
        break
    fi
    sleep 5
    echo "  …waiting (${i}/60)"
done

# Run corrected experiment.
${PYTHON} "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
    --server        "${SERVER}" \
    --model         "${MODEL_KEY}" \
    --model-path    "${MODEL_PATH}" \
    --policy        "${POLICY}" \
    --budget-Q      200 \
    --n-challenges  100 \
    --n-attacker-accounts 2 \
    --n-recovery-trials 10 \
    --n-tokens-to-recover 5 \
    --vocab-sample  20 \
    --dataset       "${DATASET}" \
    --output        "${OUT}"

echo "[run_exp3] Done. Summary:"
cat "${OUT%.csv}.summary.json" 2>/dev/null || true
