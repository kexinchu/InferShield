#!/usr/bin/env bash
# run_exp5_qwen.sh  –  Qwen30b + Qwen32b exp5_v2 (run AFTER phi4 finishes).
# Each model uses TP=2 and needs both GPUs.

set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
SAFEKV_OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-exp5-operator-key}"

LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results/exp5_v2"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

PORT=8092
SERVER="http://127.0.0.1:${PORT}"

declare -A MODEL_PATHS=(
    ["qwen30b"]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    ["qwen32b"]="/workspace/Models/Qwen3-32B"
)
declare -A POLICY_MODE=(
    ["vanilla"]="none"
    ["cache_partition"]="strict"
    ["strict"]="strict"
    ["balanced"]="balanced"
    ["balanced_public"]="balanced"
)
WORKLOADS=("single_pii" "multi_turn" "system_prompt")

kill_server() {
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        kill $(lsof -ti:${PORT}) 2>/dev/null || true; sleep 3
    fi
}

wait_server() {
    for i in $(seq 1 90); do
        curl -sf "${SERVER}/health" >/dev/null 2>&1 && echo "[orch] Ready (~$((i*5))s)" && return 0
        sleep 5
    done
    echo "[orch] ERROR: timeout"; return 1
}

run_combo() {
    local MODEL_KEY="$1"
    local POLICY="$2"
    local MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
    local MODE="${POLICY_MODE[$POLICY]}"

    local all_done=true
    for WL in "${WORKLOADS[@]}"; do
        [[ ! -f "${RESULTS_DIR}/${MODEL_KEY}_${POLICY}_${WL}.csv" ]] && all_done=false && break
    done
    $all_done && echo "[orch] SKIP ${MODEL_KEY}_${POLICY}" && return

    kill_server
    export CUDA_VISIBLE_DEVICES=0,1

    ${PYTHON} -m sglang.launch_server \
        --model-path "${MODEL_PATH}" --host 127.0.0.1 --port ${PORT} \
        --dtype bfloat16 --trust-remote-code --tp-size 2 --context-length 32768 \
        --served-model-name "${MODEL_KEY}" --attention-backend torch_native \
        --disable-cuda-graph --mem-fraction-static 0.80 --schedule-policy fcfs \
        --safekv-mode "${MODE}" --safekv-access-budget 100 \
        --safekv-operator-key "${SAFEKV_OPERATOR_KEY}" --safekv-policy-epoch 1 \
        >"${LOG_DIR}/${MODEL_KEY}_${POLICY}_exp5v2.log" 2>&1 &

    wait_server || { kill $(lsof -ti:${PORT}) 2>/dev/null; return 1; }

    for WL in "${WORKLOADS[@]}"; do
        local OUT="${RESULTS_DIR}/${MODEL_KEY}_${POLICY}_${WL}.csv"
        [[ -f "${OUT}" ]] && continue
        echo "[orch]   ${MODEL_KEY}/${POLICY}/${WL}"
        ${PYTHON} "${SCRIPT_DIR}/exp5_serving_perf.py" \
            --server "${SERVER}" --model "${MODEL_KEY}" \
            --model-path "${MODEL_PATH}" --workload "${WL}" --policy "${POLICY}" \
            --n-users 50 --rps 8 --max-new-tokens 64 \
            --dataset "${DATASET}" --output "${OUT}" \
            --operator-key "${SAFEKV_OPERATOR_KEY}" \
            2>&1 | tail -4
    done
    kill_server
}

# cache_partition and strict share the same mode=strict server
for MODEL in qwen30b qwen32b; do
    run_combo "${MODEL}" vanilla
    run_combo "${MODEL}" cache_partition
    run_combo "${MODEL}" strict      # reuses strict server if started
    run_combo "${MODEL}" balanced
    run_combo "${MODEL}" balanced_public
done

echo "Qwen runs complete."
