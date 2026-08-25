#!/bin/bash
# Rerun balanced-mode experiments that failed due to private_client None-prompt bug.
# Also runs P4, P10, and P8 which were all blocked.
set -euo pipefail

PYTHON="python3"
BASE_DIR="/workspace/docker-sys"
MODEL_DIR="${BASE_DIR}/Models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
SERVER_URL="http://127.0.0.1:8092"
PORT=8092
OPERATOR_KEY="safekv-liveexp-operator-key"

export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export SAFEKV_OPERATOR_KEY="${OPERATOR_KEY}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p "${RESULTS_DIR}/exp3" "${RESULTS_DIR}/exp4" "${RESULTS_DIR}/exp5" \
         "${RESULTS_DIR}/exp8" "${RESULTS_DIR}/exp10" "${LOG_DIR}"

declare -A MODEL_PATHS=(
    ["phi4"]="${MODEL_DIR}/Phi-4"
    ["qwen32b"]="${MODEL_DIR}/Qwen3-32B"
    ["qwen30b"]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507"
)
declare -A MODEL_TP=(["phi4"]="1" ["qwen32b"]="2" ["qwen30b"]="2")
declare -A MODEL_MAXLEN=(["phi4"]="16384" ["qwen32b"]="32768" ["qwen30b"]="32768")
declare -A MODEL_GPUS=(["phi4"]="0" ["qwen32b"]="0,1" ["qwen30b"]="0,1")

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/rerun_balanced.log"; }

kill_server() {
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        log "Killing existing server on port ${PORT}…"
        kill $(lsof -ti:${PORT}) 2>/dev/null || true
        sleep 5
    fi
}

wait_server_ready() {
    local deadline=$(( $(date +%s) + 600 ))
    log "Waiting for server on ${SERVER_URL}…"
    while true; do
        if curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; then
            log "Server ready."
            sleep 3
            return 0
        fi
        if [[ $(date +%s) -gt $deadline ]]; then
            log "ERROR: Server did not start within 10 minutes."; return 1
        fi
        sleep 5
    done
}

start_server() {
    local model_key="$1" mode="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    local tp="${MODEL_TP[$model_key]}"
    local max_len="${MODEL_MAXLEN[$model_key]}"
    local gpus="${MODEL_GPUS[$model_key]}"
    local log_file="${LOG_DIR}/${model_key}_${mode}_rerun.log"
    kill_server
    log "Starting ${model_key} [mode=${mode}, TP=${tp}, GPUs=${gpus}]…"
    export CUDA_VISIBLE_DEVICES="${gpus}"
    nohup "${PYTHON}" -m sglang.launch_server \
        --model-path "${model_path}" \
        --host 127.0.0.1 --port "${PORT}" \
        --dtype bfloat16 --trust-remote-code \
        --tp-size "${tp}" --context-length "${max_len}" \
        --served-model-name "${model_key}" \
        --attention-backend torch_native --disable-cuda-graph \
        --mem-fraction-static 0.80 --schedule-policy fcfs \
        --safekv-mode "${mode}" \
        --safekv-operator-key "${OPERATOR_KEY}" \
        --safekv-policy-epoch 1 \
        > "${log_file}" 2>&1 &
    wait_server_ready
}

run_p5() {
    local model_key="$1" policy="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    log "=== P5: ${model_key}/${policy} ==="
    for workload in single_pii multi_turn system_prompt; do
        local out="${RESULTS_DIR}/exp5/${model_key}_${policy}_${workload}.csv"
        [[ -f "${out}" ]] && log "P5 ${out} already exists, skipping" && continue
        "${PYTHON}" "${SCRIPT_DIR}/exp5_serving_perf.py" \
            --server "${SERVER_URL}" --model "${model_key}" \
            --model-path "${model_path}" --workload "${workload}" \
            --policy "${policy}" --n-users 20 --rps 8.0 \
            --dataset "${DATASET}" --operator-key "${OPERATOR_KEY}" \
            --output "${out}" && log "P5 saved: ${out}" || log "WARN: P5 ${model_key}/${policy}/${workload} failed"
    done
}

run_p3() {
    local model_key="$1" policy="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    local out="${RESULTS_DIR}/exp3/${model_key}_${policy}.csv"
    [[ -f "${out}" ]] && log "P3 ${out} already exists, skipping" && return
    log "=== P3: ${model_key}/${policy} ==="
    "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
        --server "${SERVER_URL}" --model "${model_key}" \
        --model-path "${model_path}" --policy "${policy}" \
        --budget-Q 200 --n-challenges 50 --n-attacker-accounts 2 \
        --n-recovery-trials 5 --dataset "${DATASET}" \
        --output "${out}" && log "P3 saved: ${out}" || log "WARN: P3 ${model_key}/${policy} failed"
}

run_p4() {
    local model_key="$1"
    local model_path="${MODEL_PATHS[$model_key]}"
    local out="${RESULTS_DIR}/exp4/${model_key}.csv"
    [[ -f "${out}" ]] && log "P4 ${out} already exists, skipping" && return
    log "=== P4: ${model_key} ==="
    "${PYTHON}" "${SCRIPT_DIR}/exp4_public_membership.py" \
        --server "${SERVER_URL}" --model "${model_key}" \
        --model-path "${model_path}" --n-challenges 40 \
        --dataset "${DATASET}" --operator-key "${OPERATOR_KEY}" \
        --output "${out}" && log "P4 saved: ${out}" || log "WARN: P4 ${model_key} failed"
}

run_p8() {
    local model_key="$1"
    local model_path="${MODEL_PATHS[$model_key]}"
    local out="${RESULTS_DIR}/exp8/${model_key}_full.csv"
    [[ -f "${out}" ]] && log "P8 ${out} already exists, skipping" && return
    log "=== P8: ${model_key} (auth matrix) ==="
    "${PYTHON}" "${SCRIPT_DIR}/exp8_auth_matrix.py" \
        --server "${SERVER_URL}" --model "${model_key}" \
        --model-path "${model_path}" --trials 5 \
        --dataset "${DATASET}" --operator-key "${OPERATOR_KEY}" \
        --output "${out}" && log "P8 saved: ${out}" || log "WARN: P8 ${model_key} failed"
}

run_p10() {
    local model_key="$1" policy="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    local out="${RESULTS_DIR}/exp10/${model_key}_${policy}.csv"
    [[ -f "${out}" ]] && log "P10 ${out} already exists, skipping" && return
    log "=== P10: ${model_key}/${policy} ==="
    "${PYTHON}" "${SCRIPT_DIR}/exp10_attack_robustness.py" \
        --server "${SERVER_URL}" --model "${model_key}" \
        --model-path "${model_path}" --policy "${policy}" \
        --n-challenges 30 --dataset "${DATASET}" \
        --output "${out}" && log "P10 saved: ${out}" || log "WARN: P10 ${model_key}/${policy} failed"
}

MODELS=("phi4" "qwen32b" "qwen30b")

log "======================================================"
log "Balanced-mode rerun + P4/P8/P10 — $(date)"
log "======================================================"

for MODEL in "${MODELS[@]}"; do
    log "############# Model: ${MODEL} (strict pass 2: P10) #############"
    # strict pass: run P10 which failed due to flush_cache error
    start_server "${MODEL}" "strict"
    run_p10 "${MODEL}" "strict"
    kill_server

    log "############# Model: ${MODEL} (balanced) #############"
    start_server "${MODEL}" "balanced"
    run_p5 "${MODEL}" "balanced"
    run_p5 "${MODEL}" "balanced_public"
    run_p3 "${MODEL}" "balanced"
    run_p4 "${MODEL}"
    if [[ "${MODEL}" == "phi4" ]]; then
        run_p8 "${MODEL}"
    fi
    kill_server
    log "############# Done: ${MODEL} #############"
done

log "======================================================"
log "Rerun COMPLETE — $(date)"
log "======================================================"
