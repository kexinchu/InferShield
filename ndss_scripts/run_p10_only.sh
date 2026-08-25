#!/bin/bash
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
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
mkdir -p "${RESULTS_DIR}/exp10" "${LOG_DIR}"

declare -A MODEL_PATHS=(["phi4"]="${MODEL_DIR}/Phi-4" ["qwen32b"]="${MODEL_DIR}/Qwen3-32B" ["qwen30b"]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507")
declare -A MODEL_TP=(["phi4"]="1" ["qwen32b"]="2" ["qwen30b"]="2")
declare -A MODEL_MAXLEN=(["phi4"]="16384" ["qwen32b"]="32768" ["qwen30b"]="32768")
declare -A MODEL_GPUS=(["phi4"]="0" ["qwen32b"]="0,1" ["qwen30b"]="0,1")

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "${LOG_DIR}/p10_run.log"; }

kill_server() {
    lsof -ti:${PORT} >/dev/null 2>&1 && { log "Killing server…"; kill $(lsof -ti:${PORT}) 2>/dev/null || true; sleep 5; } || true
}

wait_ready() {
    local dead=$(( $(date +%s) + 600 ))
    while ! curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; do
        [[ $(date +%s) -gt $dead ]] && { log "Server timeout"; return 1; }; sleep 5
    done; log "Server ready"; sleep 3
}

for MODEL in phi4 qwen32b qwen30b; do
    out="${RESULTS_DIR}/exp10/${MODEL}_strict.csv"
    [[ -f "$out" ]] && log "P10 ${MODEL} already exists, skipping" && continue
    log "=== P10: ${MODEL} ==="
    kill_server
    export CUDA_VISIBLE_DEVICES="${MODEL_GPUS[$MODEL]}"
    nohup "${PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATHS[$MODEL]}" --host 127.0.0.1 --port "${PORT}" \
        --dtype bfloat16 --trust-remote-code --tp-size "${MODEL_TP[$MODEL]}" \
        --context-length "${MODEL_MAXLEN[$MODEL]}" --served-model-name "${MODEL}" \
        --attention-backend torch_native --disable-cuda-graph \
        --mem-fraction-static 0.80 --schedule-policy fcfs \
        --safekv-mode strict --safekv-operator-key "${OPERATOR_KEY}" --safekv-policy-epoch 1 \
        > "${LOG_DIR}/${MODEL}_p10.log" 2>&1 &
    wait_ready
    "${PYTHON}" "${SCRIPT_DIR}/exp10_attack_robustness.py" \
        --server "${SERVER_URL}" --model "${MODEL}" \
        --model-path "${MODEL_PATHS[$MODEL]}" --policy strict \
        --n-challenges 30 --dataset "${DATASET}" --output "${out}" \
        && log "P10 saved: ${out}" || log "WARN: P10 ${MODEL} failed"
    kill_server
done
log "P10 run COMPLETE"
