#!/usr/bin/env bash
# Run P10 balanced mode for all three models sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python3"
PORT=8092
OPERATOR_KEY="safekv-liveexp-operator-key"
LOG_DIR="${SCRIPT_DIR}/logs"
RES_DIR="${SCRIPT_DIR}/results"
export PYTHONPATH="${SCRIPT_DIR}/../python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

declare -A MODEL_PATHS=(
    [phi4]="/workspace/Models/Phi-4"
    [qwen32b]="/workspace/Models/Qwen3-32B"
    [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
)
declare -A MODEL_TP=( [phi4]=1 [qwen32b]=2 [qwen30b]=2 )
declare -A MODEL_MAXLEN=( [phi4]=16384 [qwen32b]=16384 [qwen30b]=16384 )
declare -A MODEL_GPU=( [phi4]=0 [qwen32b]="0,1" [qwen30b]="0,1" )

log() { echo "[$(date '+%H:%M:%S')] $*"; }

start_server() {
    local model=$1 mode=$2
    log "=== Starting $model ($mode) ==="
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 15
    local deadline_free=$(( $(date +%s) + 60 ))
    while (( $(date +%s) < deadline_free )); do
        if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then break; fi
        sleep 2
    done
    CUDA_VISIBLE_DEVICES="${MODEL_GPU[$model]}" nohup \
        "${PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATHS[$model]}" \
        --host 127.0.0.1 --port "${PORT}" --dtype bfloat16 --trust-remote-code \
        --tp-size "${MODEL_TP[$model]}" --context-length "${MODEL_MAXLEN[$model]}" \
        --served-model-name "${model}" \
        --attention-backend torch_native --disable-cuda-graph \
        --mem-fraction-static 0.80 --schedule-policy fcfs \
        --safekv-mode "${mode}" \
        --safekv-operator-key "${OPERATOR_KEY}" --safekv-policy-epoch 1 \
        > "${LOG_DIR}/${model}_${mode}_p10bal.log" 2>&1 &
    log "Server PID=$!; waiting..."
    local deadline=$(( $(date +%s) + 720 ))
    while (( $(date +%s) < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            log "Server ready"; return 0
        fi
        sleep 5
    done
    log "WARN: server did not become ready for $model ($mode)"; return 1
}

for model in qwen32b qwen30b phi4; do
    out="${RES_DIR}/exp10/${model}_balanced.csv"
    if [[ -f "$out" ]]; then
        log "P10 balanced $model exists, skipping"
        continue
    fi
    start_server "$model" balanced || { log "WARN: $model server failed, skipping"; continue; }
    log "Running P10 balanced for $model..."
    "${PYTHON}" "${SCRIPT_DIR}/exp10_attack_robustness.py" \
        --server "http://127.0.0.1:${PORT}" \
        --model "${model}" \
        --model-path "${MODEL_PATHS[$model]}" \
        --policy balanced \
        --n-challenges 30 \
        --output "${out}" \
        && log "Saved: $out" || log "WARN: P10 balanced $model failed"
done

pkill -f "sglang.launch_server" 2>/dev/null || true
log "=== ALL DONE ==="
