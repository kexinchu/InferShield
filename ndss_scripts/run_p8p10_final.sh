#!/usr/bin/env bash
# Run P8 (qwen32b, qwen30b) and P10 balanced (all models) sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="python3"
PORT=8092
OPERATOR_KEY="safekv-liveexp-operator-key"
LOG_DIR="${SCRIPT_DIR}/logs"
RES_DIR="${SCRIPT_DIR}/results"
PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH LD_LIBRARY_PATH

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
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 15
    # Wait until port is free before launching
    local deadline_free=$(( $(date +%s) + 60 ))
    while (( $(date +%s) < deadline_free )); do
        if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    CUDA_VISIBLE_DEVICES="${MODEL_GPU[$model]}" nohup \
        "${PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATHS[$model]}" \
        --host 127.0.0.1 --port "${PORT}" --dtype bfloat16 --trust-remote-code \
        --tp-size "${MODEL_TP[$model]}" \
        --context-length "${MODEL_MAXLEN[$model]}" \
        --served-model-name "${model}" \
        --attention-backend torch_native --disable-cuda-graph \
        --mem-fraction-static 0.80 --schedule-policy fcfs \
        --safekv-mode "${mode}" \
        --safekv-operator-key "${OPERATOR_KEY}" --safekv-policy-epoch 1 \
        > "${LOG_DIR}/${model}_${mode}_p8p10.log" 2>&1 &
    local pid=$!
    log "Server PID=${pid}; waiting for ready..."
    local deadline=$(( $(date +%s) + 600 ))
    while (( $(date +%s) < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            log "Server ready"
            return 0
        fi
        sleep 5
    done
    log "WARN: server did not become ready for $model ($mode)"
    return 1
}

kill_server() {
    log "Killing server..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 10
    # Wait until port is free
    local deadline=$(( $(date +%s) + 30 ))
    while (( $(date +%s) < deadline )); do
        if ! curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
}

# ── P8: qwen32b and qwen30b ───────────────────────────────────────────────
for model in qwen32b qwen30b; do
    out="${RES_DIR}/exp8/${model}_full.csv"
    if [[ -f "$out" ]]; then
        log "P8 $model already exists, skipping"
        continue
    fi
    start_server "$model" strict || { log "WARN: $model server failed, skipping P8"; continue; }
    log "Running P8 for $model..."
    "${PYTHON}" "${SCRIPT_DIR}/exp8_auth_matrix.py" \
        --server "http://127.0.0.1:${PORT}" \
        --model "${model}" \
        --model-path "${MODEL_PATHS[$model]}" \
        --trials 3 \
        --output "${out}" && log "P8 $model saved: $out" || log "WARN: P8 $model failed"
    kill_server
done

# ── P10 balanced: all models ──────────────────────────────────────────────
for model in phi4 qwen32b qwen30b; do
    out="${RES_DIR}/exp10/${model}_balanced.csv"
    if [[ -f "$out" ]]; then
        log "P10 balanced $model already exists, skipping"
        continue
    fi
    start_server "$model" balanced || { log "WARN: $model server failed, skipping P10 balanced"; continue; }
    log "Running P10 balanced for $model..."
    "${PYTHON}" "${SCRIPT_DIR}/exp10_attack_robustness.py" \
        --server "http://127.0.0.1:${PORT}" \
        --model "${model}" \
        --model-path "${MODEL_PATHS[$model]}" \
        --policy balanced \
        --n-challenges 30 \
        --output "${out}" && log "P10 balanced $model saved: $out" || log "WARN: P10 balanced $model failed"
    kill_server
done

log "=== ALL DONE ==="
