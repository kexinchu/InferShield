#!/bin/bash
# ============================================================
# P5 Baseline runner: vanilla SGLang + cache-partition baselines
#
# Runs exp5_serving_perf.py against a server launched with
#   --safekv-mode none   (global cache sharing = vanilla SGLang)
# and separately against
#   --safekv-mode strict  with user_id isolation = cache-partition
#
# Usage:
#   ./run_p5_baseline.sh phi4
#   ./run_p5_baseline.sh qwen30b
#   ./run_p5_baseline.sh qwen32b
# ============================================================
set -euo pipefail

MODEL_KEY="${1:-phi4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SAFEKV_PYTHON:-python3}"
RESULTS="${SCRIPT_DIR}/results/exp5"
SERVER="http://127.0.0.1:8092"
PORT=8092
N_USERS=20
RPS=8.0

export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"

declare -A MODEL_PATHS=(
    ["phi4"]="/workspace/Models/Phi-4"
    ["qwen30b"]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    ["qwen32b"]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=(
    ["phi4"]="1"
    ["qwen30b"]="2"
    ["qwen32b"]="2"
)
declare -A MODEL_MAXLEN=(
    ["phi4"]="16384"
    ["qwen30b"]="32768"
    ["qwen32b"]="32768"
)

MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
TP="${MODEL_TP[$MODEL_KEY]}"
MAX_LEN="${MODEL_MAXLEN[$MODEL_KEY]}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RESULTS}"

# ── Helper: wait for server to be ready ─────────────────────────
wait_server() {
    local url="${SERVER}/health"
    echo "[baseline] Waiting for server at ${url}..."
    for i in $(seq 1 120); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "[baseline] Server ready."
            return 0
        fi
        sleep 5
    done
    echo "[ERROR] Server did not start within 10min" >&2
    return 1
}

# ── Helper: kill server ──────────────────────────────────────────
kill_server() {
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        echo "[baseline] Killing server on port ${PORT}..."
        kill $(lsof -ti:${PORT}) 2>/dev/null || true
        sleep 5
    fi
}

# ── Helper: launch server with given safekv-mode ────────────────
launch_server() {
    local mode="$1"
    local log="${LOG_DIR}/${MODEL_KEY}_${mode}.log"

    kill_server

    if [[ "${TP}" == "1" ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export CUDA_VISIBLE_DEVICES=0,1
    fi

    echo "[baseline] Launching ${MODEL_KEY} with --safekv-mode=${mode} (TP=${TP})..."
    nohup ${PYTHON} -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tp-size "${TP}" \
        --context-length "${MAX_LEN}" \
        --served-model-name "${MODEL_KEY}" \
        --attention-backend torch_native \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --schedule-policy fcfs \
        --safekv-mode "${mode}" \
        --safekv-operator-key "safekv-liveexp-operator-key" \
        --safekv-policy-epoch 1 \
        > "${log}" 2>&1 &
    SERVER_PID=$!
    echo "[baseline] Server PID=${SERVER_PID}, log=${log}"
}

# ── Helper: run one exp5 workload ────────────────────────────────
run_workload() {
    local policy="$1"
    local workload="$2"
    local output="${RESULTS}/${MODEL_KEY}_${policy}_${workload}.csv"

    echo "[baseline] Running: model=${MODEL_KEY} policy=${policy} workload=${workload}"
    ${PYTHON} "${SCRIPT_DIR}/exp5_serving_perf.py" \
        --server "${SERVER}" \
        --model "${MODEL_KEY}" \
        --model-path "${MODEL_PATH}" \
        --workload "${workload}" \
        --policy "${policy}" \
        --n-users "${N_USERS}" \
        --rps "${RPS}" \
        --output "${output}" \
        --operator-key "safekv-liveexp-operator-key"
}

echo "============================================================"
echo " P5 Baseline Run: model=${MODEL_KEY}"
echo " Results dir: ${RESULTS}"
echo "============================================================"

# ── PHASE 1: vanilla SGLang (--safekv-mode none) ─────────────────
echo ""
echo "== Phase 1: Vanilla SGLang (--safekv-mode none) =="
launch_server "none"
wait_server

for wl in single_pii multi_turn system_prompt; do
    run_workload "vanilla" "${wl}"
done

kill_server

# ── PHASE 2: Cache-Partition (strict + user_id isolation) ────────
# Strict mode already enforces per-user namespace isolation
# which is equivalent to Cache-Partition for the TTFT comparison
echo ""
echo "== Phase 2: Cache-Partition equivalent (--safekv-mode strict) =="
echo "   (Strict mode = per-user private namespace = Cache-Partition behavior)"
launch_server "strict"
wait_server

for wl in single_pii multi_turn system_prompt; do
    run_workload "cache_partition" "${wl}"
done

kill_server

echo ""
echo "============================================================"
echo " Done! Results written to ${RESULTS}/"
echo " Files: ${MODEL_KEY}_vanilla_{single_pii,multi_turn,system_prompt}.csv"
echo "        ${MODEL_KEY}_cache_partition_{single_pii,multi_turn,system_prompt}.csv"
echo "============================================================"
