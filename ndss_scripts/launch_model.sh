#!/bin/bash
# ============================================================
# SGLang Model Server Launcher (OpenAI-compatible API)
# Usage:
#   ./launch_model.sh                  # default: Qwen3-32B
#   ./launch_model.sh qwen32b          # Qwen3-32B
#   ./launch_model.sh qwen30b          # Qwen3-30B-A3B-Instruct-2507
#   ./launch_model.sh phi4             # Phi-4
# ============================================================

set -euo pipefail

PYTHON="/home/kec23008/miniconda3/envs/vllm_test/bin/python3"
echo "[INFO] Using Python: ${PYTHON}"

# nvshmem required by torch 2.9; use CUDA 12.9 nvcc for JIT compilation
export LD_LIBRARY_PATH="/home/kec23008/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

MODEL_KEY="${1:-qwen32b}"
BASE_DIR="/home/kec23008/docker-sys"
MODEL_DIR="${BASE_DIR}/Models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---- Model configs ----
declare -A MODEL_PATHS=(
    ["qwen32b"]="${MODEL_DIR}/Qwen3-32B"
    ["qwen30b"]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507"
    ["phi4"]="${MODEL_DIR}/Phi-4"
)
declare -A MODEL_TP=(
    ["qwen32b"]="2"
    ["qwen30b"]="2"
    ["phi4"]="1"
)
declare -A MODEL_PORTS=(
    ["qwen32b"]="8090"
    ["qwen30b"]="8091"
    ["phi4"]="8092"
)
declare -A MODEL_MAXLEN=(
    ["qwen32b"]="32768"
    ["qwen30b"]="32768"
    ["phi4"]="16384"
)

if [[ -z "${MODEL_PATHS[$MODEL_KEY]+x}" ]]; then
    echo "[ERROR] Unknown model key: ${MODEL_KEY}"
    echo "Available: ${!MODEL_PATHS[*]}"
    exit 1
fi

MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
TP_SIZE="${MODEL_TP[$MODEL_KEY]}"
PORT="${MODEL_PORTS[$MODEL_KEY]}"
MAX_LEN="${MODEL_MAXLEN[$MODEL_KEY]}"
LOG_FILE="${LOG_DIR}/${MODEL_KEY}.log"

echo "============================================"
echo " Model:    ${MODEL_KEY}"
echo " Path:     ${MODEL_PATH}"
echo " TP Size:  ${TP_SIZE}"
echo " Port:     ${PORT}"
echo " Log:      ${LOG_FILE}"
echo " Engine:   SGLang"
echo "============================================"

# Kill any existing server on this port
if lsof -ti:${PORT} >/dev/null 2>&1; then
    echo "[WARN] Killing existing process on port ${PORT}"
    kill $(lsof -ti:${PORT}) 2>/dev/null || true
    sleep 2
fi

# Set GPU devices based on tp_size
if [[ "${TP_SIZE}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=0
elif [[ "${TP_SIZE}" == "2" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Starting SGLang server..."

${PYTHON} -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --tp-size "${TP_SIZE}" \
    --context-length "${MAX_LEN}" \
    --served-model-name "${MODEL_KEY}" \
    --attention-backend torch_native \
    --disable-cuda-graph \
    --mem-fraction-static 0.80 \
    --context-length "${MAX_LEN}" \
    2>&1 | tee "${LOG_FILE}"
