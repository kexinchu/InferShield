#!/bin/bash
# ============================================================
# SGLang Model Server Launcher (OpenAI-compatible API)
# Usage:
#   ./launch_model.sh qwen32b          # Qwen3-32B,             TP=2, port 8090
#   ./launch_model.sh qwen30b          # Qwen3-30B-A3B FP16,    TP=2, port 8094
#   ./launch_model.sh qwen30b-int4     # Qwen3-30B-A3B INT4,    TP=1, port 8093
#   ./launch_model.sh phi4             # Phi-4 FP16,            DP=2, port 8092
# ============================================================

set -euo pipefail

PYTHON="/home/kec23008/.venv/bin/python3"
echo "[INFO] Using Python: ${PYTHON}"

export LD_LIBRARY_PATH="/home/kec23008/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

MODEL_KEY="${1:-qwen32b}"
MODEL_DIR="/home/kec23008/Models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---- Model configs ----
declare -A MODEL_PATHS=(
    ["qwen32b"]="${MODEL_DIR}/Qwen3-32B"
    ["qwen30b"]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507"
    ["qwen30b-int4"]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
    ["phi4"]="${MODEL_DIR}/Phi-4"
)
# TP size per model (number of GPUs for tensor parallelism)
declare -A MODEL_TP=(
    ["qwen32b"]="2"
    ["qwen30b"]="2"
    ["qwen30b-int4"]="1"
    ["phi4"]="1"
)
# DP size per model (data parallel replicas; each replica uses tp GPUs)
declare -A MODEL_DP=(
    ["qwen32b"]="1"
    ["qwen30b"]="1"
    ["qwen30b-int4"]="1"
    ["phi4"]="2"
)
declare -A MODEL_PORTS=(
    ["qwen32b"]="8090"
    ["qwen30b"]="8094"
    ["qwen30b-int4"]="8093"
    ["phi4"]="8092"
)
declare -A MODEL_MAXLEN=(
    ["qwen32b"]="32768"
    ["qwen30b"]="32768"
    ["qwen30b-int4"]="32768"
    ["phi4"]="16384"
)
declare -A MODEL_MEM_FRAC=(
    ["qwen32b"]="0.90"
    ["qwen30b"]="0.90"
    ["qwen30b-int4"]="0.85"
    ["phi4"]="0.90"
)

if [[ -z "${MODEL_PATHS[$MODEL_KEY]+x}" ]]; then
    echo "[ERROR] Unknown model key: ${MODEL_KEY}"
    echo "Available: ${!MODEL_PATHS[*]}"
    exit 1
fi

MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
TP_SIZE="${MODEL_TP[$MODEL_KEY]}"
DP_SIZE="${MODEL_DP[$MODEL_KEY]}"
PORT="${MODEL_PORTS[$MODEL_KEY]}"
MAX_LEN="${MODEL_MAXLEN[$MODEL_KEY]}"
MEM_FRAC="${MODEL_MEM_FRAC[$MODEL_KEY]}"
LOG_FILE="${LOG_DIR}/${MODEL_KEY}.log"

# Total GPUs needed = TP * DP
TOTAL_GPUS=$(( TP_SIZE * DP_SIZE ))

echo "============================================"
echo " Model:    ${MODEL_KEY}"
echo " Path:     ${MODEL_PATH}"
echo " TP Size:  ${TP_SIZE}  DP Size: ${DP_SIZE}  (total GPUs: ${TOTAL_GPUS})"
echo " Port:     ${PORT}"
echo " Mem Frac: ${MEM_FRAC}"
echo " Log:      ${LOG_FILE}"
echo " Engine:   SGLang"
echo "============================================"

# Kill any existing server on this port
if lsof -ti:${PORT} >/dev/null 2>&1; then
    echo "[WARN] Killing existing process on port ${PORT}"
    kill $(lsof -ti:${PORT}) 2>/dev/null || true
    sleep 2
fi

# Set CUDA_VISIBLE_DEVICES based on total GPUs needed
case "${TOTAL_GPUS}" in
    1) export CUDA_VISIBLE_DEVICES=0 ;;
    2) export CUDA_VISIBLE_DEVICES=0,1 ;;
    4) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
    *) export CUDA_VISIBLE_DEVICES=0,1 ;;
esac

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Starting SGLang server (TP=${TP_SIZE}, DP=${DP_SIZE})..."

SAFEKV_BUDGET="${SAFEKV_BUDGET:-10}"
SAFEKV_THRESHOLD="${SAFEKV_THRESHOLD:-2}"
SAFEKV_PRIVATE_ONLY="${SAFEKV_PRIVATE_ONLY:-0}"
echo "[INFO] SafeKV: access_budget=${SAFEKV_BUDGET}, creator_threshold=${SAFEKV_THRESHOLD}, private_only=${SAFEKV_PRIVATE_ONLY}"

SAFEKV_EXTRA_ARGS=""
if [[ "${SAFEKV_PRIVATE_ONLY}" == "1" || "${SAFEKV_PRIVATE_ONLY}" == "true" ]]; then
    SAFEKV_EXTRA_ARGS="--safekv-private-only"
fi

${PYTHON} -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype float16 \
    --trust-remote-code \
    --tp-size "${TP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --context-length "${MAX_LEN}" \
    --served-model-name "${MODEL_KEY}" \
    --attention-backend torch_native \
    --disable-cuda-graph \
    --mem-fraction-static "${MEM_FRAC}" \
    --enable-metrics \
    --safekv-access-budget "${SAFEKV_BUDGET}" \
    --safekv-creator-threshold "${SAFEKV_THRESHOLD}" \
    ${SAFEKV_EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"
