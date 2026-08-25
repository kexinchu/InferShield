#!/usr/bin/env bash
# Fill missing reuse-B cells for the 7-point grid:
#   B in {0, 10, 25, 50, 75, 100, 150}
# Skips any cell JSON that already exists.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${REUSE_PORT:-8093}"
SERVER="http://127.0.0.1:${PORT}"
OLD_DIR="${SCRIPT_DIR}/results/reuse_risk_curve/cells"
NEW_DIR="${SCRIPT_DIR}/results/reuse_risk_curve/cells_new_models"
LOG_DIR="${SCRIPT_DIR}/logs/reuse_risk_curve_fill"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-reuse-curve-key}"
mkdir -p "${OLD_DIR}" "${NEW_DIR}" "${LOG_DIR}"

BUDGETS=(0 10 25 50 75 100 150)

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
  [ds_r1_qwen32b]="/workspace/Models/DeepSeek-R1-Distill-Qwen-32B"
  [llama70b_awq]="/workspace/Models/Llama-3.3-70B-Instruct-AWQ"
)
declare -A MODEL_OUT=(
  [phi4]="${OLD_DIR}"
  [qwen30b]="${OLD_DIR}"
  [qwen32b]="${OLD_DIR}"
  [ds_r1_qwen32b]="${NEW_DIR}"
  [llama70b_awq]="${NEW_DIR}"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2 [ds_r1_qwen32b]=2 [llama70b_awq]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=8192 [qwen32b]=8192 [ds_r1_qwen32b]=8192 [llama70b_awq]=4096)
declare -A MODEL_DTYPE=([phi4]=bfloat16 [qwen30b]=bfloat16 [qwen32b]=bfloat16 [ds_r1_qwen32b]=bfloat16 [llama70b_awq]=float16)
declare -A MODEL_QUANT=([phi4]="" [qwen30b]="" [qwen32b]="" [ds_r1_qwen32b]="" [llama70b_awq]=awq)
declare -A MODEL_MEM=([phi4]=0.80 [qwen30b]=0.80 [qwen32b]=0.80 [ds_r1_qwen32b]=0.80 [llama70b_awq]=0.85)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(phi4 qwen30b qwen32b ds_r1_qwen32b llama70b_awq)
fi
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  local pids
  pids="$(lsof -ti:${PORT} 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 4
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  local model="$1" mode="$2" B="$3"
  cleanup
  local gpus=0
  [[ "${MODEL_TP[$model]}" == 2 ]] && gpus=0,1
  local log="${LOG_DIR}/${model}_B${B}_${mode}_server.log"
  local extra=()
  if [[ -n "${MODEL_QUANT[$model]}" ]]; then
    extra+=(--quantization "${MODEL_QUANT[$model]}")
  fi
  echo "[FILL_SERVER_START] model=${model} B=${B} mode=${mode}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype "${MODEL_DTYPE[$model]}" --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static "${MODEL_MEM[$model]}" --schedule-policy fcfs \
    --safekv-mode "${mode}" \
    --safekv-access-budget "${B}" \
    --safekv-operator-key "${OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    --safekv-experiment-autoshare \
    "${extra[@]}" \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 240); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[FILL_SERVER_READY] model=${model} B=${B} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[FILL_SERVER_FAILED] model=${model} B=${B} log=${log}" >&2
      tail -30 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[FILL_SERVER_TIMEOUT] model=${model} B=${B}" >&2
  return 1
}

for model in "${MODELS[@]}"; do
  for B in "${BUDGETS[@]}"; do
    cell="${MODEL_OUT[$model]}/${model}_B${B}.json"
    if [[ -s "${cell}" ]]; then
      echo "[FILL_SKIP] model=${model} B=${B}"
      continue
    fi
    if [[ "${B}" == "0" ]]; then
      mode=strict
      cli_budget=1
    else
      mode=balanced
      cli_budget="${B}"
    fi
    if ! start_server "${model}" "${mode}" "${cli_budget}"; then
      echo "[FILL_FAILED] model=${model} B=${B} reason=server" | tee -a "${LOG_DIR}/failed.log"
      cleanup
      continue
    fi
    if ! "${PYTHON}" "${SCRIPT_DIR}/measure_reuse_b_cell.py" \
      --server "${SERVER}" \
      --model "${model}" \
      --model-path "${MODEL_PATH[$model]}" \
      --B "${B}" \
      --output "${cell}" \
      2>&1 | tee "${LOG_DIR}/${model}_B${B}_client.log"; then
      echo "[FILL_FAILED] model=${model} B=${B} reason=client" | tee -a "${LOG_DIR}/failed.log"
    fi
    cleanup
  done
done

echo "[FILL_DONE] log=${LOG_DIR}"
