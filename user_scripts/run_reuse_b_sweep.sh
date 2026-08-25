#!/usr/bin/env bash
# Fill the missing reuse axis of the reuse-risk curve.
#
# Balanced WITHOUT Registry on a non-sensitive shared system prompt, so reuse
# is B-gated. The client waits for async Candidate admission before probing.
# Grid: B in {0, 10, 50, 100, 150, 200} on phi4 / qwen30b / qwen32b.
# B=0 starts the server in strict mode.
#
# After the sweep:
#   python user_scripts/merge_reuse_b_cells.py
#   python user_scripts/plot_reuse_risk_curve.py
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${REUSE_PORT:-8093}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/reuse_risk_curve/cells"
LOG_DIR="${SCRIPT_DIR}/logs/reuse_risk_curve"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-reuse-curve-key}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=32768 [qwen32b]=32768)

BUDGETS=(0 10 50 100 150 200)
MODELS=(phi4 qwen30b qwen32b)
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
  echo "[REUSE_SERVER_START] model=${model} B=${B} mode=${mode}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static 0.80 --schedule-policy fcfs \
    --safekv-mode "${mode}" \
    --safekv-access-budget "${B}" \
    --safekv-operator-key "${OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    --safekv-experiment-autoshare \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[REUSE_SERVER_READY] model=${model} B=${B} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[REUSE_SERVER_FAILED] model=${model} B=${B}" >&2
      return 1
    fi
    sleep 5
  done
  echo "[REUSE_SERVER_TIMEOUT] model=${model} B=${B}" >&2
  return 1
}

for model in "${MODELS[@]}"; do
  for B in "${BUDGETS[@]}"; do
    cell="${OUT_DIR}/${model}_B${B}.json"
    if [[ -s "${cell}" ]]; then
      echo "[REUSE_SKIP] model=${model} B=${B}"
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
      echo "[REUSE_CELL_FAILED] model=${model} B=${B} reason=server" | tee -a "${LOG_DIR}/failed_cells.log"
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
      echo "[REUSE_CELL_FAILED] model=${model} B=${B} reason=client" | tee -a "${LOG_DIR}/failed_cells.log"
    fi
    cleanup
  done
done

echo "[REUSE_ALL_DONE] cells=${OUT_DIR}"
echo "Next: python ${SCRIPT_DIR}/merge_reuse_b_cells.py && python ${SCRIPT_DIR}/plot_reuse_risk_curve.py"
