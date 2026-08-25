#!/usr/bin/env bash
# E5: stronger token-recovery evaluation under Balanced B=100.
# Modes: guaranteed (true token in shuffled candidates) and logit_topk.
# Recovery-only with full Q=200; 10 trials × 5 tokens; k=20.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${E5_PORT:-8092}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/submission_gap_experiments/e5_strong_recovery"
LOG_DIR="${SCRIPT_DIR}/logs/submission_gap_experiments/e5_strong_recovery"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-e5-operator-key}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=32768 [qwen32b]=32768)

MODES=(guaranteed logit_topk)
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
  local model="$1"
  cleanup
  local gpus=0
  [[ "${MODEL_TP[$model]}" == 2 ]] && gpus=0,1
  local log="${LOG_DIR}/${model}_balanced_B100_server.log"
  echo "[E5_SERVER_START] model=${model}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static 0.80 --schedule-policy fcfs \
    --safekv-mode balanced \
    --safekv-access-budget 100 \
    --safekv-operator-key "${OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[E5_SERVER_READY] model=${model} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[E5_SERVER_FAILED] model=${model}" >&2
      return 1
    fi
    sleep 5
  done
  echo "[E5_SERVER_TIMEOUT] model=${model}" >&2
  return 1
}

for model in phi4 qwen30b qwen32b; do
  need_server=0
  for mode in "${MODES[@]}"; do
    out="${OUT_DIR}/${model}_balanced_B100_${mode}.csv"
    summ="${OUT_DIR}/${model}_balanced_B100_${mode}.summary.json"
    if [[ ! -s "${out}" || ! -s "${summ}" ]]; then
      need_server=1
      break
    fi
  done
  if [[ "${need_server}" == "0" ]]; then
    echo "[E5_SKIP_MODEL] ${model}"
    continue
  fi

  start_server "${model}"
  for mode in "${MODES[@]}"; do
    out="${OUT_DIR}/${model}_balanced_B100_${mode}.csv"
    summ="${OUT_DIR}/${model}_balanced_B100_${mode}.summary.json"
    if [[ -s "${out}" && -s "${summ}" ]]; then
      echo "[E5_SKIP] model=${model} mode=${mode}"
      continue
    fi
    rm -f "${out}" "${summ}"
    echo "[E5_CELL_START] model=${model} mode=${mode}"
    "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
      --server "${SERVER}" \
      --model "${model}" \
      --model-path "${MODEL_PATH[$model]}" \
      --policy balanced \
      --access-budget-B 100 \
      --budget-Q 200 \
      --n-challenges 2 \
      --n-attacker-accounts 2 \
      --n-recovery-trials 10 \
      --n-tokens-to-recover 5 \
      --vocab-sample 20 \
      --recovery-mode "${mode}" \
      --recovery-only \
      --post-victim-settle-ms 750 \
      --dataset "${DATASET}" \
      --seed 20260821 \
      --output "${out}" \
      2>&1 | tee "${LOG_DIR}/${model}_${mode}_client.log"
    echo "[E5_CELL_DONE] model=${model} mode=${mode}"
  done
  cleanup
done

echo "[E5_ALL_DONE] out=${OUT_DIR}"
