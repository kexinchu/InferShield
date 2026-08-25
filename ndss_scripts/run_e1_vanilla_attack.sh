#!/usr/bin/env bash
# E1: corrected-protocol membership baseline under vanilla SGLang (safekv-mode=none).
# Outputs only under results/submission_gap_experiments/e1_vanilla_attack/.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${E1_PORT:-8092}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/submission_gap_experiments/e1_vanilla_attack"
LOG_DIR="${SCRIPT_DIR}/logs/submission_gap_experiments/e1_vanilla_attack"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=32768 [qwen32b]=32768)

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
  local log="${LOG_DIR}/${model}_vanilla_server.log"
  echo "[E1_SERVER_START] model=${model}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static 0.80 --schedule-policy fcfs \
    --safekv-mode none --safekv-policy-epoch 1 \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[E1_SERVER_READY] model=${model} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[E1_SERVER_FAILED] model=${model}" >&2
      return 1
    fi
    sleep 5
  done
  echo "[E1_SERVER_TIMEOUT] model=${model}" >&2
  return 1
}

for model in phi4 qwen30b qwen32b; do
  out="${OUT_DIR}/${model}_vanilla_membership.csv"
  summ="${OUT_DIR}/${model}_vanilla_membership.summary.json"
  if [[ -s "${out}" && -s "${summ}" ]]; then
    echo "[E1_SKIP] ${model}"
    continue
  fi
  rm -f "${out}" "${summ}"
  start_server "${model}"
  echo "[E1_CELL_START] model=${model}"
  "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
    --server "${SERVER}" \
    --model "${model}" \
    --model-path "${MODEL_PATH[$model]}" \
    --policy vanilla \
    --access-budget-B -1 \
    --budget-Q 200 \
    --n-challenges 100 \
    --n-attacker-accounts 2 \
    --membership-only \
    --dataset "${DATASET}" \
    --seed 20260821 \
    --output "${out}" \
    2>&1 | tee "${LOG_DIR}/${model}_vanilla_client.log"
  echo "[E1_CELL_DONE] model=${model}"
  cleanup
done

echo "[E1_ALL_DONE] out=${OUT_DIR}"
