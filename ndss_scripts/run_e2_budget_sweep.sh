#!/usr/bin/env bash
# E2: revised B-security-cost sweep.
# B=0 → strict; B∈{1,10,50,100} → balanced --safekv-access-budget B.
# Per cell: membership-only (n=100) + one system_prompt serving measurement.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${E2_PORT:-8092}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/submission_gap_experiments/e2_budget_sweep"
LOG_DIR="${SCRIPT_DIR}/logs/submission_gap_experiments/e2_budget_sweep"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-e2-operator-key}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=32768 [qwen32b]=32768)

BUDGETS=(0 1 10 50 100)
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

mode_for_B() {
  local B="$1"
  if [[ "${B}" == "0" ]]; then
    echo strict
  else
    echo balanced
  fi
}

policy_label() {
  local B="$1"
  if [[ "${B}" == "0" ]]; then
    echo "B0_strict"
  else
    echo "B${B}_balanced"
  fi
}

start_server() {
  local model="$1" mode="$2" B="$3"
  cleanup
  local gpus=0
  [[ "${MODEL_TP[$model]}" == 2 ]] && gpus=0,1
  local log="${LOG_DIR}/${model}_B${B}_${mode}_server.log"
  echo "[E2_SERVER_START] model=${model} B=${B} mode=${mode}"
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
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[E2_SERVER_READY] model=${model} B=${B} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[E2_SERVER_FAILED] model=${model} B=${B}" >&2
      return 1
    fi
    sleep 5
  done
  echo "[E2_SERVER_TIMEOUT] model=${model} B=${B}" >&2
  return 1
}

for model in phi4 qwen30b qwen32b; do
  for B in "${BUDGETS[@]}"; do
    mode="$(mode_for_B "${B}")"
    # Access budget 0 is invalid for the ledger; use budget=1 only as a
    # placeholder CLI value while mode=strict enforces B=0 semantics.
    cli_budget="${B}"
    [[ "${B}" == "0" ]] && cli_budget=1
    policy="$(policy_label "${B}")"
    mi_out="${OUT_DIR}/${model}_B${B}_membership.csv"
    mi_summ="${OUT_DIR}/${model}_B${B}_membership.summary.json"
    srv_out="${OUT_DIR}/${model}_B${B}_system_prompt.csv"

    if [[ -s "${mi_out}" && -s "${mi_summ}" && -s "${srv_out}" ]]; then
      echo "[E2_SKIP] model=${model} B=${B}"
      continue
    fi

    start_server "${model}" "${mode}" "${cli_budget}"

    if [[ ! -s "${mi_out}" || ! -s "${mi_summ}" ]]; then
      rm -f "${mi_out}" "${mi_summ}"
      echo "[E2_MI_START] model=${model} B=${B}"
      "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
        --server "${SERVER}" \
        --model "${model}" \
        --model-path "${MODEL_PATH[$model]}" \
        --policy "${policy}" \
        --access-budget-B "${B}" \
        --budget-Q 200 \
        --n-challenges 100 \
        --n-attacker-accounts 2 \
        --membership-only \
        --post-victim-settle-ms 750 \
        --dataset "${DATASET}" \
        --seed $((20260821 + B)) \
        --output "${mi_out}" \
        2>&1 | tee "${LOG_DIR}/${model}_B${B}_mi_client.log"
      echo "[E2_MI_DONE] model=${model} B=${B}"
    else
      echo "[E2_MI_SKIP] model=${model} B=${B}"
    fi

    if [[ ! -s "${srv_out}" ]]; then
      echo "[E2_SRV_START] model=${model} B=${B}"
      "${PYTHON}" "${SCRIPT_DIR}/exp5_serving_perf.py" \
        --server "${SERVER}" \
        --model "${model}" \
        --model-path "${MODEL_PATH[$model]}" \
        --workload system_prompt \
        --policy "$([[ "${mode}" == strict ]] && echo strict || echo balanced)" \
        --n-users 20 --rps 8 --max-new-tokens 64 \
        --seed $((20260821 + B)) \
        --dataset "${DATASET}" \
        --operator-key "${OPERATOR_KEY}" \
        --output "${srv_out}" \
        2>&1 | tee "${LOG_DIR}/${model}_B${B}_serving_client.log"
      # Stamp B into a sidecar for aggregation (serving CSV schema has no B field).
      "${PYTHON}" - "${srv_out}" "${B}" "${mode}" <<'PY'
import csv, json, sys
from pathlib import Path
path = Path(sys.argv[1]); B = int(sys.argv[2]); mode = sys.argv[3]
rows = list(csv.DictReader(path.open()))
meta = {
    "access_budget_B": B,
    "safekv_mode": mode,
    "source_csv": path.name,
    "row": rows[-1] if rows else {},
}
sidecar = path.with_suffix(".meta.json")
sidecar.write_text(json.dumps(meta, indent=2) + "\n")
PY
      echo "[E2_SRV_DONE] model=${model} B=${B}"
    else
      echo "[E2_SRV_SKIP] model=${model} B=${B}"
    fi

    cleanup
  done
done

echo "[E2_ALL_DONE] out=${OUT_DIR}"
