#!/usr/bin/env bash
# Rerun Strict P3 and E2 B=0 on the current isolation code.
# Writes to a new directory; does not overwrite paper artifacts.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${STRICT_RERUN_PORT:-8092}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/strict_isolation_rerun"
LOG_DIR="${SCRIPT_DIR}/logs/strict_isolation_rerun"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-strict-rerun-operator-key}"
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
  pids="$(lsof -ti:"${PORT}" 2>/dev/null || true)"
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
  local log="${LOG_DIR}/${model}_strict_server.log"
  echo "[RERUN_SERVER_START] model=${model} mode=strict B=0"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static 0.80 --schedule-policy fcfs \
    --safekv-mode strict \
    --safekv-access-budget 1 \
    --safekv-operator-key "${OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[RERUN_SERVER_READY] model=${model} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[RERUN_SERVER_FAILED] model=${model} see ${log}" >&2
      tail -n 40 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[RERUN_SERVER_TIMEOUT] model=${model} see ${log}" >&2
  return 1
}

write_status() {
  "${PYTHON}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cells = []
for model in ("phi4", "qwen30b", "qwen32b"):
    for tag, name in (
        ("p3", f"{model}_strict_p3.summary.json"),
        ("e2_b0", f"{model}_B0_membership.summary.json"),
    ):
        path = root / name
        if not path.exists():
            cells.append({"model": model, "experiment": tag, "status": "pending"})
            continue
        data = json.loads(path.read_text())
        cells.append({
            "model": model,
            "experiment": tag,
            "status": "done",
            "roc_auc": data.get("roc_auc"),
            "roc_auc_ci_lo": data.get("roc_auc_ci_lo"),
            "roc_auc_ci_hi": data.get("roc_auc_ci_hi"),
            "adv_mi": data.get("adv_mi"),
            "tpr": data.get("tpr"),
            "fpr": data.get("fpr"),
            "tokens_recovered": data.get("tokens_recovered"),
            "tokens_attempted": data.get("tokens_attempted"),
        })
(root / "status.json").write_text(json.dumps({
    "note": "Isolation-fix rerun. Not paper artifacts.",
    "cells": cells,
}, indent=2) + "\n")
print(json.dumps(cells, indent=2))
PY
}

for model in phi4 qwen30b qwen32b; do
  p3_out="${OUT_DIR}/${model}_strict_p3.csv"
  p3_summ="${OUT_DIR}/${model}_strict_p3.summary.json"
  e2_out="${OUT_DIR}/${model}_B0_membership.csv"
  e2_summ="${OUT_DIR}/${model}_B0_membership.summary.json"

  if [[ -s "${p3_out}" && -s "${p3_summ}" && -s "${e2_out}" && -s "${e2_summ}" ]]; then
    echo "[RERUN_SKIP] model=${model} both cells exist"
    continue
  fi

  start_server "${model}"

  if [[ ! -s "${p3_out}" || ! -s "${p3_summ}" ]]; then
    rm -f "${p3_out}" "${p3_summ}"
    echo "[RERUN_P3_START] model=${model}"
    "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
      --server "${SERVER}" \
      --model "${model}" \
      --model-path "${MODEL_PATH[$model]}" \
      --policy strict \
      --access-budget-B 0 \
      --budget-Q 200 \
      --n-challenges 100 \
      --n-attacker-accounts 2 \
      --n-recovery-trials 10 \
      --n-tokens-to-recover 5 \
      --vocab-sample 20 \
      --dataset "${DATASET}" \
      --seed 20260811 \
      --output "${p3_out}" \
      2>&1 | tee "${LOG_DIR}/${model}_p3_client.log"
    echo "[RERUN_P3_DONE] model=${model}"
  else
    echo "[RERUN_P3_SKIP] model=${model}"
  fi

  if [[ ! -s "${e2_out}" || ! -s "${e2_summ}" ]]; then
    rm -f "${e2_out}" "${e2_summ}"
    echo "[RERUN_E2B0_START] model=${model}"
    "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
      --server "${SERVER}" \
      --model "${model}" \
      --model-path "${MODEL_PATH[$model]}" \
      --policy B0_strict \
      --access-budget-B 0 \
      --budget-Q 200 \
      --n-challenges 100 \
      --n-attacker-accounts 2 \
      --membership-only \
      --post-victim-settle-ms 750 \
      --dataset "${DATASET}" \
      --seed 20260821 \
      --output "${e2_out}" \
      2>&1 | tee "${LOG_DIR}/${model}_e2_b0_client.log"
    echo "[RERUN_E2B0_DONE] model=${model}"
  else
    echo "[RERUN_E2B0_SKIP] model=${model}"
  fi

  write_status
  cleanup
done

write_status
echo "[RERUN_ALL_DONE] out=${OUT_DIR}"
cat "${OUT_DIR}/status.json"
