#!/usr/bin/env bash
# Combined Table-7 + Table-16 grid.
# B in {0,10,25,50,75,100,150} on 5 models.
# Membership: exp3 n=100 Q=200 A=2 (reuse e2 summaries when present).
# Serving: TTFT / TPS / Coverage=defense_rate on the same server.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${COMBINED_PORT:-8092}"
SERVER="http://127.0.0.1:${PORT}"
OUT="${SCRIPT_DIR}/results/combined_b_table"
E2="${SCRIPT_DIR}/results/submission_gap_experiments/e2_budget_sweep"
LOG="${SCRIPT_DIR}/logs/combined_b_table"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-e2-operator-key}"
mkdir -p "${OUT}" "${LOG}"

BUDGETS=(0 10 25 50 75 100 150)

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
  [ds_r1_qwen32b]="/workspace/Models/DeepSeek-R1-Distill-Qwen-32B"
  [llama70b_awq]="/workspace/Models/Llama-3.3-70B-Instruct-AWQ"
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

has_membership() {
  local model="$1" B="$2"
  [[ -s "${OUT}/${model}_B${B}_membership.summary.json" ]] && return 0
  [[ -s "${E2}/${model}_B${B}_membership.summary.json" ]] && return 0
  return 1
}

has_serving() {
  local model="$1" B="$2"
  [[ -s "${OUT}/${model}_B${B}_serving.json" ]]
}

start_server() {
  local model="$1" mode="$2" B="$3"
  cleanup
  local gpus=0
  [[ "${MODEL_TP[$model]}" == 2 ]] && gpus=0,1
  local log="${LOG}/${model}_B${B}_${mode}_server.log"
  local extra=()
  if [[ -n "${MODEL_QUANT[$model]}" ]]; then
    extra+=(--quantization "${MODEL_QUANT[$model]}")
  fi
  echo "[COMBINED_SERVER_START] model=${model} B=${B} mode=${mode}"
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
    --safekv-creator-threshold 2 \
    --safekv-operator-key "${OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    "${extra[@]}" \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 240); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[COMBINED_SERVER_READY] model=${model} B=${B} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[COMBINED_SERVER_FAILED] model=${model} B=${B} log=${log}" >&2
      tail -30 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[COMBINED_SERVER_TIMEOUT] model=${model} B=${B}" >&2
  return 1
}

for model in "${MODELS[@]}"; do
  for B in "${BUDGETS[@]}"; do
    need_mi=0
    need_srv=0
    has_membership "${model}" "${B}" || need_mi=1
    has_serving "${model}" "${B}" || need_srv=1
    if [[ "${need_mi}" == 0 && "${need_srv}" == 0 ]]; then
      echo "[COMBINED_SKIP] model=${model} B=${B}"
      continue
    fi
    if [[ "${B}" == "0" ]]; then
      mode=strict
      cli_budget=1
      policy=B0_strict
    else
      mode=balanced
      cli_budget="${B}"
      policy="B${B}_balanced"
    fi
    if ! start_server "${model}" "${mode}" "${cli_budget}"; then
      echo "[COMBINED_FAILED] model=${model} B=${B} reason=server" | tee -a "${LOG}/failed.log"
      cleanup
      continue
    fi
    if [[ "${need_mi}" == 1 ]]; then
      echo "[COMBINED_MI_START] model=${model} B=${B}"
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
        --output "${OUT}/${model}_B${B}_membership.csv" \
        2>&1 | tee "${LOG}/${model}_B${B}_mi_client.log"
      echo "[COMBINED_MI_DONE] model=${model} B=${B}"
    else
      echo "[COMBINED_MI_SKIP] model=${model} B=${B}"
    fi
    if [[ "${need_srv}" == 1 ]]; then
      echo "[COMBINED_SRV_START] model=${model} B=${B}"
      srv_csv="${OUT}/${model}_B${B}_system_prompt.csv"
      e2_meta="${E2}/${model}_B${B}_system_prompt.meta.json"
      ttft_s=""
      tps=""
      if [[ -s "${e2_meta}" ]]; then
        read -r ttft_s tps < <("${PYTHON}" - "${e2_meta}" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
row = m.get("row") or {}
ttft = float(row["mean_ttft_ms"]) / 1000.0
tps = float(row["throughput_tok_s"])
print(f"{ttft:.6f} {tps:.6f}")
PY
)
        echo "[COMBINED_SRV_FROM_E2] model=${model} B=${B} ttft=${ttft_s} tps=${tps}"
      else
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
          --output "${srv_csv}" \
          2>&1 | tee "${LOG}/${model}_B${B}_exp5_client.log"
        read -r ttft_s tps < <("${PYTHON}" - "${srv_csv}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
row = rows[-1]
print(f"{float(row['mean_ttft_ms'])/1000.0:.6f} {float(row['throughput_tok_s']):.6f}")
PY
)
      fi
      if "${PYTHON}" "${SCRIPT_DIR}/measure_serving_coverage.py" \
        --server "${SERVER}" \
        --model "${model}" \
        --B "${B}" \
        --ttft-s "${ttft_s}" \
        --tps "${tps}" \
        --output "${OUT}/${model}_B${B}_serving.json" \
        2>&1 | tee "${LOG}/${model}_B${B}_srv_client.log"; then
        echo "[COMBINED_SRV_DONE] model=${model} B=${B}"
      else
        echo "[COMBINED_SRV_FAILED] model=${model} B=${B} writing ttft/tps only" | tee -a "${LOG}/failed.log"
        "${PYTHON}" - "${OUT}/${model}_B${B}_serving.json" "${model}" "${B}" "${ttft_s}" "${tps}" <<'PY'
import json, sys
from pathlib import Path
path, model, B, ttft, tps = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
Path(path).write_text(json.dumps({
    "model": model, "B": B, "ttft_s": ttft, "tps": tps,
    "coverage_pct": None, "source": "exp5_only_coverage_failed",
}, indent=2) + "\n")
PY
      fi
    else
      echo "[COMBINED_SRV_SKIP] model=${model} B=${B}"
    fi
    cleanup
    "${PYTHON}" "${SCRIPT_DIR}/export_combined_b_table.py" \
      >"${OUT}/combined_b_table.txt" || true
  done
done

"${PYTHON}" "${SCRIPT_DIR}/export_combined_b_table.py"
echo "[COMBINED_ALL_DONE] out=${OUT}"
