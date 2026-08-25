#!/usr/bin/env bash
# Repeated-probe Table 7: one PII prefix, k probes, B in {0,10,25,50,75,100,150}.
#
#   ADMISSION=detector   ./ndss_scripts/run_table7_repeated_probe.sh phi4
#   ADMISSION=autoshare  ./ndss_scripts/run_table7_repeated_probe.sh phi4
#
# detector: conservative gate holds PII; expect ~0 hits at every B.
# autoshare: forced FN / residual path (server --safekv-experiment-autoshare);
#            hits should flatten at min(B, k). This is the B-knee figure.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${T7_PORT:-8094}"
SERVER="http://127.0.0.1:${PORT}"
OUT="${SCRIPT_DIR}/results/table7_repeated_probe"
LOG="${SCRIPT_DIR}/logs/table7_repeated_probe"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-e2-operator-key}"
ADMISSION="${ADMISSION:-detector}"
K="${K:-160}"
N_GAMES="${N_GAMES:-12}"
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
  MODELS=(phi4)
fi
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  local pids
  pids="$(lsof -ti:${PORT} 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} 2>/dev/null || true
    sleep 4
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  local model="$1" mode="$2" B="$3"
  cleanup
  local gpus=0
  [[ "${MODEL_TP[$model]}" == 2 ]] && gpus=0,1
  local log="${LOG}/${model}_${ADMISSION}_B${B}_${mode}_server.log"
  local extra=()
  if [[ -n "${MODEL_QUANT[$model]}" ]]; then
    extra+=(--quantization "${MODEL_QUANT[$model]}")
  fi
  if [[ "${ADMISSION}" == "autoshare" ]]; then
    extra+=(--safekv-experiment-autoshare)
  fi
  echo "[T7_SERVER_START] model=${model} B=${B} mode=${mode} admission=${ADMISSION}"
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
      echo "[T7_SERVER_READY] model=${model} B=${B} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[T7_SERVER_FAILED] model=${model} B=${B} log=${log}" >&2
      tail -30 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[T7_SERVER_TIMEOUT] model=${model} B=${B}" >&2
  return 1
}

for model in "${MODELS[@]}"; do
  for B in "${BUDGETS[@]}"; do
    out="${OUT}/${model}_${ADMISSION}_B${B}_repeated.json"
    if [[ -s "${out}" ]]; then
      echo "[T7_SKIP] ${out}"
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
      echo "[T7_FAILED] model=${model} B=${B} reason=server" | tee -a "${LOG}/failed.log"
      continue
    fi
    PYTHONUNBUFFERED=1 "${PYTHON}" "${SCRIPT_DIR}/exp_table7_repeated_probe.py" \
      --server "${SERVER}" \
      --model "${model}" \
      --B "${B}" \
      --k "${K}" \
      --n-games "${N_GAMES}" \
      --admission "${ADMISSION}" \
      --timeout 300 \
      --output "${out}" \
      2>&1 | tee "${LOG}/${model}_${ADMISSION}_B${B}_client.log"
    cleanup
  done
done

"${PYTHON}" "${SCRIPT_DIR}/plot_table7_repeated_probe.py" \
  --indir "${OUT}" \
  --output "${PROJECT_DIR}/ndss_scripts/figures/table7_repeated_probe.pdf" || true
echo "[T7_DONE] out=${OUT}"
