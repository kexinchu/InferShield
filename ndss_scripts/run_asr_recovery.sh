#!/usr/bin/env bash
# PromptPeek-style token-recovery ASR.
#
# Default: vanilla SGLang first (positive control), 50 trials x 5 tokens.
#   POLICY=vanilla MODELS="phi4" ./ndss_scripts/run_asr_recovery.sh
#   POLICY=strict  MODELS="phi4 qwen32b" ./ndss_scripts/run_asr_recovery.sh
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${ASR_PORT:-8096}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/asr_recovery"
LOG_DIR="${SCRIPT_DIR}/logs/asr_recovery"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-asr-operator-key}"
POLICY="${POLICY:-vanilla}"
TRIALS="${TRIALS:-50}"
TOKENS="${TOKENS:-5}"
VOCAB="${VOCAB:-6}"
REPEATS="${REPEATS:-15}"
CHUNK="${CHUNK:-64}"
JITTER="${JITTER:-12}"
OPEN_MISS="${OPEN_MISS:-0.05}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

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
declare -A MODEL_GPU=(
  [phi4]="${ASR_GPU_PHI4:-1}"
  [qwen30b]="${ASR_GPU_QWEN:-0,1}"
  [qwen32b]="${ASR_GPU_QWEN:-0,1}"
  [ds_r1_qwen32b]="${ASR_GPU_QWEN:-0,1}"
  [llama70b_awq]="${ASR_GPU_QWEN:-0,1}"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  # shellcheck disable=SC2206
  MODELS=(${MODELS:-phi4})
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
    sleep 3
  fi
}
trap cleanup EXIT INT TERM

safekv_flags() {
  case "${POLICY}" in
    vanilla|sglang|none)
      echo --safekv-mode none --safekv-policy-epoch 1
      ;;
    strict)
      echo --safekv-mode strict --safekv-access-budget 0 --safekv-policy-epoch 1 --safekv-operator-key "${OPERATOR_KEY}"
      ;;
    balanced)
      echo --safekv-mode balanced --safekv-access-budget "${B:-100}" \
        --safekv-policy-epoch 1 --safekv-operator-key "${OPERATOR_KEY}"
      ;;
    autoshare)
      echo --safekv-mode balanced --safekv-access-budget "${B:-100}" \
        --safekv-policy-epoch 1 --safekv-operator-key "${OPERATOR_KEY}" \
        --safekv-experiment-autoshare
      ;;
    *)
      echo "[ASR] unknown POLICY=${POLICY}" >&2
      return 1
      ;;
  esac
}

start_server() {
  local model="$1"
  cleanup
  local log="${LOG_DIR}/${model}_${POLICY}_server.log"
  echo "[ASR_SERVER_START] model=${model} policy=${POLICY} gpu=${MODEL_GPU[$model]}"
  local quant_flags=()
  if [[ -n "${MODEL_QUANT[$model]:-}" ]]; then
    quant_flags+=(--quantization "${MODEL_QUANT[$model]}")
  fi
  CUDA_VISIBLE_DEVICES="${MODEL_GPU[$model]}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype "${MODEL_DTYPE[$model]:-bfloat16}" --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --disable-overlap-schedule \
    --max-running-requests 1 \
    --mem-fraction-static "${MODEL_MEM[$model]:-0.80}" --schedule-policy lpm \
    "${quant_flags[@]}" \
    $(safekv_flags) \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[ASR_SERVER_READY] model=${model} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ASR_SERVER_FAILED] model=${model} log=${log}" >&2
      tail -40 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[ASR_SERVER_TIMEOUT] model=${model}" >&2
  tail -40 "${log}" >&2 || true
  return 1
}

for model in "${MODELS[@]}"; do
  tag_b=""
  if [[ "${POLICY}" == "balanced" || "${POLICY}" == "autoshare" ]]; then
    tag_b="_B${B:-100}"
  fi
  out="${OUT_DIR}/${model}_${POLICY}${tag_b}_n${TRIALS}_k${VOCAB}_r${REPEATS}_j${JITTER}_ttft.csv"
  summ="${out%.csv}.summary.json"
  if [[ -s "${summ}" ]]; then
    echo "[ASR_SKIP] ${summ}"
    continue
  fi
  start_server "${model}"
  echo "[ASR_CELL_START] model=${model} policy=${POLICY} trials=${TRIALS}"
  "${PYTHON}" "${SCRIPT_DIR}/exp_asr_recovery.py" \
    --server "${SERVER}" \
    --model "${model}" \
    --model-path "${MODEL_PATH[$model]}" \
    --policy "${POLICY}" \
    --access-budget-B "${B:--1}" \
    --budget-Q 100000 \
    --n-recovery-trials "${TRIALS}" \
    --n-tokens-to-recover "${TOKENS}" \
    --vocab-sample "${VOCAB}" \
    --repeats "${REPEATS}" \
    --chunk-len "${CHUNK}" \
    --rtt-jitter-ms "${JITTER}" \
    --open-miss "${OPEN_MISS}" \
    --recovery-mode guaranteed \
    --post-victim-settle-ms 200 \
    --dataset "${DATASET}" \
    --seed 20260820 \
    --output "${out}" \
    2>&1 | tee "${LOG_DIR}/${model}_${POLICY}_client.log"
  echo "[ASR_CELL_DONE] model=${model} policy=${POLICY}"
  cleanup
done

echo "[ASR_ALL_DONE] out=${OUT_DIR}"
