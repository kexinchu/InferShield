#!/usr/bin/env bash
# PromptPeek-aligned ASR (open LM set, per-token TTFT, repeats, LPM).
#
#   POLICY=vanilla TRIALS=50 ./user_scripts/run_asr_promptpeek.sh phi4
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${ASR_PORT:-8096}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/asr_promptpeek"
LOG_DIR="${SCRIPT_DIR}/logs/asr_promptpeek"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-asr-operator-key}"
POLICY="${POLICY:-vanilla}"
TRIALS="${TRIALS:-50}"
TOKENS="${TOKENS:-5}"
VOCAB="${VOCAB:-10}"
REPEATS="${REPEATS:-15}"
JITTER="${JITTER:-12}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=8192 [qwen32b]=8192)
declare -A MODEL_GPU=([phi4]="${ASR_GPU_PHI4:-1}" [qwen30b]="${ASR_GPU_QWEN:-0,1}" [qwen32b]="${ASR_GPU_QWEN:-0,1}")

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
      echo --safekv-mode strict --safekv-policy-epoch 1 --safekv-operator-key "${OPERATOR_KEY}"
      ;;
    *)
      echo "[PP] unknown POLICY=${POLICY}" >&2
      return 1
      ;;
  esac
}

start_server() {
  local model="$1"
  cleanup
  local log="${LOG_DIR}/${model}_${POLICY}_server.log"
  echo "[PP_SERVER_START] model=${model} policy=${POLICY} gpu=${MODEL_GPU[$model]}"
  # LPM + concurrent batch: PromptPeek's scheduling channel.
  CUDA_VISIBLE_DEVICES="${MODEL_GPU[$model]}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --disable-overlap-schedule \
    --max-running-requests 16 \
    --mem-fraction-static 0.80 --schedule-policy lpm \
    $(safekv_flags) \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[PP_SERVER_READY] model=${model} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[PP_SERVER_FAILED] model=${model} log=${log}" >&2
      tail -40 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[PP_SERVER_TIMEOUT] model=${model}" >&2
  tail -40 "${log}" >&2 || true
  return 1
}

for model in "${MODELS[@]}"; do
  out="${OUT_DIR}/${model}_${POLICY}_n${TRIALS}_k${VOCAB}_r${REPEATS}_j${JITTER}.csv"
  summ="${out%.csv}.summary.json"
  if [[ -s "${out}" && -s "${summ}" ]]; then
    echo "[PP_SKIP] ${out}"
    continue
  fi
  start_server "${model}"
  echo "[PP_CELL_START] model=${model} policy=${POLICY} n=${TRIALS} k=${VOCAB} r=${REPEATS}"
  "${PYTHON}" "${SCRIPT_DIR}/exp_asr_promptpeek.py" \
    --server "${SERVER}" \
    --model "${model}" \
    --model-path "${MODEL_PATH[$model]}" \
    --policy "${POLICY}" \
    --n-recovery-trials "${TRIALS}" \
    --n-tokens-to-recover "${TOKENS}" \
    --vocab-k "${VOCAB}" \
    --repeats "${REPEATS}" \
    --rtt-jitter-ms "${JITTER}" \
    --dataset "${DATASET}" \
    --seed 20260820 \
    --output "${out}" \
    2>&1 | tee "${LOG_DIR}/${model}_${POLICY}_client.log"
  echo "[PP_CELL_DONE] model=${model} policy=${POLICY}"
  cleanup
done

echo "[PP_ALL_DONE] out=${OUT_DIR}"
