#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="python3"
PORT=8092
KEY="safekv-exp8-operator-key"
OUT="${ROOT}/user_scripts/results/exp8_v2"
LOG="${ROOT}/user_scripts/logs"
mkdir -p "${OUT}" "${LOG}"

export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

declare -A PATHS=(
  [phi4]="/workspace/Models/Phi-4"
  [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
  [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A TPS=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A GPUS=([phi4]=0 [qwen30b]="0,1" [qwen32b]="0,1")

stop_server() {
  local pids
  pids="$(lsof -ti:${PORT} 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 5
  fi
}

start_server() {
  local model="$1"
  stop_server
  CUDA_VISIBLE_DEVICES="${GPUS[$model]}" \
    "${PYTHON}" -m sglang.launch_server \
      --model-path "${PATHS[$model]}" \
      --host 127.0.0.1 --port "${PORT}" \
      --dtype bfloat16 --trust-remote-code \
      --tp-size "${TPS[$model]}" --context-length 16384 \
      --served-model-name "${model}" \
      --attention-backend torch_native --disable-cuda-graph \
      --mem-fraction-static 0.80 --schedule-policy fcfs \
      --safekv-mode strict \
      --safekv-operator-key "${KEY}" --safekv-policy-epoch 1 \
      >"${LOG}/${model}_p8_v2_server.log" 2>&1 &

  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR P8 server timeout: ${model}"
  return 1
}

for model in phi4 qwen30b qwen32b; do
  echo "P8_START ${model}"
  rm -f "${OUT}/${model}.csv" "${OUT}/${model}_revocation_bench.json"
  start_server "${model}"
  "${PYTHON}" "${ROOT}/user_scripts/exp8_auth_matrix.py" \
    --server "http://127.0.0.1:${PORT}" \
    --model "${model}" \
    --model-path "${PATHS[$model]}" \
    --trials 5 \
    --operator-key "${KEY}" \
    --output "${OUT}/${model}.csv"
  echo "P8_DONE ${model}"
done

stop_server
echo "P8_ALL_DONE"
