#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="python3"
MODEL_PATH="/workspace/Models/Phi-4"
PORT=8092
OUT="${ROOT}/ndss_scripts/results/exp11_v2"
LOG="${ROOT}/ndss_scripts/logs"
mkdir -p "${OUT}" "${LOG}"

export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export SAFEKV_OPERATOR_KEY="safekv-exp11-key"

pids="$(lsof -ti:${PORT} 2>/dev/null || true)"
if [[ -n "${pids}" ]]; then
  kill ${pids} 2>/dev/null || true
  sleep 5
fi

CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size 1 --context-length 16384 \
    --served-model-name phi4 \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static 0.80 --schedule-policy fcfs \
    --safekv-mode balanced --safekv-access-budget 100 \
    --safekv-operator-key "${SAFEKV_OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    >"${LOG}/phi4_p11_v2_server.log" 2>&1 &

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR P11 server timeout"
  exit 1
fi

"${PYTHON}" "${ROOT}/ndss_scripts/exp11_registry_scaling.py" \
  --server "http://127.0.0.1:${PORT}" \
  --model phi4 \
  --model-path "${MODEL_PATH}" \
  --n-samples 300 \
  --public-token-levels 0 512 2048 8192 \
  --live-requests 20 \
  --output "${OUT}/scaling.json"

pids="$(lsof -ti:${PORT} 2>/dev/null || true)"
[[ -n "${pids}" ]] && kill ${pids} 2>/dev/null || true
echo "P11_ALL_DONE"
