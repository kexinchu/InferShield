#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL_PATH="${PHI4_MODEL_PATH:-/workspace/Models/Phi-4}"
PORT="${PORT:-8094}"
OUT="${ROOT}/user_scripts/results/submission_gap_experiments/e4_principal_binding"
CREDS="$(mktemp)"

mkdir -p "${OUT}"
chmod 600 "${CREDS}"
export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

stop_server() {
  local pids
  pids="$(lsof -ti:"${PORT}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti:"${PORT}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      kill -9 ${pids} 2>/dev/null || true
    fi
  fi
  for _ in $(seq 1 30); do
    if ! lsof -ti:"${PORT}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "E4 port ${PORT} did not release" >&2
  return 1
}

cleanup() {
  stop_server
  rm -f "${CREDS}"
}
trap cleanup EXIT

"${PYTHON}" - "${CREDS}" <<'PY'
import json
import secrets
import sys

with open(sys.argv[1], "w", encoding="utf-8") as file:
    json.dump(
        {
            secrets.token_urlsafe(32): "victim",
            secrets.token_urlsafe(32): "attacker",
        },
        file,
    )
PY

start_server() {
  local phase="$1"
  local server_log="${OUT}/phi4_strict_${phase}_server.log"
  stop_server
  local binding_args=()
  if [[ "${phase}" == "enabled" ]]; then
    binding_args=(--principal-binding-file "${CREDS}")
  fi

  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name phi4 \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --trust-remote-code \
    --tp-size 1 --context-length 16384 \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static 0.80 --schedule-policy fcfs \
    --enable-cache-report \
    --safekv-mode strict --safekv-policy-epoch 1 \
    "${binding_args[@]}" \
    >"${server_log}" 2>&1 &

  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      # Verify the intended binding mode is actually live.
      local code
      code="$(curl -s -o /dev/null -w '%{http_code}' \
        "http://127.0.0.1:${PORT}/safekv/effective_principal" || true)"
      if [[ "${phase}" == "disabled" && "${code}" == "404" ]]; then
        # Wait until warmup finishes so flush_cache is usable.
        for _ in $(seq 1 60); do
          if curl -sf -X POST "http://127.0.0.1:${PORT}/flush_cache" >/dev/null 2>&1; then
            return 0
          fi
          sleep 1
        done
        echo "E4 disabled server ready but flush never succeeded" >&2
        return 1
      fi
      if [[ "${phase}" == "enabled" && "${code}" == "401" ]]; then
        for _ in $(seq 1 60); do
          # Authenticated flush using the first configured token.
          local token
          token="$("${PYTHON}" - "${CREDS}" <<'PY'
import json, sys
print(next(iter(json.load(open(sys.argv[1])))))
PY
)"
          if curl -sf -X POST "http://127.0.0.1:${PORT}/flush_cache" \
              -H "Authorization: Bearer ${token}" >/dev/null 2>&1 \
            || curl -sf -X POST "http://127.0.0.1:${PORT}/flush_cache" >/dev/null 2>&1; then
            return 0
          fi
          sleep 1
        done
        echo "E4 enabled server ready but flush never succeeded" >&2
        return 1
      fi
      echo "E4 health up but binding mode mismatch phase=${phase} code=${code}" >&2
    fi
    sleep 5
  done
  echo "E4 server startup timed out (${phase})" >&2
  return 1
}

# Refuse to run if another E4/server already owns the port.
if lsof -ti:"${PORT}" >/dev/null 2>&1; then
  echo "E4 port ${PORT} already in use; refusing to race" >&2
  exit 1
fi

rm -f "${OUT}/disabled.json" "${OUT}/enabled.json" "${OUT}/manifest.json"

start_server disabled
"${PYTHON}" "${ROOT}/user_scripts/exp4_principal_binding.py" \
  --server "http://127.0.0.1:${PORT}" \
  --phase disabled \
  --output-dir "${OUT}"
[[ -s "${OUT}/disabled.json" ]] || { echo "E4 missing disabled.json" >&2; exit 1; }

start_server enabled
"${PYTHON}" "${ROOT}/user_scripts/exp4_principal_binding.py" \
  --server "http://127.0.0.1:${PORT}" \
  --phase enabled \
  --credentials "${CREDS}" \
  --output-dir "${OUT}"
[[ -s "${OUT}/enabled.json" ]] || { echo "E4 missing enabled.json" >&2; exit 1; }

"${PYTHON}" "${ROOT}/user_scripts/exp4_principal_binding.py" \
  --finalize \
  --output-dir "${OUT}"
[[ -s "${OUT}/manifest.json" ]] || { echo "E4 missing manifest.json" >&2; exit 1; }

echo "E4 complete: ${OUT}/manifest.json"
