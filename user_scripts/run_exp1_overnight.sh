#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SAFEKV_PYTHON:-python3}"
SERVER="http://127.0.0.1:8092"
RESULT_DIR="${SCRIPT_DIR}/results/exp1_revised"
PHI_DRIVER_PID="${1:-}"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export SAFEKV_OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-exp1-operator-key}"

SERVER_LAUNCHER_PID=""

cleanup_server() {
  pkill -TERM -f "python3 -m sglang.launch_server.*--port 8092" 2>/dev/null || true
  if [[ -n "${SERVER_LAUNCHER_PID}" ]]; then
    kill "${SERVER_LAUNCHER_PID}" 2>/dev/null || true
    wait "${SERVER_LAUNCHER_PID}" 2>/dev/null || true
    SERVER_LAUNCHER_PID=""
  fi
  for _ in $(seq 1 120); do
    if ! curl -fsS "${SERVER}/health" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
}

fail() {
  echo "OVERNIGHT_FAILED: $*" >&2
  cleanup_server
  exit 1
}
trap 'fail "line ${LINENO}"' ERR

validate_raw() {
  local file="$1"
  "${PYTHON}" - "${file}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 800:
    raise SystemExit(f"{path}: expected 800 rows, found {len(rows)}")
if any(row["error"] for row in rows):
    raise SystemExit(f"{path}: contains failed rows")
if not all(int(row["pass"]) for row in rows):
    raise SystemExit(f"{path}: contains invariant failures")
print(f"validated {path}: 800/800 passed", flush=True)
PY
}

wait_for_server() {
  local model="$1"
  for _ in $(seq 1 1800); do
    if curl -fsS "${SERVER}/health" >/dev/null 2>&1; then
      echo "${model} server ready" >&2
      return
    fi
    if ! kill -0 "${SERVER_LAUNCHER_PID}" 2>/dev/null; then
      fail "${model} server exited during startup"
    fi
    sleep 1
  done
  fail "${model} server startup timed out"
}

run_model() {
  local model="$1"
  local model_path="$2"
  local output="$3"
  cleanup_server
  bash "${SCRIPT_DIR}/launch_model.sh" "${model}" \
    >"${SCRIPT_DIR}/logs/${model}_exp1_launcher.log" 2>&1 &
  SERVER_LAUNCHER_PID=$!
  wait_for_server "${model}"
  "${PYTHON}" "${SCRIPT_DIR}/exp1_promotion_integrity.py" \
    --server "${SERVER}" \
    --model "${model}" \
    --model-path "${model_path}" \
    --trials 20 \
    --output "${output}"
  validate_raw "${output}"
}

cd "${PROJECT_DIR}"

if [[ -n "${PHI_DRIVER_PID}" ]]; then
  while kill -0 "${PHI_DRIVER_PID}" 2>/dev/null; do
    sleep 10
  done
fi
validate_raw "${RESULT_DIR}/phi4_raw.csv"

run_model \
  qwen30b \
  /workspace/Models/Qwen3-30B-A3B-Instruct-2507 \
  "${RESULT_DIR}/qwen30b_raw.csv"

run_model \
  qwen32b \
  /workspace/Models/Qwen3-32B \
  "${RESULT_DIR}/qwen32b_raw.csv"

"${PYTHON}" "${SCRIPT_DIR}/aggregate_exp1.py" \
  "${RESULT_DIR}/phi4_raw.csv" \
  "${RESULT_DIR}/qwen30b_raw.csv" \
  "${RESULT_DIR}/qwen32b_raw.csv" \
  --trials 20 \
  --output-dir "${RESULT_DIR}"

cleanup_server
trap - ERR
echo "ALL_EXPERIMENTS_COMPLETE"
