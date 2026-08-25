#!/usr/bin/env bash
# Resume only E2 + E3 after stream TTFT fix (skip-completed inside runners).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/submission_gap_experiments"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_e2_e3.log"

log() { echo "$*" | tee -a "${MASTER_LOG}"; }

LOCK="${LOG_DIR}/orchestrator.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another submission-gap orchestrator holds the lock" >&2
  exit 1
fi

for port in 8092 8094; do
  pids="$(lsof -ti:${port} 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 3
  fi
done

FAILED=()
run_stage() {
  local name="$1" script="$2"
  log "[RESUME_START] ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if bash "${script}" >>"${MASTER_LOG}" 2>&1; then
    log "[RESUME_DONE] ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    local rc=$?
    log "[RESUME_FAIL] ${name} exit=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    FAILED+=("${name}:${rc}")
  fi
}

log "[E2E3_ORCH_START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_stage E2 "${SCRIPT_DIR}/run_e2_budget_sweep.sh"
run_stage E3 "${SCRIPT_DIR}/run_e3_serving_repeated.sh"

SAFEKV_PYTHON="${SAFEKV_PYTHON:-python3}"
"${SAFEKV_PYTHON}" "${SCRIPT_DIR}/aggregate_submission_gap_experiments.py" | tee -a "${MASTER_LOG}"

if ((${#FAILED[@]})); then
  log "[E2E3_PARTIAL] failed=${FAILED[*]} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
log "[E2E3_ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "E2E3_ALL_DONE"
