#!/usr/bin/env bash
# Resume incomplete submission-gap stages (skip-completed inside each runner).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/submission_gap_experiments"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_resume.log"

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

log "[RESUME_ORCH_START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_stage E4 "${SCRIPT_DIR}/run_e4_principal_binding_phi4.sh"
run_stage E2 "${SCRIPT_DIR}/run_e2_budget_sweep.sh"
run_stage E5 "${SCRIPT_DIR}/run_e5_strong_recovery.sh"
run_stage E3 "${SCRIPT_DIR}/run_e3_serving_repeated.sh"

SAFEKV_PYTHON="${SAFEKV_PYTHON:-python3}"
"${SAFEKV_PYTHON}" "${SCRIPT_DIR}/aggregate_submission_gap_experiments.py" | tee -a "${MASTER_LOG}"

if ((${#FAILED[@]})); then
  log "[RESUME_PARTIAL] failed=${FAILED[*]} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
log "[RESUME_ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "RESUME_ALL_DONE"
