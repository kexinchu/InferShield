#!/usr/bin/env bash
# Serial orchestrator for submission-gap experiments.
# Order: E4 (fast trust boundary) → E1 → E2 → E5 → E3 (longest serving matrix).
# Does not modify paper files. All outputs under results/submission_gap_experiments/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/submission_gap_experiments"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator.log"

log() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

FAILED_STAGES=()

run_stage() {
  local name="$1"
  local script="$2"
  log "[STAGE_START] ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if bash "${script}" >>"${MASTER_LOG}" 2>&1; then
    log "[STAGE_DONE] ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    local rc=$?
    log "[STAGE_FAIL] ${name} exit=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    FAILED_STAGES+=("${name}:${rc}")
  fi
}

# Ensure a single orchestrator instance.
LOCK="${LOG_DIR}/orchestrator.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another submission-gap orchestrator is already running" >&2
  exit 1
fi

# Clear any leftover servers on the experiment ports.
for port in 8092 8094; do
  pids="$(lsof -ti:${port} 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 3
  fi
done

log "[ORCH_START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_stage E4 "${SCRIPT_DIR}/run_e4_principal_binding_phi4.sh"
run_stage E1 "${SCRIPT_DIR}/run_e1_vanilla_attack.sh"
run_stage E2 "${SCRIPT_DIR}/run_e2_budget_sweep.sh"
run_stage E5 "${SCRIPT_DIR}/run_e5_strong_recovery.sh"
run_stage E3 "${SCRIPT_DIR}/run_e3_serving_repeated.sh"

SAFEKV_PYTHON="${SAFEKV_PYTHON:-python3}"
"${SAFEKV_PYTHON}" "${SCRIPT_DIR}/aggregate_submission_gap_experiments.py" \
  | tee -a "${MASTER_LOG}"

if ((${#FAILED_STAGES[@]})); then
  log "[ORCH_PARTIAL] failed=${FAILED_STAGES[*]} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ORCH_PARTIAL failed=${FAILED_STAGES[*]}"
  exit 1
fi
log "[ORCH_ALL_DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ORCH_ALL_DONE"
