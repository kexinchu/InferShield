#!/usr/bin/env bash
# Chain: wait for the live campaign, then run Table5+Table7 (skips done cells).
# Restart the follow-on if it dies before CAMPAIGN_DONE.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/logs/asr_recovery/campaign_table5_table7.log"
LIVE_PID="${1:-}"
mkdir -p "${SCRIPT_DIR}/logs/asr_recovery"

if [[ -n "${LIVE_PID}" ]]; then
  echo "[WATCH] $(date -Is) waiting for live campaign pid=${LIVE_PID}"
  while kill -0 "${LIVE_PID}" 2>/dev/null; do
    sleep 20
  done
  echo "[WATCH] $(date -Is) live campaign exited"
fi

while true; do
  if grep -q '\[CAMPAIGN_DONE\]' "${LOG}" 2>/dev/null; then
    echo "[WATCH] $(date -Is) Table5+7 campaign already done"
    exit 0
  fi
  echo "[WATCH] $(date -Is) starting Table5+7 campaign"
  "${SCRIPT_DIR}/run_asr_table5_table7.sh" >>"${LOG}" 2>&1
  rc=$?
  echo "[WATCH] $(date -Is) campaign exit=${rc}"
  if grep -q '\[CAMPAIGN_DONE\]' "${LOG}" 2>/dev/null; then
    exit 0
  fi
  echo "[WATCH] retry in 120s"
  sleep 120
done
