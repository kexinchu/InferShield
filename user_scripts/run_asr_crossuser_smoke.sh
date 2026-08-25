#!/usr/bin/env bash
# Cross-principal isolation smoke, then full campaign if Strict is not vanilla-like.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/asr_recovery"
mkdir -p "${LOG_DIR}"

export VOCAB="${VOCAB:-6}"
export REPEATS="${REPEATS:-6}"
export CHUNK="${CHUNK:-64}"
export JITTER="${JITTER:-8}"
export OPEN_MISS="${OPEN_MISS:-0.05}"
export TOKENS="${TOKENS:-5}"

echo "[SMOKE] $(date -Is) Strict n=3 (must be near chance, not ~vanilla)"
POLICY=strict TRIALS=3 "${SCRIPT_DIR}/run_asr_recovery.sh" phi4
echo "[SMOKE] $(date -Is) vanilla n=3 (must stay high)"
POLICY=vanilla TRIALS=3 "${SCRIPT_DIR}/run_asr_recovery.sh" phi4
echo "[SMOKE_DONE] $(date -Is)"
