#!/usr/bin/env bash
# PromptPeek-aligned TTFT recovery campaign.
# 50 trials × 5 slots, TTFT-only, LPM races, remote jitter. No cached_tokens.
#
#   nohup ./ndss_scripts/run_asr_ttft_campaign.sh > ndss_scripts/logs/asr_recovery/campaign.log 2>&1 &
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

run_cell() {
  local policy="$1" model="$2" trials="$3"
  echo "[CAMPAIGN] $(date -Is) POLICY=${policy} model=${model} n=${trials} j=${JITTER} r=${REPEATS} k=${VOCAB}"
  POLICY="${policy}" TRIALS="${trials}" "${SCRIPT_DIR}/run_asr_recovery.sh" "${model}"
}

# Pairwise LPM + 5% open-miss already calibrated on n=3 (15/15 closed, ~95% open).
for policy in vanilla strict; do
  run_cell "${policy}" phi4 50
done
for model in qwen32b qwen30b; do
  for policy in vanilla strict; do
    run_cell "${policy}" "${model}" 50
  done
done

export B=100
for model in phi4 qwen32b qwen30b; do
  run_cell autoshare "${model}" 50
done
for b in 0 10 50 150; do
  export B=$b
  run_cell autoshare phi4 50
done

echo "[CAMPAIGN_DONE] $(date -Is)"
python3 "${SCRIPT_DIR}/plot_table7_asr.py" || true
