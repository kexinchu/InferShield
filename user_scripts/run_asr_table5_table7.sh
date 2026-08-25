#!/usr/bin/env bash
# Table 5 (3 models × SGLang/Strict/Balanced-B100) then
# Table 7 (5 models × B ∈ {0,10,50,100,150}).
# Skips cells that already have a summary. Safe to resume.
#
#   nohup ./user_scripts/run_asr_table5_table7.sh \
#     > user_scripts/logs/asr_recovery/campaign_table5_table7.log 2>&1 &
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
  echo "[CAMPAIGN] $(date -Is) POLICY=${policy} model=${model} n=${trials} B=${B:--} j=${JITTER}"
  POLICY="${policy}" TRIALS="${trials}" "${SCRIPT_DIR}/run_asr_recovery.sh" "${model}"
}

echo "[PHASE_TABLE5] $(date -Is) 3 models × vanilla/strict/autoshare@B100"
for model in phi4 qwen32b qwen30b; do
  for policy in vanilla strict; do
    run_cell "${policy}" "${model}" 50
  done
done
export B=100
for model in phi4 qwen32b qwen30b; do
  run_cell autoshare "${model}" 50
done
echo "[PHASE_TABLE5_DONE] $(date -Is)"
python3 "${SCRIPT_DIR}/update_asr_tables.py" --write-tex || true

echo "[PHASE_TABLE7] $(date -Is) 5 models × B in 0,10,50,100,150"
TABLE7_MODELS=(phi4 qwen30b qwen32b llama70b_awq ds_r1_qwen32b)
# B=0 is Strict isolation. Running autoshare with budget 0 crashes the
# server and fails the share-required preflight. Reuse a Strict summary
# when one exists; otherwise collect Strict (not autoshare B=0).
for model in "${TABLE7_MODELS[@]}"; do
  for b in 0 10 50 100 150; do
    if [[ "${b}" == "0" ]]; then
      strict_summ="${SCRIPT_DIR}/results/asr_recovery/${model}_strict_n50_k${VOCAB}_r${REPEATS}_j${JITTER}_ttft.summary.json"
      if [[ -s "${strict_summ}" ]]; then
        echo "[CAMPAIGN] skip B=0 model=${model}: reuse ${strict_summ}"
        continue
      fi
      unset B
      run_cell strict "${model}" 50
      continue
    fi
    export B=$b
    run_cell autoshare "${model}" 50
  done
done
echo "[PHASE_TABLE7_DONE] $(date -Is)"
python3 "${SCRIPT_DIR}/update_asr_tables.py" --write-tex || true
python3 "${SCRIPT_DIR}/plot_table7_asr.py" || true
echo "[CAMPAIGN_DONE] $(date -Is)"
