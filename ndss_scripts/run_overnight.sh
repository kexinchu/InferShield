#!/bin/bash
# Overnight experiment pipeline — Phi-4 / Qwen3-30B / Qwen3-32B / (Llama-70B pending)
# Experiments: throughput_ablation, safekv_ablation, E6 multi-tenant, E2 budget-B
# Usage: bash run_overnight.sh [--skip-sweep-wait]

set -uo pipefail

PYTHON=/home/kec23008/.venv/bin/python3
SCRIPTS=/home/kec23008/InferShield/ndss_scripts
LOGDIR=/tmp/overnight
mkdir -p "$LOGDIR"
PIPE_LOG="$LOGDIR/pipeline.log"

# Models to run (llama70b omitted — weights not downloaded)
MODELS_FULL="phi4 qwen30b qwen32b"
MODELS_E6="phi4"          # E6 multi-tenant: phi4 only (SGLang baselines exist)
MODELS_E2="phi4"          # E2 budget-B: phi4 only

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PIPE_LOG"
}

kill_servers() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "python.*launch_server" 2>/dev/null || true
    sleep 8
}

run_step() {
    local name="$1"; shift
    log "=== START: $name ==="
    kill_servers
    if "$@" 2>&1 | tee "$LOGDIR/${name}.log"; then
        log "=== DONE:  $name ==="
    else
        log "=== ERROR: $name (exit $?) — continuing ==="
    fi
    kill_servers
}

# ── 0. Wait for any running pii_ratio_sweep to finish ───────────────────────
if [[ "${1:-}" != "--skip-sweep-wait" ]] && pgrep -f "test_pii_ratio_sweep.py" > /dev/null 2>&1; then
    log "Waiting for pii_ratio_sweep to finish..."
    while pgrep -f "test_pii_ratio_sweep.py" > /dev/null 2>&1; do sleep 30; done
    log "pii_ratio_sweep finished."
    kill_servers
fi

# ── 1. throughput_ablation — all 3 models ───────────────────────────────────
for MODEL in $MODELS_FULL; do
    run_step "throughput_${MODEL}" \
        $PYTHON $SCRIPTS/test_throughput_ablation.py --model "$MODEL"
done

# ── 2. safekv_ablation — all 3 models ───────────────────────────────────────
for MODEL in $MODELS_FULL; do
    run_step "ablation_${MODEL}" \
        $PYTHON $SCRIPTS/test_safekv_ablation.py --model "$MODEL"
done

# ── 3. pii_ratio_sweep — qwen32b only (phi4 done, qwen30b running) ──────────
# Check if bug-fixed qwen32b sweep exists; if not, rerun
QWEN32B_SWEEP=$(ls "$SCRIPTS/logs/pii_ratio_sweep_qwen32b_20260420_230333.csv" 2>/dev/null || true)
if [[ -z "$QWEN32B_SWEEP" ]]; then
    run_step "pii_sweep_qwen32b" \
        $PYTHON $SCRIPTS/test_pii_ratio_sweep.py --model qwen32b
else
    log "=== SKIP: pii_sweep_qwen32b (bug-fixed result exists) ==="
fi

# ── 4. E6 Multi-tenant — phi4, SafeKV + Cache-Partition ─────────────────────
log "=== START: E6 multi-tenant phi4 ==="
kill_servers
for C in 1 4 8 16 32; do
    log "  E6 phi4 concurrency=$C"
    kill_servers
    $PYTHON $SCRIPTS/test_throughput_ablation.py \
        --model phi4 \
        --max-workers "$C" \
        --modes private_default full_safekv \
        2>&1 | tee -a "$LOGDIR/e6_phi4_c${C}.log"
    kill_servers
done
log "=== DONE: E6 multi-tenant phi4 ==="

# ── 5. E2 Budget-B — phi4, full_safekv, B = 10 / 100 / 500 ─────────────────
log "=== START: E2 budget-B phi4 ==="
kill_servers
for B in 10 100 500; do
    log "  E2 phi4 B=$B"
    kill_servers
    $PYTHON $SCRIPTS/test_throughput_ablation.py \
        --model phi4 \
        --modes full_safekv \
        --budget "$B" \
        2>&1 | tee -a "$LOGDIR/e2_phi4_B${B}.log"
    kill_servers
done
log "=== DONE: E2 budget-B phi4 ==="

log "================================================================"
log "ALL EXPERIMENTS COMPLETE"
log "Results: $SCRIPTS/logs/"
log "Logs:    $LOGDIR/"
log "================================================================"
echo ""
echo "Generated CSVs (newest first):"
ls -lt "$SCRIPTS/logs/"*.csv 2>/dev/null | head -30
