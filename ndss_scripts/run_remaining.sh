#!/bin/bash
# Remaining experiments for SafeKV paper (post-overnight run)
# Covers: E4 qwen30b re-run, E5 multi-tenant (qwen30b/qwen32b),
#         E2a private_detector at c=16, E2b budget transition time-series
#
# Usage: bash run_remaining.sh [--skip-e4] [--skip-e5] [--skip-e2a] [--skip-e2b]

set -uo pipefail

PYTHON=/home/kec23008/.venv/bin/python3
SCRIPTS=/home/kec23008/InferShield/ndss_scripts
LOGDIR=/tmp/remaining
mkdir -p "$LOGDIR"
PIPE_LOG="$LOGDIR/pipeline.log"

SKIP_E4=0; SKIP_E5=0; SKIP_E2A=0; SKIP_E2B=0
for arg in "$@"; do
    case "$arg" in
        --skip-e4)  SKIP_E4=1  ;;
        --skip-e5)  SKIP_E5=1  ;;
        --skip-e2a) SKIP_E2A=1 ;;
        --skip-e2b) SKIP_E2B=1 ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PIPE_LOG"
}

kill_servers() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "python.*launch_server" 2>/dev/null || true
    sleep 8
}

log "================================================================"
log "Remaining experiments pipeline"
log "================================================================"

# ── E4: qwen30b ablation re-run ──────────────────────────────────────────────
if [[ "$SKIP_E4" -eq 0 ]]; then
    log "=== START: E4 ablation qwen30b re-run ==="
    kill_servers
    $PYTHON "$SCRIPTS/test_safekv_ablation.py" \
        --model qwen30b \
        2>&1 | tee "$LOGDIR/e4_ablation_qwen30b.log"
    kill_servers
    log "=== DONE:  E4 ablation qwen30b ==="
else
    log "=== SKIP: E4 ablation qwen30b ==="
fi

# ── E5: Multi-tenant qwen30b (c=1,4,8,16,32) ────────────────────────────────
if [[ "$SKIP_E5" -eq 0 ]]; then
    log "=== START: E5 multi-tenant qwen30b ==="
    for C in 1 4 8 16 32; do
        log "  E5 qwen30b concurrency=$C"
        kill_servers
        $PYTHON "$SCRIPTS/test_throughput_ablation.py" \
            --model qwen30b \
            --max-workers "$C" \
            --modes private_default full_safekv \
            --n-requests 300 \
            2>&1 | tee "$LOGDIR/e5_qwen30b_c${C}.log"
        kill_servers
    done
    log "=== DONE:  E5 multi-tenant qwen30b ==="

    log "=== START: E5 multi-tenant qwen32b ==="
    for C in 1 4 8 16 32; do
        log "  E5 qwen32b concurrency=$C"
        kill_servers
        $PYTHON "$SCRIPTS/test_throughput_ablation.py" \
            --model qwen32b \
            --max-workers "$C" \
            --modes private_default full_safekv \
            --n-requests 300 \
            2>&1 | tee "$LOGDIR/e5_qwen32b_c${C}.log"
        kill_servers
    done
    log "=== DONE:  E5 multi-tenant qwen32b ==="
else
    log "=== SKIP: E5 multi-tenant ==="
fi

# ── E2a: private_detector at c=16 (all 3 models, for Re-Promoted TTFT) ───────
if [[ "$SKIP_E2A" -eq 0 ]]; then
    log "=== START: E2a private_detector c=16 ==="
    for MODEL in phi4 qwen30b qwen32b; do
        log "  E2a $MODEL c=16"
        kill_servers
        $PYTHON "$SCRIPTS/test_throughput_ablation.py" \
            --model "$MODEL" \
            --max-workers 16 \
            --modes private_detector \
            --n-requests 300 \
            2>&1 | tee "$LOGDIR/e2a_${MODEL}_c16.log"
        kill_servers
    done
    log "=== DONE:  E2a private_detector c=16 ==="
else
    log "=== SKIP: E2a private_detector c=16 ==="
fi

# ── E2b: Budget transition time-series (phi4, N=2,5,10,20, B=20) ─────────────
if [[ "$SKIP_E2B" -eq 0 ]]; then
    log "=== START: E2b budget transition phi4 ==="
    kill_servers
    $PYTHON "$SCRIPTS/test_budget_transition.py" \
        --model phi4 \
        --N-values 2 5 10 20 \
        --budget 20 \
        --K 2 \
        2>&1 | tee "$LOGDIR/e2b_phi4_budget_transition.log"
    kill_servers
    log "=== DONE:  E2b budget transition phi4 ==="
else
    log "=== SKIP: E2b budget transition ==="
fi

log "================================================================"
log "ALL REMAINING EXPERIMENTS COMPLETE"
log "================================================================"
echo ""
echo "Generated CSVs (newest first):"
ls -lt "$SCRIPTS/logs/"*.csv 2>/dev/null | head -30
