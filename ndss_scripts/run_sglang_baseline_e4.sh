#!/bin/bash
# Run SGLang baseline (no SafeKV) at each concurrency level for Exp 4 multi-tenant
# Fills in the missing SGLang column in the multi-tenant figure
set -uo pipefail

PY=/home/kec23008/.venv/bin/python3
SCRIPTS=/home/kec23008/InferShield/ndss_scripts
LOG=/tmp/sglang_baseline_e4.log

cd "$SCRIPTS"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== SGLang Baseline E4 Multi-tenant ==="
log "Models: phi4 qwen30b qwen32b  |  Concurrency: 1 4 8 16 32"

for MODEL in phi4 qwen30b qwen32b; do
    log ">>> Model: $MODEL"
    for C in 1 4 8 16 32; do
        log "  c=$C start"
        $PY test_throughput_ablation.py \
            --model "$MODEL" \
            --modes baseline \
            --max-workers "$C" \
            --n-requests 300 \
            >> "$LOG" 2>&1
        log "  c=$C done"
        sleep 5
    done
    log ">>> $MODEL all concurrencies done"
done

log "=== All baseline runs complete ==="
