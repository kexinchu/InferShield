#!/bin/bash
# Run PII ratio sweep for CacheSolidarity only (all 3 models)
set -e
cd "$(dirname "$0")"

PY=/home/kec23008/.venv/bin/python3
LOG=/tmp/cache_solidarity_pii.log

echo "[$(date '+%H:%M:%S')] Starting CacheSolidarity PII ratio sweep" | tee "$LOG"

for MODEL in phi4 qwen30b qwen32b; do
    echo "[$(date '+%H:%M:%S')] >>> Model: $MODEL" | tee -a "$LOG"
    $PY test_pii_ratio_sweep.py --model "$MODEL" --systems cache_solidarity 2>&1 | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] >>> $MODEL done" | tee -a "$LOG"
    sleep 5
done

echo "[$(date '+%H:%M:%S')] All done." | tee -a "$LOG"
