#!/bin/bash
set -e
cd /workspace/ndss_scripts
PY=python3

echo "========================================"
echo " PII Ratio Sweep — All Models"
echo " Started: $(date)"
echo "========================================"

for MODEL in qwen32b phi4 qwen30b; do
    echo ""
    echo ">>> Model: $MODEL  ($(date))"
    $PY test_pii_ratio_sweep.py --model $MODEL --port 8090
    echo ">>> $MODEL done ($(date))"
    sleep 5
done

echo ""
echo "========================================"
echo " ALL DONE: $(date)"
echo "========================================"
