#!/bin/bash
# ============================================================
# Full Benchmark Orchestrator
#
# Usage:
#   ./run_bench.sh                    # benchmark all 3 models sequentially
#   ./run_bench.sh qwen32b            # benchmark only Qwen3-32B
#   ./run_bench.sh qwen32b --quick    # quick test (10 queries)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="/home/kec23008/.venv/bin/python3"

# ---- Config ----
MODELS=("qwen32b" "qwen30b" "qwen30b-int4" "phi4")
declare -A PORTS=(["qwen32b"]="8090" ["qwen30b"]="8094" ["qwen30b-int4"]="8093" ["phi4"]="8092")

NUM_QUERIES=50
MAX_TOKENS=128
CONCURRENCY=1

# Parse args
if [[ "${1:-}" != "" && "${1:-}" != "--"* ]]; then
    MODELS=("$1")
    shift
fi
if [[ "${1:-}" == "--quick" ]]; then
    NUM_QUERIES=10
    MAX_TOKENS=64
    shift
fi

DATASET="${SCRIPT_DIR}/results/benchmark_data.jsonl"

# Generate dataset if not exists
if [[ ! -f "${DATASET}" ]]; then
    echo "[INFO] Generating benchmark dataset (ShareGPT + PII) ..."
    ${PYTHON} "${SCRIPT_DIR}/prepare_benchmark_data.py" --num-samples 200 --output "${DATASET}"
fi

echo "============================================"
echo " NDSS Benchmark Runner"
echo " Models: ${MODELS[*]}"
echo " Queries: ${NUM_QUERIES}  Max tokens: ${MAX_TOKENS}"
echo " Dataset: ${DATASET}"
echo "============================================"

for MODEL_KEY in "${MODELS[@]}"; do
    PORT="${PORTS[$MODEL_KEY]}"
    echo ""
    echo "############################################"
    echo "# Model: ${MODEL_KEY}  Port: ${PORT}"
    echo "############################################"

    # 1. Check if server is already running
    SERVER_READY=false
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[INFO] Server already running on port ${PORT}"
        SERVER_READY=true
    else
        echo "[INFO] Starting server for ${MODEL_KEY} ..."
        bash "${SCRIPT_DIR}/launch_model.sh" "${MODEL_KEY}" &
        SERVER_PID=$!
        echo "[INFO] Server PID: ${SERVER_PID}"

        # Wait for server to be ready (up to 10 minutes for large models)
        echo "[INFO] Waiting for server to load model ..."
        for i in $(seq 1 120); do
            if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
                echo "[INFO] Server ready after ~$((i*5))s"
                SERVER_READY=true
                break
            fi
            sleep 5
        done

        if [[ "${SERVER_READY}" != "true" ]]; then
            echo "[ERROR] Server failed to start for ${MODEL_KEY}. Skipping."
            kill ${SERVER_PID} 2>/dev/null || true
            continue
        fi
    fi

    # 2. Run benchmark
    echo "[INFO] Running benchmark ..."
    ${PYTHON} "${SCRIPT_DIR}/bench_metrics.py" \
        --server "127.0.0.1:${PORT}" \
        --model-name "${MODEL_KEY}" \
        --dataset "${DATASET}" \
        --num-queries "${NUM_QUERIES}" \
        --max-tokens "${MAX_TOKENS}" \
        --concurrency "${CONCURRENCY}"

    # 3. Also run a concurrency test
    echo "[INFO] Running concurrency=4 benchmark ..."
    ${PYTHON} "${SCRIPT_DIR}/bench_metrics.py" \
        --server "127.0.0.1:${PORT}" \
        --model-name "${MODEL_KEY}" \
        --num-queries "${NUM_QUERIES}" \
        --max-tokens "${MAX_TOKENS}" \
        --concurrency 4

    # 4. If we started the server, stop it to free GPU for next model
    if [[ -v SERVER_PID ]]; then
        echo "[INFO] Stopping server (PID ${SERVER_PID}) ..."
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        sleep 5
        unset SERVER_PID
    fi

    echo "[INFO] Done with ${MODEL_KEY}"
done

echo ""
echo "============================================"
echo " All benchmarks complete!"
echo " Results in: ${SCRIPT_DIR}/results/"
echo "============================================"
ls -la "${SCRIPT_DIR}/results/" 2>/dev/null || echo "(no results yet)"
