#!/usr/bin/env bash
# run_exp5_all.sh  –  Full P5 serving-performance matrix.
#
# For each (model, policy) pair, starts the server, runs all 3 workloads, kills server.
# Results go to user_scripts/results/exp5_v2/<model>_<policy>_<workload>.csv
#
# Changes vs v1:
#   * max_new_tokens=64  (meaningful throughput measurement)
#   * n_users=50         (more samples, lower noise)
#   * rps=8              (moderate load; matches placeholder spec)
#   * flush before each workload run

set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
SAFEKV_OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-exp5-operator-key}"

LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results/exp5_v2"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

PORT=8092
SERVER="http://127.0.0.1:${PORT}"

declare -A MODEL_PATHS=(
    ["phi4"]="/workspace/Models/Phi-4"
    ["qwen30b"]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    ["qwen32b"]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=( ["phi4"]="1" ["qwen30b"]="2" ["qwen32b"]="2" )
declare -A MODEL_MAXLEN=( ["phi4"]="16384" ["qwen30b"]="32768" ["qwen32b"]="32768" )

# SafeKV mode that each policy key maps to
declare -A POLICY_MODE=(
    ["vanilla"]="none"
    ["cache_partition"]="strict"
    ["strict"]="strict"
    ["balanced"]="balanced"
    ["balanced_public"]="balanced"
)

WORKLOADS=("single_pii" "multi_turn" "system_prompt")

kill_server() {
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        echo "[orch] Killing server on port ${PORT}"
        kill $(lsof -ti:${PORT}) 2>/dev/null || true
        sleep 3
    fi
}

wait_server() {
    for i in $(seq 1 90); do
        if curl -sf "${SERVER}/health" >/dev/null 2>&1 || \
           curl -sf "${SERVER}/v1/models" >/dev/null 2>&1; then
            echo "[orch] Server ready (~$((i*5))s)"
            return 0
        fi
        sleep 5
    done
    echo "[orch] ERROR: server timeout"
    return 1
}

run_combo() {
    local MODEL_KEY="$1"
    local POLICY="$2"
    local MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
    local TP="${MODEL_TP[$MODEL_KEY]}"
    local MAXLEN="${MODEL_MAXLEN[$MODEL_KEY]}"
    local SAFEKV_MODE_VAL="${POLICY_MODE[$POLICY]}"

    # Check if all 3 workloads already done for this combo
    local all_done=true
    for WL in "${WORKLOADS[@]}"; do
        [[ ! -f "${RESULTS_DIR}/${MODEL_KEY}_${POLICY}_${WL}.csv" ]] && all_done=false && break
    done
    if $all_done; then
        echo "[orch] SKIP ${MODEL_KEY}_${POLICY} (all workloads exist)"
        return
    fi

    kill_server

    if [[ "${TP}" == "1" ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export CUDA_VISIBLE_DEVICES=0,1
    fi

    local SERVER_LOG="${LOG_DIR}/${MODEL_KEY}_${POLICY}_exp5.log"
    echo "[orch] Starting ${MODEL_KEY} policy=${POLICY} safekv-mode=${SAFEKV_MODE_VAL}"

    # For cache_partition: strict mode but treat as Cache-Partition semantics
    # (SafeKV-Strict = per-principal private namespace = same as Cache-Partition)
    SAFEKV_MODE="${SAFEKV_MODE_VAL}" \
    ${PYTHON} -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 127.0.0.1 --port "${PORT}" \
        --dtype bfloat16 --trust-remote-code \
        --tp-size "${TP}" --context-length "${MAXLEN}" \
        --served-model-name "${MODEL_KEY}" \
        --attention-backend torch_native \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --schedule-policy fcfs \
        --safekv-mode "${SAFEKV_MODE_VAL}" \
        --safekv-access-budget 100 \
        --safekv-operator-key "${SAFEKV_OPERATOR_KEY}" \
        --safekv-policy-epoch 1 \
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    wait_server || { kill $SERVER_PID 2>/dev/null; return 1; }

    for WL in "${WORKLOADS[@]}"; do
        local OUT="${RESULTS_DIR}/${MODEL_KEY}_${POLICY}_${WL}.csv"
        if [[ -f "${OUT}" ]]; then
            echo "[orch]   SKIP ${WL} (exists)"
            continue
        fi

        echo "[orch]   Running workload: ${WL}"
        local CLIENT_LOG="${LOG_DIR}/${MODEL_KEY}_${POLICY}_${WL}_exp5_client.log"
        ${PYTHON} "${SCRIPT_DIR}/exp5_serving_perf.py" \
            --server        "${SERVER}" \
            --model         "${MODEL_KEY}" \
            --model-path    "${MODEL_PATH}" \
            --workload      "${WL}" \
            --policy        "${POLICY}" \
            --n-users       50 \
            --rps           8 \
            --max-new-tokens 64 \
            --dataset       "${DATASET}" \
            --output        "${OUT}" \
            --operator-key  "${SAFEKV_OPERATOR_KEY}" \
            2>&1 | tee "${CLIENT_LOG}"
        echo "[orch]   Done: ${WL}"
    done

    kill_server
}

MODELS=("phi4" "qwen30b" "qwen32b")
POLICIES=("vanilla" "cache_partition" "strict" "balanced" "balanced_public")

for MODEL in "${MODELS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        run_combo "${MODEL}" "${POLICY}"
    done
done

echo ""
echo "============================== SUMMARY =============================="
python3 - <<'PYEOF'
import csv, statistics
from pathlib import Path
rd = Path("user_scripts/results/exp5_v2")
models = ["phi4","qwen30b","qwen32b"]
policies = ["vanilla","cache_partition","strict","balanced","balanced_public"]
workloads = ["single_pii","multi_turn","system_prompt"]
data = {}
for fpath in rd.glob("*.csv"):
    parts = fpath.stem.split("_")
    for m in models:
        if not fpath.stem.startswith(m): continue
        rest = fpath.stem[len(m)+1:]
        for wl in workloads:
            if rest.endswith(wl):
                pol = rest[:-(len(wl)+1)]
                rows = list(csv.DictReader(fpath.open()))
                if rows:
                    r = rows[-1]
                    data[(m,pol,wl)] = (float(r["mean_ttft_ms"]), float(r["throughput_tok_s"]))
                break

print(f"{'Model':8} {'Policy':18}  {'Sgl-PII TTFT(ms)':>16}  {'Multi-Turn':>10}  {'SysPrompt':>9}  {'Multi-Turn tput':>15}")
print("-"*80)
for m in models:
    for p in policies:
        cells = [data.get((m,p,w)) for w in workloads]
        row = [f"{c[0]:8.0f}" if c else f"{'—':>8}" for c in cells]
        tput = f"{cells[1][1]:8.1f} tok/s" if cells[1] else "—"
        print(f"{m:8} {p:18}  {'  '.join(row)}  {tput}")
    print()
PYEOF
