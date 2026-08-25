#!/usr/bin/env bash
# run_exp3_all.sh  –  Full P3 run for all models/policies in sequence.
# Starts + kills the SGLang server for each (model, policy) pair.

set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
SAFEKV_OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-exp1-operator-key}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

declare -A MODEL_PATHS=(
    ["phi4"]="/workspace/Models/Phi-4"
    ["qwen30b"]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    ["qwen32b"]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=( ["phi4"]="1" ["qwen30b"]="2" ["qwen32b"]="2" )
declare -A MODEL_MAXLEN=( ["phi4"]="16384" ["qwen30b"]="32768" ["qwen32b"]="32768" )

PORT=8092
SERVER="http://127.0.0.1:${PORT}"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
RESULTS_DIR="${SCRIPT_DIR}/results/exp3"
mkdir -p "${RESULTS_DIR}"

kill_server() {
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        echo "[orch] Killing server on port ${PORT}"
        kill $(lsof -ti:${PORT}) 2>/dev/null || true
        sleep 3
    fi
}

wait_server() {
    echo "[orch] Waiting for server..."
    for i in $(seq 1 90); do
        if curl -sf "${SERVER}/health" >/dev/null 2>&1 || \
           curl -sf "${SERVER}/v1/models" >/dev/null 2>&1; then
            echo "[orch] Server ready after ~$((i*5))s"
            return 0
        fi
        sleep 5
    done
    echo "[orch] ERROR: Server did not start in time"
    return 1
}

run_combo() {
    local MODEL_KEY="$1"
    local POLICY="$2"
    local MODEL_PATH="${MODEL_PATHS[$MODEL_KEY]}"
    local TP="${MODEL_TP[$MODEL_KEY]}"
    local MAXLEN="${MODEL_MAXLEN[$MODEL_KEY]}"
    local OUT="${RESULTS_DIR}/${MODEL_KEY}_${POLICY}_v2.csv"

    if [[ -f "${OUT}" ]]; then
        echo "[orch] SKIP ${MODEL_KEY}_${POLICY}_v2 (output exists)"
        return
    fi

    kill_server

    if [[ "$TP" == "1" ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export CUDA_VISIBLE_DEVICES=0,1
    fi

    local LOG="${LOG_DIR}/${MODEL_KEY}_${POLICY}_exp3.log"
    echo "[orch] Starting ${MODEL_KEY} (${POLICY}) on port ${PORT}..."
    SAFEKV_MODE="${POLICY}" \
    ${PYTHON} -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tp-size "${TP}" \
        --context-length "${MAXLEN}" \
        --served-model-name "${MODEL_KEY}" \
        --attention-backend torch_native \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --schedule-policy fcfs \
        --safekv-mode "${POLICY}" \
        --safekv-operator-key "${SAFEKV_OPERATOR_KEY}" \
        --safekv-policy-epoch 1 \
        >"${LOG}" 2>&1 &
    SERVER_PID=$!

    if ! wait_server; then
        echo "[orch] FAIL: server did not start for ${MODEL_KEY}_${POLICY}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    echo "[orch] Running P3 for ${MODEL_KEY} / ${POLICY}..."
    ${PYTHON} "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
        --server        "${SERVER}" \
        --model         "${MODEL_KEY}" \
        --model-path    "${MODEL_PATH}" \
        --policy        "${POLICY}" \
        --budget-Q      200 \
        --n-challenges  100 \
        --n-attacker-accounts 2 \
        --n-recovery-trials 10 \
        --n-tokens-to-recover 5 \
        --vocab-sample  20 \
        --dataset       "${DATASET}" \
        --output        "${OUT}" \
        2>&1 | tee "${LOG_DIR}/${MODEL_KEY}_${POLICY}_exp3_client.log"

    echo "[orch] Finished ${MODEL_KEY}_${POLICY}"
    kill_server
}

# Run all combos
MODELS=("phi4" "qwen30b" "qwen32b")
POLICIES=("strict" "balanced")

for MODEL in "${MODELS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        run_combo "${MODEL}" "${POLICY}"
    done
done

echo ""
echo "=============================== SUMMARY ==============================="
python3 - <<'PYEOF'
import csv, json, math, random
from pathlib import Path

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return (max(0,c-h), min(1,c+h))

results_dir = Path("user_scripts/results/exp3")
print(f"{'Model':<10} {'Policy':<10} {'N':>4} {'TPR':>5} {'FPR':>5} {'AdvMI':>6} {'AUC':>6}")
print("-"*55)
for fpath in sorted(results_dir.glob("*_v2.csv")):
    rows = list(csv.DictReader(fpath.open()))
    mi = [r for r in rows if r["experiment_type"]=="membership" and r.get("predicted_bit") not in ("","-1")]
    if not mi: continue
    yt = [int(r["challenge_bit"]) for r in mi]
    yp = [int(r["predicted_bit"]) for r in mi]
    tt = [float(r["ttft_ms"]) for r in mi]
    n = len(yt); n_pos = sum(yt); n_neg = n-n_pos
    tp = sum(1 for a,b in zip(yt,yp) if a==1 and b==1)
    fp = sum(1 for a,b in zip(yt,yp) if a==0 and b==1)
    tpr = tp/n_pos if n_pos else 0; fpr = fp/n_neg if n_neg else 0
    adv = abs(tpr-fpr)
    pos_tt = [s for t,s in zip(yt,tt) if t==1]
    neg_tt = [s for t,s in zip(yt,tt) if t==0]
    conc = sum(1 for p in pos_tt for q in neg_tt if p < q)
    tie  = sum(1 for p in pos_tt for q in neg_tt if p == q)
    auc = (conc + 0.5*tie)/(n_pos*n_neg) if n_pos and n_neg else float('nan')
    stem = fpath.stem.replace("_v2","")
    model, policy = stem.rsplit("_",1)
    print(f"{model:<10} {policy:<10} {n:4d} {tpr:5.2f} {fpr:5.2f} {adv:6.3f} {auc:6.3f}")
PYEOF
