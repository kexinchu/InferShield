#!/usr/bin/env bash
# run_exp5_complete.sh  –  Serialized full matrix: phi4 + qwen30b + qwen32b
# Runs all (model, policy, workload) combos sequentially.  No port conflicts.
# Results → user_scripts/results/exp5_v2/

set -uo pipefail

PYTHON="python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
SAFEKV_OPERATOR_KEY="safekv-exp5-operator-key"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
LOG_DIR="${SCRIPT_DIR}/logs"
RD="${SCRIPT_DIR}/results/exp5_v2"
PORT=8092
SERVER="http://127.0.0.1:${PORT}"

declare -A MODEL_PATH=(
    [phi4]="/workspace/Models/Phi-4"
    [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=32768 [qwen32b]=32768)
declare -A POLICY_MODE=([vanilla]=none [cache_partition]=strict [strict]=strict [balanced]=balanced [balanced_public]=balanced)

kill_server() {
    local pids
    pids=$(lsof -ti:${PORT} 2>/dev/null) || true
    [[ -n "$pids" ]] && { echo "[srv] killing $pids"; kill $pids 2>/dev/null; sleep 4; }
}

start_server() {
    local model="$1" mode="$2"
    kill_server
    local tp="${MODEL_TP[$model]}"
    local maxlen="${MODEL_MAXLEN[$model]}"
    local mpath="${MODEL_PATH[$model]}"
    local gpus="0"
    [[ "$tp" == "2" ]] && gpus="0,1"
    echo "[srv] Starting ${model} mode=${mode} tp=${tp}"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    ${PYTHON} -m sglang.launch_server \
        --model-path "${mpath}" --host 127.0.0.1 --port ${PORT} \
        --dtype bfloat16 --trust-remote-code \
        --tp-size "${tp}" --context-length "${maxlen}" \
        --served-model-name "${model}" \
        --attention-backend torch_native --disable-cuda-graph \
        --mem-fraction-static 0.80 --schedule-policy fcfs \
        --safekv-mode "${mode}" --safekv-access-budget 100 \
        --safekv-operator-key "${SAFEKV_OPERATOR_KEY}" --safekv-policy-epoch 1 \
        >"${LOG_DIR}/${model}_${mode}_exp5v2.log" 2>&1 &
    # Wait up to 7 minutes for health
    for i in $(seq 1 84); do
        curl -sf "${SERVER}/health" >/dev/null 2>&1 && echo "[srv] Ready after $((i*5))s" && return 0
        sleep 5
    done
    echo "[srv] ERROR: timeout for ${model}/${mode}"; return 1
}

run_wl() {
    local model="$1" policy="$2" workload="$3"
    local out="${RD}/${model}_${policy}_${workload}.csv"
    if [[ -f "$out" ]]; then
        echo "  [skip] ${model}/${policy}/${workload}"
        return
    fi
    echo "  [run] ${model}/${policy}/${workload}"
    ${PYTHON} "${SCRIPT_DIR}/exp5_serving_perf.py" \
        --server "${SERVER}" \
        --model "${model}" \
        --model-path "${MODEL_PATH[$model]}" \
        --workload "${workload}" \
        --policy "${policy}" \
        --n-users 20 --rps 8 --max-new-tokens 64 \
        --dataset "${DATASET}" \
        --output "${out}" \
        --operator-key "${SAFEKV_OPERATOR_KEY}" \
        2>&1 | tail -6
}

# ---------- main matrix ----------
# Group by (model, safekv_mode) to minimize server restarts.
# Each group:  start_server, run all policies that share that mode, kill.

run_model_mode() {
    local model="$1" mode="$2"
    shift 2
    local policies=("$@")
    start_server "${model}" "${mode}" || return 1
    for pol in "${policies[@]}"; do
        for wl in single_pii multi_turn system_prompt; do
            run_wl "${model}" "${pol}" "${wl}"
        done
    done
    kill_server
}

for model in phi4 qwen30b qwen32b; do
    run_model_mode "${model}" none         vanilla
    run_model_mode "${model}" strict       cache_partition strict
    run_model_mode "${model}" balanced     balanced balanced_public
done

# ---------- summary ----------
echo ""
echo "=============================== SUMMARY ================================"
python3 - <<'PYEOF'
import csv
from pathlib import Path
rd = Path("/workspace/user_scripts/results/exp5_v2")
models = ["phi4","qwen30b","qwen32b"]
policies = ["vanilla","cache_partition","strict","balanced","balanced_public"]
workloads = ["single_pii","multi_turn","system_prompt"]
data = {}
for fpath in rd.glob("*.csv"):
    rows = list(csv.DictReader(fpath.open()))
    if not rows: continue
    r = rows[-1]
    data[(r["model"],r["policy"],r["workload"])] = (float(r["mean_ttft_ms"]),float(r["throughput_tok_s"]))
print(f"{'Model':8} {'Policy':20}  {'Sgl-PII':>8}  {'Multi-Turn':>10}  {'SysPrompt':>9}  {'MT tput':>8}")
print("-"*75)
for m in models:
    van = {w: data.get((m,"vanilla",w),(None,None))[0] for w in workloads}
    for p in policies:
        cells = [data.get((m,p,w)) for w in workloads]
        def fmt(c, w):
            if not c or not c[0]: return "—"
            v = van.get(w)
            if p=="vanilla" or not v: return f"{c[0]:.0f}ms"
            return f"{c[0]:.0f}ms({(c[0]/v-1)*100:+.0f}%)"
        row = "  ".join(fmt(c,w) for c,w in zip(cells,workloads))
        tput = f"{cells[1][1]:.1f}" if cells[1] and cells[1][1] else "—"
        print(f"{m:8} {p:20}  {row}  {tput}")
    print()
PYEOF
echo "Files collected: $(ls ${RD}/*.csv 2>/dev/null | wc -l)/45"
