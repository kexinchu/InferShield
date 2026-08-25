#!/bin/bash
# ============================================================
# SafeKV Live Experiment Orchestrator
# Runs P3/P4/P5/P8/P10 overnight on the current server.
#
# Requires: 2× NVIDIA RTX A6000 GPUs, safekv_exp conda env.
# Execution order:
#   For each model in [phi4, qwen32b, qwen30b]:
#     1. Server (strict)   → P5-strict  + P3-strict  + P10
#     2. Server (balanced) → P5-balanced + P3-balanced + P4
#     3. Server (balanced) → P5-balanced_public (system_prompt)
#   After Phi-4 balanced run → P8 (auth matrix, model-agnostic)
# ============================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="python3"
BASE_DIR="/workspace/docker-sys"
MODEL_DIR="${BASE_DIR}/Models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
SERVER_URL="http://127.0.0.1:8092"
PORT=8092
OPERATOR_KEY="safekv-liveexp-operator-key"

export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export SAFEKV_OPERATOR_KEY="${OPERATOR_KEY}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p "${RESULTS_DIR}/exp3" "${RESULTS_DIR}/exp4" "${RESULTS_DIR}/exp5" \
         "${RESULTS_DIR}/exp8" "${RESULTS_DIR}/exp10" "${LOG_DIR}"

# ── Model configs ─────────────────────────────────────────────────────────────
declare -A MODEL_PATHS=(
    ["phi4"]="${MODEL_DIR}/Phi-4"
    ["qwen32b"]="${MODEL_DIR}/Qwen3-32B"
    ["qwen30b"]="${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507"
)
declare -A MODEL_TP=(
    ["phi4"]="1"
    ["qwen32b"]="2"
    ["qwen30b"]="2"
)
declare -A MODEL_MAXLEN=(
    ["phi4"]="16384"
    ["qwen32b"]="32768"
    ["qwen30b"]="32768"
)
declare -A MODEL_GPUS=(
    ["phi4"]="0"
    ["qwen32b"]="0,1"
    ["qwen30b"]="0,1"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/orchestrator.log"; }

kill_server() {
    if lsof -ti:${PORT} >/dev/null 2>&1; then
        log "Killing existing server on port ${PORT}…"
        kill $(lsof -ti:${PORT}) 2>/dev/null || true
        sleep 5
    fi
}

wait_server_ready() {
    local deadline=$(( $(date +%s) + 600 ))   # 10 min timeout
    log "Waiting for server on ${SERVER_URL}…"
    while true; do
        if curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; then
            log "Server ready."
            sleep 3
            return 0
        fi
        if [[ $(date +%s) -gt $deadline ]]; then
            log "ERROR: Server did not start within 10 minutes."
            return 1
        fi
        sleep 5
    done
}

start_server() {
    local model_key="$1"
    local mode="$2"       # strict | balanced
    local model_path="${MODEL_PATHS[$model_key]}"
    local tp="${MODEL_TP[$model_key]}"
    local max_len="${MODEL_MAXLEN[$model_key]}"
    local gpus="${MODEL_GPUS[$model_key]}"
    local log_file="${LOG_DIR}/${model_key}_${mode}.log"

    kill_server

    log "Starting ${model_key} server [mode=${mode}, TP=${tp}, GPUs=${gpus}]…"
    export CUDA_VISIBLE_DEVICES="${gpus}"

    nohup "${PYTHON}" -m sglang.launch_server \
        --model-path "${model_path}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tp-size "${tp}" \
        --context-length "${max_len}" \
        --served-model-name "${model_key}" \
        --attention-backend torch_native \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --schedule-policy fcfs \
        --safekv-mode "${mode}" \
        --safekv-operator-key "${OPERATOR_KEY}" \
        --safekv-policy-epoch 1 \
        > "${log_file}" 2>&1 &

    wait_server_ready
}

# ── Experiment runners ────────────────────────────────────────────────────────

run_p5() {
    local model_key="$1"
    local policy="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    log "=== P5: ${model_key} / ${policy} ==="
    for workload in single_pii multi_turn system_prompt; do
        local out="${RESULTS_DIR}/exp5/${model_key}_${policy}_${workload}.csv"
        "${PYTHON}" "${SCRIPT_DIR}/exp5_serving_perf.py" \
            --server "${SERVER_URL}" \
            --model "${model_key}" \
            --model-path "${model_path}" \
            --workload "${workload}" \
            --policy "${policy}" \
            --n-users 20 \
            --rps 8.0 \
            --dataset "${DATASET}" \
            --operator-key "${OPERATOR_KEY}" \
            --output "${out}" \
            && log "P5 saved: ${out}" \
            || log "WARN: P5 ${model_key}/${policy}/${workload} failed"
    done
}

run_p3() {
    local model_key="$1"
    local policy="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    log "=== P3: ${model_key} / ${policy} ==="
    local out="${RESULTS_DIR}/exp3/${model_key}_${policy}.csv"
    "${PYTHON}" "${SCRIPT_DIR}/exp3_endtoend_attack.py" \
        --server "${SERVER_URL}" \
        --model "${model_key}" \
        --model-path "${model_path}" \
        --policy "${policy}" \
        --budget-Q 200 \
        --n-challenges 50 \
        --n-attacker-accounts 2 \
        --n-recovery-trials 5 \
        --dataset "${DATASET}" \
        --output "${out}" \
        && log "P3 saved: ${out}" \
        || log "WARN: P3 ${model_key}/${policy} failed"
}

run_p4() {
    local model_key="$1"
    local model_path="${MODEL_PATHS[$model_key]}"
    log "=== P4: ${model_key} (balanced) ==="
    local out="${RESULTS_DIR}/exp4/${model_key}.csv"
    "${PYTHON}" "${SCRIPT_DIR}/exp4_public_membership.py" \
        --server "${SERVER_URL}" \
        --model "${model_key}" \
        --model-path "${model_path}" \
        --n-challenges 40 \
        --dataset "${DATASET}" \
        --operator-key "${OPERATOR_KEY}" \
        --output "${out}" \
        && log "P4 saved: ${out}" \
        || log "WARN: P4 ${model_key} failed"
}

run_p8() {
    local model_key="$1"
    local model_path="${MODEL_PATHS[$model_key]}"
    log "=== P8: ${model_key} (auth matrix) ==="
    local out="${RESULTS_DIR}/exp8/${model_key}.csv"
    "${PYTHON}" "${SCRIPT_DIR}/exp8_auth_matrix.py" \
        --server "${SERVER_URL}" \
        --model "${model_key}" \
        --model-path "${model_path}" \
        --trials 5 \
        --dataset "${DATASET}" \
        --operator-key "${OPERATOR_KEY}" \
        --output "${out}" \
        && log "P8 saved: ${out}" \
        || log "WARN: P8 ${model_key} failed"
}

run_p10() {
    local model_key="$1"
    local policy="$2"
    local model_path="${MODEL_PATHS[$model_key]}"
    log "=== P10: ${model_key} / ${policy} ==="
    local out="${RESULTS_DIR}/exp10/${model_key}_${policy}.csv"
    "${PYTHON}" "${SCRIPT_DIR}/exp10_attack_robustness.py" \
        --server "${SERVER_URL}" \
        --model "${model_key}" \
        --model-path "${model_path}" \
        --policy "${policy}" \
        --n-challenges 30 \
        --dataset "${DATASET}" \
        --output "${out}" \
        && log "P10 saved: ${out}" \
        || log "WARN: P10 ${model_key}/${policy} failed"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
MODELS=("phi4" "qwen32b" "qwen30b")

log "======================================================"
log "SafeKV Live Experiment Overnight Run — $(date)"
log "Models: ${MODELS[*]}"
log "======================================================"

for MODEL in "${MODELS[@]}"; do
    log "############### Model: ${MODEL} ###############"

    # ── Round 1: strict mode ──────────────────────────────────────────────────
    start_server "${MODEL}" "strict"
    run_p5 "${MODEL}" "strict"
    run_p3 "${MODEL}" "strict"
    # Run P10 once per model under strict (all conditions are measured in one run)
    run_p10 "${MODEL}" "strict"
    kill_server

    # ── Round 2: balanced mode ────────────────────────────────────────────────
    start_server "${MODEL}" "balanced"
    run_p5 "${MODEL}" "balanced"
    run_p3 "${MODEL}" "balanced"
    run_p4 "${MODEL}"
    # P8 is model-agnostic: run only once (with phi4)
    if [[ "${MODEL}" == "phi4" ]]; then
        run_p8 "${MODEL}"
    fi
    # balanced_public is just a policy flag in the client (same server mode=balanced)
    run_p5 "${MODEL}" "balanced_public"
    kill_server

    log "############### Done: ${MODEL} ###############"
done

log "======================================================"
log "All live experiments complete — $(date)"
log "Results in: ${RESULTS_DIR}"
log "======================================================"

# ── Aggregate summary ─────────────────────────────────────────────────────────
log "Running result aggregation…"
"${PYTHON}" - <<'PYEOF'
import csv, json, os, glob, statistics
from pathlib import Path

RESULTS = Path("/workspace/user_scripts/results")

def summarize_csvs(pattern, key_field, value_fields):
    rows = []
    for f in sorted(glob.glob(str(RESULTS / pattern))):
        for row in csv.DictReader(open(f)):
            rows.append(row)
    return rows

# P3 summary
p3_rows = summarize_csvs("exp3/*.csv", "policy", [])
p3_mi = [r for r in p3_rows if r.get("experiment_type","") == "membership"]
if p3_mi:
    by_policy = {}
    for r in p3_mi:
        k = (r.get("model",""), r.get("policy",""))
        by_policy.setdefault(k, []).append(int(r["correct"]) if r.get("correct","") not in ("","None") else None)
    p3_summary = {}
    for (model, policy), corrects in by_policy.items():
        vals = [v for v in corrects if v is not None]
        acc = statistics.mean(vals) if vals else None
        adv = abs(2 * acc - 1) if acc is not None else None
        p3_summary[f"{model}/{policy}"] = {"accuracy": acc, "adv_mi": adv, "n": len(vals)}
    with open(RESULTS / "exp3_summary.json", "w") as f:
        json.dump(p3_summary, f, indent=2)
    print(f"P3 summary: {len(p3_summary)} (model,policy) pairs")

# P5 summary
p5_rows = summarize_csvs("exp5/*.csv", "policy", [])
if p5_rows:
    p5_summary = {}
    for r in p5_rows:
        k = f"{r.get('model','')}/{r.get('policy','')}/{r.get('workload','')}"
        p5_summary[k] = {
            "mean_ttft_ms": float(r.get("mean_ttft_ms", 0)),
            "p50_ttft_ms": float(r.get("p50_ttft_ms", 0)),
            "p95_ttft_ms": float(r.get("p95_ttft_ms", 0)),
            "throughput_tok_s": float(r.get("throughput_tok_s", 0)),
        }
    with open(RESULTS / "exp5_summary.json", "w") as f:
        json.dump(p5_summary, f, indent=2)
    print(f"P5 summary: {len(p5_summary)} (model,policy,workload) cells")

print("Aggregation done.")
PYEOF

log "Overnight run COMPLETE."
