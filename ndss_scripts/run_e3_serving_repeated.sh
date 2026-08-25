#!/usr/bin/env bash
# Experiment 3 repeated serving-performance runner.
# This script is intentionally not started by setup or test commands.

set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${SCRIPT_DIR}/results/submission_gap_experiments/e3_serving_repeated"
RUN_DIR="${RESULT_ROOT}/runs"
AGG_DIR="${RESULT_ROOT}/aggregated"
LOG_DIR="${RESULT_ROOT}/logs"
DATASET="${PROJECT_DIR}/datasets/english_pii_43k.jsonl"
PORT="${E3_PORT:-8092}"
SERVER="http://127.0.0.1:${PORT}"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-e3-operator-key}"

export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p "${RUN_DIR}" "${AGG_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH=(
    [phi4]="/workspace/Models/Phi-4"
    [qwen30b]="/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
    [qwen32b]="/workspace/Models/Qwen3-32B"
)
declare -A MODEL_TP=([phi4]=1 [qwen30b]=2 [qwen32b]=2)
declare -A MODEL_MAXLEN=([phi4]=16384 [qwen30b]=32768 [qwen32b]=32768)

MODELS=(phi4 qwen30b qwen32b)
SEEDS=(20260821 20260822 20260823)
WORKLOADS=(single_pii multi_turn system_prompt)
SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

cell_stem() {
    local model="$1" policy="$2" workload="$3" rep="$4" seed="$5"
    printf '%s/%s/%s__%s__rep%s__seed%s' \
        "${RUN_DIR}" "${model}" "${policy}" "${workload}" "${rep}" "${seed}"
}

cell_complete() {
    local stem
    stem="$(cell_stem "$@")"
    [[ -s "${stem}.csv" && -s "${stem}.complete.json" ]]
}

group_complete() {
    local model="$1" group="$2" rep="$3" seed="$4"
    local policy workload
    case "${group}" in
        vanilla) policies=(vanilla) ;;
        strict) policies=(strict shared_system_prompt_emulation) ;;
        balanced) policies=(balanced balanced_public) ;;
        *) echo "[E3] Unknown group: ${group}" >&2; return 2 ;;
    esac
    for policy in "${policies[@]}"; do
        if [[ "${policy}" == "shared_system_prompt_emulation" ]]; then
            cell_complete "${model}" "${policy}" system_prompt "${rep}" "${seed}" || return 1
        else
            for workload in "${WORKLOADS[@]}"; do
                cell_complete "${model}" "${policy}" "${workload}" "${rep}" "${seed}" || return 1
            done
        fi
    done
    return 0
}

wait_server() {
    local i
    for i in $(seq 1 90); do
        if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
            echo "[E3_SERVER_READY] url=${SERVER} wait_seconds=$((i * 5))"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[E3_SERVER_FAILED] pid=${SERVER_PID}" >&2
            return 1
        fi
        sleep 5
    done
    echo "[E3_SERVER_TIMEOUT] url=${SERVER}" >&2
    return 1
}

start_server() {
    local model="$1" mode="$2" rep="$3"
    local gpus=0
    cleanup_server
    if lsof -tiTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[E3_PORT_IN_USE] port=${PORT}; refusing to kill an unrelated process" >&2
        return 1
    fi
    if [[ "${MODEL_TP[$model]}" == 2 ]]; then
        gpus=0,1
    fi
    local server_log="${LOG_DIR}/server__${model}__${mode}__rep${rep}.log"
    echo "[E3_SERVER_START] model=${model} mode=${mode} rep=${rep} log=${server_log}"
    CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATH[$model]}" \
        --host 127.0.0.1 --port "${PORT}" \
        --dtype bfloat16 --trust-remote-code \
        --tp-size "${MODEL_TP[$model]}" \
        --context-length "${MODEL_MAXLEN[$model]}" \
        --served-model-name "${model}" \
        --attention-backend torch_native \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --schedule-policy fcfs \
        --safekv-mode "${mode}" \
        --safekv-access-budget 100 \
        --safekv-operator-key "${OPERATOR_KEY}" \
        --safekv-policy-epoch 1 \
        >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_server
}

run_cell() {
    local model="$1" policy="$2" workload="$3" rep="$4" seed="$5"
    local stem tmp_csv client_log
    stem="$(cell_stem "${model}" "${policy}" "${workload}" "${rep}" "${seed}")"
    mkdir -p "$(dirname "${stem}")"
    if cell_complete "${model}" "${policy}" "${workload}" "${rep}" "${seed}"; then
        echo "[E3_CELL_SKIP] model=${model} policy=${policy} workload=${workload} rep=${rep} seed=${seed}"
        return
    fi
    rm -f "${stem}.csv.tmp" "${stem}.complete.json.tmp"
    tmp_csv="${stem}.csv.tmp"
    client_log="${LOG_DIR}/client__${model}__${policy}__${workload}__rep${rep}__seed${seed}.log"
    echo "[E3_CELL_START] model=${model} policy=${policy} workload=${workload} rep=${rep} seed=${seed}"
    "${PYTHON}" "${SCRIPT_DIR}/exp5_serving_perf.py" \
        --server "${SERVER}" \
        --model "${model}" \
        --model-path "${MODEL_PATH[$model]}" \
        --workload "${workload}" \
        --policy "${policy}" \
        --n-users 20 \
        --rps 8 \
        --max-new-tokens 64 \
        --seed "${seed}" \
        --dataset "${DATASET}" \
        --output "${tmp_csv}" \
        --operator-key "${OPERATOR_KEY}" \
        2>&1 | tee "${client_log}"
    "${PYTHON}" - "${tmp_csv}" "${model}" "${policy}" "${workload}" <<'PY'
import csv
import sys

path, model, policy, workload = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1:
    raise SystemExit(f"expected exactly one result row in {path}, found {len(rows)}")
row = rows[0]
expected = {"model": model, "policy": policy, "workload": workload}
for key, value in expected.items():
    if row.get(key) != value:
        raise SystemExit(f"{path}: expected {key}={value!r}, found {row.get(key)!r}")
for key in ("mean_ttft_ms", "throughput_tok_s"):
    if float(row[key]) < 0:
        raise SystemExit(f"{path}: invalid {key}={row[key]!r}")
PY
    mv "${tmp_csv}" "${stem}.csv"
    printf '{"status":"complete","model":"%s","policy":"%s","workload":"%s","repetition":%s,"seed":%s}\n' \
        "${model}" "${policy}" "${workload}" "${rep}" "${seed}" \
        >"${stem}.complete.json.tmp"
    mv "${stem}.complete.json.tmp" "${stem}.complete.json"
    echo "[E3_CELL_COMPLETE] model=${model} policy=${policy} workload=${workload} rep=${rep} seed=${seed}"
}

run_group() {
    local model="$1" group="$2" mode="$3" rep="$4" seed="$5"
    if group_complete "${model}" "${group}" "${rep}" "${seed}"; then
        echo "[E3_GROUP_SKIP] model=${model} group=${group} rep=${rep} seed=${seed}"
        return
    fi
    start_server "${model}" "${mode}" "${rep}"
    case "${group}" in
        vanilla)
            for workload in "${WORKLOADS[@]}"; do
                run_cell "${model}" vanilla "${workload}" "${rep}" "${seed}"
            done
            ;;
        strict)
            for workload in "${WORKLOADS[@]}"; do
                run_cell "${model}" strict "${workload}" "${rep}" "${seed}"
            done
            run_cell "${model}" shared_system_prompt_emulation system_prompt "${rep}" "${seed}"
            ;;
        balanced)
            for policy in balanced balanced_public; do
                for workload in "${WORKLOADS[@]}"; do
                    run_cell "${model}" "${policy}" "${workload}" "${rep}" "${seed}"
                done
            done
            ;;
    esac
    cleanup_server
}

for rep_index in "${!SEEDS[@]}"; do
    rep=$((rep_index + 1))
    seed="${SEEDS[$rep_index]}"
    for model in "${MODELS[@]}"; do
        run_group "${model}" vanilla none "${rep}" "${seed}"
        run_group "${model}" strict strict "${rep}" "${seed}"
        run_group "${model}" balanced balanced "${rep}" "${seed}"
    done
done

"${PYTHON}" "${SCRIPT_DIR}/aggregate_e3_serving_repeated.py"
echo "[E3_ALL_COMPLETE] result_root=${RESULT_ROOT}"
