#!/usr/bin/env bash
# Smoke-test + reuse-B cells for the two newly downloaded backbones.
# Llama-3.3-70B-Instruct-FP8 uses modelopt FP8; Distill-Qwen-32B is BF16.
# Grid is B in {0, 100, 150} so we can see Strict vs the paper B* vs overshoot.
set -euo pipefail

PYTHON="${SAFEKV_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH}"

PORT="${REUSE_PORT:-8093}"
SERVER="http://127.0.0.1:${PORT}"
OUT_DIR="${SCRIPT_DIR}/results/reuse_risk_curve/cells_new_models"
LOG_DIR="${SCRIPT_DIR}/logs/reuse_risk_curve_new_models"
OPERATOR_KEY="${SAFEKV_OPERATOR_KEY:-safekv-reuse-curve-key}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LLAMA_FP8_DIR="/workspace/Models/Llama-3.3-70B-Instruct-FP8"
LLAMA_AWQ_DIR="/workspace/Models/Llama-3.3-70B-Instruct-AWQ"
DS32_DIR="/workspace/Models/DeepSeek-R1-Distill-Qwen-32B"

declare -A MODEL_PATH=(
  [llama70b_fp8]="${LLAMA_FP8_DIR}"
  [llama70b_awq]="${LLAMA_AWQ_DIR}"
  [ds_r1_qwen32b]="${DS32_DIR}"
)
declare -A MODEL_TP=([llama70b_fp8]=2 [llama70b_awq]=2 [ds_r1_qwen32b]=2)
declare -A MODEL_MAXLEN=([llama70b_fp8]=4096 [llama70b_awq]=4096 [ds_r1_qwen32b]=8192)
declare -A MODEL_DTYPE=([llama70b_fp8]=auto [llama70b_awq]=float16 [ds_r1_qwen32b]=bfloat16)
declare -A MODEL_QUANT=([llama70b_fp8]=modelopt [llama70b_awq]=awq [ds_r1_qwen32b]="")
declare -A MODEL_MEM=([llama70b_fp8]=0.85 [llama70b_awq]=0.85 [ds_r1_qwen32b]=0.80)

BUDGETS=(0 100 150)
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(ds_r1_qwen32b llama70b_awq)
fi
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  local pids
  pids="$(lsof -ti:${PORT} 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 4
  fi
}
trap cleanup EXIT INT TERM

weights_ready() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || return 1
  [[ -f "${dir}/model.safetensors.index.json" ]] || return 1
  if find "${dir}" -name "*.incomplete" | grep -q .; then
    return 1
  fi
  local expected actual
  expected="$(python3 -c "import json; print(len(set(json.load(open('${dir}/model.safetensors.index.json'))['weight_map'].values())))")"
  actual="$(find "${dir}" -maxdepth 1 \( -name 'model-*.safetensors' -o -name '*.safetensors' \) ! -name 'model.safetensors' | wc -l)"
  if [[ "${actual}" -eq 0 && -f "${dir}/model.safetensors" ]]; then
    actual=1
  fi
  [[ "${actual}" -ge "${expected}" ]] || return 1
  echo "[WEIGHTS_OK] dir=${dir} shards=${actual}/${expected}"
}

start_server() {
  local model="$1" mode="$2" B="$3"
  cleanup
  local gpus=0,1
  local log="${LOG_DIR}/${model}_B${B}_${mode}_server.log"
  local extra=()
  if [[ -n "${MODEL_QUANT[$model]}" ]]; then
    extra+=(--quantization "${MODEL_QUANT[$model]}")
  fi
  echo "[NEW_SERVER_START] model=${model} B=${B} mode=${mode} quant=${MODEL_QUANT[$model]:-none}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH[$model]}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dtype "${MODEL_DTYPE[$model]}" --trust-remote-code \
    --tp-size "${MODEL_TP[$model]}" \
    --context-length "${MODEL_MAXLEN[$model]}" \
    --served-model-name "${model}" \
    --attention-backend torch_native --disable-cuda-graph \
    --mem-fraction-static "${MODEL_MEM[$model]}" --schedule-policy fcfs \
    --safekv-mode "${mode}" \
    --safekv-access-budget "${B}" \
    --safekv-operator-key "${OPERATOR_KEY}" \
    --safekv-policy-epoch 1 \
    --safekv-experiment-autoshare \
    "${extra[@]}" \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 240); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
      echo "[NEW_SERVER_READY] model=${model} B=${B} wait_s=$((i*5))"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[NEW_SERVER_FAILED] model=${model} B=${B} log=${log}" >&2
      tail -40 "${log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[NEW_SERVER_TIMEOUT] model=${model} B=${B} log=${log}" >&2
  return 1
}

smoke_generate() {
  local model="$1"
  local out="${OUT_DIR}/${model}_smoke.json"
  echo "[SMOKE] model=${model}"
  "${PYTHON}" - <<PY
import json, time, requests
from pathlib import Path
server = "${SERVER}"
t0 = time.time()
r = requests.post(
    f"{server}/generate",
    json={
        "text": "Say the word ready and stop.",
        "sampling_params": {"max_new_tokens": 8, "temperature": 0.0, "user_id": "smoke"},
        "stream": False,
    },
    timeout=180,
)
r.raise_for_status()
body = r.json()
text = body[0]["text"] if isinstance(body, list) else body.get("text", "")
meta = (body[0] if isinstance(body, list) else body).get("meta_info") or {}
rec = {
    "model": "${model}",
    "ok": True,
    "elapsed_s": round(time.time() - t0, 3),
    "text": text[:200],
    "prompt_tokens": meta.get("prompt_tokens"),
    "completion_tokens": meta.get("completion_tokens"),
}
Path("${out}").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
PY
}

echo "=== weight check ==="
for model in "${MODELS[@]}"; do
  if ! weights_ready "${MODEL_PATH[$model]}"; then
    echo "[WEIGHTS_MISSING] model=${model} path=${MODEL_PATH[$model]}" >&2
    exit 1
  fi
done

run_cell() {
  local model="$1" B="$2"
  local cell="${OUT_DIR}/${model}_B${B}.json"
  if [[ -s "${cell}" ]]; then
    echo "[REUSE_SKIP] model=${model} B=${B}"
    return 0
  fi
  if ! "${PYTHON}" "${SCRIPT_DIR}/measure_reuse_b_cell.py" \
    --server "${SERVER}" \
    --model "${model}" \
    --model-path "${MODEL_PATH[$model]}" \
    --B "${B}" \
    --output "${cell}" \
    2>&1 | tee "${LOG_DIR}/${model}_B${B}_client.log"; then
    echo "[REUSE_CELL_FAILED] model=${model} B=${B} reason=client" | tee -a "${LOG_DIR}/failed.log"
    return 1
  fi
}

for model in "${MODELS[@]}"; do
  smoked=0
  for B in "${BUDGETS[@]}"; do
    if [[ "${B}" == "0" ]]; then
      mode=strict
      cli_budget=1
    else
      mode=balanced
      cli_budget="${B}"
    fi
    if ! start_server "${model}" "${mode}" "${cli_budget}"; then
      echo "[REUSE_CELL_FAILED] model=${model} B=${B} reason=server" | tee -a "${LOG_DIR}/failed.log"
      cleanup
      continue
    fi
    if [[ "${smoked}" == "0" ]]; then
      if ! smoke_generate "${model}"; then
        echo "[SMOKE_FAILED] model=${model} reason=generate" | tee -a "${LOG_DIR}/failed.log"
        cleanup
        continue
      fi
      smoked=1
    fi
    run_cell "${model}" "${B}" || true
    cleanup
  done
done

echo "[NEW_MODELS_DONE] cells=${OUT_DIR}"
