#!/bin/bash

# 定义模型部署配置数组
declare -a MODEL_NAMES=("qwen-235b")
declare -a MODEL_PATHS=(
    # "/dcar-vepfs-trans-models/Qwen3-30B-A3B"
    # "/dcar-vepfs-trans-models/Qwen3-32B"
    # "/dcar-vepfs-trans-models/Llama-3.3-70B-Instruct"
    "/dcar-vepfs-trans-models/Qwen3-235B-A22B"
    # "/dcar-vepfs-trans-models/Phi-4"
    # "/dev/shm/DeepSeek-R1-Int4"
)
declare -a CUDA_DEVICES=("0,1,2,3,4,5,6,7")
declare -a TPS=(8)
declare -a PORTS=("8083")

# 启动模型函数
start_model() {
    local idx=$1
    local log_file="logs/${MODEL_NAMES[$idx]}.log"
    mkdir -p logs

    echo "[INFO] Starting model ${MODEL_NAMES[$idx]} on port ${PORTS[$idx]}"

    LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$idx]} \
    /usr/bin/python3 python/sglang/launch_server.py \
        --model-path ${MODEL_PATHS[$idx]} \
        --host 127.0.0.1 \
        --port ${PORTS[$idx]} \
        --max-running-requests 32 \
        --max-total-tokens 40960 \
        --dtype bfloat16 \
        --trust-remote-code \
        --attention-backend torch_native \
        --sampling-backend pytorch \
        --disable-cuda-graph \
        --disable-cuda-graph-padding \
        --tp-size ${TPS[$idx]} \
        --dp-size 1 \
        --kv-cache-dtype auto \
        --allow-auto-truncate \
        --context-length 65536 \
        --chunked-prefill-size 16384 \
        >> "${log_file}" 2>&1 &

    PIDS[$idx]=$!
}


# --model-path ${MODEL_PATHS[$idx]} \
# --host 127.0.0.1 \
# --port ${PORTS[$idx]} \
# --max-running-requests 32 \
# --max-total-tokens 40960 \
# --dtype float16 \
# --trust-remote-code \
# --attention-backend flashinfer \
# --sampling-backend pytorch \
# --disable-cuda-graph \
# --disable-cuda-graph-padding \
# --tp-size ${TPS[$idx]} \
# --pp-size 1 \
# --dp-size 1 \
# --kv-cache-dtype auto \
# --allow-auto-truncate \
# --context-length 65536 \
# --chunked-prefill-size 16384 \
# --quantization awq \

# 启动所有模型
declare -a PIDS
for i in "${!MODEL_NAMES[@]}"; do
    start_model $i
done

# 监控循环
while true; do
    sleep 5
    for i in "${!MODEL_NAMES[@]}"; do
        if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
            echo "[WARN] Model ${MODEL_NAMES[$i]} crashed. Restarting..."
            start_model $i
        fi
    done
done
