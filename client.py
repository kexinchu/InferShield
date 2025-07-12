import requests
import uuid
import json
import threading
import queue
import time
import random
import sys
from transformers import AutoTokenizer
import multiprocessing as mp
from load_requests import load_requests

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# === 参数 ===
NUM_THREADS = 1          # 线程数
QPS_PER_THREAD = 1       # 每个线程的QPS
ISOLATION_MODE = True

# === 每个线程对应一个Queue和Session ID ===
thread_queues = []
thread_session_ids = []

# === 新增统计信息 ===
thread_latencies = [[] for _ in range(NUM_THREADS)]  # 每个线程独立的延迟记录列表

def shuffle_and_assign_requests(user_requests, num_threads):
    """将requests打乱后，均匀分配到每个thread的queue"""
    random.shuffle(user_requests)
    queues = [queue.Queue() for _ in range(num_threads)]
    for idx, req in enumerate(user_requests):
        queues[idx % num_threads].put(req)
    return queues

def worker(thread_id, user_id, req_queue, qps, model_name):
    """每个线程的逻辑"""
    interval = 1.0 / qps  # 控制速率
    while True:
        try:
            request_text = req_queue.get(timeout=5)
        except queue.Empty:
            print(f"[Thread {thread_id}] No more requests. Exiting.")
            break

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": request_text}
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "user_id": user_id,
        }
        start = time.perf_counter()
        response = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload))
        end = time.perf_counter()

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(content)
            thread_latencies[thread_id].append(end - start)
        else:
            continue

        # 控制qps
        time.sleep(interval)

def report_latency():
    """最后统计所有线程的延迟数据"""
    all_latencies = [lat for thread in thread_latencies for lat in thread]
    if not all_latencies:
        print("No latency data collected.")
        return

    avg_latency = sum(all_latencies) / len(all_latencies)
    
    # Calculate percentiles
    sorted_latencies = sorted(all_latencies)
    p50_idx = int(0.5 * len(sorted_latencies))
    p95_idx = int(0.95 * len(sorted_latencies))
    p99_idx = int(0.99 * len(sorted_latencies))
    
    p50_latency = sorted_latencies[p50_idx]
    p95_latency = sorted_latencies[p95_idx]
    p99_latency = sorted_latencies[p99_idx]

    print("\n=== Latency Statistics ===")
    print(f"Total Requests: {len(all_latencies)}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"50th Percentile Latency: {p50_latency:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency:.2f} ms")
    print(f"99th Percentile Latency: {p99_latency:.2f} ms")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    model_name = "llama3-8b" 
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    file_name = "configurable-system-prompt-multitask.parquet"
    if len(sys.argv) > 2:
        file_name = sys.argv[2]
        
    local_path = {
        "llama3-8b": "../Models/Llama-3.2-8B",
        "qwen3-8b": "../Models/Qwen3-8B"
    }
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_path[model_name])
    
    print("Loading requests...")
    user_requests = load_requests(
        file_name, 
        tokenizer, 
        max_embedding_positions=4090
    )
    sampled_requests = user_requests[:2]
    random.shuffle(sampled_requests)
    # Take first 1000 requests
    prompts = []
    for prompt, _, _, session_id in sampled_requests:
        prompts.append(prompt)

    # 1. 打乱并分配requests到queues
    thread_queues = shuffle_and_assign_requests(prompts, NUM_THREADS)

    # 2. 为每个线程分配一个session_id
    if not ISOLATION_MODE:
        thread_session_ids = [str(1) for _ in range(NUM_THREADS)]
    else:
        thread_session_ids = [str(uuid.uuid4()) for _ in range(NUM_THREADS)]

    # 3. 启动线程
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(
            target=worker,
            args=(i, thread_session_ids[i], thread_queues[i], QPS_PER_THREAD, model_name)
        )
        t.start()
        threads.append(t)

    # 4. 等待所有线程完成
    for t in threads:
        t.join()

    # 5. 汇总延迟统计
    report_latency()
