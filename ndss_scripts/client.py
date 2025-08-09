import requests
import uuid
import json
import threading
import queue
import time
import random
import sys
import re
import numpy as np
import multiprocessing as mp
from load_requests import load_jsonl_dataset

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
thread_scores = [[] for _ in range(NUM_THREADS)]

def make_prompt_llama_1B(user_input):
    return f"""You are a privacy classification assistant.

Your task is to determine if a given sentence contains any Personally Privacy Information.
Privacy Info includes:
**1. Personal Privacy Information**
- General Info: nickname, avatar, personal signature
- Basic Info: third-party account usernames, platform handles
- Identity Info: ID card, passport, driver’s license, social security number (SSN)
- Location Info: country, city, province/state, region, address, ZIP code
- Biometric Info: face, fingerprint, iris, gene sequence, voiceprint
- System/Network ID: UserID, IP address, Cookie, password, certificate, RFID
**2. Device Information**
- Software Info: Android ID, IDFA, IDFV, OS version, software region
- Hardware Info: MAC address, IMEI, GUID, serial number, device model
**3. Profile & Financial Info**
- Cultural & Social: education level, job title, certificates, likes, follows, browsing interests
- Financial: bank account, balance, payment history, loan records, cryptocurrency addresses
- Social & Behavior: contact list, collections, browsing/purchase/download records, login logs
**4. Temporal or Numeric Identifiers**
- Dates: date of birth, login time, purchase date
- Numbers: phone number, account number, credit card, PIN, IBAN, CVV, masked digits

## Instructions:
1. Carefully read the input text.
2. Decide whether it includes any privacy infomation.
3. Output a number between 0 and 1 representing how confident you are that the text contains any privacy info.
4. The score must be in ahead of your answer.

## Examples:
Input: "Her passport number is XJ32981921."
Answer: 0.97

Input: "User: kevinchu@protonmail.com. IP: 192.168.2.1"
Answer: 0.96

Input: "Contact me at john.doe@example.com or 555-123-4567."
Answer: 0.97

Input: "Her birthday is January 14, 1993."
Answer: 0.93

Input: "The capital of France is Paris."
Answer: 0.47

Input: "This is a list of animals: dog, cat, lion."
Answer: 0.02

Input: "James recently moved to 456 Oak Avenue."
Answer: 0.96
"""

def make_prompt_deepseek(user_input: str) -> str:
    prompt = f"""You are an expert privacy classifier.
Your ONLY task is to decide whether the **exact** piece of text below contains **personally identifiable information (PII)** or **any sensitive personal data**.

PII includes (non-exhaustive):
**1. Personal Privacy Information**
- General Info: nickname, avatar, personal signature
- Basic Info: third-party account usernames, platform handles
- Identity Info: ID card, passport, driver’s license, social security number (SSN)
- Location Info: country, city, province/state, region, address, ZIP code
- Biometric Info: face, fingerprint, iris, gene sequence, voiceprint
- System/Network ID: UserID, IP address, Cookie, password, certificate, RFID
**2. Device Information**
- Software Info: Android ID, IDFA, IDFV, OS version, software region
- Hardware Info: MAC address, IMEI, GUID, serial number, device model
**3. Profile & Financial Info**
- Cultural & Social: education level, job title, certificates, likes, follows, browsing interests
- Financial: bank account, balance, payment history, loan records, cryptocurrency addresses
- Social & Behavior: contact list, collections, browsing/purchase/download records, login logs
**4. Temporal or Numeric Identifiers**
- Dates: date of birth, login time, purchase date
- Numbers: phone number, account number, credit card, PIN, IBAN, CVV, masked digits

Non-PII examples:
• Generic place names (“Paris”, “California”)
• Common first names standing alone (“John”, “Maria”)
• Public company names, product names
• Aggregated or anonymized statistics

Instructions:
1. **Read the input carefully** and extract any potential PII snippets.
2. **Think step-by-step** (Chain-of-Thought) inside `<reasoning>` tags.
3. **Output ONLY a JSON object** in the exact format:
   {{"reasoning":"<your short reasoning>","score":<float 0-1 rounded to two decimals>}}

Scoring rubric:
- 1.00  absolutely certain PII is present
- ≥0.90 very high confidence
- 0.70–0.89 likely but not 100 % sure
- 0.10–0.69 probably not PII
- 0.00  definitely no PII

Examples:

Input: "My name is John Smith and my email is john@smith.com"
Answer: {{"reasoning":"Full name plus email address → direct PII","score":1.00}}

Input: "The meeting is at 123 Baker Street, London."
Answer: {{"reasoning":"Exact street address → PII","score":0.95}}

Input: "Contact me on Telegram @alice_2024"
Answer: {{"reasoning":"Username tied to a person on messaging platform → PII","score":0.91}}

Input: "Her passport number is E12345678."
Answer: {{"reasoning":"Passport number → direct PII","score":1.00}}

Input: "I was born on 1990-12-05."
Answer: {{"reasoning":"DOB reveals age/identity → PII","score":0.93}}

Input: "Employee ID 48291 just got promoted."
Answer: {{"reasoning":"Employee ID unique to person → PII","score":0.87}}

Input: "BTC wallet 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
Answer: {{"reasoning":"Cryptocurrency address traceable to owner → PII","score":0.96}}

Input: "The capital of France is Paris."
Answer: {{"reasoning":"Generic geographic fact, no person → no PII","score":0.00}}

Input: "James is a common first name."
Answer: {{"reasoning":"Name alone, no linkage → no PII","score":0.05}}

Input: "User likes apples and oranges."
Answer: {{"reasoning":"Generic preferences → no PII","score":0.00}}

Input: "Female, 28, Austin, works at Meta."
Answer: {{"reasoning":"Quasi-identifiers together likely identify a person → PII","score":0.84}}

Input: "Bank card ending in **** 1234"
Answer: {{"reasoning":"Masked but still partial card number → sensitive","score":0.72}}

Input: "Latitude 39.9042, Longitude 116.4074"
Answer: {{"reasoning":"Exact GPS coordinates → PII","score":0.97}}

Input: "Invoice #INV-2024-001"
Answer: {{"reasoning":"Invoice number not tied to person without context → no PII","score":0.12}}
"""
    return prompt

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

        system_text = make_prompt_deepseek(request_text)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": request_text}
            ],
            "temperature": 0.0,
            "max_tokens": 100,
            "user_id": user_id,
        }
        start = time.perf_counter()
        response = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload))
        end = time.perf_counter()

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # print(content)
            thread_latencies[thread_id].append((end - start) * 1000)
            # 尝试解析成 float
            try:
                score = [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', content)]
                score = max(0.0, min(1.0, score[0]))
            except:
                # print(f"No score in result: {result}")
                # break
                score = 0.5
            thread_scores[thread_id].append(score)
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

    all_scores = [score for scores in thread_scores for score in scores]
    if not all_scores:
        print("No scores data collected")
        return
    preds = np.array([1 if s >= 0.7 else 0 for s in all_scores])
    labels_np = np.array([1 for s in all_scores])
    acc = (preds == labels_np).mean()


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
    print(f"Average Accuracy: {acc:.4f}")
    print(f"Remain Request: {len(all_latencies) - (acc * len(all_latencies))}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"50th Percentile Latency: {p50_latency:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency:.2f} ms")
    print(f"99th Percentile Latency: {p99_latency:.2f} ms")



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    data_list = [
        "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl"
    ]
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        if len(sys.argv) > 2:
            SERVER_URL = "http://" + sys.argv[2] + "/v1/chat/completions"
        print(f"Loading requests...: {file_name}")
        SAMPLE_N = 500
        thread_latencies = [[] for _ in range(NUM_THREADS)]
        user_requests, labels = load_jsonl_dataset(
            file_name,
            sample_n=SAMPLE_N
        )
        sampled_requests = user_requests[:SAMPLE_N]
        random.shuffle(sampled_requests)
        # Take first 1000 requests
        prompts = sampled_requests

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
                args=(i, thread_session_ids[i], thread_queues[i], QPS_PER_THREAD, "llama3-70b")
            )
            t.start()
            threads.append(t)

        # 4. 等待所有线程完成
        for t in threads:
            t.join()

        # 5. 汇总延迟统计
        report_latency()
    else:
        for file_name in data_list:
            print(f"Loading requests...: {file_name}")
            SAMPLE_N = 5000
            thread_latencies = [[] for _ in range(NUM_THREADS)]
            user_requests, labels = load_jsonl_dataset(
                file_name,
                sample_n=SAMPLE_N
            )
            sampled_requests = user_requests[:SAMPLE_N]
            random.shuffle(sampled_requests)
            # Take first 1000 requests
            prompts = sampled_requests

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
                    args=(i, thread_session_ids[i], thread_queues[i], QPS_PER_THREAD, "llama3-70b")
                )
                t.start()
                threads.append(t)

            # 4. 等待所有线程完成
            for t in threads:
                t.join()

            # 5. 汇总延迟统计
            report_latency()
