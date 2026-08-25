from os import stat_result
import requests
import json
import sys
import time
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures
from load_requests import load_jsonl_dataset
from collections import defaultdict

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# === 参数 ===
NUM_THREADS = 10          # 线程数
SAMPLE_N = 20

TOTAL_TOKENS = []
PREFILL_TOKENS = []
DECODE_TOKENS = []
LATENCIES = []

def worker(request_text, user_id, model_name, system_p):
    """每个线程的逻辑"""
    num_retry = 3
    SYSTEM_PROMPT = ""
    if system_p :
        SYSTEM_PROMPT = make_prompt_deepseek()
        # sys_list = []
        # for i in range(14):
        #     sys_list.append(SYSTEM_PROMPT_)
        # SYSTEM_PROMPT = "\n".join(sys_list)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": SYSTEM_PROMPT + "\n" + request_text}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "user_id": str(user_id),
    }

    while num_retry > 0:
        try:
            start = time.perf_counter()
            response = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload), timeout=60)
            latency = time.perf_counter() - start
            LATENCIES.append(latency)

            if response.status_code == 200:
                result = response.json()
                total_tokens = result["usage"]["total_tokens"]
                prefill_tokens = result["usage"]["prompt_tokens"]
                decode_tokens = result["usage"]["completion_tokens"]
                TOTAL_TOKENS.append(total_tokens)
                PREFILL_TOKENS.append(prefill_tokens)
                DECODE_TOKENS.append(decode_tokens)
                break
        except:
            time.sleep(60)
            num_retry-=1
            continue

def report_latency(latency):
    """最后统计throughput"""
    total_tokens = sum(TOTAL_TOKENS)
    prefill_tokens = sum(PREFILL_TOKENS)
    decode_tokens = sum(DECODE_TOKENS)

    print("\n=== TPS Statistics ===")
    print(f"Latency: {latency}")
    print(f"Total TPS: {float(total_tokens) / latency}")
    print(f"Prefill TPS: {float(prefill_tokens) / latency}")
    print(f"Decode TPS: {float(decode_tokens) / latency}")

def make_prompt_deepseek() -> str:
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

def reorder(requests):
    history_dict = defaultdict(list)
    print(len(requests))
    for req in requests:
        history = ''
        if '\n' in req:
            lines = req.split("\n")
            last_user_idx = max(i for i, line in enumerate(lines) if len(line) > 10)
            for i in range(3):
                history += '\n'.join(lines[:last_user_idx])
            req = history + '\n'.join(lines[last_user_idx:])
        else:
            req = req
        history_dict[history].append(req)

    # reorder
    new_requests = []
    for h, reqs in history_dict.items():
        new_requests.append(h)
        for req in reqs:
            new_requests.append(req)
    return new_requests

if __name__ == "__main__":
    time.sleep(60*30)
    mp.set_start_method('spawn', force=True)
    file_name = sys.argv[1]
    if len(sys.argv) > 2:
        SERVER_URL = "http://" + sys.argv[2] + "/v1/chat/completions"
    print(f"Loading requests...: {file_name}")
    TOTAL_TOKENS = []
    # system-p
    user_requests, _ = load_jsonl_dataset(
        "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
        sample_n=SAMPLE_N
    )
    sampled_requests_base = []
    for i in range(SAMPLE_N):
        sys_prompt = make_prompt_deepseek()
        sampled_requests_base.append(sys_prompt + "\n" + user_requests[i])
    # multi-session
    user_requests, _ = load_jsonl_dataset(
        "/root/code/sglang-security/results/english_pii_43k-actual.jsonl",
        sample_n=1000
    )
    sampled_requests_multi = reorder(user_requests)[:SAMPLE_N]
    for prompt in sampled_requests_multi:
        sampled_requests_base.append(prompt)
    # origin
    user_requests, _ = load_jsonl_dataset(
        "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
        sample_n=SAMPLE_N
    )
    for prompt in user_requests:
        sampled_requests_base.append(prompt)

    NewPrompts = reorder(sampled_requests_base)
    print(len(NewPrompts))

    system_p = False
    if len(sys.argv) > 3:
        system_p = True

    # 3. 启动线程
    # request_text, user_id, model_name, system_p
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for i, prompt in enumerate(NewPrompts):
            futures.append(executor.submit(worker, prompt, i, "llama3-70b", system_p))

        for future in tqdm(concurrent.futures.as_completed(futures)):
            _ = future.result()

    # 5. 汇总延迟统计
    report_latency(sum(LATENCIES) / NUM_THREADS)
