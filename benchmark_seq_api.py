#!/usr/bin/env python3
"""
Usage:
python3 benchmark_seq_api.py --model_name doubao-1-5-pro-32k-250115

python3 benchmark_seq_api.py --model_name deepseek-r1-250528
"""
import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
import argparse
import time
import numpy as np
from load_requests import load_jsonl_dataset

# 构造 prompt
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

## Now analyze the input below:
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

Now analyze the user's input:
"""
    return prompt

# 模型处理
def detect_privacy_llm(client: OpenAI, input_texts, model_name):
    """
    发送单个流式请求，返回包含时间戳和 token 统计的字典。
    字段：
      start_ts, first_chunk, chunk_ts_list, end_ts,
      prompt_tokens, total_tokens, completion_tokens
    """
    scores = []
    latencies = []
    for i in range(0, len(input_texts)):
        text = input_texts[i]
        system_prompt = make_prompt_deepseek(text)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
        }

        start = time.perf_counter()
        stream = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            stream=False,
            max_tokens=4096
        )
        latencies.append((time.perf_counter() - start) * 1000)

        result = stream.choices[0].message.content.strip()
        # 尝试解析成 float
        try:
            score = [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', result)]
            score = max(0.0, min(1.0, score[0]))
        except:
            # print(f"No score in result: {result}")
            # break
            score = 0.5
        scores.append(score)

    return scores, latencies

# ---------- 每个子进程的工作 ----------
def worker(client, model_name, texts, labels, save_dir):
    scores, latencies = detect_privacy_llm(client, texts, model_name)

    # 写结果
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"res_file-{model_name.split('/')[-1]}.txt")
    with open(fname, "w") as f:
        for idx, score in enumerate(scores):
            f.write(f"Source:{texts[idx]}\tpredict:{score}\tlabel:{labels[idx]}\n")

    # 统计指标
    preds = np.array([1 if s >= 0.7 else 0 for s in scores])
    labels_np = np.array(labels)
    acc = (preds == labels_np).mean()

    # 打印
    print(f"{model_name}  Acc: {acc:.4f}")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p50_idx = int(0.5 * len(sorted_latencies))
        p95_idx = int(0.95 * len(sorted_latencies))
        p99_idx = int(0.99 * len(sorted_latencies))
        p50_latency = sorted_latencies[p50_idx]
        p95_latency = sorted_latencies[p95_idx]
        p99_latency = sorted_latencies[p99_idx]
        print(f"{model_name} Average Latency: {avg_latency:.2f} ms")
        print(f"{model_name} 50th Percentile Latency: {p50_latency:.2f} ms")
        print(f"{model_name} 95th Percentile Latency: {p95_latency:.2f} ms")
        print(f"{model_name} 99th Percentile Latency: {p99_latency:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="models name")
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="efb7ff5c-5dd4-446b-b73d-aa5910913f7c"
    )

    # 简单的提示列表，可根据实际情况替换
    SAMPLE_N = 200
    texts, labels = load_jsonl_dataset("/dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl", sample_n=SAMPLE_N)
    # 检查字段名
    print(texts[0])
    print(labels[0])

    worker(client, args.model_name, texts, labels, "./results/pii-detection")


if __name__ == "__main__":
    main()
