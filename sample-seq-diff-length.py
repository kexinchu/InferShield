import os
import re
import time
import torch
import numpy as np
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_requests import load_jsonl_dataset

# 模型和tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval().half().cuda()

    return model, tokenizer

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
Input: "{user_input}"
Answer:
"""

# 模型推理 - 批处理版本
def detect_privacy_llm_batch(input_texts, tokenizer, model, batch_size=16):
    scores = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        prompts = [make_prompt_llama_1B(text) for text in batch_texts]

        # 批处理编码
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        # 解码每个样本的输出
        for j, output in enumerate(outputs):
            # 获取新生成的token
            input_length = inputs["input_ids"].shape[1] if len(inputs["input_ids"].shape) > 1 else inputs["input_ids"].shape[0]
            result = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()

            # 尝试解析成 float
            try:
                score = [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', result)]
                score = max(0.0, min(1.0, score[0]))
            except:
                score = 0.5
            scores.append(score)

    return scores

# 保持原有的单样本处理函数作为备用
def detect_privacy_llm(input_texts, tokenizer, model, model_name):
    scores = []
    latencies = []
    for i in range(0, len(input_texts)):
        text = input_texts[i]
        prompts = [make_prompt_llama_1B(text)]

        # max_output_length
        max_output_length = 10
        if "Qwen3" in model_name:
            max_output_length = 100
        if "70B" in model_name or "32B" in model_name:
            max_output_length = 4096

        # 批处理编码
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(model.device)
        with torch.no_grad():
            start = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=max_output_length)
            latencies.append((time.perf_counter() - start) * 1000)

        # 解码每个样本的输出
        for j, output in enumerate(outputs):
            # 获取新生成的token
            input_length = inputs["input_ids"].shape[1] if len(inputs["input_ids"].shape) > 1 else inputs["input_ids"].shape[0]
            result = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()

            # 尝试解析成 float
            try:
                score = [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', result)]
                score = max(0.0, min(1.0, score[0]))
            except:
                score = 0.5
            scores.append(score)

    return scores, latencies

# ---------- 每个子进程的工作 ----------
def worker(model, tokenizer, model_name, texts):
    scores, latencies = detect_privacy_llm(texts, tokenizer, model, model_name)

    # 打印
    print(f"{model_name}  Acc: Not Care")
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
    torch.cuda.empty_cache()

def shuffle_and_assign_requests(num_request, length):
    """将requests打乱后，均匀分配到每个thread的queue"""
    import string
    import random
    queue = []
    for _ in range(num_request):
        # 生成指定长度的随机prompt - 使用字母、数字和常见标点符号
        chars = string.ascii_letters + string.digits + " .,!?;:"
        req = ''.join(random.choice(chars) for _ in range(length))
        queue.append(req)
    return queue

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    SAMPLE_N = 200

    model_names = [
        # "/dcar-vepfs-trans-models/Qwen3-0.6B",
        # "/dcar-vepfs-trans-models/Qwen3-4B",
        # "/dcar-vepfs-trans-models/Qwen3-8B",
        # "/dcar-vepfs-trans-models/Qwen3-32B",
        # "/dcar-vepfs-trans-models/Qwen3-30B-A3B",
        "/dcar-vepfs-trans-models/Llama-3.2-1B",
        # "/dcar-vepfs-trans-models/Llama-3.2-3B",
        # "/dcar-vepfs-trans-models/Llama-3.1-8B",
        # "/dcar-vepfs-trans-models/Llama-3.3-70B-Instruct",
    ]

    device_map = {
        "/dcar-vepfs-trans-models/Llama-3.3-70B-Instruct": [4,5,6,7],
        "/dcar-vepfs-trans-models/Qwen3-32B": [3],
        "/dcar-vepfs-trans-models/Llama-3.2-1B": [0],
    }

    for rank, mdl in enumerate(model_names):
        if mdl in device_map:
            rank = device_map[mdl]
        else:
            rank = [rank]
        # set cuda
        if len(rank) > 1:
            str_rank = [str(i) for i in rank]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_rank)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank[0])  # 让进程只看到 rank 这张卡
            torch.cuda.set_device(0)                           # 因为上面只暴露一张卡，所以 cuda:0 就是物理卡 rank

        tokenizer = AutoTokenizer.from_pretrained(mdl, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mdl, trust_remote_code=True, device_map="auto"
        )
        model.eval()

        for length in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            if length < 32768:
                continue
            texts = shuffle_and_assign_requests(SAMPLE_N, length)
            # 检查字段名
            print(f"length: {length}")

            worker(model, tokenizer, mdl, texts)
