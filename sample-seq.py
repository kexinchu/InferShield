from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import numpy as np
import os
import time

latencies = []

# 模型和tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval().half().cuda()

    return model, tokenizer

# 构造 prompt
def make_prompt(user_input):
    return f"""You are a privacy classification assistant.

Your task is to determine if a given sentence contains any Personally Privacy Information.

Privacy Info includes: Real names (full or partial), Phone numbers, Email addresses, National ID, passport, social security numbers, Home or mailing addresses, Account numbers, License plates, IP addresses, GPS locations, Dates of birth or death, etc.

## Instructions:
1. Carefully read the input text.
2. Decide whether it includes any privacy infomation.
3. Output a number between 0 and 1 representing how confident you are that the text contains any privacy info.

## Examples:
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

## Now evaluate this input:
Input: "{user_input}"
Answer:"""

def make_prompt_llama(user_input):
    return f"""You are a privacy classification assistant.

# Your task is to determine if a given sentence contains any Personally Privacy Information.
Privacy Info includes:
---
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

---
### Examples:
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
---

Now analyze the input below:
Input: "{user_input}"
Answer:
"""

# 模型推理 - 批处理版本
def detect_privacy_llm_batch(input_texts, tokenizer, model, batch_size=16):
    scores = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        prompts = [make_prompt_llama(text) for text in batch_texts]
        
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
                score = max(0.0, min(1.0, score[0])) if score else 0.5
            except:
                score = 0.5
            scores.append(score)
    
    return scores

# 保持原有的单样本处理函数作为备用
def detect_privacy_llm(input_text, tokenizer, model):
    prompt = make_prompt_llama(input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=10)
        latencies.append((time.perf_counter() - start) * 1000)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    # 尝试解析成 float
    try:
        score = [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', result)]
        score = max(0.0, min(1.0, score[0]))  # 限制范围
    except:
        score = 0.5
    return score

if __name__ == "__main__":
    SAMPLE_N = 2000
    texts, labels = load_jsonl_dataset("./Datasets/english_pii_43k.jsonl", sample_n=SAMPLE_N)
    # 检查字段名
    print(texts[0])
    print(labels[1])

    # 创建结果目录
    os.makedirs("./results/pii-detection", exist_ok=True)

    model_names = [
        # "./Models/Qwen3-0.6B", 
        # "./Models/Qwen3-4B", 
        # "./Models/Qwen3-8B",
        "./Models/Llama-3.2-1B",
        # "./Models/Llama-3.2-3B",
        # "./Models/Llama-3.2-8B"
    ]

    batch_size = 16
    
    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        file_ = open("./results/pii-detection/res_file-" + model_name.split("/")[-1] + ".txt", "w")
        model, tokenizer = load_model(model_name)
        
        # 使用批处理
        scores = detect_privacy_llm_batch(texts, tokenizer, model, batch_size=batch_size)
        preds = []
        for idx, score in enumerate(scores):
            file_.write(f"Source:{texts[idx]}\tpredict:{score}\tlabel:{labels[idx]}\n")
            pred = 1 if score >= 0.7 else 0
            preds.append(pred)
        preds = np.array(preds)
        labels_np = np.array(labels)
        acc = (preds == labels_np).mean()
        file_.close()
        print(f"Accuracy for {model_name}: {acc:.4f};")

        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p50_idx = int(0.5 * len(sorted_latencies))
        p95_idx = int(0.95 * len(sorted_latencies))
        p99_idx = int(0.99 * len(sorted_latencies))
        p50_latency = sorted_latencies[p50_idx]
        p95_latency = sorted_latencies[p95_idx]
        p99_latency = sorted_latencies[p99_idx]
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"50th Percentile Latency: {p50_latency:.2f} ms")
        print(f"95th Percentile Latency: {p95_latency:.2f} ms")
        print(f"99th Percentile Latency: {p99_latency:.2f} ms")
