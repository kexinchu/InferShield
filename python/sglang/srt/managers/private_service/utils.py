import random
import json

def load_jsonl_dataset(path, sample_n=1000, seed=42):
    # 读取jsonl文件
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 打乱并采样
    random.seed(seed)
    random.shuffle(lines)
    lines = lines[:sample_n]
    # 解析
    texts = []
    labels = []
    for line in lines:
        item = json.loads(line)

        # 判断是否有PII
        if "new_prompts" in path:
            label = item["label"]
            texts.append(item["rewritten"])
            labels.append(item["label"])
        else:
            texts.append(item["source_text"])
            bio_labels = item["mbert_bio_labels"]
            if isinstance(bio_labels, str):
                bio_labels = eval(bio_labels)  # 兼容字符串格式
            label = 1 if any(l != "O" for l in bio_labels) else 0
            labels.append(label)
    return texts, labels

def make_prompt(user_input):
    """针对Llama-3.2-1B 优化后"""
    return f"""You are a privacy classification assistant.

# Your task is to determine if a given sentence contains any Personally Privacy Information.
Privacy Info includes — but is not limited to — the following categories:
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
