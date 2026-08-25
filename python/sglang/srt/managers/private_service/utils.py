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

PRIVACY_SYSTEM_PROMPT = (
    "You are a security-first privacy filter for KV-cache sharing. "
    "Answer with a single word: yes or no. "
    "If the text might contain personal or sensitive information, or you are unsure, answer yes. "
    "Answer no only when you are confident the text is purely public."
)

PRIVACY_USER_TEMPLATE = """Decide whether this text is safe to treat as Public (reusable across tenants).

Answer yes (keep Private) if the text looks like a message, form, account notice,
or record about a private individual — especially with email, phone, address,
id/account numbers, password, username, or personal health/financial details.
If it could be someone's private data and you are not sure, answer yes.

Answer no (allow Public) if the text is clearly news, Wikipedia, sports, or
business reporting about public events, companies, or public figures. Naming a
well-known person, city, or organization in a news/wiki style is not private.

If you cannot tell whether it is a private record or public reporting, answer yes.

Text:
{text}

Is this a private record that should stay Private? yes or no."""


def make_prompt(user_input):
    """Compact instruction for Llama-3.2-1B yes/no PII classification."""
    text = (user_input or "").replace("\n", " ").strip()
    if len(text) > 1200:
        text = text[:1200]
    return PRIVACY_USER_TEMPLATE.format(text=text)


def make_chat_messages(user_input):
    """Chat-template messages for Instruct models."""
    return [
        {"role": "system", "content": PRIVACY_SYSTEM_PROMPT},
        {"role": "user", "content": make_prompt(user_input)},
    ]

