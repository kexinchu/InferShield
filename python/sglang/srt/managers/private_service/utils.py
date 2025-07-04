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
        texts.append(item["source_text"])
        # 判断是否有PII
        bio_labels = item["mbert_bio_labels"]
        if isinstance(bio_labels, str):
            bio_labels = eval(bio_labels)  # 兼容字符串格式
        label = 1 if any(l != "O" for l in bio_labels) else 0
        labels.append(label)
    return texts, labels