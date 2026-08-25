from re import X
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import torch.nn.functional as F

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained('/dcar-vepfs-trans-models/piiranha-v1')
# 3. 模型定义
model = AutoModelForTokenClassification.from_pretrained("/dcar-vepfs-trans-models/piiranha-v1")

# 获取豁免标签（仅在首次初始化时转换为 tensor）
ignore_labels = ["O"] #, "I-CITY"]
ignore_label_ids = torch.tensor(
    [model.config.label2id[l] for l in ignore_labels],
    dtype=torch.long
)

# 6. 推理：判断是否包含隐私信息
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=256, padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, num_labels]
        preds = torch.argmax(probs, dim=-1)   # [batch_size, seq_len]

    # 高效地判断是否存在非豁免标签：即 preds 中是否有不属于 ignore_label_ids 的元素
    # mask[i] = True 表示该 token 是敏感标签
    is_private_mask = ~torch.isin(preds, ignore_label_ids)
    double_check_privacy = []
    for i in range(is_private_mask.shape[0]):  # 遍历batch
        token_mask = is_private_mask[i]  # [seq_len]
        contains_privacy = token_mask.any().item()

        if contains_privacy:
            double_check = token_mask.int()
            double_check_result = (double_check[:-1] & double_check[1:]).any().item()
            double_check_privacy.append(double_check_result)

            # 对敏感 token 的 max(prob) 求均值作为不信任度
            selected_probs = probs[i][token_mask]  # [num_sensitive_tokens, num_labels]
            untrust_score = selected_probs.max(dim=-1).values.mean().item()
        else:
            double_check_privacy.append(False)

            # 所有 token 的 "O" 类概率均值作为信任度
            o_label_id = model.config.label2id["O"]
            trust_score = probs[i, :, o_label_id].mean().item()
            untrust_score = 1.0 - trust_score

    return double_check_privacy

# 示例预测
# 示例
batch_ = []
num_true = 0
num_total = 0
with open("./results/english_non_17.jsonl", "r") as fr:
    lines = fr.readlines()
    for line in lines:
        if len(batch_) >= 16:
            contains = predict(batch_)
            for x in contains:
                if x:
                    num_true += 1
            num_total += len(contains)
            batch_ = []
        else:
            batch_.append(line)

print(f"accuracy: {float(num_true)/num_total}")

