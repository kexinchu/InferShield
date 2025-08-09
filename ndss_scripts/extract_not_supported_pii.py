import json

def load_jsonl_dataset(path):
    # 读取jsonl文件
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 解析
    texts = []
    labels = []
    for line in lines:
        item = json.loads(line)
        texts.append(item)

    #     # 判断是否有PII
    #     texts.append(item["source_text"])
    #     bio_labels = item["mbert_bio_labels"]
    #     if isinstance(bio_labels, str):
    #         bio_labels = eval(bio_labels)  # 兼容字符串格式
    #     label = 1 if any(l != "O" for l in bio_labels) else 0
    #     labels.append(label)
    return texts, labels

# 1. 加载数据集
ds, _ = load_jsonl_dataset("/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl")

# 2. 收集所有出现的 PII label
all_labels = set()
for example in ds:  # 可根据需求调整扫描样本量
    for m in example["privacy_mask"]:
        all_labels.add(m["label"])

piiranha_types = {
    "ACCOUNTNUM","BUILDINGNUM","CITY","CREDITCARDNUMBER","DATEOFBIRTH",
    "DRIVERLICENSENUM","EMAIL","GIVENNAME","SURNAME","IDCARDNUM",
    "PASSWORD","SOCIALNUM","STREET","TAXNUM","TELEPHONENUM","USERNAME","ZIPCODE",
    # 下面的是不太敏感的
    "JOBAREA", "FIRSTNAME", "AGE", "GENDER", "HEIGHT", "BUILDINGNUMBER", "MASKEDNUMBER",
    "DOB", "TIME", "JOBTITLE", "COUNTY", "EYECOLOR", "LASTNAME", "DATE", "PREFIX", "MIDDLENAME",
    "CREDITCARDISSUER", "STATE", "ORDINALDIRECTION", "SEX", "JOBTYPE", "CURRENCYCODE", "CURRENCYSYMBOL",
    "AMOUNT", "ACCOUNTNAME", "PHONENUMBER", "CURRENCY", "COMPANYNAME", "URL", "SECONDARYADDRESS",
    "ETHEREUMADDRESS", "CREDITCARDCVV", "BIC"
}

# 3. 筛选出 Piiranha 不支持的 label
unsupported = all_labels - piiranha_types
print("Piiranha-v1 Unsupported PII Types:", unsupported)

# 4. 为每种 unsupported label 随机抽 5 个样本输出
from collections import defaultdict
samples = defaultdict(list)
for example in ds:
    for m in example["privacy_mask"]:
        label = m["label"]
        if label in unsupported:
            samples[label].append(example["target_text"].replace("[" + label + "]", m["value"]))

# 打印结果
for label, texts in samples.items():
    for t in texts:
        print(t)

