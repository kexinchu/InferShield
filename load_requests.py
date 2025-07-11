import os
import json
import random
from typing import List, Tuple
# from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import pandas as pd

# default output length for LLM
default_max_output_length = 256
default_min_prompt_length = 4

# get current abs path
current_file_path = os.path.dirname(__file__)
# print(f"当前文件的绝对路径是: {current_file_path}")
request_root_dir = current_file_path + "/"

def read_chatGPT(file_path):
    # Load the dataset.
    requests = []
    session_id = 0
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        for data in dataset:
            for i in range(len(data["conversations"]) - 1):
                context = ""
                if i % 2 == 0:
                    if context == "":
                        context = data["conversations"][i]["value"]
                    else:
                        context = context + "\n" + data["conversations"][i]["value"]
                    answer = data["conversations"][i+1]["value"]
                    requests.append((context, answer, session_id))
            # context = data["conversations"][0]["value"]
            # answer = data["conversations"][1]["value"]
            # requests.append((context, answer, session_id))
            session_id += 1
            if session_id > 50:
                break
    return requests

def read_txt(file_path, max_num=-1):
    # 从txt文件中读取请求；每行一个request
    requests = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            req = line.strip()
            if req:
                requests.append((req, ""))
    return requests

def read_chatgpt_paraphrases(file_path, max_num=-1):
    requests = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = row.get("text", "").strip()
            paraphrase = row.get("paraphrases", "").strip()
            if prompt:
                requests.append((prompt, paraphrase, i))
    return requests

def read_multiturn_chat(file_path, max_num=-1):
    requests = []
    session_id = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            full_context = data.get("instruction", "").strip()
            final_answer = data.get("output", "").strip()
            
            # 如果context为空，跳过
            if not full_context:
                continue
                
            # 分割对话轮次
            turns = full_context.split("\n")
            current_context = ""
            
            # 处理每一轮对话
            for i in range(0, len(turns), 2):
                if i + 1 >= len(turns):  # 如果是最后一轮且没有回答，使用final_answer
                    if turns[i].startswith("Human:"):
                        current_context += turns[i] + "\n"
                        requests.append((current_context.strip(), final_answer, session_id))
                else:
                    # 添加当前轮次的对话
                    current_context += turns[i] + "\n"
                    if i + 1 < len(turns):
                        # 提取当前轮次的回答
                        current_answer = turns[i + 1].replace("Assistant:", "").strip()
                        requests.append((current_context.strip(), current_answer, session_id))
                        current_context += turns[i + 1] + "\n"
            
            session_id += 1
                
    return requests

def read_configurable_system_prompt_multitask(file_path, max_num=-1):
    requests = []
    df = pd.read_parquet(file_path)
    for idx, row in df.iterrows():
        context = str(row.get("system", "")).strip() + "\n" + str(row.get("prompt", "")).strip()
        answer = str(row.get("chosen", "")).strip()
        if context:
            requests.append((context, answer, idx))
    return requests

def load_requests(
        req_file, 
        tokenizer, 
        max_embedding_positions, 
        max_nums=-1
    ) -> List[Tuple[str, int, int, int]]:
    requests = []
    file_path = request_root_dir + "../Datasets/" + req_file
    if not os.path.exists(file_path):
        print(f"文件不存在, 请检查路径：{file_path}")
        return requests

    # 处理逻辑，根据数据集确定
    if "GPT" in req_file:
        requests = read_chatGPT(file_path)
    elif "paraphrases" in req_file:
        requests = read_chatgpt_paraphrases(file_path, max_nums)
    elif "multiturn" in req_file:
        requests = read_multiturn_chat(file_path, max_nums)
    elif "configurable-system-prompt-multitask" in req_file:
        requests = read_configurable_system_prompt_multitask(file_path, max_nums)
    else:
        requests = read_txt(file_path, max_nums)

    # tokenizers 处理
    prompts = [prompt for prompt, _, _ in requests]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion, _ in requests]
    session_ids = [session_id for _, _, session_id in requests]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(requests)):
        output_len = len(completion_token_ids[i])
        # 当数据集未提供output时，或者output过于短时
        if output_len < 4:
            output_len = default_max_output_length
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len, session_ids[i]))

    # Filter out too long snd too short equences. (select 512 ~ 2k)
    filtered_dataset: List[Tuple[str, int, int, int]] = []
    for prompt, prompt_token_ids, output_len, session_id in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < default_min_prompt_length:
            continue
        if prompt_len > max_embedding_positions:
            continue
        # print(prompt_len)
        filtered_dataset.append((prompt, prompt_len, output_len, session_id))

    return filtered_dataset

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

if __name__ == "__main__":
    # 重写 load_tokenizer 方法，直接加载 llama-3-8b 的 tokenizer
    # def load_tokenizer(model_name):
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     return tokenizer
    
    # def local_model_path(model_root_dir: str) -> str:
    #     # model 路径
    #     local_model_dir = model_root_dir
    #     for dir_name in os.listdir(local_model_dir):
    #         if "models" in dir_name:
    #             local_model_dir = local_model_dir + "/" + dir_name + "/snapshots/"
    #     local_model_dir = local_model_dir + os.listdir(local_model_dir)[0]

    #     return local_model_dir

    # # # 使用 llama-3-8b 模型
    # model_name = local_model_path("./llama-3-8b-hf")
    # tokenizer = load_tokenizer(model_name)

    # requests = load_requests("ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer, max_embedding_positions=1000000)

    # requests = read_configurable_system_prompt_multitask("./configurable-system-prompt-multitask.parquet")
    requests = read_multiturn_chat("./multiturn_chat_0.8M.json")
    print(len(requests))
    print(requests[0:5])

