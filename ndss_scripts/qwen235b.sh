#/bin/sh

# CUDA_VISIBLE_DEVICES=6,7 /usr/bin/python3 -m sglang.launch_server --model-path /dcar-vepfs-trans-models/Qwen3-32B --host 127.0.0.1 --port 8082 \
# --max-running-requests 32 --max-total-tokens 40960 --dtype bfloat16 --trust-remote-code --attention-backend torch_native --sampling-backend pytorch \
# --disable-cuda-graph --disable-cuda-graph-padding --tp-size 2 --allow-auto-truncate --context-length 65536

# sample_req
# python3 client.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8080
# python3 client.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8080
# python3 client.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8080
# python3 client.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8080

# more_complex_req
# python3 client_level2.py /root/code/sglang-security/results/english_pii_43k-new_prompts.txt 127.0.0.1:8080
# python3 client_level2.py /root/code/sglang-security/results/french_pii_62k-new_prompts.txt 127.0.0.1:8080
# python3 client_level2.py /root/code/sglang-security/results/german_pii_52k-new_prompts.txt 127.0.0.1:8080
# python3 client_level2.py /root/code/sglang-security/results/italian_pii_50k-new_prompts.txt 127.0.0.1:8080


# # diff length
# python3 client_diff_length.py 127.0.0.1:8080

# perf-overall
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8080
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8080
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8080
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8080

# # # multi-session
# python3 client_perf_ori.py /root/code/sglang-security/results/english_pii_43k-actual.jsonl 127.0.0.1:8080 # test
# python3 client_perf_ori.py /root/code/sglang-security/results/french_pii_62k-actual.jsonl 127.0.0.1:8080
# python3 client_perf_ori.py /root/code/sglang-security/results/german_pii_52k-actual.jsonl 127.0.0.1:8080
# python3 client_perf_ori.py /root/code/sglang-security/results/italian_pii_50k-actual.jsonl 127.0.0.1:8080

# # # system-prompt
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8080 1
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8080 1
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8080 1
# python3 client_perf_ori.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8080 1

# python3 client.py /root/code/sglang-security/results/english_pii_43k-after_level_2.jsonl 127.0.0.1:8080
# python3 client.py /root/code/sglang-security/results/french_pii_62k-after_level_2.jsonl 127.0.0.1:8080
# python3 client.py /root/code/sglang-security/results/german_pii_52k-after_level_2.jsonl 127.0.0.1:8080
# python3 client.py /root/code/sglang-security/results/italian_pii_50k-after_level_2.jsonl 127.0.0.1:8080

# system-prompt
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8083 1
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8083 1
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8083 1
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8083 1

# multi-session
python3 client_perf_safekv.py /root/code/sglang-security/results/english_pii_43k-actual.jsonl 127.0.0.1:8083 # test
python3 client_perf_safekv.py /root/code/sglang-security/results/french_pii_62k-actual.jsonl 127.0.0.1:8083
python3 client_perf_safekv.py /root/code/sglang-security/results/german_pii_52k-actual.jsonl 127.0.0.1:8083
python3 client_perf_safekv.py /root/code/sglang-security/results/italian_pii_50k-actual.jsonl 127.0.0.1:8083


# # perf-overall
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8083
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8083
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8083
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8083
