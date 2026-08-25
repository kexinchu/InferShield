#/bin/sh

# system-prompt
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8084 1
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8084 1
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8084 1
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8084 1

# multi-session
python3 client_perf_safekv.py /root/code/sglang-security/results/english_pii_43k-actual.jsonl 127.0.0.1:8084 # test
python3 client_perf_safekv.py /root/code/sglang-security/results/french_pii_62k-actual.jsonl 127.0.0.1:8084
python3 client_perf_safekv.py /root/code/sglang-security/results/german_pii_52k-actual.jsonl 127.0.0.1:8084
python3 client_perf_safekv.py /root/code/sglang-security/results/italian_pii_50k-actual.jsonl 127.0.0.1:8084


# perf-overall
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8084
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8084
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8084
python3 client_perf_safekv.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8084
