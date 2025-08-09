#/bin/sh

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
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8083
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8083
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8083
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8083

# # multi-session
# python3 client_perf_test.py /root/code/sglang-security/results/english_pii_43k-actual.jsonl 127.0.0.1:8083 # test
# python3 client_perf_test.py /root/code/sglang-security/results/french_pii_62k-actual.jsonl 127.0.0.1:8083
# python3 client_perf_test.py /root/code/sglang-security/results/german_pii_52k-actual.jsonl 127.0.0.1:8083
# python3 client_perf_test.py /root/code/sglang-security/results/italian_pii_50k-actual.jsonl 127.0.0.1:8083

# # system-prompt
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8083 1
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8083 1
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8083 1
# python3 client_perf_test.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8083 1

# system-prompt
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8083 1
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8083 1
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8083 1
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8083 1

# multi-session
# python3 client_perf_throughput.py /root/code/sglang-security/results/english_pii_43k-actual.jsonl 127.0.0.1:8083 # test
# python3 client_perf_throughput.py /root/code/sglang-security/results/french_pii_62k-actual.jsonl 127.0.0.1:8083
# python3 client_perf_throughput.py /root/code/sglang-security/results/german_pii_52k-actual.jsonl 127.0.0.1:8083
# python3 client_perf_throughput.py /root/code/sglang-security/results/italian_pii_50k-actual.jsonl 127.0.0.1:8083

# perf-overall
python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl 127.0.0.1:8083
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl 127.0.0.1:8083
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl 127.0.0.1:8083
# python3 client_perf_throughput.py /dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl 127.0.0.1:8083
