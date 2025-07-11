## docker
```shell
cd ./sglang/docker
docker build -t sglang .  # build images

# init container
docker run -d --shm-size=8g --entrypoint bash --gpus all --name kv_security -v /home/kec23008/docker-sys/llm-security:/workspace sglang -c "tail -f /dev/null"

## enter the container
docker exec -it kv_security /bin/bash
```

## sglang使用
```shell
# download models
pip install protobuf google-api-python-client sentencepiece
# awq 依赖vllm for llama-3-70b-hf-gptq
pip install vllm==0.7.2
pip install optimum gptqmodel
pip install xformers
pip install sgl-kernel==0.1.5

# 使用本地sglang
cd /workspace/sglang-security/python
pip3 install -e .
pip install vllm==0.7.2
pip install sgl-kernel==0.1.5
```