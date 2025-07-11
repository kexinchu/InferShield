import sys
from python.sglang import Runtime, set_default_backend
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    model_name = "llama3-8b"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    local_path = {
        "llama3-8b": "../Models/Llama-3.2-8B",
        "qwen3-8b": "../Models/Qwen3-8B"
    }

    print(f"Loading model: {model_name}")
    runtime = Runtime(
        model_path=local_path[model_name],
        max_running_requests=32,
        max_total_tokens=40960,
        dtype="bfloat16",
        trust_remote_code=True,
        attention_backend="torch_native",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        disable_cuda_graph_padding=True,
        tp_size=2,
    )
    set_default_backend(runtime)
    print("Starting HTTP server on port 8080...")
    runtime.run_http_server(port=8080)
