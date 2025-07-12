import sys
import os
import subprocess

if __name__ == "__main__":
    model_name = "llama3-8b"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    local_path = {
        "llama3-8b": "../Models/Llama-3.2-8B",
        "qwen3-8b": "../Models/Qwen3-8B"
    }

    print(f"Loading model: {model_name}")
    
    # Use the existing launch_server module
    cmd = [
        sys.executable, "python/sglang/launch_server.py",
        "--model-path", local_path[model_name],
        "--host", "127.0.0.1",
        "--port", "8080",
        "--max-running-requests", "32",
        "--max-total-tokens", "40960",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--attention-backend", "torch_native",
        "--sampling-backend", "pytorch",
        "--disable-cuda-graph",
        "--disable-cuda-graph-padding",
        "--tp-size", "2"
    ]
    
    print("Starting HTTP server on port 8080...")
    print(f"Command: {' '.join(cmd)}")
    
    # Add the python directory to the path
    env = os.environ.copy()
    python_path = os.path.join(os.path.dirname(__file__), 'python')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = python_path + ':' + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = python_path
    
    subprocess.run(cmd, env=env)
