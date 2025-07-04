import threading
import time
from privacy_detector_piiranha import PiiPrivacyService
from pii_bert_client import PiiBERTClient
from utils import load_jsonl_dataset
import sys
import os
import numpy as np
# Add the project root to Python path to use local sglang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from server_args import ServerArgs, PortArgs

# Define required PortArgs values (use unique IPC names for this test)
def get_test_port_args():
    # Create a dummy ServerArgs for PortArgs.init_new
    dummy_server_args = ServerArgs(model_path="/workspace/Models/piiranha-v1")
    return PortArgs.init_new(dummy_server_args)

# Start the server in a background thread
def start_server(port_args):
    server_args = ServerArgs(model_path="/workspace/Models/piiranha-v1")
    service = PiiPrivacyService(
        server_args=server_args,
        port_args=port_args,
        pii_model_name="/workspace/Models/piiranha-v1",
        gene_model_name="/workspace/Models/Qwen3-0.6B",
        confidence_threshold=0.7,
        device=None
    )
    # Keep the service running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.close()

if __name__ == "__main__":
    # Start server in background
    port_args = get_test_port_args()
    server_thread = threading.Thread(target=start_server, args=(port_args,), daemon=True)
    server_thread.start()
    time.sleep(5)  # Wait for server to start

    # Load test data
    texts, labels = load_jsonl_dataset("/workspace/Datasets/english_pii_43k.jsonl", sample_n=2000)
    # 检查字段名
    print(texts[0])
    print(labels[1])
    
    # Start client
    server_args = ServerArgs(model_path="/workspace/Models/piiranha-v1")
    client = PiiBERTClient(server_args, port_args)

    # Send requests and collect results
    results = []
    request_ids = []
    preds = []
    file_ = open("/workspace/results/pii-detection/res_file-pii.txt", "w")
    for text in texts:
        request_id = client.detect_privacy(text)
        request_ids.append(request_id)
    
    print(len(request_ids))
    file_ = open("/workspace/results/pii-detection/res_file-pii.txt", "w")
    for i, request_id in enumerate(request_ids):
        res = client.detect_privacy_sync(request_id)
        file_.write(f"Text:{text}\tstatus:{res.is_private}\tscore:{res.confidence}\tlabel:{labels[i]}\n")
        preds.append(1 if res.is_private else 0)
        client.free_cache(request_id)
    file_.close()

    # 显示统计信息
    preds = np.array(preds)
    labels_np = np.array(labels)
    acc = (preds == labels_np).mean()
    print(f"Accuracy for Custom Detctor: {acc:.4f}")