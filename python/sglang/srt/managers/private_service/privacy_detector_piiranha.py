"""
Privacy Detector Service based on Piiranha-v1
使用Piiranha-v1模型进行隐私信息检测的第二级检测服务
"""
import time
import logging
import threading
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import zmq
import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM
from torch.nn.functional import softmax

from sglang.srt.utils import get_zmq_socket
from sglang.srt.server_args import PortArgs, ServerArgs
from .utils import make_chat_messages, make_prompt
from .global_task_queue import tier_2_task_queue, tier_2_result_queue

# Security-first: anything that can identify a person stays Private.
# Public is allowed only when no risk tags remain and Llama is confident.
_HIGH_PRECISION_LABEL_MARKERS = (
    "EMAIL", "PHONE", "TELEPHONE", "SOCIAL", "SSN", "CREDIT", "PASSWORD",
    "ACCOUNTNUM", "ACCOUNT_NUM", "ACCOUNTNUMBER", "IDCARD", "DRIVER",
    "TAXNUM", "IBAN", "IPV", "IPADDRESS", "MAC", "CVV", "BITCOIN",
    "ETHEREUM", "LITECOIN", "PASSPORT", "PIN", "IMEI", "USERNAME",
)

logger = logging.getLogger(__name__)

@dataclass
class PiiDetectionResult:
    """Pii检测结果"""
    is_private: bool
    confidence: float
    score: float
    model_name: str = "Piiranha-v1"

@dataclass
class PiiRequest:
    """Pii检测请求"""
    text: str
    request_id: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class PiiResponse:
    """Pii检测响应"""
    request_id: str
    result: PiiDetectionResult
    status: str = "success"
    error: Optional[str] = None

class PiiPrivacyDetector:
    """
    基于Pii的隐私检测器

    特性:
    1. 使用预训练的Pii模型进行文本分类
    2. 支持批量处理
    3. 可配置的置信度阈值
    4. 模型热重载
    5. 性能监控
    """
    def __init__(self,
                 pii_model_name: str = "/dcar-vepfs-trans-models/piiranha-v1",
                 gene_model_name: str = "/dcar-vepfs-trans-models/Llama-3.2-1B",
                 max_length: int = 256,
                 confidence_threshold: float = 0.55,
                 device: Optional[str] = None):

        self.pii_model_name = pii_model_name
        self.gene_model_name = gene_model_name
        self.max_length = max_length
        # Yes-prob bar for Private. Values near 0.5 stay Private.
        self.confidence_threshold = confidence_threshold
        self.high_precision_label_ids = torch.tensor([], dtype=torch.long)
        self.escalate_label_ids = torch.tensor([], dtype=torch.long)
        self.street_label_ids = torch.tensor([], dtype=torch.long)
        self.addr_part_label_ids = torch.tensor([], dtype=torch.long)
        self._yes_token_ids: Tuple[int, ...] = ()
        self._no_token_ids: Tuple[int, ...] = ()

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 初始化模型和tokenizer
        self.labels = ["public", "private"]  # 二分类标签
        self.ignore_labels = ["O", "I-CITY"]

        # 性能统计
        self.stats = {
            'total_requests': 0,
            'total_private_detected': 0,
            'avg_processing_time': 0.0,
            'model_load_time': 0.0
        }

        # 加载Pii模型和tokenizer
        self._load_model()

    def _load_model(self):
        start_time = time.time()

        logger.info(f"Loading Pii model: {self.pii_model_name}")

        # 加载tokenizer & 模型
        self.pii_tokenizer = AutoTokenizer.from_pretrained(self.pii_model_name)
        self.pii_model = AutoModelForTokenClassification.from_pretrained(self.pii_model_name)
        label2id = getattr(self.pii_model.config, "label2id", {}) or {}
        high_ids = []
        street_ids = []
        addr_part_ids = []
        for name, idx in label2id.items():
            upper = str(name).upper()
            if any(m in upper for m in _HIGH_PRECISION_LABEL_MARKERS):
                high_ids.append(int(idx))
            if "STREET" in upper:
                street_ids.append(int(idx))
            if any(m in upper for m in ("BUILDING", "ZIPCODE", "ZIP")):
                addr_part_ids.append(int(idx))
        self.high_precision_label_ids = torch.tensor(
            high_ids, dtype=torch.long
        ).to(self.device)
        self.street_label_ids = torch.tensor(street_ids, dtype=torch.long).to(self.device)
        self.addr_part_label_ids = torch.tensor(
            addr_part_ids, dtype=torch.long
        ).to(self.device)

        logger.info(f"Loading Pii model: {self.gene_model_name}")

        # 加载通用模型
        self.gene_tokenizer = AutoTokenizer.from_pretrained(self.gene_model_name, trust_remote_code=True)
        self.gene_model = AutoModelForCausalLM.from_pretrained(self.gene_model_name, trust_remote_code=True)
        if self.gene_tokenizer.pad_token is None:
            self.gene_tokenizer.pad_token = self.gene_tokenizer.eos_token
        self._yes_token_ids = self._first_tokens(("yes", "Yes", "YES", " yes"))
        self._no_token_ids = self._first_tokens(("no", "No", "NO", " no"))

        # 移动到指定设备
        self.pii_model.to(self.device)
        self.gene_model.to(self.device)
        self.pii_model.eval()
        self.gene_model.eval()

        self.stats['model_load_time'] = time.time() - start_time
        logger.info(f"Model loaded successfully in {self.stats['model_load_time']:.2f}s")

    def _first_tokens(self, words: Sequence[str]) -> Tuple[int, ...]:
        ids = []
        for word in words:
            pieces = self.gene_tokenizer.encode(word, add_special_tokens=False)
            if pieces:
                ids.append(int(pieces[0]))
        return tuple(dict.fromkeys(ids))

    def detect_privacy_pii(self, texts):
        """Token-level PII tags. Returns per-sample high-precision masks and probs."""
        inputs = self.pii_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.pii_model.device)

        with torch.no_grad():
            outputs = self.pii_model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        high_precision_mask = torch.isin(preds, self.high_precision_label_ids)
        street_mask = torch.isin(preds, self.street_label_ids)
        addr_part_mask = torch.isin(preds, self.addr_part_label_ids)
        address_combo_mask = street_mask.any(dim=-1) & addr_part_mask.any(dim=-1)
        return high_precision_mask, address_combo_mask, probs

    def detect_privacy_gene(self, texts):
        """Yes/no PII score from Llama next-token logits (not generated floats)."""
        if hasattr(self.gene_tokenizer, "apply_chat_template"):
            prompts = [
                self.gene_tokenizer.apply_chat_template(
                    make_chat_messages(text),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for text in texts
            ]
        else:
            prompts = [make_prompt(text) + "\n" for text in texts]

        inputs = self.gene_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(self.gene_model.device)

        with torch.no_grad():
            logits = self.gene_model(**inputs).logits[:, -1, :]

        yes_logit = logits[:, list(self._yes_token_ids)].max(dim=-1).values
        no_logit = logits[:, list(self._no_token_ids)].max(dim=-1).values
        pair = torch.stack([no_logit, yes_logit], dim=-1)
        yes_prob = softmax(pair, dim=-1)[:, 1]
        return [float(x) for x in yes_prob.detach().cpu()]

    def detect_privacy(self, texts) -> List[PiiDetectionResult]:
        """Two-stage semantic check: structured tags, then Llama on the rest."""
        start_time = time.time()
        high_precision_mask, address_combo_mask, probs = self.detect_privacy_pii(texts)
        batched_result: List[Optional[PiiDetectionResult]] = [None] * len(texts)
        llama_texts = []
        llama_index = []

        for idx in range(len(texts)):
            token_mask = high_precision_mask[idx]
            if token_mask.any().item() or bool(address_combo_mask[idx].item()):
                selected = token_mask
                if selected.any().item():
                    untrust_score = (
                        probs[idx][selected].max(dim=-1).values.mean().item()
                    )
                else:
                    untrust_score = 0.85
                batched_result[idx] = PiiDetectionResult(
                    is_private=True,
                    confidence=untrust_score,
                    score=untrust_score,
                    model_name=self.pii_model_name,
                )
            else:
                # No clear identifier: Llama must be confident it is Public.
                llama_texts.append(texts[idx])
                llama_index.append(idx)

        if llama_texts:
            scores = self.detect_privacy_gene(llama_texts)
            for j, score in enumerate(scores):
                batched_result[llama_index[j]] = PiiDetectionResult(
                    is_private=score >= self.confidence_threshold,
                    confidence=score,
                    score=score,
                    model_name=self.gene_model_name,
                )

        self._update_stats(len(texts), time.time() - start_time)
        return list(batched_result)


    def _update_stats(self, req_num, processing_time: float):
        """更新统计信息"""
        self.stats['total_requests'] += req_num

        # 更新平均处理时间
        current_avg = self.stats['avg_processing_time']
        total_requests = self.stats['total_requests']
        self.stats['avg_processing_time'] = (current_avg * (total_requests - 1) + processing_time) / total_requests

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['private_detection_rate'] = (
            stats['total_private_detected'] / stats['total_requests']
            if stats['total_requests'] > 0 else 0.0
        )
        return stats

    def reload_model(self, model_name: Optional[str] = None):
        """重新加载模型"""
        if model_name:
            self.model_name = model_name

        logger.info(f"Reloading model: {self.model_name}")
        self._load_model()

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to: {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

class PiiPrivacyService:
    """
    Pii隐私检测服务
    提供ZMQ接口的隐私检测服务，支持异步处理
    """
    def __init__(self,
                 server_args: ServerArgs,
                 port_args: PortArgs,
                 pii_model_name: str = "/dcar-vepfs-trans-models/piiranha-v1",
                 gene_model_name: str = "/dcar-vepfs-trans-models/Llama-3.2-1B",
                 max_length: int = 256,
                 confidence_threshold: float = 0.55,
                 device: Optional[str] = None):

        self.server_args = server_args
        self.port_args = port_args

        # 初始化检测器
        self.detector = PiiPrivacyDetector(
            pii_model_name=pii_model_name,
            gene_model_name=gene_model_name,
            max_length=max_length,
            confidence_threshold=confidence_threshold,
            device=device
        )

        # 初始化ZMQ
        # self.context = zmq.Context(2)
        # self.recv_socket = get_zmq_socket(
        #     self.context, zmq.PULL, port_args.distillbert_service_port, True  # bind=True for server
        # )
        # self.send_socket = get_zmq_socket(
        #     self.context, zmq.PUSH, port_args.distillbert_client_port, True  # bind=True for server
        # )

        # print(f"Server bound to:")
        # print(f"  Service port: {port_args.distillbert_service_port}")
        # print(f"  Client port: {port_args.distillbert_client_port}")

        # 初始化处理线程
        self.processing_thread = threading.Thread(
            target=self._process_requests,
            daemon=True
        )
        self.running = True

        # 启动处理线程
        self.processing_thread.start()

        logger.info("Pii Privacy Service started")

    def _process_requests(self):
        """处理请求的主循环"""
        while self.running:
            # 接收请求
            # message = self.recv_socket.recv_json()
            try:
                message = tier_2_task_queue.get(timeout=0.1)
                if 'batch' not in message:
                    logger.error("Invalid message format: missing 'batch' field")
                    continue
            except:
                time.sleep(0.1)
                continue

            # 处理批量请求
            responses = self._handle_requests(message['batch'])

            # 发送响应
            response_message = {'batch': responses}
            # self.send_socket.send_json(response_message)
            tier_2_result_queue.put(response_message)


    def _handle_requests(self, request_datas: List) -> List:
        """处理单个请求"""
        texts = []
        request_ids = []
        for request_data in request_datas:
            request_id = request_data["request_id"]
            text = request_data["text"]
            texts.append(text)
            request_ids.append(request_id)

        # 执行检测
        results = self.detector.detect_privacy(texts)

        final_results = []
        for idx, result in enumerate(results):
            final_results.append({
                'request_id': request_ids[idx],
                'status': 'success',
                'result': {
                    'is_private': result.is_private,
                    'confidence': result.confidence,
                    'score': result.score,
                    'model_name': result.model_name
                }
            })

        return final_results

    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return self.detector.get_stats()

    def reload_model(self, model_name: Optional[str] = None):
        """重新加载模型"""
        self.detector.reload_model(model_name)

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.detector.set_confidence_threshold(threshold)

    def close(self):
        """关闭服务"""
        self.running = False
        self.context.term()
        logger.info("Pii Privacy Service stopped")

def main():
    """服务启动入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Pii Privacy Detection Service")
    parser.add_argument("--model_path", default="/dcar-vepfs-trans-models/Qwen3-4B",
                       help="model name")
    parser.add_argument("--pii_model_name", default="/dcar-vepfs-trans-models/piiranha-v1",
                       help="Pii model name")
    parser.add_argument("--gene_model_name", default="/dcar-vepfs-trans-models/Llama-3.2-1B",
                       help="General LLM model name")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--confidence_threshold", type=float, default=0.55,
                       help="Confidence threshold for privacy detection")
    parser.add_argument("--device", default=None,
                       help="Device to run model on (cuda/cpu)")

    args = parser.parse_args()

    # 创建服务配置
    server_args = ServerArgs()
    port_args = PortArgs()

    # 启动服务
    service = PiiPrivacyService(
        server_args=server_args,
        port_args=port_args,
        pii_model_name=args.pii_model_name,
        gene_model_name=args.gene_model_name,
        confidence_threshold=args.confidence_threshold,
        device=args.device
    )

    try:
        # 保持服务运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down service...")
        service.close()

if __name__ == "__main__":
    main()
