"""
Privacy Detector Service based on Piiranha-v1
使用Piiranha-v1模型进行隐私信息检测的第二级检测服务
"""
from ast import ListComp
import time
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import zmq
import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM
from torch.nn.functional import softmax

from sglang.srt.utils import get_zmq_socket
from sglang.srt.server_args import PortArgs, ServerArgs

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
                 pii_model_name: str = "/workspace/Models/piiranha-v1",
                 gene_model_name: str = "/workspace/Models/Qwen3-0.6B",
                 max_length: int = 256,
                 confidence_threshold: float = 0.7,
                 device: Optional[str] = None):
        
        self.pii_model_name = pii_model_name
        self.gene_model_name = gene_model_name
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        
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
        # 获取豁免标签
        self.ignore_label_ids = torch.tensor(
            [self.pii_model.config.label2id[l] for l in self.ignore_labels],
            dtype=torch.long
        )
        # 加载通用模型
        self.gene_tokenizer = AutoTokenizer.from_pretrained(self.gene_model_name, trust_remote_code=True)
        self.gene_model = AutoModelForCausalLM.from_pretrained(self.gene_model_name, trust_remote_code=True)

        # 移动到指定设备
        self.pii_model.to(self.device)
        self.gene_model.to(self.device)
        self.pii_model.eval()
        self.gene_model.eval()
        
        self.stats['model_load_time'] = time.time() - start_time
        logger.info(f"Model loaded successfully in {self.stats['model_load_time']:.2f}s")
    
    def detect_privacy_pii(self, texts):
        """
        使用pii finetuned model检测文本中的隐私信息
        """
        # 预处理文本 - Tokenize
        inputs = self.pii_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.pii_model.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.pii_model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)[0]  # [batch_size, seq_len, num_labels]
            preds = torch.argmax(probs, dim=-1)   # [batch_size, seq_len]
        
        # 高效地判断是否存在非豁免标签：即 preds 中是否有不属于 ignore_label_ids 的元素
        # mask[i] = True 表示该 token 是敏感标签
        is_private_mask = ~torch.isin(preds, self.ignore_label_ids)
        return is_private_mask, probs
    
    # 构造 prompt
    def make_prompt(self, user_input):
        return f"""You are a privacy classification assistant.

Your task is to determine if a given sentence contains any Personally Privacy Information.

Privacy Info includes: Real names (full or partial), Phone numbers, Email addresses, National ID, passport, social security numbers, Home or mailing addresses, Account numbers, License plates, IP addresses, GPS locations, Dates of birth or death, etc.

## Instructions:
1. Carefully read the input text.
2. Decide whether it includes any privacy infomation.
3. Output a number between 0 and 1 representing how confident you are that the text contains any privacy info.

## Examples:
Input: "Contact me at john.doe@example.com or 555-123-4567."
Answer: 0.97

Input: "Her birthday is January 14, 1993."
Answer: 0.93

Input: "The capital of France is Paris."
Answer: 0.47

Input: "This is a list of animals: dog, cat, lion."
Answer: 0.02

Input: "James recently moved to 456 Oak Avenue."
Answer: 0.96

## Now evaluate this input:
Input: "{user_input}"
Answer:"""

    def detect_privacy_gene(self, texts):
        """使用 general LLM model detect privacy info"""
        prompts = [self.make_prompt(text) for text in texts]

        # 批处理编码
        inputs = self.gene_tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(self.gene_model.device)
        
        with torch.no_grad():
            outputs = self.gene_model.generate(**inputs, max_new_tokens=10)

        # 解码每个样本的输出
        scores = []
        for j, output in enumerate(outputs):
            # 获取新生成的token
            input_length = inputs["input_ids"].shape[1] if len(inputs["input_ids"].shape) > 1 else inputs["input_ids"].shape[0]
            result = self.gene_tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            
            # 尝试解析成 float
            try:
                score = [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', result)]
                score = max(0.0, min(1.0, score[0])) if score else 0.5
            except:
                score = 0.5
            scores.append(score)
        
        return scores

    def detect_privacy(self, texts) -> List[PiiDetectionResult]:
        """
        检测文本中的隐私信息
        Args:
            text: 待检测的文本
        Returns:
            PiiDetectionResult: 检测结果
        """
        start_time = time.time()
        batched_result = []
        batched_text_gene = []
        batched_index_gene = []
        # 使用pii检测
        is_private_mask, probs = self.detect_privacy_pii(texts)
        for idx in range(is_private_mask.shape[0]):  # 遍历batch
            token_mask = is_private_mask[idx]  # [seq_len]
            contains_privacy = token_mask.any().item()
            if contains_privacy:
                # 对敏感 token 的 max(prob) 求均值作为不信任度
                selected_probs = probs[idx][token_mask]  # [num_sensitive_tokens, num_labels]
                untrust_score = selected_probs.max(dim=-1).values.mean().item()
                batched_result.append(PiiDetectionResult(
                    is_private=True,
                    confidence=untrust_score,
                    score=untrust_score,
                    model_name=self.pii_model_name
                ))
            else:
                batched_text_gene.append(texts[idx])
                batched_index_gene.append(idx)
                batched_result.append(PiiDetectionResult(
                    is_private=True,
                    confidence=0.5,
                    score=0.5,
                    model_name=self.pii_model_name
                ))
        
        if len(batched_index_gene) > 0:
            scores = self.detect_privacy_gene(batched_text_gene)

            for j, score in enumerate(scores):
                # 获取新生成的token
                batched_result[batched_index_gene[j]] = PiiDetectionResult(
                    is_private=True if score > self.confidence_threshold else False,
                    confidence=score,
                    score=score,
                    model_name=self.gene_model_name
                )
        
        processing_time = time.time() - start_time
        
        # 更新统计信息
        self._update_stats(len(texts), processing_time)
        
        return batched_result
     

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
                 pii_model_name: str = "/workspace/Models/piiranha-v1",
                 gene_model_name: str = "/workspace/Models/Qwen3-0.6B",
                 max_length: int = 256,
                 confidence_threshold: float = 0.7,
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
        self.context = zmq.Context(2)
        self.recv_socket = get_zmq_socket(
            self.context, zmq.PULL, port_args.distillbert_service_port, False
        )
        self.send_socket = get_zmq_socket(
            self.context, zmq.PUSH, port_args.distillbert_client_port, False
        )
        
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
            message = self.recv_socket.recv_json()
            
            if 'batch' not in message:
                logger.error("Invalid message format: missing 'batch' field")
                continue
            
            # 处理批量请求
            responses = self._handle_requests(message['batch'])
            
            # 发送响应
            response_message = {'batch': responses}
            self.send_socket.send_json(response_message)
    
    def _handle_requests(self, request_datas: List) -> List:
        """处理单个请求"""
        texts = []
        request_ids = []
        for request_data in request_datas:
            request_id = request_data.get('request_id', 'unknown')
            text = request_data.get('text', '')
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
    parser.add_argument("--model_path", default="/workspace/Models/Qwen3-4B", 
                       help="model name")
    parser.add_argument("--pii_model_name", default="/workspace/Models/piiranha-v1", 
                       help="Pii model name")
    parser.add_argument("--gene_model_name", default="/workspace/Models/Qwen3-0.6B", 
                       help="General LLM model name")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
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