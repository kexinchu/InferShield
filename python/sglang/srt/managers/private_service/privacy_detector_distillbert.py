"""
Privacy Detector Service based on DistilBERT
使用DistilBERT模型进行隐私信息检测的第二级检测服务
"""
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import zmq
import numpy as np

try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from torch.nn.functional import softmax
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch and transformers not available. DistilBERT service will not work.")

from sglang.srt.utils import get_zmq_socket
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

@dataclass
class DistilBERTDetectionResult:
    """DistilBERT检测结果"""
    is_private: bool
    confidence: float
    label: str
    scores: Dict[str, float]
    processing_time: float
    model_name: str = "distilbert-base-uncased"

@dataclass
class DistilBERTRequest:
    """DistilBERT检测请求"""
    text: str
    request_id: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class DistilBERTResponse:
    """DistilBERT检测响应"""
    request_id: str
    result: DistilBERTDetectionResult
    status: str = "success"
    error: Optional[str] = None

class DistilBERTPrivacyDetector:
    """
    基于DistilBERT的隐私检测器
    
    特性:
    1. 使用预训练的DistilBERT模型进行文本分类
    2. 支持批量处理
    3. 可配置的置信度阈值
    4. 模型热重载
    5. 性能监控
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 max_length: int = 512,
                 confidence_threshold: float = 0.7,
                 device: Optional[str] = None):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers are required for DistilBERT service")
        
        self.model_name = model_name
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 初始化模型和tokenizer
        self.tokenizer = None
        self.model = None
        self.labels = ["public", "private"]  # 二分类标签
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'total_private_detected': 0,
            'avg_processing_time': 0.0,
            'model_load_time': 0.0
        }
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载DistilBERT模型和tokenizer"""
        start_time = time.time()
        
        try:
            logger.info(f"Loading DistilBERT model: {self.model_name}")
            
            # 加载tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            
            # 加载模型
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.labels)
            )
            
            # 移动到指定设备
            self.model.to(self.device)
            self.model.eval()
            
            self.stats['model_load_time'] = time.time() - start_time
            logger.info(f"Model loaded successfully in {self.stats['model_load_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """预处理文本"""
        # 截断过长的文本
        if len(text) > self.max_length * 4:  # 粗略估计token数量
            text = text[:self.max_length * 4]
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def detect_privacy(self, text: str) -> DistilBERTDetectionResult:
        """
        检测文本中的隐私信息
        Args:
            text: 待检测的文本
        Returns:
            DistilBERTDetectionResult: 检测结果
        """
        start_time = time.time()
        
        try:
            # 预处理文本
            inputs = self.preprocess_text(text)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = softmax(logits, dim=1)
            
            # 获取预测结果
            scores = probabilities[0].cpu().numpy()
            predicted_label_idx = np.argmax(scores)
            predicted_label = self.labels[predicted_label_idx]
            confidence = float(scores[predicted_label_idx])
            
            # 判断是否为隐私信息
            is_private = predicted_label == "private" and confidence >= self.confidence_threshold
            
            # 构建分数字典
            scores_dict = {label: float(score) for label, score in zip(self.labels, scores)}
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._update_stats(is_private, processing_time)
            
            return DistilBERTDetectionResult(
                is_private=is_private,
                confidence=confidence,
                label=predicted_label,
                scores=scores_dict,
                processing_time=processing_time,
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error during privacy detection: {e}")
            raise
    
    def batch_detect(self, texts: List[str]) -> List[DistilBERTDetectionResult]:
        """
        批量检测隐私信息
        
        Args:
            texts: 待检测的文本列表
            
        Returns:
            List[DistilBERTDetectionResult]: 检测结果列表
        """
        results = []
        
        for text in texts:
            try:
                result = self.detect_privacy(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch detection: {e}")
                # 返回默认结果
                results.append(DistilBERTDetectionResult(
                    is_private=False,
                    confidence=0.0,
                    label="public",
                    scores={"public": 1.0, "private": 0.0},
                    processing_time=0.0,
                    model_name=self.model_name
                ))
        
        return results
    
    def _update_stats(self, is_private: bool, processing_time: float):
        """更新统计信息"""
        self.stats['total_requests'] += 1
        if is_private:
            self.stats['total_private_detected'] += 1
        
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

class DistilBERTPrivacyService:
    """
    DistilBERT隐私检测服务
    
    提供ZMQ接口的隐私检测服务，支持异步处理
    """
    
    def __init__(self, 
                 server_args: ServerArgs,
                 port_args: PortArgs,
                 model_name: str = "distilbert-base-uncased",
                 max_length: int = 512,
                 confidence_threshold: float = 0.7,
                 device: Optional[str] = None):
        
        self.server_args = server_args
        self.port_args = port_args
        
        # 初始化检测器
        self.detector = DistilBERTPrivacyDetector(
            model_name=model_name,
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
        
        logger.info("DistilBERT Privacy Service started")
    
    def _process_requests(self):
        """处理请求的主循环"""
        while self.running:
            try:
                # 接收请求
                message = self.recv_socket.recv_json()
                
                if 'batch' not in message:
                    logger.error("Invalid message format: missing 'batch' field")
                    continue
                
                # 处理批量请求
                responses = []
                for request_data in message['batch']:
                    response = self._handle_single_request(request_data)
                    responses.append(response)
                
                # 发送响应
                response_message = {'batch': responses}
                self.send_socket.send_json(response_message)
                
            except Exception as e:
                logger.error(f"Error in request processing: {e}")
                time.sleep(1)  # 防止错误循环
    
    def _handle_single_request(self, request_data: Dict) -> Dict:
        """处理单个请求"""
        try:
            request_id = request_data.get('request_id', 'unknown')
            text = request_data.get('text', '')
            
            if not text:
                return {
                    'request_id': request_id,
                    'status': 'error',
                    'error': 'Empty text provided'
                }
            
            # 执行检测
            result = self.detector.detect_privacy(text)
            
            return {
                'request_id': request_id,
                'status': 'success',
                'result': {
                    'is_private': result.is_private,
                    'confidence': result.confidence,
                    'label': result.label,
                    'scores': result.scores,
                    'processing_time': result.processing_time,
                    'model_name': result.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                'request_id': request_data.get('request_id', 'unknown'),
                'status': 'error',
                'error': str(e)
            }
    
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
        logger.info("DistilBERT Privacy Service stopped")

def main():
    """服务启动入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DistilBERT Privacy Detection Service")
    parser.add_argument("--model_name", default="distilbert-base-uncased", 
                       help="DistilBERT model name")
    parser.add_argument("--max_length", type=int, default=512,
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
    service = DistilBERTPrivacyService(
        server_args=server_args,
        port_args=port_args,
        model_name=args.model_name,
        max_length=args.max_length,
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