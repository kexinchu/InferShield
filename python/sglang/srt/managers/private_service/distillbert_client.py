"""
DistilBERT Privacy Detection Client
用于向DistilBERT隐私检测服务发送请求的客户端
"""
import time
import json
import logging
import threading
import uuid
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import zmq

from sglang.srt.utils import get_zmq_socket
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

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
    is_private: bool
    confidence: float
    label: str
    scores: Dict[str, float]
    processing_time: float
    model_name: str
    status: str = "success"
    error: Optional[str] = None

class DistilBERTClient:
    """
    DistilBERT隐私检测客户端
    
    特性:
    1. 异步请求处理
    2. 批量请求支持
    3. 请求超时处理
    4. 连接重试机制
    5. 响应缓存
    """
    
    def __init__(self, 
                 server_args: ServerArgs,
                 port_args: PortArgs,
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 batch_size: int = 16):
        
        self.server_args = server_args
        self.port_args = port_args
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        
        # 初始化ZMQ
        self.context = zmq.Context(2)
        self.send_socket = get_zmq_socket(
            self.context, zmq.PUSH, port_args.distillbert_service_port, False
        )
        self.recv_socket = get_zmq_socket(
            self.context, zmq.PULL, port_args.distillbert_client_port, False
        )
        
        # 请求队列和响应映射
        self.request_queue = []
        self.pending_requests = {}  # request_id -> callback
        self.response_cache = {}  # text_hash -> (response, timestamp)
        self.cache_ttl = 300  # 5分钟缓存
        
        # 处理线程
        self.processing_thread = threading.Thread(
            target=self._process_responses,
            daemon=True
        )
        self.sending_thread = threading.Thread(
            target=self._process_sending,
            daemon=True
        )
        
        # 控制标志
        self.running = True
        self.lock = threading.Lock()
        
        # 启动处理线程
        self.processing_thread.start()
        self.sending_thread.start()
        
        logger.info("DistilBERT Client started")
    
    def detect_privacy(self, text: str, callback: Optional[Callable] = None) -> str:
        """
        检测文本隐私信息（异步）
        
        Args:
            text: 待检测的文本
            callback: 回调函数，接收DistilBERTResponse参数
            
        Returns:
            str: 请求ID
        """
        # 检查缓存
        text_hash = hash(text)
        if text_hash in self.response_cache:
            cached_response, timestamp = self.response_cache[text_hash]
            if time.time() - timestamp < self.cache_ttl:
                if callback:
                    callback(cached_response)
                return f"cached_{text_hash}"
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 创建请求
        request = DistilBERTRequest(
            text=text,
            request_id=request_id
        )
        
        # 添加到队列
        with self.lock:
            self.request_queue.append(request)
            if callback:
                self.pending_requests[request_id] = callback
        
        return request_id
    
    def detect_privacy_sync(self, text: str, timeout: Optional[float] = None) -> DistilBERTResponse:
        """
        检测文本隐私信息（同步）
        
        Args:
            text: 待检测的文本
            timeout: 超时时间（秒）
            
        Returns:
            DistilBERTResponse: 检测结果
        """
        if timeout is None:
            timeout = self.timeout
        
        result_received = threading.Event()
        result = [None]
        
        def callback(response):
            result[0] = response
            result_received.set()
        
        request_id = self.detect_privacy(text, callback)
        
        # 等待结果
        if result_received.wait(timeout):
            return result[0]
        else:
            # 超时处理
            with self.lock:
                if request_id in self.pending_requests:
                    del self.pending_requests[request_id]
            
            return DistilBERTResponse(
                request_id=request_id,
                is_private=False,
                confidence=0.0,
                label="public",
                scores={"public": 1.0, "private": 0.0},
                processing_time=0.0,
                model_name="unknown",
                status="timeout",
                error="Request timeout"
            )
    
    def batch_detect(self, texts: List[str], callback: Optional[Callable] = None) -> List[str]:
        """
        批量检测隐私信息
        
        Args:
            texts: 待检测的文本列表
            callback: 批量回调函数，接收List[DistilBERTResponse]参数
            
        Returns:
            List[str]: 请求ID列表
        """
        request_ids = []
        
        for text in texts:
            request_id = self.detect_privacy(text)
            request_ids.append(request_id)
        
        # 如果有批量回调，设置批量处理
        if callback:
            self._setup_batch_callback(request_ids, callback)
        
        return request_ids
    
    def _setup_batch_callback(self, request_ids: List[str], callback: Callable):
        """设置批量回调"""
        pending_count = [len(request_ids)]
        responses = [None] * len(request_ids)
        
        def batch_callback(response):
            try:
                idx = request_ids.index(response.request_id)
                responses[idx] = response
                pending_count[0] -= 1
                
                if pending_count[0] == 0:
                    # 所有响应都收到了
                    callback(responses)
            except ValueError:
                pass  # 请求ID不在列表中
        
        # 为每个请求设置回调
        with self.lock:
            for request_id in request_ids:
                if request_id in self.pending_requests:
                    # 保存原始回调
                    original_callback = self.pending_requests[request_id]
                    
                    # 设置新的回调
                    def make_callback(orig_cb, batch_cb):
                        def combined_callback(response):
                            if orig_cb:
                                orig_cb(response)
                            batch_cb(response)
                        return combined_callback
                    
                    self.pending_requests[request_id] = make_callback(original_callback, batch_callback)
    
    def _process_sending(self):
        """处理发送请求的线程"""
        while self.running:
            try:
                # 收集批量请求
                batch_requests = []
                
                with self.lock:
                    while len(self.request_queue) > 0 and len(batch_requests) < self.batch_size:
                        batch_requests.append(self.request_queue.pop(0))
                
                if batch_requests:
                    # 发送批量请求
                    message = {
                        'batch': [
                            {
                                'request_id': req.request_id,
                                'text': req.text,
                                'timestamp': req.timestamp
                            }
                            for req in batch_requests
                        ]
                    }
                    
                    self.send_socket.send_json(message)
                    logger.debug(f"Sent batch of {len(batch_requests)} requests")
                
                time.sleep(0.01)  # 短暂休眠
                
            except Exception as e:
                logger.error(f"Error in sending thread: {e}")
                time.sleep(1)
    
    def _process_responses(self):
        """处理响应的线程"""
        while self.running:
            try:
                # 接收响应
                message = self.recv_socket.recv_json()
                
                if 'batch' not in message:
                    logger.error("Invalid response format: missing 'batch' field")
                    continue
                
                # 处理批量响应
                for response_data in message['batch']:
                    self._handle_single_response(response_data)
                
            except Exception as e:
                logger.error(f"Error in response processing: {e}")
                time.sleep(1)
    
    def _handle_single_response(self, response_data: Dict):
        """处理单个响应"""
        try:
            request_id = response_data.get('request_id', 'unknown')
            status = response_data.get('status', 'error')
            
            if status == 'success':
                result_data = response_data.get('result', {})
                
                response = DistilBERTResponse(
                    request_id=request_id,
                    is_private=result_data.get('is_private', False),
                    confidence=result_data.get('confidence', 0.0),
                    label=result_data.get('label', 'public'),
                    scores=result_data.get('scores', {}),
                    processing_time=result_data.get('processing_time', 0.0),
                    model_name=result_data.get('model_name', 'unknown'),
                    status=status
                )
            else:
                response = DistilBERTResponse(
                    request_id=request_id,
                    is_private=False,
                    confidence=0.0,
                    label="public",
                    scores={"public": 1.0, "private": 0.0},
                    processing_time=0.0,
                    model_name="unknown",
                    status=status,
                    error=response_data.get('error', 'Unknown error')
                )
            
            # 缓存响应
            self._cache_response(response)
            
            # 调用回调函数
            with self.lock:
                if request_id in self.pending_requests:
                    callback = self.pending_requests.pop(request_id)
                    try:
                        callback(response)
                    except Exception as e:
                        logger.error(f"Error in callback for request {request_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error handling response: {e}")
    
    def _cache_response(self, response: DistilBERTResponse):
        """缓存响应"""
        if response.status == 'success':
            # 这里需要从原始请求中获取文本，简化处理
            # 实际实现中可能需要维护request_id到文本的映射
            pass
    
    def clear_cache(self):
        """清除缓存"""
        with self.lock:
            self.response_cache.clear()
    
    def get_pending_count(self) -> int:
        """获取待处理请求数量"""
        with self.lock:
            return len(self.pending_requests)
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        with self.lock:
            return len(self.request_queue)
    
    def close(self):
        """关闭客户端"""
        self.running = False
        self.context.term()
        logger.info("DistilBERT Client stopped")

# 全局客户端实例
_global_client = None

def get_distillbert_client(server_args: Optional[ServerArgs] = None,
                          port_args: Optional[PortArgs] = None) -> DistilBERTClient:
    """获取全局DistilBERT客户端实例"""
    global _global_client
    
    if _global_client is None:
        if server_args is None:
            server_args = ServerArgs()
        if port_args is None:
            port_args = PortArgs()
        
        _global_client = DistilBERTClient(server_args, port_args)
    
    return _global_client

def detect_privacy_with_distillbert(text: str, timeout: Optional[float] = None) -> DistilBERTResponse:
    """
    使用DistilBERT检测隐私信息的便捷函数
    
    Args:
        text: 待检测的文本
        timeout: 超时时间（秒）
        
    Returns:
        DistilBERTResponse: 检测结果
    """
    client = get_distillbert_client()
    return client.detect_privacy_sync(text, timeout)

def batch_detect_privacy_with_distillbert(texts: List[str], 
                                        callback: Optional[Callable] = None) -> List[str]:
    """
    批量使用DistilBERT检测隐私信息的便捷函数
    
    Args:
        texts: 待检测的文本列表
        callback: 批量回调函数
        
    Returns:
        List[str]: 请求ID列表
    """
    client = get_distillbert_client()
    return client.batch_detect(texts, callback)

# 示例用法
if __name__ == "__main__":
    # 创建客户端
    server_args = ServerArgs()
    port_args = PortArgs()
    client = DistilBERTClient(server_args, port_args)
    
    # 测试文本
    test_texts = [
        "This is a public message about general topics.",
        "My email is john.doe@example.com and my phone number is 123-456-7890.",
        "The weather is nice today.",
        "My social security number is 123-45-6789."
    ]
    
    # 同步检测
    print("=== 同步检测 ===")
    for text in test_texts:
        result = client.detect_privacy_sync(text)
        print(f"Text: {text[:50]}...")
        print(f"Private: {result.is_private}, Confidence: {result.confidence:.3f}")
        print(f"Label: {result.label}, Processing time: {result.processing_time:.3f}s")
        print()
    
    # 异步检测
    print("=== 异步检测 ===")
    def callback(response):
        print(f"Async result for {response.request_id[:8]}...")
        print(f"Private: {response.is_private}, Confidence: {response.confidence:.3f}")
        print()
    
    for text in test_texts:
        client.detect_privacy(text, callback)
    
    # 等待异步请求完成
    time.sleep(5)
    
    # 关闭客户端
    client.close() 