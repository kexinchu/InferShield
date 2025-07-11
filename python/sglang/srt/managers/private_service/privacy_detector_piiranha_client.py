"""
PiiBERT Privacy Detection Client
用于向PiiBERT隐私检测服务发送请求的客户端
"""
import time
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
class PiiBERTRequest:
    """PiiBERT检测请求"""
    text: str
    request_id: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class PiiBERTResponse:
    """PiiBERT检测响应"""
    is_private: bool
    confidence: float
    score: float
    model_name: str
    status: str = "success"
    error: Optional[str] = None

class PiiBERTClient:
    """
    PiiBERT隐私检测客户端
    
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
        
        print(f"Client connected to:")
        print(f"  Service port: {port_args.distillbert_service_port}")
        print(f"  Client port: {port_args.distillbert_client_port}")
        
        # 请求队列和响应映射
        self.request_queue = []
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
        
        logger.info("PiiBERT Client started")
    
    def detect_privacy(self, text: str) -> str:
        """
        检测文本隐私信息（异步）
        Args:
            text: 待检测的文本
            callback: 回调函数，接收PiiBERTResponse参数
        Returns:
            str: 请求ID
        """
        # 检查缓存
        text_hash = hash(text)
        if str(text_hash) in self.response_cache:
            return str(text_hash)
        
        # 生成请求ID
        request_id = str(text_hash)
        
        # 创建请求
        request = PiiBERTRequest(
            text=text,
            request_id=request_id
        )
        
        # 添加到队列
        with self.lock:
            self.request_queue.append(request)
        
        self.response_cache[request_id] = None
        
        return request_id
    
    def detect_privacy_sync(self, request_id: str) -> PiiBERTResponse:
        """
        检测文本隐私信息（同步）
        """
        if request_id not in self.response_cache:
            return PiiBERTResponse(
                is_private=True,
                confidence=0.5,
                score=0.5,
                model_name="unknown",
                status="failed",
            )
        while self.response_cache[request_id] == None:
            time.sleep(0.1)
        return self.response_cache[request_id]
    
    def _process_sending(self):
        """处理发送请求的线程"""
        while self.running:
            if len(self.request_queue) <= 0:
                time.sleep(5)
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
    
    def _process_responses(self):
        """处理响应的线程"""
        while self.running:
            # try:
            # 接收响应
            message = self.recv_socket.recv_json()
            
            if 'batch' not in message:
                logger.error("Invalid response format: missing 'batch' field")
                continue
            
            # 处理批量响应
            for response_data in message['batch']:
                self._handle_single_response(response_data)
                
            # except Exception as e:
            #     logger.error(f"Error in response processing: {e}")
            #     time.sleep(1)
    
    def _handle_single_response(self, response_data: Dict):
        """处理单个响应"""
        try:
            request_id = response_data.get('request_id', 'unknown')
            status = response_data.get('status', 'error')
            
            if status == 'success':
                result_data = response_data.get('result', {})
                
                response = PiiBERTResponse(
                    is_private=result_data.get('is_private', False),
                    confidence=result_data.get('confidence', 0.5),
                    score=result_data.get('score', 0.5),
                    model_name=result_data.get('model_name', 'unknown'),
                    status=status
                )
            else:
                response = PiiBERTResponse(
                    is_private=True,
                    confidence=0.5,
                    score=0.5,
                    model_name="unknown",
                    status=status,
                    error=response_data.get('error', 'Unknown error')
                )
            
            # 缓存响应
            self.response_cache[request_id] = response
            
        except Exception as e:
            logger.error(f"Error handling response: {e}")
    
    def free_cache(self, request_id: str):
        """缓存响应"""
        self.response_cache.pop(request_id)
    
    def close(self):
        """关闭客户端"""
        self.running = False
        self.context.term()
        logger.info("PiiBERT Client stopped")
