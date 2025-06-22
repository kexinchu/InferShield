"""
async private judge service
在创建KV-Cache时, 触发本任务, 异步处理node的private状态 
add by kexinchu
"""
import time
import zmq
import json
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass

from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket

# 导入隐私检测器
from .privacy_detector_custom import PrivacyDetector
# 导入DistilBERT客户端
from .distillbert_client import DistilBERTClient

BATCH_SIZE = 16
LOW_QUALITY_THRESHOLD = 0.3
HIGH_QUALITY_THRESHOLD = 0.7

@dataclass
class PrivateNodeTask:
    node: TreeNode
    task_type: str  # 'check_private', 'update_private', 'cleanup_private'
    context: str    # 上下文信息
    prompt: str     # 提示词
    timestamp: float = time.time()

@dataclass
class BatchTasks:
    tasks: List[PrivateNodeTask]
    timestamp: float

class PrivateJudgeService:
    def __init__(self, 
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # init private service
        self.server_args = server_args
        self.port_args = port_args
        
        # 初始化隐私检测器
        self.privacy_detector = PrivacyDetector("./privacy_patterns_config.json")
        
        # 初始化DistilBERT客户端
        try:
            self.distillbert_client = DistilBERTClient(server_args, port_args)
            self.distillbert_available = True
        except Exception as e:
            print(f"Failed to initialize DistilBERT client: {e}")
            self.distillbert_available = False
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.PULL, port_args.private_judge_to_server, False
        )
        self.send_to_client = get_zmq_socket(
            self.context, zmq.PUSH, port_args.private_judge_to_client, False
        )
        
        # Initialize processing thread
        self.processing_thread = threading.Thread(
            target=self._process_requests,
            daemon=True
        )
        self.first_level_thread = threading.Thread(
            target=self._process_first_level_tasks,
            daemon=True
        )
        self.second_level_thread = threading.Thread(
            target=self._process_second_level_tasks,
            daemon=True
        )
        self.third_level_thread = threading.Thread(
            target=self._process_third_level_tasks,
            daemon=True
        )
        self.result_thread = threading.Thread(
            target=self._process_result_tasks,
            daemon=True
        )
        self.running = True

        # three level detection
        self.first_level_task_queue = []
        self.second_level_task_queue = []
        self.third_level_task_queue = []
        self.result_queue = []

        # 启动处理线程
        self.processing_thread.start()
        self.first_level_thread.start()
        self.second_level_thread.start()
        self.third_level_thread.start()
        self.result_thread.start()

    def _process_requests(self):
        """Process incoming requests from clients"""
        while self.running:
            try:
                # Receive batch message
                message = self.recv_from_client.recv_json()
                
                if 'batch' not in message:
                    print("Invalid message format: missing 'batch' field")
                    continue
                
                # get tasks from client
                for i, task_data in enumerate(message['batch']):
                    """Handle a single task"""
                    task_type = task_data.get('task_type')
                    
                    if task_type == 'update_private':
                        self.first_level_task_queue.append(task_data)
                    else:
                        return {
                            'status': 'error',
                            'error': f'Unknown task type: {task_type}'
                        }
                    
            except Exception as e:
                print(f"Error in request processing: {e}")
                time.sleep(5)  # Prevent tight loop on error
    
    def _process_result_tasks(self):
        """Process result tasks"""
        while self.running:
            if len(self.result_queue) == 0:
                time.sleep(5)
            
            # get result queue
            while len(self.result_queue) > 0:
                result = self.result_queue.pop(0)
                self.send_to_client.send_json(result)

    def _process_first_level_tasks(self):
        """Process first level detection: 正则化/trieTree 检测"""
        if len(self.first_level_task_queue) == 0:
            time.sleep(5)
            return
        
        # get task from queue
        while len(self.first_level_task_queue) > 0:
            task_data = self.first_level_task_queue.pop(0)
            # Extract task data
            node_id = task_data['node_id']
            context = task_data.get('context', '')
            prompt = task_data.get('prompt', '')
            
            try:
                # 使用隐私检测器进行检测
                prompt_result = self.privacy_detector.detect_privacy(prompt)
                
                # 如果未检测到privacy，并不一定未public，需要进行第二级检测
                if not prompt_result.is_private:
                    self.second_level_task_queue.append(task_data)
                else:
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': 'private',
                        'confidence': prompt_result.confidence,
                        'detected_patterns': prompt_result.detected_patterns,
                        'node_id': node_id,
                        'detection_level': 'first_level'
                    }) 
            
            except KeyError as e:
                # 检测失败，进行第二级别检测
                self.second_level_task_queue.append(task_data)

    def _process_second_level_tasks(self):
        """Process second level detection: DistilBERT小模型检测"""
        if len(self.first_level_task_queue) == 0:
            time.sleep(5)
            return
        
        # get task from queue # 处理成batch任务
        batch_tasks = []
        while len(self.first_level_task_queue) > 0 or len(batch_tasks) < BATCH_SIZE:
            task_data = self.first_level_task_queue.pop(0)
            # 检查DistilBERT客户端是否可用
            if not self.distillbert_available:
                # 如果DistilBERT不可用，直接进入第三级检测
                self.third_level_task_queue.append(task_data)
            batch_tasks.append(task_data)
            
        try:
            # 使用DistilBERT客户端进行检测
            response = self.distillbert_client.detect_privacy_sync(batch_tasks, timeout=10.0)
            for i, res in enumerate(response):
                # 如果未检测到privacy，进行第三级检测
                if LOW_QUALITY_THRESHOLD < res.confidence < HIGH_QUALITY_THRESHOLD:
                    self.third_level_task_queue.append(batch_tasks[i])
                else:
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': 'private' if res.is_private else 'public',
                        'confidence': res.confidence,
                        'detected_patterns': {
                            'name': 'distilbert_detection',
                            'pattern_type': 'ml_model',
                            'severity': 'high' if response.confidence > 0.8 else 'medium',
                            'description': f'DistilBERT detected as {response.label} with confidence {response.confidence:.3f}'
                        },
                        'node_id': batch_tasks[i]['node_id'],
                        'detection_level': 'second_level',
                        'model_name': res.model_name,
                    })
        
        except Exception as e:
            # 检测失败，进行第三级别检测
            self.third_level_task_queue.append(task_data)

    def _process_third_level_tasks(self):
        """Process third level detection: LLM大模型检测 => 请求合并到普通结果一起"""
        pass

    def add_custom_privacy_pattern(self, pattern_name: str, pattern_type: str, 
                                 pattern: str, severity: str = 'high', 
                                 description: str = ""):
        """添加自定义隐私模式"""
        from .privacy_detector_custom import PrivacyPattern
        privacy_pattern = PrivacyPattern(
            name=pattern_name,
            pattern=pattern,
            pattern_type=pattern_type,
            severity=severity,
            description=description
        )
        self.privacy_detector.add_pattern(privacy_pattern)

    def add_custom_handler(self, name: str, handler):
        """添加自定义处理器"""
        self.privacy_detector.add_custom_handler(name, handler)

    def get_privacy_stats(self) -> Dict:
        """获取隐私检测统计信息"""
        return self.privacy_detector.get_stats()

    def close(self):
        """Close the service and cleanup resources"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        if hasattr(self, 'first_level_thread'):
            self.first_level_thread.join()
        if hasattr(self, 'second_level_thread'):
            self.second_level_thread.join()
        if hasattr(self, 'third_level_thread'):
            self.third_level_thread.join()
        if hasattr(self, 'result_thread'):
            self.result_thread.join()
        
        # 关闭DistilBERT客户端
        if hasattr(self, 'distillbert_client') and self.distillbert_available:
            self.distillbert_client.close()
        
        # 关闭ZMQ连接
        if hasattr(self, 'recv_from_client'):
            self.recv_from_client.close()
        if hasattr(self, 'send_to_client'):
            self.send_to_client.close()
        if hasattr(self, 'context'):
            self.context.term()

if __name__ == "__main__":
    # Example usage
    server_args = ServerArgs()
    port_args = PortArgs()
    service = PrivateJudgeService(server_args, port_args)
    try:
        # Keep the service running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down service...")
    finally:
        service.close() 