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
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput, SamplingParams

# 导入隐私检测器
from .privacy_detector_custom import PrivacyDetector
# 导入PiiBERT客户端
from .pii_bert_client import PiiBERTClient

BATCH_SIZE = 16
LOW_QUALITY_THRESHOLD = 0.3
HIGH_QUALITY_THRESHOLD = 0.7

@dataclass
class PrivateNodeTask:
    node: TreeNode
    task_type: str  # 'check_private', 'update_private', 'cleanup_private'
    context: str    # 上下文信息
    prompt: str     # 提示词
    request_id: str
    timestamp: float = time.time()

# @dataclass
# class BatchTasks:
#     tasks: List[PrivateNodeTask]
#     timestamp: float

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
        
        # 初始化PiiBERT客户端
        self.pii_bert_client = PiiBERTClient(server_args, port_args)
        self.pii_bert_available = True
        
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
            prompt = task_data.get('prompt', '')
            
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
            

    def _process_second_level_tasks(self):
        """Process second level detection: PiiBERT小模型检测"""
        if len(self.second_level_task_queue) == 0:
            time.sleep(5)
            return
        
        # send req to pii model
        wait_for_answer = []
        while len(self.second_level_task_queue) > 0:
            task_data = self.second_level_task_queue.pop(0)
            # 检查PiiBERT客户端是否可用
            if not self.pii_bert_available:
                # 如果PiiBERT不可用，直接进入第三级检测
                self.third_level_task_queue.append(task_data)
                continue
            task_data.request_id = self.pii_bert_client.detect_privacy(task_data.prompt)
            wait_for_answer.append(task_data)
        
        # 获取结果
        while(len(wait_for_answer) > 0):
            task_data = wait_for_answer.pop(0)
            # 使用PiiBERT客户端进行检测
            res = self.pii_bert_client.detect_privacy_sync(task_data.request_id)
            
            if LOW_QUALITY_THRESHOLD < res.confidence < HIGH_QUALITY_THRESHOLD:
                self.third_level_task_queue.append(task_data)
            else:
                self.result_queue.append({
                    'status': 'success',
                    'privacy': 'private' if res.is_private else 'public',
                    'confidence': res.confidence,
                    'detected_patterns': {
                        'name': 'PiiBERT_detection',
                        'pattern_type': 'ml_model',
                        'severity': 'high' if res.confidence > 0.9 else 'medium',
                    },
                    'node_id': task_data.node.node_id,
                    'detection_level': 'second_level',
                    'model_name': res.model_name,
                })
            # 清理memory 空间
            self.pii_bert_client.free_cache(task_data.request_id)

    def _process_third_level_tasks(self):
        """Process third level detection: LLM大模型检测 => 请求合并到普通结果一起"""
        if len(self.third_level_task_queue) == 0:
            time.sleep(5)
            return
        
        # 获取任务队列中的任务
        batch_tasks = []
        while len(self.third_level_task_queue) > 0 and len(batch_tasks) < BATCH_SIZE:
            task_data = self.third_level_task_queue.pop(0)
            batch_tasks.append(task_data)
        
        if not batch_tasks:
            return
            
        try:
            # 通过ZMQ发送给SGLang自身的scheduler进行处理
            # 创建LLM检测请求，与常规request一起排队处理
            llm_detection_results = self._send_llm_detection_requests(batch_tasks)
            
            # 处理检测结果
            for i, result in enumerate(llm_detection_results):
                task_data = batch_tasks[i]
                node_id = task_data['node_id']
                
                if result['status'] == 'success':
                    # 解析LLM检测结果
                    llm_response = result.get('llm_response', '')
                    is_private = self._parse_llm_privacy_result(llm_response)
                    confidence = result.get('confidence', 0.8)  # LLM检测的置信度较高
                    
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': 'private' if is_private else 'public',
                        'confidence': confidence,
                        'detected_patterns': {
                            'name': 'llm_detection',
                            'pattern_type': 'llm_model',
                            'severity': 'high',
                            'description': f'LLM detected as {"private" if is_private else "public"} with response: {llm_response[:100]}...'
                        },
                        'node_id': node_id,
                        'detection_level': 'third_level',
                        'model_name': 'llm_detection',
                        'llm_response': llm_response
                    })
                else:
                    # 检测失败，标记为需要进一步处理
                    self.result_queue.append({
                        'status': 'error',
                        'error': f'LLM detection failed: {result.get("error", "Unknown error")}',
                        'node_id': node_id,
                        'detection_level': 'third_level'
                    })
                    
        except Exception as e:
            print(f"Error in third level processing: {e}")
            # 将失败的任务重新放回队列
            for task_data in batch_tasks:
                self.third_level_task_queue.append(task_data)
    
    def _send_llm_detection_requests(self, batch_tasks):
        """通过ZMQ发送LLM检测请求给SGLang scheduler"""
        results = []
        
        # 初始化ZMQ连接（如果还没有初始化）
        if not hasattr(self, 'llm_detection_socket'):
            self.llm_detection_socket = get_zmq_socket(
                self.context, zmq.PUSH, self.port_args.scheduler_input_ipc_name, False
            )
            self.llm_result_socket = get_zmq_socket(
                self.context, zmq.PULL, self.port_args.tokenizer_ipc_name, False
            )
        
        # 为每个任务创建LLM检测请求
        for task_data in batch_tasks:
            node_id = task_data['node_id']
            context = task_data.get('context', '')
            prompt = task_data.get('prompt', '')
            
            # 构建LLM检测的prompt
            detection_prompt = self._build_llm_detection_prompt(context, prompt)
            
            # 使用简单的字符编码作为input_ids
            # 这里我们使用简单的字符到数字的映射，实际应用中应该使用真实的tokenizer
            input_ids = []
            for char in detection_prompt[:1000]:  # 限制长度
                char_code = ord(char)
                if char_code < 65536:  # 限制在Unicode BMP范围内
                    input_ids.append(char_code)
                else:
                    input_ids.append(ord(' '))  # 替换为空格
            
            # 创建采样参数
            sampling_params = SamplingParams(
                temperature=0.1,  # 低温度以获得确定性结果
                max_new_tokens=100,  # 限制输出长度
                top_p=0.9,
                user_id=node_id  # 使用node_id作为user_id来标识请求
            )
            
            # 创建请求
            llm_request = TokenizedGenerateReqInput(
                rid=f"PRIVACY_DETECTION_LLM_{node_id}_{int(time.time())}",
                input_text=detection_prompt,
                input_ids=input_ids,
                mm_inputs={},  # 空的多模态输入
                sampling_params=sampling_params,
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=[],
                stream=False,
                lora_path=None,
                input_embeds=None,
                session_params=None,
                custom_logit_processor=None,
                return_hidden_states=False,
                bootstrap_host=None,
                bootstrap_port=None,
                bootstrap_room=None
            )
            
            # 发送请求
            try:
                self.llm_detection_socket.send_pyobj(llm_request)
                
                # 等待结果（设置超时）
                # 设置非阻塞接收，带超时
                self.llm_result_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30秒超时
                response = self.llm_result_socket.recv_pyobj()
                
                # 解析响应
                if hasattr(response, 'text') and response.text:
                    llm_response = response.text[0] if isinstance(response.text, list) else response.text
                    results.append({
                        'status': 'success',
                        'llm_response': llm_response,
                        'confidence': 0.85  # LLM检测的置信度
                    })
                else:
                    results.append({
                        'status': 'error',
                        'error': 'Empty response from LLM'
                    })
                    
            except zmq.ZMQError as e:
                results.append({
                    'status': 'error',
                    'error': f'ZMQ timeout or error: {str(e)}'
                })
            except Exception as e:
                results.append({
                    'status': 'error',
                    'error': f'Unexpected error: {str(e)}'
                })
        
        return results
    
    def _build_llm_detection_prompt(self, context, prompt):
        """构建LLM隐私检测的prompt"""
        detection_prompt = f"""请分析以下文本是否包含隐私信息。请只回答"是"或"否"。
上下文: {context}
文本: {prompt}
请判断这个文本是否包含隐私信息（如个人信息、敏感数据、密码等）。只回答"是"或"否"。
回答:"""
        return detection_prompt
    
    def _parse_llm_privacy_result(self, llm_response):
        """解析LLM的隐私检测结果"""
        if not llm_response:
            return False
        
        # 清理响应文本
        response_clean = llm_response.strip().lower()
        
        # 检查是否包含"是"、"private"、"隐私"等关键词
        private_keywords = ['是', 'yes', 'private', '隐私', '敏感', 'sensitive']
        public_keywords = ['否', 'no', 'public', '公开', '非隐私']
        
        for keyword in private_keywords:
            if keyword in response_clean:
                return True
        
        for keyword in public_keywords:
            if keyword in response_clean:
                return False
        
        # 如果没有明确的关键词，根据响应长度和内容判断
        # 通常"是"的回答比较简短
        if len(response_clean) <= 3:
            return True  # 短回答通常是"是"
        
        return False  # 默认认为是公开的

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
        
        # 关闭PiiBERT客户端
        if hasattr(self, 'pii_bert_client') and self.pii_bert_available:
            self.pii_bert_client.close()
        
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