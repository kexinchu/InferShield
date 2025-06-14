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
        self.running = True
        self.processing_thread.start()

    def _process_requests(self):
        """Process incoming requests from clients"""
        while self.running:
            try:
                # Receive batch message
                message = self.recv_from_client.recv_json()
                
                if 'batch' not in message:
                    print("Invalid message format: missing 'batch' field")
                    continue
                
                # Process each task in the batch
                responses = []
                for i, task_data in enumerate(message['batch']):
                    try:
                        response = self._handle_task(task_data)
                        response['node_id'] = task_data['node_id']
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing task {i}: {e}")
                        responses.append({
                            'status': 'error',
                            'error': str(e),
                            'node_id': task_data['node_id']
                        })
                
                # Send responses back to client
                for response in responses:
                    self.send_to_client.send_json(response)
                    
            except Exception as e:
                print(f"Error in request processing: {e}")
                time.sleep(0.1)  # Prevent tight loop on error

    def _handle_task(self, task_data: Dict) -> Dict:
        """Handle a single task"""
        task_type = task_data.get('task_type')
        
        if task_type == 'update_private':
            return self._handle_update_private(task_data)
        else:
            return {
                'status': 'error',
                'error': f'Unknown task type: {task_type}'
            }

    def _handle_update_private(self, task_data: Dict) -> Dict:
        """Handle update_private task"""
        try:
            # Extract task data
            node_id = task_data['node_id']
            context = task_data['context']
            prompt = task_data['prompt']
            
            # TODO: Implement actual privacy check logic here
            # For now, we'll use a simple heuristic based on context length
            is_private = len(context) > 10  # Example condition
            
            return {
                'status': 'success',
                'privacy': 'private' if is_private else 'public',
                'node_id': node_id
            }
            
        except KeyError as e:
            return {
                'status': 'error',
                'error': f'Missing required field: {str(e)}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def close(self):
        """Close the service and cleanup resources"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        if hasattr(self, 'recv_from_client'):
            self.recv_from_client.close()
        if hasattr(self, 'send_to_client'):
            self.send_to_client.close()
        if hasattr(self, 'context'):
            self.context.term()

if __name__ == "__main__":
    # Example usage
    service = PrivateJudgeService()
    try:
        # Keep the service running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down service...")
    finally:
        service.close() 