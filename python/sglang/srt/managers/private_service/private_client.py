"""
private client
add by kexinchu
"""
from dataclasses import dataclass
import time
import zmq
import queue
import threading
import traceback
from typing import List
from dataclasses import dataclass

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.mem_cache.tree_node import TreeNode
from sglang.srt.utils import get_zmq_socket
from .private_service import PrivateJudgeService

from .global_task_queue import tier_1_task_queue, result_final_queue

@dataclass
class PrivateNodeTask:
    node: TreeNode
    task_type: str  # 'check_private', 'update_private', 'cleanup_private'
    context: str    # 上下文信息
    prompt: str     # 提示词
    request_id: str
    timestamp: float = time.time()
    privacy: bool = True

class PrivateJudgeClient:
    def __init__(self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # init private service
        self.server_args = server_args
        self.port_args = port_args
        self.batch_size = 1
        self.batch_timeout = 0.1 # seconds

        # Initialize ZMQ context and sockets
        # self.context = zmq.Context(2)
        # self.send_to_server = get_zmq_socket(
        #     self.context, zmq.PUSH, port_args.private_judge_to_server, False
        # )
        # self.recv_from_server = get_zmq_socket(
        #     self.context, zmq.PULL, port_args.private_judge_to_client, False
        # )

        # Flag to control thread execution
        self.running = True
        self.private_server = PrivateJudgeService(server_args, port_args)

        # Initialize task queue and processing thread
        self.task_queue = queue.Queue()
        # self.processing_thread = threading.Thread(
        #     target=self._process_tasks,
        #     daemon=True
        # )
        # self.processing_thread.start()

        # response
        self.response_thread = threading.Thread(
            target=self._response_task,
            daemon=True
        )
        self.response_thread.start()

    def update_privacy(self, node, context: str, prompt: str) -> None:
        """Update node privacy status asynchronously"""
        # check the parents's node: privacy status
        if node.parent is not None and \
           not node.parent.need_check_privacy and \
            node.parent.private:
            node.need_check_privacy = False
            return

        if len(prompt.strip().split("\n")) > 1:
            prompt_ = prompt.strip().split("\n")[-2]
        else:
            prompt_ = prompt.strip().split("\n")[-1]
        if "Assistant" in prompt_ or "im_end" in prompt_:
            prompt_ = "hello, you are an simple assistant"

        # Create task and add to queue
        task = PrivateNodeTask(
            node=node,
            task_type='update_private',
            context=prompt_,
            prompt=prompt_,
            request_id="",
            timestamp=time.time(),
            privacy=True,
        )
        # self.task_queue.put(task)
        tier_1_task_queue.put(task)

    def _process_tasks(self):
        """Background thread to process tasks from queue"""
        current_batch = []
        last_batch_time = time.time()
        try:
            while self.running:
                try:
                    task = self.task_queue.get(timeout=self.batch_timeout)
                    # current_batch.append(task)
                except queue.Empty:
                    time.sleep(0.1) # wait for new task
                    continue

                current_time = time.time()
                # Process batch if it's full or timeout reached
                if (len(current_batch) >= self.batch_size or
                    (current_batch and current_time - last_batch_time \
                        >= self.batch_timeout)):
                    # self._send_batch_tasks(current_batch)
                    tier_1_task_queue.put()
                    current_batch = []
                    last_batch_time = current_time
        except Exception as outer:
            print("[ERROR] processing thread crashed:")
            traceback.print_exc()

    def _response_task(self) -> None:
        try:
            while self.running:
                try:
                    task = result_final_queue.get(timeout=self.batch_timeout)
                except queue.Empty:
                    time.sleep(0.1) # wait for new task
                    continue

                if task.prompt == "hello, you are an simple assistant":
                    task.privacy = False

                # print(f"change states: ori: {task.node.private}; new {task.privacy}")
                if not task.privacy:
                    task.node.private = False
                task.node.need_check_privacy = False

        except Exception as outer:
            print("[ERROR] processing thread crashed:")
            traceback.print_exc()

    # def _send_batch_tasks(self, tasks: List[PrivateNodeTask]) -> None:
    #     """Send a batch of tasks to the server"""
    #     if not tasks:
    #         return
    #     try:
    #         # Prepare batch message
    #         batch_message = {
    #             'batch': [{
    #                 'node_id': task.node.id,
    #                 'task_type': task.task_type,
    #                 'context': task.context,
    #                 'prompt': task.prompt,
    #                 'timestamp': task.timestamp
    #             } for task in tasks]
    #         }

    #         # Send batch
    #         self.send_to_server.send_json(batch_message)

    #         # Get responses for all tasks in batch
    #         for _ in range(len(tasks)):
    #             response = self.recv_from_server.recv_json()

    #             if response['status'] == 'error':
    #                 print(f"Error in batch processing: {response['error']}")
    #                 continue

    #             # Update node status based on response
    #             task = tasks[response['task_index']]
    #             if response['privacy'] == 'public':
    #                 task.node.private = False
    #             task.node.need_check_privacy = False

    #     except Exception as e:
    #         print(f"Error sending batch tasks: {e}")
    #         raise

    def close(self):
        """Close the client connection and stop processing thread"""
        self.running = False
        # if hasattr(self, 'processing_thread'):
        #     self.processing_thread.join()
        # if hasattr(self, 'send_to_server'):
        #     self.send_to_server.close()
        # if hasattr(self, 'recv_from_server'):
        #     self.recv_from_server.close()
        if hasattr(self, 'context'):
            self.context.term()
