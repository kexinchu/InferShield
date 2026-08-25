"""
private client
SafeKV: bridges KV cache manager and privacy detection service.
On detection result, promotes non-private nodes to shareable with access budget.
"""
from dataclasses import dataclass
import time
import queue
import threading
import traceback
from typing import List
from dataclasses import dataclass

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.mem_cache.tree_node import TreeNode
from sglang.srt.mem_cache.safekv_policy import Visibility
from .private_service import PrivateJudgeService

from .global_task_queue import tier_1_task_queue, result_final_queue

@dataclass
class PrivateNodeTask:
    node: TreeNode
    task_type: str  # 'check_private', 'update_private', 'cleanup_private'
    context: str
    prompt: str
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

        # SafeKV safeguard: access budget B for promotion
        self.access_budget_B = getattr(server_args, 'safekv_access_budget', 10)

        # Flag to control thread execution
        self.running = True
        self.private_server = PrivateJudgeService(server_args, port_args)

        # Initialize task queue
        self.task_queue = queue.Queue()

        # Response thread
        self.response_thread = threading.Thread(
            target=self._response_task,
            daemon=True
        )
        self.response_thread.start()

    def update_privacy(self, node, context: str, prompt: str) -> None:
        """Update node privacy status asynchronously"""
        # Parent inheritance: if parent is confirmed private, child inherits
        if node.parent is not None and \
           not node.parent.need_check_privacy and \
            node.parent.private_tag == 1:
            node.need_check_privacy = False
            return

        if not prompt or not isinstance(prompt, str):
            # Token-only requests have no text prompt; skip privacy detection.
            # Do not treat this as "non-private" — the node stays Private.
            node.need_check_privacy = False
            return

        if len(prompt.strip().split("\n")) > 1:
            prompt_ = prompt.strip().split("\n")[-2]
        else:
            prompt_ = prompt.strip().split("\n")[-1]

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
        tier_1_task_queue.put(task)

    def _response_task(self) -> None:
        """Process detection results and update node privacy status.
        Non-private nodes are promoted to shareable with access budget B.
        Permanently private nodes are never re-promoted.
        """
        try:
            while self.running:
                try:
                    task = result_final_queue.get(timeout=self.batch_timeout)
                except queue.Empty:
                    time.sleep(0.1)
                    continue

                if not task.privacy:
                    # Detection says non-private: promote for Balanced reuse.
                    # Visibility (not private_tag) is authoritative for cross-tenant hits.
                    if not task.node.permanently_private:
                        task.node.private_tag = 0
                        task.node.access_budget = self.access_budget_B
                        if getattr(task.node, "visibility", None) == Visibility.PRIVATE.value:
                            task.node.visibility = Visibility.BUDGETED_SHARED.value
                    # else: permanently private after budget demotion, do not promote
                else:
                    task.node.private_tag = 1  # confirmed private
                task.node.need_check_privacy = False

        except Exception as outer:
            print("[ERROR] processing thread crashed:")
            traceback.print_exc()

    def close(self):
        """Close the client connection and stop processing thread"""
        self.running = False
        if hasattr(self, 'context'):
            self.context.term()
