"""
async private judge service
Two-tier privacy detection pipeline:
  Tier 1: Rule-based pattern matching (regex/trie)
  Tier 2: Semantic screening with compact language model (Piiranha/Llama-3.2-1B)
When Tier 2 is uncertain, conservatively keep the block as private.
add by kexinchu
"""
import os
import time
import threading
from typing import Dict

from sglang.srt.mem_cache.tree_node import TreeNode
from sglang.srt.server_args import PortArgs, ServerArgs

# Import privacy detectors
from .privacy_detector_custom import PrivacyDetector
from .global_task_queue import tier_1_task_queue, result_final_queue

BATCH_SIZE = 16
LOW_QUALITY_THRESHOLD = 0.3
HIGH_QUALITY_THRESHOLD = 0.7

class PrivateJudgeService:
    def __init__(self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # init private service
        self.server_args = server_args
        self.port_args = port_args

        # Initialize rule-based privacy detector
        config_path = os.path.join(os.path.dirname(__file__), "privacy_patterns_config.json")
        self.privacy_detector = PrivacyDetector(config_path)

        # Initialize PiiBERT client (Tier 2) — lazy import to avoid crash if models missing
        try:
            from .privacy_detector_piiranha_client import PiiBERTClient
            from .privacy_detector_piiranha import PiiPrivacyService
            self.pii_bert_server = PiiPrivacyService(server_args, port_args)
            self.pii_bert_client = PiiBERTClient(server_args, port_args)
            self.pii_bert_available = True
        except Exception as e:
            print(f"[SafeKV] Tier-2 ML detection unavailable: {e}")
            print("[SafeKV] Falling back to Tier-1 (rule-based) detection only")
            self.pii_bert_server = None
            self.pii_bert_client = None
            self.pii_bert_available = False

        # Initialize processing threads (two-tier pipeline)
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
        self.result_thread = threading.Thread(
            target=self._process_result_tasks,
            daemon=True
        )
        self.running = True

        # Two-tier detection queues
        self.first_level_task_queue = []
        self.second_level_task_queue = []
        self.result_queue = []

        # Start processing threads
        self.processing_thread.start()
        self.first_level_thread.start()
        self.second_level_thread.start()
        self.result_thread.start()

    def _process_requests(self):
        """Process incoming requests from clients"""
        while self.running:
            try:
                task = tier_1_task_queue.get(timeout=0.1)
                if task.task_type == 'update_private':
                    self.first_level_task_queue.append(task)
            except Exception as e:
                time.sleep(0.1)

    def _process_result_tasks(self):
        """Process result tasks"""
        while self.running:
            if len(self.result_queue) == 0:
                time.sleep(0.1)

            while len(self.result_queue) > 0:
                result = self.result_queue.pop(0)
                if result['privacy']:
                    result["task"].privacy = True
                else:
                    result["task"].privacy = False
                result_final_queue.put(result["task"])

    def _process_first_level_tasks(self):
        """Tier 1: Rule-based detection (regex/trie pattern matching)"""
        while self.running:
            try:
                task = self.first_level_task_queue.pop(0)
                prompt = task.prompt

                prompt_result = self.privacy_detector.detect_privacy(prompt)

                if not prompt_result.is_private:
                    # Not detected by rules, escalate to Tier 2
                    self.second_level_task_queue.append(task)
                else:
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': True,
                        'confidence': prompt_result.confidence,
                        'detected_patterns': prompt_result.detected_patterns,
                        'task': task,
                        'detection_level': 'first_level'
                    })
            except:
                time.sleep(0.1)

    def _process_second_level_tasks(self):
        """Tier 2: Semantic screening with compact language model (Piiranha)
        When uncertain, conservatively mark as private (no Level 3 escalation).
        """
        wait_for_answer = []
        while self.running:
            try:
                task = self.second_level_task_queue.pop(0)
                if not self.pii_bert_available:
                    # PiiBERT unavailable: conservatively mark as private
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': True,
                        'confidence': 0.5,
                        'task': task,
                        'detection_level': 'second_level_conservative',
                    })
                    continue
                task.request_id = self.pii_bert_client.detect_privacy(task.prompt)
                wait_for_answer.append(task)
            except:
                time.sleep(0.1)

            # Collect results
            while (len(wait_for_answer) > 0):
                task = wait_for_answer.pop(0)
                res = self.pii_bert_client.detect_privacy_sync(task.request_id)
                if LOW_QUALITY_THRESHOLD < res.confidence < HIGH_QUALITY_THRESHOLD:
                    # Uncertain: conservatively mark as private (safeguard handles false negatives)
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': True,
                        'confidence': res.confidence,
                        'task': task,
                        'detection_level': 'second_level_uncertain',
                    })
                else:
                    self.result_queue.append({
                        'status': 'success',
                        'privacy': True if res.is_private else False,
                        'confidence': res.confidence,
                        'detected_patterns': {
                            'name': 'PiiBERT_detection',
                            'pattern_type': 'ml_model',
                            'severity': 'high' if res.confidence > 0.9 else 'medium',
                        },
                        'task': task,
                        'detection_level': 'second_level',
                        'model_name': res.model_name,
                    })
                # Free memory
                self.pii_bert_client.free_cache(task.request_id)

    def add_custom_privacy_pattern(self, pattern_name: str, pattern_type: str,
                                 pattern: str, severity: str = 'high',
                                 description: str = ""):
        """Add custom privacy pattern"""
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
        """Add custom handler"""
        self.privacy_detector.add_custom_handler(name, handler)

    def get_privacy_stats(self) -> Dict:
        """Get privacy detection stats"""
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
        if hasattr(self, 'result_thread'):
            self.result_thread.join()

        # Close PiiBERT client
        if hasattr(self, 'pii_bert_client') and self.pii_bert_available:
            self.pii_bert_client.close()

        if hasattr(self, 'context'):
            self.context.term()

if __name__ == "__main__":
    server_args = ServerArgs()
    port_args = PortArgs()
    service = PrivateJudgeService(server_args, port_args)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down service...")
    finally:
        service.close()
