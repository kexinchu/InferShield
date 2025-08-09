"""
Privacy Detector based on Trie Tree and Regex patterns
支持基于Trie Tree的快速敏感词检测和正则表达式匹配
"""
import re
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class PrivacyPattern:
    """隐私模式定义"""
    name: str
    pattern: str
    pattern_type: str
    severity: str = 'high'
    description: str = ""
    custom_handler: Optional[Callable] = None

@dataclass
class DetectionResult:
    """检测结果"""
    is_private: bool
    detected_patterns: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    processing_time: float = 0.0

class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.pattern_info: Optional[Dict] = None

class PrivacyDetector:
    """
    基于Trie Tree和正则表达式的隐私检测器

    特性:
    1. 使用Trie Tree进行快速字符串匹配
    2. 支持正则表达式模式匹配
    3. 可插拔的规则系统
    4. 支持自定义处理器
    5. 批量检测优化
    """
    def __init__(self, config_file: Optional[str] = "./privacy_patterns_config.json"):
        self.trie_root = TrieNode()
        self.regex_patterns: List[PrivacyPattern] = []
        self.trie_patterns: List[PrivacyPattern] = []
        self.custom_handlers: Dict[str, Callable] = {}

        # 性能统计
        self.stats = {
            'total_checks': 0,
            'total_matches': 0,
            'avg_processing_time': 0.0
        }

        # 如果提供了配置文件，加载自定义规则
        if config_file:
            self.load_config(config_file)

    def add_pattern(self, pattern: PrivacyPattern):
        """添加隐私模式"""
        if pattern.pattern_type == "trie":
            self._add_trie_pattern(pattern)
        elif pattern.pattern_type == "regex":
            self._add_regex_pattern(pattern)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern.pattern_type}")

        # 注册自定义处理器
        if pattern.custom_handler:
            self.custom_handlers[pattern.name] = pattern.custom_handler

    def _add_trie_pattern(self, pattern: PrivacyPattern):
        """添加Trie模式"""
        # 对于trie模式，pattern字段包含逗号分隔的词汇
        words = [word.strip() for word in pattern.pattern.split(',')]

        for word in words:
            if word:
                self._insert_to_trie(word.lower(), pattern)

        self.trie_patterns.append(pattern)

    def _add_regex_pattern(self, pattern: PrivacyPattern):
        """添加正则表达式模式"""
        try:
            compiled_regex = re.compile(pattern.pattern, re.IGNORECASE)
            pattern.compiled_regex = compiled_regex
            self.regex_patterns.append(pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern.pattern}': {e}")

    def _insert_to_trie(self, word: str, pattern_info: PrivacyPattern):
        """向Trie树插入词汇"""
        node = self.trie_root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True
        node.pattern_info = {
            'name': pattern_info.name,
            'severity': pattern_info.severity,
            'description': pattern_info.description
        }

    def detect_privacy(self, text: str) -> DetectionResult:
        """
        检测文本中的隐私信息
        Args:
            text: 待检测的文本
        Returns:
            DetectionResult: 检测结果
        """
        start_time = time.time()

        detected_patterns = []

        # 1. Trie树匹配
        trie_matches = self._check_trie_patterns(text)
        detected_patterns.extend(trie_matches)

        # 2. 正则表达式匹配
        regex_matches = self._check_regex_patterns(text)
        detected_patterns.extend(regex_matches)

        # 3. 自定义处理器
        custom_matches = self._check_custom_handlers(text)
        detected_patterns.extend(custom_matches)

        # 计算置信度和隐私状态
        is_private = len(detected_patterns) > 0
        confidence = self._calculate_confidence(detected_patterns)

        processing_time = time.time() - start_time

        # 更新统计信息
        self._update_stats(is_private, processing_time)

        return DetectionResult(
            is_private=is_private,
            detected_patterns=detected_patterns,
            confidence=confidence,
            processing_time=processing_time
        )

    def _check_trie_patterns(self, text: str) -> List[Dict]:
        """使用Trie树检查模式"""
        matches = []
        text_lower = text.lower()

        for i in range(len(text_lower)):
            node = self.trie_root
            j = i

            while j < len(text_lower) and text_lower[j] in node.children:
                node = node.children[text_lower[j]]
                if node.is_end and node.pattern_info:
                    matches.append({
                        'type': 'trie',
                        'pattern_name': node.pattern_info['name'],
                        'severity': node.pattern_info['severity'],
                        'description': node.pattern_info['description'],
                        'matched_text': text[i:j+1],
                        'start_pos': i,
                        'end_pos': j+1
                    })
                j += 1

        return matches

    def _check_regex_patterns(self, text: str) -> List[Dict]:
        """使用正则表达式检查模式"""
        matches = []

        for pattern in self.regex_patterns:
            if hasattr(pattern, 'compiled_regex'):
                for match in pattern.compiled_regex.finditer(text):
                    matches.append({
                        'type': 'regex',
                        'pattern_name': pattern.name,
                        'severity': pattern.severity,
                        'description': pattern.description,
                        'matched_text': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })

        return matches

    def _check_custom_handlers(self, text: str) -> List[Dict]:
        """检查自定义处理器"""
        matches = []

        for pattern_name, handler in self.custom_handlers.items():
            try:
                result = handler(text)
                if result:
                    if isinstance(result, dict):
                        matches.append(result)
                    elif isinstance(result, list):
                        matches.extend(result)
            except Exception as e:
                logger.error(f"Error in custom handler '{pattern_name}': {e}")

        return matches

    def _calculate_confidence(self, detected_patterns: List[Dict]) -> float:
        """计算检测置信度"""
        if not detected_patterns:
            return 0.0

        # 基于严重程度和匹配数量计算置信度
        severity_weights = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }

        total_weight = 0.0
        for pattern in detected_patterns:
            severity = pattern.get('severity', 'medium')
            weight = severity_weights.get(severity, 0.5)
            total_weight += weight

        # 归一化到0-1范围
        confidence = min(total_weight / len(detected_patterns), 1.0)
        return confidence

    def _update_stats(self, is_private: bool, processing_time: float):
        """更新统计信息"""
        self.stats['total_checks'] += 1
        if is_private:
            self.stats['total_matches'] += 1

        # 更新平均处理时间
        total_time = self.stats['avg_processing_time'] * (self.stats['total_checks'] - 1)
        self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['total_checks']

    def add_custom_handler(self, name: str, handler: Callable):
        """添加自定义处理器"""
        self.custom_handlers[name] = handler

    def load_config(self, config_file: str):
        """从配置文件加载规则"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 加载Trie模式
            trie_patterns = config.get('trie_patterns', [])
            for pattern_data in trie_patterns:
                pattern = PrivacyPattern(**pattern_data)
                # 目前这里只有一些关键词，may导致误判
                # self.add_pattern(pattern)

            # 加载正则模式
            regex_patterns = config.get('regex_patterns', [])
            for pattern_data in regex_patterns:
                pattern = PrivacyPattern(**pattern_data)
                self.add_pattern(pattern)

        except Exception as e:
            logger.error(f"Error loading config file '{config_file}': {e}")

    def save_config(self, config_file: str):
        """保存配置到文件"""
        config = {
            'trie_patterns': [
                {
                    'name': p.name,
                    'pattern': p.pattern,
                    'pattern_type': p.pattern_type,
                    'severity': p.severity,
                    'description': p.description
                }
                for p in self.trie_patterns
            ],
            'regex_patterns': [
                {
                    'name': p.name,
                    'pattern': p.pattern,
                    'pattern_type': p.pattern_type,
                    'severity': p.severity,
                    'description': p.description
                }
                for p in self.regex_patterns
            ]
        }

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config file '{config_file}': {e}")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    def clear_stats(self):
        """清除统计信息"""
        self.stats = {
            'total_checks': 0,
            'total_matches': 0,
            'avg_processing_time': 0.0
        }

    def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        """批量检测多个文本"""
        results = []
        for text in texts:
            result = self.detect_privacy(text)
            results.append(result)
        return results


# 示例自定义处理器
def detect_internal_documents(text: str) -> Optional[Dict]:
    """检测内部文档标识"""
    internal_keywords = ['内部', '机密', '保密', '绝密', 'confidential', 'secret', 'internal']

    for keyword in internal_keywords:
        if keyword.lower() in text.lower():
            return {
                'type': 'custom',
                'pattern_name': 'internal_documents',
                'severity': 'high',
                'description': '内部文档标识',
                'matched_text': keyword,
                'start_pos': text.lower().find(keyword.lower()),
                'end_pos': text.lower().find(keyword.lower()) + len(keyword)
            }
    return None


def test_acc(detector):
    # 测试文本
    from utils import load_jsonl_dataset
    import numpy as np
    for file_name in [
        # "/root/code/sglang-security/results/english_pii_43k-new_prompts.txt",
        # "/root/code/sglang-security/results/french_pii_62k-new_prompts.txt",
        # "/root/code/sglang-security/results/german_pii_52k-new_prompts.txt",
        # "/root/code/sglang-security/results/italian_pii_50k-new_prompts.txt"
        "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl"
    ]:
        texts, labels = load_jsonl_dataset(file_name, sample_n=40000)
        # 检查字段名
        print(len(texts))

        # 批量检测
        batch_size = 16
        preds = []
        for i in range(0, len(texts), batch_size):
            test_texts = texts[i:i+batch_size]
            results = detector.batch_detect(test_texts)

            for i, (text, result) in enumerate(zip(test_texts, results)):
                preds.append(1 if result.is_private else 0)

        # 显示统计信息
        stats = detector.get_stats()
        preds = np.array(preds)
        labels_np = np.array(labels)
        acc = (preds == labels_np).mean()
        print(f"Accuracy for Custom Detctor: {acc:.4f}")

def test_perf(detector, is_length_):
    import string
    import random
    def shuffle_and_assign_requests(num_request, length):
        """将requests打乱后，均匀分配到每个thread的queue"""
        queue = []
        for _ in range(num_request):
            # 生成指定长度的随机prompt - 使用字母、数字和常见标点符号
            chars = string.ascii_letters + string.digits + " .,!?;:"
            req = ''.join(random.choice(chars) for _ in range(length))
            queue.append(req)
        return queue
    # 简单的提示列表，可根据实际情况替换
    SAMPLE_N = 2000
    if is_length_:
        for length in [512, 1024, 2048, 4096, 8192, 16384, 32758, 65536]:
            texts = shuffle_and_assign_requests(SAMPLE_N, length)
            # 检查字段名
            print(f"length: {length}")

            # 批量检测
            latencies = []
            for i in range(0, len(texts), 1):
                test_texts = texts[i]
                start = time.perf_counter()
                results = detector.batch_detect([test_texts])
                latencies.append((time.perf_counter() - start) * 1000)

            # 打印
            print(f"layer-1  Acc: Not Care")
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                sorted_latencies = sorted(latencies)
                p50_idx = int(0.5 * len(sorted_latencies))
                p95_idx = int(0.95 * len(sorted_latencies))
                p99_idx = int(0.99 * len(sorted_latencies))
                p50_latency = sorted_latencies[p50_idx]
                p95_latency = sorted_latencies[p95_idx]
                p99_latency = sorted_latencies[p99_idx]
                print(f"layer-1 Average Latency: {avg_latency:.2f} ms")
                print(f"layer-1 50th Percentile Latency: {p50_latency:.2f} ms")
                print(f"layer-1 95th Percentile Latency: {p95_latency:.2f} ms")
                print(f"layer-1 99th Percentile Latency: {p99_latency:.2f} ms")
    else:
        from utils import load_jsonl_dataset
        for file_name in [
            "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
            "/dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl",
            "/dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl",
            "/dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl"
        ]:
            texts, labels = load_jsonl_dataset(file_name, sample_n=SAMPLE_N)
            # 检查字段名

            # 批量检测
            latencies = []
            for i in range(0, len(texts), 1):
                test_texts = texts[i]
                start = time.perf_counter()
                results = detector.batch_detect([test_texts])
                latencies.append((time.perf_counter() - start) * 1000)

            # 打印
            print(f"layer-1  Acc: Not Care")
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                sorted_latencies = sorted(latencies)
                p50_idx = int(0.5 * len(sorted_latencies))
                p95_idx = int(0.95 * len(sorted_latencies))
                p99_idx = int(0.99 * len(sorted_latencies))
                p50_latency = sorted_latencies[p50_idx]
                p95_latency = sorted_latencies[p95_idx]
                p99_latency = sorted_latencies[p99_idx]
                print(f"layer-1 Average Latency: {avg_latency:.2f} ms")
                print(f"layer-1 50th Percentile Latency: {p50_latency:.2f} ms")
                print(f"layer-1 95th Percentile Latency: {p95_latency:.2f} ms")
                print(f"layer-1 99th Percentile Latency: {p99_latency:.2f} ms")

# 获取未检测的数据集合
def get_after_level_1(detector):
    # 测试文本
    from utils import load_jsonl_dataset
    import numpy as np
    for file_name in [
        "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl"
    ]:
        texts, labels = load_jsonl_dataset(file_name, sample_n=5000)
        batch_size = 16

        # 写结果
        ori_name = file_name.split("/")[-1].split(".")[0]
        fname = f"/root/code/sglang-security/results/{ori_name}-after_level_1.jsonl"
        with open(fname, "w") as f:
            # 批量检测
            for i in range(0, len(texts), batch_size):
                test_texts = texts[i:i+batch_size]
                results = detector.batch_detect(test_texts)

                for i, (text, result) in enumerate(zip(test_texts, results)):
                    if result.is_private:
                        continue

                    tmp = {
                        "prompt": text,
                        "label": labels[i],
                    }
                    f.write(json.dumps(tmp, ensure_ascii=False) + "\n")

# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = PrivacyDetector()

    # 添加自定义处理器
    detector.add_custom_handler('internal_documents', detect_internal_documents)

    test_perf(detector, False)
    # test_acc(detector)
    # get_after_level_1(detector)
