"""
PrivacyDetector 使用示例
展示如何使用基于Trie Tree的隐私检测器
"""
import os
import sys
from typing import Dict, List, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from privacy_detector import PrivacyDetector, PrivacyPattern, DetectionResult

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建检测器
    detector = PrivacyDetector()
    
    # 测试文本
    test_texts = [
        "我的邮箱是 test@example.com，请保密",
        "手机号是 13812345678",
        "身份证号是 110101199001011234",
        "这是一个内部机密文档",
        "普通文本，没有敏感信息",
        "信用卡号是 1234-5678-9012-3456",
        "IP地址是 192.168.1.1"
    ]
    
    # 批量检测
    results = detector.batch_detect(test_texts)
    
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"文本 {i+1}: {text}")
        print(f"  隐私状态: {'🔒 私有' if result.is_private else '✅ 公开'}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  处理时间: {result.processing_time:.4f}s")
        if result.detected_patterns:
            print(f"  检测到的模式:")
            for pattern in result.detected_patterns:
                severity_icon = {
                    'low': '🟢',
                    'medium': '🟡', 
                    'high': '🟠',
                    'critical': '🔴'
                }.get(pattern['severity'], '⚪')
                print(f"    {severity_icon} {pattern['pattern_name']}: {pattern['matched_text']}")
        print()

def example_custom_patterns():
    """自定义模式示例"""
    print("=== 自定义模式示例 ===")
    
    # 创建检测器
    detector = PrivacyDetector()
    
    # 添加自定义Trie模式
    custom_trie_pattern = PrivacyPattern(
        name="custom_keywords",
        pattern="project_alpha,beta_test,gamma_release,delta_version",
        pattern_type="trie",
        severity="high",
        description="自定义项目关键词"
    )
    detector.add_pattern(custom_trie_pattern)
    
    # 添加自定义正则模式
    custom_regex_pattern = PrivacyPattern(
        name="custom_id",
        pattern=r"\b[A-Z]{2}\d{6}\b",
        pattern_type="regex",
        severity="medium",
        description="自定义ID格式"
    )
    detector.add_pattern(custom_regex_pattern)
    
    # 测试文本
    test_texts = [
        "我们在进行project_alpha的开发",
        "用户ID是 AB123456",
        "beta_test阶段已经完成",
        "这是gamma_release版本"
    ]
    
    results = detector.batch_detect(test_texts)
    
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"文本 {i+1}: {text}")
        print(f"  隐私状态: {'🔒 私有' if result.is_private else '✅ 公开'}")
        if result.detected_patterns:
            print(f"  检测到的模式:")
            for pattern in result.detected_patterns:
                print(f"    - {pattern['pattern_name']}: {pattern['matched_text']}")
        print()

def example_custom_handlers():
    """自定义处理器示例"""
    print("=== 自定义处理器示例 ===")
    
    # 创建检测器
    detector = PrivacyDetector()
    
    # 定义自定义处理器
    def detect_company_secrets(text: str) -> Optional[Dict]:
        """检测公司机密信息"""
        company_keywords = ['季度报告', '财务预测', '并购计划', '战略规划']
        
        for keyword in company_keywords:
            if keyword in text:
                return {
                    'type': 'custom',
                    'pattern_name': 'company_secrets',
                    'severity': 'high',
                    'description': '公司机密信息',
                    'matched_text': keyword,
                    'start_pos': text.find(keyword),
                    'end_pos': text.find(keyword) + len(keyword)
                }
        return None
    
    def detect_code_patterns(text: str) -> Optional[Dict]:
        """检测代码模式"""
        code_patterns = [
            r'\b(api_key|secret_key|password|token)\s*=\s*["\'][^"\']+["\']',
            r'\b(https?://[^\s]+)',
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ]
        
        import re
        for pattern in code_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                return {
                    'type': 'custom',
                    'pattern_name': 'code_patterns',
                    'severity': 'high',
                    'description': '代码中的敏感信息',
                    'matched_text': match.group(),
                    'start_pos': match.start(),
                    'end_pos': match.end()
                }
        return None
    
    # 注册自定义处理器
    detector.add_custom_handler('company_secrets', detect_company_secrets)
    detector.add_custom_handler('code_patterns', detect_code_patterns)
    
    # 测试文本
    test_texts = [
        "本季度报告显示收入增长20%",
        "api_key = 'sk-1234567890abcdef'",
        "战略规划将在下月公布",
        "服务器IP是 10.0.0.1"
    ]
    
    results = detector.batch_detect(test_texts)
    
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"文本 {i+1}: {text}")
        print(f"  隐私状态: {'🔒 私有' if result.is_private else '✅ 公开'}")
        if result.detected_patterns:
            print(f"  检测到的模式:")
            for pattern in result.detected_patterns:
                print(f"    - {pattern['pattern_name']}: {pattern['matched_text']}")
        print()

def example_config_file():
    """配置文件示例"""
    print("=== 配置文件示例 ===")
    
    # 从配置文件加载规则
    config_file = "privacy_patterns_config.json"
    if os.path.exists(config_file):
        detector = PrivacyDetector(config_file)
        
        # 测试文本
        test_texts = [
            "联系邮箱: user@company.com",
            "美国电话: (555) 123-4567",
            "SSN: 123-45-6789",
            "信用卡: 1234 5678 9012 3456",
            "文件路径: C:\\Users\\admin\\secret.txt"
        ]
        
        results = detector.batch_detect(test_texts)
        
        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"文本 {i+1}: {text}")
            print(f"  隐私状态: {'🔒 私有' if result.is_private else '✅ 公开'}")
            if result.detected_patterns:
                print(f"  检测到的模式:")
                for pattern in result.detected_patterns:
                    print(f"    - {pattern['pattern_name']}: {pattern['matched_text']}")
            print()
    else:
        print(f"配置文件 {config_file} 不存在")

def example_performance_test():
    """性能测试示例"""
    print("=== 性能测试示例 ===")
    
    detector = PrivacyDetector()
    
    # 生成大量测试文本
    test_texts = [
        f"这是第{i}个测试文本，包含邮箱test{i}@example.com和手机号13812345678" 
        for i in range(1000)
    ]
    
    import time
    start_time = time.time()
    
    # 批量检测
    results = detector.batch_detect(test_texts)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    private_count = sum(1 for r in results if r.is_private)
    total_matches = sum(len(r.detected_patterns) for r in results)
    
    print(f"处理文本数量: {len(test_texts)}")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每文本处理时间: {total_time/len(test_texts)*1000:.2f}毫秒")
    print(f"检测到私有文本: {private_count}")
    print(f"总匹配数: {total_matches}")
    
    # 显示统计信息
    stats = detector.get_stats()
    print(f"检测器统计: {stats}")

def example_integration_with_private_service():
    """与PrivateService集成示例"""
    print("=== 与PrivateService集成示例 ===")
    
    # 创建检测器
    detector = PrivacyDetector()
    
    # 模拟PrivateService的检测逻辑
    def check_privacy_for_node(context: str, prompt: str) -> Dict:
        """为节点检查隐私状态"""
        # 检测上下文和提示词
        context_result = detector.detect_privacy(context)
        prompt_result = detector.detect_privacy(prompt)
        
        # 合并结果
        is_private = context_result.is_private or prompt_result.is_private
        all_patterns = context_result.detected_patterns + prompt_result.detected_patterns
        
        # 计算综合置信度
        total_confidence = max(context_result.confidence, prompt_result.confidence)
        
        return {
            'is_private': is_private,
            'confidence': total_confidence,
            'detected_patterns': all_patterns,
            'context_private': context_result.is_private,
            'prompt_private': prompt_result.is_private
        }
    
    # 测试用例
    test_cases = [
        {
            'context': "用户询问关于公司财务信息",
            'prompt': "请提供本季度的收入数据"
        },
        {
            'context': "用户询问天气情况",
            'prompt': "今天天气怎么样？"
        },
        {
            'context': "用户提供个人信息",
            'prompt': "我的邮箱是 user@example.com，手机号是 13812345678"
        }
    ]
    
    for i, case in enumerate(test_cases):
        result = check_privacy_for_node(case['context'], case['prompt'])
        
        print(f"测试用例 {i+1}:")
        print(f"  上下文: {case['context']}")
        print(f"  提示词: {case['prompt']}")
        print(f"  隐私状态: {'🔒 私有' if result['is_private'] else '✅ 公开'}")
        print(f"  置信度: {result['confidence']:.2f}")
        if result['detected_patterns']:
            print(f"  检测到的模式:")
            for pattern in result['detected_patterns']:
                print(f"    - {pattern['pattern_name']}: {pattern['matched_text']}")
        print()

if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_custom_patterns()
    example_custom_handlers()
    example_config_file()
    example_performance_test()
    example_integration_with_private_service() 