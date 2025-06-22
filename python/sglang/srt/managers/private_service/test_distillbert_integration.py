#!/usr/bin/env python3
"""
测试DistilBERT隐私检测集成系统的脚本
"""
import time
import json
import logging
import sys
import os
import threading

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from private_service import PrivateJudgeService
from distillbert_client import DistilBERTClient, detect_privacy_with_distillbert
from sglang.srt.server_args import ServerArgs, PortArgs

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_distillbert_client():
    """测试DistilBERT客户端"""
    print("=== 测试DistilBERT客户端 ===")
    
    # 创建客户端
    server_args = ServerArgs()
    port_args = PortArgs()
    client = DistilBERTClient(server_args, port_args)
    
    # 测试文本
    test_texts = [
        "This is a public message about general topics.",
        "My email is john.doe@example.com and my phone number is 123-456-7890.",
        "The weather is nice today.",
        "My social security number is 123-45-6789.",
        "This is a confidential document containing sensitive information.",
        "The company's internal strategy document should not be shared."
    ]
    
    print("同步检测结果:")
    for i, text in enumerate(test_texts):
        try:
            result = client.detect_privacy_sync(text, timeout=5.0)
            print(f"{i+1}. Text: {text[:50]}...")
            print(f"   Private: {result.is_private}, Confidence: {result.confidence:.3f}")
            print(f"   Label: {result.label}, Processing time: {result.processing_time:.3f}s")
            print()
        except Exception as e:
            print(f"{i+1}. Error: {e}")
            print()
    
    # 异步检测
    print("异步检测结果:")
    results_received = []
    
    def callback(response):
        results_received.append(response)
        print(f"Async: {response.request_id[:8]}... - Private: {response.is_private}, Confidence: {response.confidence:.3f}")
    
    for text in test_texts:
        client.detect_privacy(text, callback)
    
    # 等待异步请求完成
    time.sleep(10)
    
    client.close()
    print(f"异步检测完成，收到 {len(results_received)} 个结果\n")

def test_private_service():
    """测试隐私检测服务"""
    print("=== 测试隐私检测服务 ===")
    
    # 创建服务
    server_args = ServerArgs()
    port_args = PortArgs()
    service = PrivateJudgeService(server_args, port_args)
    
    # 测试任务
    test_tasks = [
        {
            'task_type': 'update_private',
            'node_id': 'node_1',
            'context': 'User conversation',
            'prompt': 'This is a public message about general topics.'
        },
        {
            'task_type': 'update_private',
            'node_id': 'node_2',
            'context': 'User conversation',
            'prompt': 'My email is john.doe@example.com and my phone number is 123-456-7890.'
        },
        {
            'task_type': 'update_private',
            'node_id': 'node_3',
            'context': 'User conversation',
            'prompt': 'This is a confidential document containing sensitive information.'
        }
    ]
    
    # 模拟发送任务到服务
    print("发送测试任务到服务...")
    for task in test_tasks:
        service.first_level_task_queue.append(task)
    
    # 等待处理
    print("等待处理完成...")
    time.sleep(5)
    
    # 检查结果
    print("处理结果:")
    while len(service.result_queue) > 0:
        result = service.result_queue.pop(0)
        print(f"Node {result['node_id']}: {result['privacy']} (confidence: {result.get('confidence', 0):.3f})")
        print(f"  Detection level: {result.get('detection_level', 'unknown')}")
        print(f"  Model: {result.get('model_name', 'unknown')}")
        print()
    
    service.close()

def test_convenience_functions():
    """测试便捷函数"""
    print("=== 测试便捷函数 ===")
    
    test_texts = [
        "This is a public message.",
        "My email is test@example.com.",
        "This contains sensitive information."
    ]
    
    for text in test_texts:
        try:
            result = detect_privacy_with_distillbert(text, timeout=5.0)
            print(f"Text: {text}")
            print(f"Private: {result.is_private}, Confidence: {result.confidence:.3f}")
            print(f"Label: {result.label}")
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()

def main():
    """主函数"""
    setup_logging()
    
    print("DistilBERT隐私检测集成系统测试")
    print("=" * 50)
    
    try:
        # 测试DistilBERT客户端
        test_distillbert_client()
        
        # 测试便捷函数
        test_convenience_functions()
        
        # 测试隐私检测服务
        test_private_service()
        
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 