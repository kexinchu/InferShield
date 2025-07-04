#!/usr/bin/env python3
"""
启动DistilBERT隐私检测服务的脚本
"""
import argparse
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from privacy_detector_piiranha import PiiPrivacyDetector
from sglang.srt.server_args import ServerArgs, PortArgs

def setup_logging(log_level: str = "info"):
    """设置日志"""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    
    logging.basicConfig(
        level=level_map.get(log_level.lower(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="启动DistilBERT隐私检测服务")
    parser.add_argument("--model_name", default="distilbert-base-uncased",
                       help="DistilBERT模型名称")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                       help="隐私检测置信度阈值")
    parser.add_argument("--device", default=None,
                       help="运行设备 (cuda/cpu)")
    parser.add_argument("--log_level", default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 创建服务配置
        server_args = ServerArgs()
        port_args = PortArgs()
        
        # 启动服务
        logger.info("正在启动DistilBERT隐私检测服务...")
        service = PiiPrivacyDetector(
            server_args=server_args,
            port_args=port_args,
            model_name=args.model_name,
            max_length=args.max_length,
            confidence_threshold=args.confidence_threshold,
            device=args.device
        )
        
        logger.info("DistilBERT隐私检测服务已启动")
        logger.info(f"模型: {args.model_name}")
        logger.info(f"设备: {args.device or 'auto'}")
        logger.info(f"置信度阈值: {args.confidence_threshold}")
        
        # 保持服务运行
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("正在关闭服务...")
            service.close()
            logger.info("服务已关闭")
            
    except Exception as e:
        logger.error(f"启动服务失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 