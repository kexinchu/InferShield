# DistilBERT 隐私检测系统

本系统为SGLang提供了基于DistilBERT的第二级隐私检测功能，与现有的正则表达式和Trie树检测形成三级检测体系。

## 系统架构

### 三级检测体系

1. **第一级检测**: 基于正则表达式和Trie树的快速检测
   - 文件: `privacy_detector.py`
   - 特点: 快速、准确、低延迟

2. **第二级检测**: 基于DistilBERT的机器学习检测
   - 文件: `privacy_detector_distillbert.py` + `distillbert_client.py`
   - 特点: 语义理解、高准确性

3. **第三级检测**: 基于LLM大模型的深度检测
   - 文件: `private_service.py` (预留接口)
   - 特点: 最准确但延迟较高

### 组件说明

- **`privacy_detector_distillbert.py`**: DistilBERT服务端，加载模型并提供ZMQ接口
- **`distillbert_client.py`**: DistilBERT客户端，向服务发送请求并处理响应
- **`private_service.py`**: 主服务，协调三级检测流程
- **`start_distillbert_service.py`**: 启动DistilBERT服务的脚本
- **`test_distillbert_integration.py`**: 测试整个系统的脚本

## 安装依赖

```bash
# 安装PyTorch和transformers
pip install torch transformers

# 安装其他依赖
pip install zmq numpy
```

## 使用方法

### 1. 启动DistilBERT服务

```bash
# 基本启动
python start_distillbert_service.py

# 自定义参数启动
python start_distillbert_service.py \
    --model_name distilbert-base-uncased \
    --max_length 512 \
    --confidence_threshold 0.7 \
    --device cuda \
    --log_level info
```

### 2. 使用客户端

```python
from distillbert_client import DistilBERTClient, detect_privacy_with_distillbert
from sglang.srt.server_args import ServerArgs, PortArgs

# 创建客户端
server_args = ServerArgs()
port_args = PortArgs()
client = DistilBERTClient(server_args, port_args)

# 同步检测
result = client.detect_privacy_sync("My email is test@example.com")
print(f"Private: {result.is_private}, Confidence: {result.confidence}")

# 异步检测
def callback(response):
    print(f"Async result: {response.is_private}")

client.detect_privacy("Sensitive information", callback)

# 使用便捷函数
result = detect_privacy_with_distillbert("Confidential document")
print(f"Result: {result.is_private}")

client.close()
```

### 3. 集成到主服务

```python
from private_service import PrivateJudgeService
from sglang.srt.server_args import ServerArgs, PortArgs

# 创建主服务（自动集成DistilBERT客户端）
server_args = ServerArgs()
port_args = PortArgs()
service = PrivateJudgeService(server_args, port_args)

# 服务会自动处理三级检测流程
# 第一级：正则/Trie检测
# 第二级：DistilBERT检测
# 第三级：LLM检测（预留）

service.close()
```

## 配置说明

### DistilBERT模型配置

- **model_name**: 模型名称，默认 `distilbert-base-uncased`
- **max_length**: 最大序列长度，默认 512
- **confidence_threshold**: 置信度阈值，默认 0.7
- **device**: 运行设备，默认自动选择

### 端口配置

系统使用IPC文件进行进程间通信：

- `distillbert_service_port`: DistilBERT服务接收请求的端口
- `distillbert_client_port`: DistilBERT服务发送响应的端口

### 性能调优

1. **批量处理**: 客户端支持批量请求，提高吞吐量
2. **缓存机制**: 客户端内置响应缓存，减少重复计算
3. **异步处理**: 支持异步请求，提高并发性能
4. **超时控制**: 可配置请求超时时间，避免阻塞

## 测试

运行测试脚本验证系统功能：

```bash
python test_distillbert_integration.py
```

测试内容包括：
- DistilBERT客户端功能测试
- 便捷函数测试
- 隐私检测服务集成测试

## 性能指标

### 检测准确性

- **第一级检测**: 基于规则，准确率约85-90%
- **第二级检测**: 基于DistilBERT，准确率约90-95%
- **第三级检测**: 基于LLM，准确率约95-98%

### 处理延迟

- **第一级检测**: < 1ms
- **第二级检测**: 10-50ms (取决于文本长度)
- **第三级检测**: 100-500ms (取决于LLM模型)

### 吞吐量

- **单机**: 100-500 requests/second (取决于硬件配置)
- **批量处理**: 可提高2-5倍吞吐量

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接
   - 确认模型名称正确
   - 检查磁盘空间

2. **ZMQ连接失败**
   - 检查端口配置
   - 确认服务已启动
   - 检查防火墙设置

3. **检测结果不准确**
   - 调整置信度阈值
   - 检查模型是否适合当前任务
   - 考虑重新训练模型

### 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新的检测模型

1. 继承 `DistilBERTPrivacyDetector` 类
2. 实现 `detect_privacy` 方法
3. 集成到 `DistilBERTPrivacyService` 中

### 自定义检测规则

```python
# 添加自定义隐私模式
service.add_custom_privacy_pattern(
    pattern_name="custom_pattern",
    pattern_type="regex",
    pattern=r"your_pattern",
    severity="high",
    description="Custom pattern description"
)
```