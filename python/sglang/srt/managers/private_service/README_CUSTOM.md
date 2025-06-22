# Privacy Detection System

基于Trie Tree和正则表达式的隐私检测系统，用于SGLang中的敏感信息识别。

## 功能特性

### 1. 基于Trie Tree的快速匹配
- 使用Trie树结构进行字符串匹配，时间复杂度为O(m)，其中m是文本长度
- 支持大量敏感词汇的快速检测
- 内存效率高，适合大规模词汇库

### 2. 正则表达式模式匹配
- 支持复杂的正则表达式模式
- 预编译正则表达式以提高性能
- 适用于邮箱、身份证号、银行卡号等格式化的敏感信息

### 3. 可扩展的规则系统
- 支持动态添加新的检测规则
- 配置文件驱动的规则管理
- 自定义处理器支持复杂的检测逻辑

### 4. 多级严重程度
- low: 低风险信息
- medium: 中等风险信息  
- high: 高风险信息
- critical: 极高风险信息

## 文件结构

```
private_service/
├── privacy_detector.py          # 核心隐私检测器
├── private_service.py           # 隐私服务集成
├── private_client.py            # 客户端接口
├── privacy_patterns_config.json # 配置文件示例
├── example_usage.py             # 使用示例
└── README.md                    # 本文档
```

## 快速开始

### 1. 基本使用

```python
from privacy_detector import PrivacyDetector

# 创建检测器
detector = PrivacyDetector()

# 检测文本
text = "我的邮箱是 test@example.com，手机号是 13812345678"
result = detector.detect_privacy(text)

print(f"隐私状态: {'私有' if result.is_private else '公开'}")
print(f"置信度: {result.confidence:.2f}")
print(f"检测到的模式: {len(result.detected_patterns)}")
```

### 2. 使用配置文件

```python
# 从配置文件加载规则
detector = PrivacyDetector("privacy_patterns_config.json")

# 检测文本
result = detector.detect_privacy("包含敏感信息的文本")
```

### 3. 添加自定义规则

```python
from privacy_detector import PrivacyPattern

# 添加Trie模式
trie_pattern = PrivacyPattern(
    name="custom_keywords",
    pattern="project_alpha,beta_test,gamma_release",
    pattern_type="trie",
    severity="high",
    description="自定义项目关键词"
)
detector.add_pattern(trie_pattern)

# 添加正则模式
regex_pattern = PrivacyPattern(
    name="custom_id",
    pattern=r"\b[A-Z]{2}\d{6}\b",
    pattern_type="regex",
    severity="medium",
    description="自定义ID格式"
)
detector.add_pattern(regex_pattern)
```

### 4. 自定义处理器

```python
def detect_company_secrets(text: str):
    """检测公司机密信息"""
    keywords = ['季度报告', '财务预测', '并购计划']
    
    for keyword in keywords:
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

# 注册自定义处理器
detector.add_custom_handler('company_secrets', detect_company_secrets)
```

## 配置文件格式

配置文件使用JSON格式，支持两种类型的规则：

### Trie模式配置

```json
{
  "trie_patterns": [
    {
      "name": "sensitive_words",
      "pattern": "password,secret,private,confidential",
      "pattern_type": "trie",
      "severity": "medium",
      "description": "常见敏感词"
    }
  ]
}
```

### 正则模式配置

```json
{
  "regex_patterns": [
    {
      "name": "email",
      "pattern": "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
      "pattern_type": "regex",
      "severity": "high",
      "description": "邮箱地址"
    }
  ]
}
```

## 与PrivateService集成

### 1. 初始化服务

```python
from private_service import PrivateJudgeService

# 创建服务实例
service = PrivateJudgeService(
    server_args=server_args,
    port_args=port_args,
    privacy_config_file="privacy_patterns_config.json"
)
```

### 2. 添加自定义规则

```python
# 添加自定义隐私模式
service.add_custom_privacy_pattern(
    pattern_name="custom_pattern",
    pattern_type="trie",
    pattern="keyword1,keyword2,keyword3",
    severity="high",
    description="自定义模式"
)

# 添加自定义处理器
def custom_handler(text):
    # 自定义检测逻辑
    pass

service.add_custom_handler("custom_handler", custom_handler)
```

### 3. 获取统计信息

```python
# 获取隐私检测统计
stats = service.get_privacy_stats()
print(f"总检测次数: {stats['total_checks']}")
print(f"匹配次数: {stats['total_matches']}")
print(f"平均处理时间: {stats['avg_processing_time']:.4f}s")
```

## 性能优化

### 1. Trie树优化
- 使用Trie树进行字符串匹配，时间复杂度为O(m)
- 支持大量词汇的快速检测
- 内存使用效率高

### 2. 正则表达式优化
- 预编译正则表达式
- 使用非贪婪匹配
- 避免回溯爆炸

### 3. 批量处理
```python
# 批量检测多个文本
texts = ["文本1", "文本2", "文本3", ...]
results = detector.batch_detect(texts)
```

### 4. 缓存机制
- 检测结果包含处理时间统计
- 支持性能监控和优化

## 默认规则

系统内置以下默认规则：

### 正则表达式规则
- 邮箱地址
- 中国手机号
- 中国身份证号
- 银行卡号
- 美国社会安全号
- 信用卡号
- IP地址
- MAC地址
- 日期格式
- URL地址
- 文件路径

### Trie规则
- 常见敏感词
- 公司机密信息
- 财务敏感信息
- 个人身份信息

## 扩展开发

### 1. 添加新的检测模式

```python
class CustomDetector:
    def __init__(self):
        self.detector = PrivacyDetector()
    
    def add_custom_patterns(self):
        # 添加自定义模式
        pass
    
    def detect(self, text):
        return self.detector.detect_privacy(text)
```

### 2. 集成到其他系统

```python
# 集成到Web服务
from flask import Flask, request, jsonify

app = Flask(__name__)
detector = PrivacyDetector()

@app.route('/detect', methods=['POST'])
def detect_privacy():
    text = request.json.get('text', '')
    result = detector.detect_privacy(text)
    return jsonify({
        'is_private': result.is_private,
        'confidence': result.confidence,
        'patterns': result.detected_patterns
    })
```

## 注意事项

1. **性能考虑**: 大量正则表达式可能影响性能，建议优先使用Trie模式
2. **误报处理**: 根据业务场景调整严重程度和置信度阈值
3. **规则更新**: 定期更新敏感词汇库和模式规则
4. **日志记录**: 建议记录检测结果用于分析和优化

## 示例运行

运行示例代码：

```bash
cd python/sglang/srt/managers/private_service
python example_usage.py
```

这将展示各种使用场景和功能特性。 

