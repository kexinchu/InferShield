# SafeKV Time Channel Attack Model

这个项目实现了一个针对SafeKV系统的时基侧信道攻击模型，用于演示共享KV-Cache系统中的隐私泄露风险。

## 攻击原理

### 背景
在LLM多用户环境中，共享KV-Cache可以大幅提升推理性能。然而，这种优化会产生时间侧信道：

1. 当多用户共享KV缓存时，如果攻击者的请求与另一用户的请求有相同的前缀，系统可能直接重用之前计算的KV缓存
2. 这种重用会导致响应时间显著缩短
3. 攻击者可以通过测量首次令牌延迟（TTFT）的细微差异来推断缓存是否命中
4. 通过多次试探，攻击者可以猜测其他用户提示中的隐私内容

### 攻击流程
1. **基线建立**: 发送随机请求建立正常的TTFT基线
2. **候选集生成**: 生成包含潜在隐私信息的候选集
3. **时序攻击**: 发送候选请求并测量TTFT
4. **结果分析**: 通过TTFT差异判断缓存命中情况
5. **隐私推断**: 识别可能导致缓存命中的隐私信息

## 文件结构

```
time_channel_attacker/
├── README.md                 # 项目说明文档
├── attack_model.py           # 核心攻击模型
├── candidate_generator.py    # 候选集生成器
├── main_attack.py           # 主攻击脚本
└── requirements.txt         # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 快速演示

运行演示攻击（使用少量候选集）：

```bash
python main_attack.py --demo
```

### 2. 完整攻击

运行针对所有类别的完整攻击：

```bash
python main_attack.py --api-url http://localhost:30000/generate
```

### 3. 针对性攻击

攻击特定类别的候选集：

```bash
python main_attack.py --categories emails passwords --max-candidates 10
```

### 4. 自定义配置

```bash
python main_attack.py \
    --api-url http://your-llm-server:30000/generate \
    --output-dir my_attack_results \
    --baseline-requests 20 \
    --request-delay 0.3 \
    --max-candidates 15
```

## 候选集类别

攻击模型支持以下类型的候选集：

| 类别 | 描述 | 示例 |
|------|------|------|
| `emails` | 邮箱地址 | user@example.com |
| `passwords` | 密码 | password123 |
| `ssns` | 社会安全号 | 123-45-6789 |
| `credit_cards` | 信用卡号 | 4111-1111-1111-1111 |
| `phones` | 电话号码 | (555) 123-4567 |
| `addresses` | 地址 | 123 Main St |
| `api_keys` | API密钥 | sk-1234567890abcdef... |
| `company_info` | 公司信息 | Project-A123 |
| `medical_info` | 医疗信息 | diabetes, insulin |

## 输出结果

攻击完成后，会在输出目录生成以下文件：

- `candidate_sets.json`: 生成的候选集
- `attack_results.json`: 详细的攻击结果
- `attack_summary.json`: 攻击统计摘要
- `attack_report.txt`: 人类可读的攻击报告

### 结果示例

```json
{
  "emails": {
    "session_id": "attack_emails_1234567890",
    "statistics": {
      "total_attempts": 15,
      "hit_count": 3,
      "hit_rate": 0.2,
      "avg_ttft": 0.045,
      "avg_hit_ttft": 0.032,
      "avg_miss_ttft": 0.048
    },
    "hit_candidates": ["admin@company.org", "user@example.com"],
    "likely_private_info": ["admin@company.org"],
    "attack_success": true
  }
}
```

## 攻击参数

### 关键参数说明

- `--api-url`: 目标LLM API的URL
- `--baseline-requests`: 建立基线所需的请求数量
- `--request-delay`: 请求间的延迟时间（秒）
- `--max-candidates`: 每个候选集的最大候选数量
- `--categories`: 要攻击的特定类别

### 性能调优

- **提高准确性**: 增加 `--baseline-requests` 和 `--max-candidates`
- **减少检测风险**: 增加 `--request-delay`
- **快速测试**: 使用 `--demo` 模式

## 防御建议

基于攻击结果，建议采取以下防御措施：

1. **实现SafeKV隐私检测机制**
   - 使用多级检测（规则+模型+上下文）
   - 异步隐私评估pipeline

2. **KV-Cache分区管理**
   - 为敏感数据使用私有缓存
   - 实现动态public/private缓存管理

3. **时序攻击防护**
   - 添加响应时间随机化
   - 实现请求延迟和批处理

4. **攻击检测**
   - 监控缓存访问模式
   - 检测异常的访问频率和分布

5. **访问控制**
   - 实现速率限制
   - 用户行为异常检测

## 技术细节

### TTFT测量

攻击模型通过以下方式测量TTFT：

```python
start_time = time.perf_counter()
# 发送请求并等待第一个token
ttft = time.perf_counter() - start_time
```

### 缓存命中判断

基于TTFT差异判断缓存命中：

```python
if ttft <= hit_threshold:
    # 可能缓存命中
    confidence = (baseline_ttft - ttft) / (baseline_ttft - hit_threshold)
else:
    # 可能缓存未命中
    confidence = (ttft - hit_threshold) / (baseline_ttft - hit_threshold)
```

### 候选集生成

支持多种模式的候选集生成：

- 基于常见模式的生成
- 随机变体生成
- 自定义模式匹配

## 注意事项

⚠️ **重要提醒**:

1. 此工具仅用于安全研究和教育目的
2. 请勿在未经授权的系统上使用
3. 使用前请确保获得适当的授权
4. 遵守相关法律法规和道德准则

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目仅供学术研究使用。 