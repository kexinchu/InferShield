# SafeKV Agent Experience Summary

> 本文档记录了跨 session 的开发、测试、踩坑经验，供下次启动时快速恢复上下文。

---

## 1. 项目背景

**SafeKV v2**：为多租户 LLM 推理服务设计的隐私感知 KV-Cache 共享系统，基于 SGLang。

核心机制：
- 所有 KV-cache 节点默认 `private_tag=1`（私有）
- 异步两级检测管线（Tier-1: Regex/Trie → Tier-2: Piiranha ML）
- 检测为非 PII → 升级为 `shareable`（`private_tag=0`），赋予访问预算 B
- 跨租户命中递减 budget；budget 耗尽且 creator_count < K → `permanently_private`
- 防御目标：阻断基于 TTFT 差异的 timing side-channel 攻击

---

## 2. 代码结构

### 2.1 核心 patch 位置（均已应用到 venv 0.5.8）

| 文件 | 关键修改内容 |
|------|------------|
| `sglang/srt/mem_cache/radix_cache.py` | TreeNode 中新增 7 个隐私字段；`_match_prefix_helper` 中加入跨租户可见性检查和 budget 递减逻辑；`_insert_helper` 中追踪 `creator_set` |
| `sglang/srt/mem_cache/base_prefix_cache.py` | `MatchPrefixParams` 新增 `user_id: Optional[str]` |
| `sglang/srt/mem_cache/cache_init_params.py` | `CacheInitParams` 新增 `private_judge_client`、`safekv_access_budget`、`safekv_creator_threshold` |
| `sglang/srt/server_args.py` | 新增 `safekv_access_budget: int = 10`，`safekv_creator_threshold: int = 2`，`safekv_private_only: bool = False` 及对应 CLI 参数 |
| `sglang/srt/managers/schedule_batch.py` | `Req.__init__` 新增 `user_id: Optional[str]` |
| `sglang/srt/managers/schedule_policy.py` | `MatchPrefixParams` 调用时传入 `user_id` |
| `sglang/srt/managers/scheduler.py` | 初始化 `PrivateJudgeClient`；创建 `Req` 时传入 `user_id` |
| `sglang/srt/managers/private_service/` | 完整目录，含 `private_service.py`、`private_client.py`、`global_task_queue.py`、`tree_node.py`（shim） |
| `sglang/srt/entrypoints/openai/protocol.py` | `ChatCompletionRequest` 新增 `user_id: Optional[str]` |
| `sglang/srt/entrypoints/openai/serving_chat.py` | 将请求中的 `user_id` 传给调度器 |

**所有 patch 路径前缀**：`/home/kec23008/.venv/lib/python3.10/site-packages/sglang/srt/`

### 2.2 `private_service` 子系统

```
global_task_queue.py   # 4 个 queue.Queue()（无界，无背压）
private_service.py     # 5 线程：processing / first_level / second_level / result / response
                       # BATCH_SIZE=16
private_client.py      # _response_task：读 result_final_queue，对非 PII 节点升级 private_tag=0
                       # safekv_private_only=True 时跳过所有升级（ablation 用）
tree_node.py           # shim：from sglang.srt.mem_cache.radix_cache import TreeNode
```

### 2.3 TreeNode 隐私字段（`radix_cache.py` 中内联）

```python
self.private_tag = 1          # 1=私有, 0=可共享
self.need_check_privacy = True
self.creator_id = None        # 首个创建者 user_id
self.creator_set = set()      # 所有创建者 user_id 集合
self.creator_count = 0        # len(creator_set)
self.access_budget = 0        # 跨租户访问预算（升级时设为 B）
self.permanently_private = False  # budget 耗尽后禁止重新升级
```

---

## 3. 启动服务器

### 脚本

```bash
# 标准 SafeKV（B=10, K=2）
./launch_model.sh qwen32b       # port 8090, TP=2
./launch_model.sh qwen30b       # port 8094, TP=2
./launch_model.sh qwen30b-int4  # port 8093, TP=1
./launch_model.sh phi4          # port 8092, DP=2

# Ablation 模式（环境变量控制）
SAFEKV_BUDGET=10    SAFEKV_THRESHOLD=2 ./launch_model.sh qwen32b   # full_safekv
SAFEKV_PRIVATE_ONLY=1              ./launch_model.sh qwen32b        # private_default
SAFEKV_BUDGET=999999 SAFEKV_THRESHOLD=1 ./launch_model.sh qwen32b  # private_detector
```

### 日志位置

`ndss_scripts/logs/<model_key>.log`（标准启动）  
`ndss_scripts/logs/ablation_qwen32b_<mode>.log`（ablation 启动）

---

## 4. 测试脚本

| 脚本 | 用途 |
|------|------|
| `test_safekv_functional.py` | 6 个功能测试：隔离/升级/budget/重升级/timing/sysprompt |
| `test_safekv_10k.py` | 大规模吞吐测试（已修复 log 路径 hardcode 问题） |
| `test_safekv_ablation.py` | **4 模式 ablation 对比**（见下节） |

### 运行功能测试

```bash
cd ndss_scripts
/home/kec23008/.venv/bin/python3 test_safekv_functional.py --model qwen32b --port 8090
```

---

## 5. Ablation 测试（test_safekv_ablation.py）

### 4 个模式

| Mode | 服务器参数 | user_id | 预期行为 |
|------|-----------|---------|---------|
| `baseline` | 无 SafeKV 参数 | 不发送 | 全共享，attacker 命中所有缓存 |
| `private_default` | `--safekv-private-only` | 发送 | 无升级，attacker 零命中 |
| `private_detector` | `--safekv-access-budget 999999 --safekv-creator-threshold 1` | 发送 | PII 保护，非 PII 自由共享 |
| `full_safekv` | `--safekv-access-budget 10 --safekv-creator-threshold 2` | 发送 | 完整保护，budget 限制 |

### 运行方式

```bash
# 全部 4 个模式（自动重启服务器，约 30-45 分钟）
/home/kec23008/.venv/bin/python3 test_safekv_ablation.py --model qwen32b

# 单个模式，不重启服务器
/home/kec23008/.venv/bin/python3 test_safekv_ablation.py \
    --model qwen32b --mode full_safekv --no-restart
```

### 测试设计（重要）

**工作负载（每个 mode）：**
1. Phase 1：5 个 victim，每人发 3 次自己的 PII prompt（热身 cache）
2. Phase 2：等待 20s 让异步 PII 检测完成
3. Phase 3：测量参考冷基准 TTFT（空闲状态，仅用于报告）
4. Phase 4：4 个 regular user 并发发非 PII 请求 → 测 TTFT/TPS
5. Phase 5：attacker 对每个 victim 顺序发 5 次配对 probe → 测 defense rate

**Defense rate 计算（配对 ratio 法）：**
```
对每次 probe：
  ① 发唯一冷 prompt（保证 cache miss） → cold_ttft_local
  ② 发 victim 的 PII prompt           → probe_ttft
  ratio = probe_ttft / cold_ttft_local
  ratio < 0.80  →  cache 命中  →  防御失败
defense_rate = (blocked_probes / total_probes) × 100%
```

**为什么用配对 ratio 而不是绝对 TTFT 阈值：**
- 并发负载下，调度延迟远大于 cache hit 带来的加速（~15%）
- 绝对阈值无法区分"cache hit"和"高负载慢"
- ratio 方法在同等负载下对比，消除调度抖动

**PII prompt 设计：**
- 每个 prompt ~200 词（医疗记录/银行账户/移民信息等）
- 长 prompt 使 prefill 主导 TTFT，cache hit 加速比 ≥ 50%
- 短 prompt（<30 词）时 cache hit 仅带来 ~13% 加速，低于 0.80 阈值，**无法检测**

---

## 6. 已知问题与踩坑

### 6.1 PYTHONPATH 方式失败
- **问题**：设 `PYTHONPATH=/home/kec23008/InferShield/python` 让 0.4.6 版本覆盖 venv 0.5.8 → 循环 import 崩溃（`sglang.srt.layers.linear → quantization/__init__ → linear`）
- **解决**：直接 patch venv 中的 0.5.8 文件

### 6.2 0.4.6 → 0.5.8 API 变化
- `RadixKey` 对象（0.5.8）vs 原始 list（0.4.6）
- `CacheInitParams` 包装了所有初始化参数
- `MatchPrefixParams` 包装了 match_prefix 参数
- `TreeNode` 在 0.5.8 中直接内联在 `radix_cache.py`，没有单独文件 → 创建了 `tree_node.py` shim

### 6.3 测试中 defense rate 全为 100%（包括 baseline）
- **原因**：冷基准在空闲状态测（~0.22s），attacker probe 在并发负载下测（~0.4-0.85s），阈值 = 0.22 × 0.80 = 0.176s，所有 probe 都高于此阈值
- **解决**：改用配对 ratio 法（见 5 节），PII prompt 改为长文档

### 6.4 test_safekv_10k.py hardcode phi4.log
- **问题**：无论测哪个 model，日志路径都写死为 `phi4.log`
- **解决**：加了 model 到 log 文件名的映射字典

---

## 7. 代码统计（SafeKV 修改量）

| 维度 | 数值 |
|------|------|
| 修改模块数 | ~10 个文件 |
| 新增代码行数（估算） | ~1110 LOC |
| 新增线程数 | 5（异步检测管线） |
| TreeNode 新增字段数 | 7 个隐私字段 + 1 个 prompt 字段 |
| 队列实现 | `queue.Queue()`（无界，无背压） |
| 并发控制 | CPython GIL（无显式锁，依赖 GIL 的隐式原子性） |
| TreeNode 内存开销 | ~400B/节点（set + int + bool × 4）|

---

## 8. 当前状态（2026-04-16）

- **已完成**：功能测试（Qwen30B / Qwen32B 均通过），前两轮 ablation（均有缺陷）
- **第三轮 ablation**：待运行（已修复 cold baseline + 加入 calibration 诊断）
- **结果文件**：`ndss_scripts/logs/ablation_qwen32b_<timestamp>.csv`
- **前两轮结果**（defense rate 无效，所有模式均 80%）：
  ```
  Mode              TTFT_mean  TTFT_P50  TTFT_P95  TPS
  baseline          0.283s     0.301s    0.379s    37.8
  private_default   0.288s     0.306s    0.388s    37.4
  private_detector  0.295s     0.301s    0.399s    37.5
  full_safekv       0.310s     0.307s    0.503s    36.9
  ```

## 9. 重要：已确认的两个 Bug（2026-04-16）

### Bug 1（已修复）：cold baseline 被 KV cache 污染

- **问题**：Phase 5 中每个 victim 的 5 次 probe 都用同一个 `DUMMY_PROMPTS_BASE[v]` base + unique suffix
- **后果**：probe[0] 把 DUMMY base 发到 server → 被缓存到 attacker 的 KV cache
  - probe[1-4] 的 cold prompt 仍用同一 base：cold_ttft ≈ 0.155s（调度开销 ≈ 纯 cache hit）
  - probe TTFT 也 ≈ 0.155s → ratio ≈ 1.0 → 永远检测不到命中（false negative）
  - 结果：所有模式均只 5/25 hits（仅 probe[0]），defense rate 80%（虚假）
- **修复**：`_generate_cold_baseline_prompt()` 动态生成 1000 个随机 8 位数（≈3000 tokens），每次调用唯一，永远不在 cache 中

### Bug 2（已发现，待确认）：SafeKV 隐私保护可能失效

- **证据**：
  - Phase 5 调度基线（probe[1] cold_ttft）= 0.158s（DUMMY 已缓存，纯调度）
  - private_default 模式下 probe[0] 的 probe_ttft = 0.156s ≈ 0.158s（0 prefill = full cache hit）
  - private_default 应有 100% 防御（`--safekv-private-only` 禁止所有升级），但 attacker 仍能读 victim 的 PII cache
- **影响**：如果确认，则 SafeKV 在所有 4 种模式下的 defense rate 都接近 0%（和 baseline 相同）
- **诊断手段**：Phase 2.5 加入 calibration 步骤，用 `calib_user_never_seen_pii` 测量 PII 冷 TTFT
  - 若 calib TTFT ≈ 0.35s 而 attacker probe TTFT ≈ 0.15s → 确认 SafeKV 失效（cache hit）
  - 若 calib TTFT ≈ 0.15s → GPU Phase 5 本来就这么快，无 bug
- **可疑代码路径**：`_match_prefix_helper` 中 private_tag=1 的 BREAK 逻辑，以及 `cache_finished_req` 中 attacker insert 后是否意外修改了 victim 节点的 value

---

---

## 11. NDSS 吞吐量 Ablation（EXP2）+ 保护延迟（EXP1）进展（2026-04-18）

### 11.1 实验目标

| 实验 | 脚本 | 目的 |
|------|------|------|
| EXP2 | `test_throughput_ablation.py` | 4 模式 × 3 模型的 KV 缓存效率对比 |
| EXP1 | `test_time_to_protection.py` | PII 检测完成后何时开始阻断攻击者 |

**重要**：两个实验自动串行执行，脚本路径 `/tmp/run_exp2_then_exp1.sh`（PID=2251317）

---

### 11.2 数据集设计（EXP2）

- 所有 1000 个请求均为 PII 请求
- 每个请求 = **共享非 PII 前缀（~2048 token）** + **唯一 PII 后缀（~2048 token）**
- 共享前缀来自 ShareGPT 对话；唯一后缀来自 `english_pii_43k.jsonl`
- **预期 cumul_kv 比率**：`private_default / baseline ≈ 2×`；`full_safekv / baseline ≈ 1×`（如修复生效）

---

### 11.3 代码修改汇总（已应用）

#### A. `test_throughput_ablation.py`
1. **加入 `pii_task` 指令**：在每个 PII prompt 末尾追加任务要求，确保模型生成 ≥128 个 token（否则 qwen30b-A3B 等模型输出为空）
2. **关闭所有模型 thinking 模式**：
   ```python
   MODEL_COMPLETION_OVERRIDES = {
       "qwen30b": {"enable_thinking": False},
       "qwen32b": {"enable_thinking": False},
       "phi4":    {"enable_thinking": False},
   }
   ```
   在 `_send_one()` 中通过 `**extra` 合并到 payload。

#### B. `radix_cache.py`（venv 安装版本）
1. **`insert()` 新增 `prompt: str = ""` 参数**，传入 `req.origin_input_text`
2. **`cache_finished_req()` / `cache_unfinished_req()` 传入 prompt**：
   ```python
   prompt=getattr(req, "origin_input_text", "") or ""
   ```
3. **`_insert_helper()` 改为 64-token 分块插入**：
   - 每 64 token 创建一个独立 `TreeNode`
   - 每个节点异步提交 PII 检测（proportional 字符切片）
   - 只有包含 PII 的块保持 `private_tag=1`；非 PII 块可被异步提升为 `private_tag=0`
   - **意义**：修复了"整条共享前缀因后缀 PII 而被标记为 private"的核心缺陷

---

### 11.4 当前实验结果（2026-04-18 23:17 EDT 时刻）

#### qwen32b（EXP2 完成，结果在 `throughput_ablation_qwen32b_20260418_152421.csv`）

| Mode | TTFT PII (mean) | TPOP (mean) | Throughput | cumul_kv (tokens) |
|------|----------------|-------------|------------|-------------------|
| baseline | 2.776 s | 0.0909 s/tok | 26.62 tok/s | **2,375,847** |
| private_default | 4.971 s | 0.0912 s/tok | 23.05 tok/s | **4,520,091** |
| private_detector | 4.967 s | 0.0909 s/tok | 23.10 tok/s | **4,520,091** ← 同 private_default（问题！） |
| full_safekv | 4.970 s | 0.0907 s/tok | 23.13 tok/s | **4,520,091** ← 同 private_default（问题！） |

- baseline/private_default 比率 = **1.90×** ✓（验证了数据集设计正确）
- private_detector/full_safekv 与 private_default 相同 → 64-chunk 修复未生效（详见 §11.6）

#### qwen30b（EXP2 完成，但数据无效，结果在 `throughput_ablation_qwen30b_20260418_213144.csv`）

| Mode | TPOP | Throughput | cumul_kv |
|------|------|------------|---------|
| baseline | nan | 4.00 tok/s | 2,351,333 |
| private_default | nan | 2.48 tok/s | 4,520,091 |
| private_detector | nan | 2.49 tok/s | 4,520,091 |
| full_safekv | nan | 2.49 tok/s | 4,520,091 |

- **原因**：qwen30b 在修复前运行，pii_task 和 enable_thinking=False 均未生效 → 输出近乎空 → TPOP/Throughput 无效
- **需要重跑**

#### phi4（EXP2 大部分完成，full_safekv 正在运行）

数据来自 master log（`master_20260418_152420.log`），**不是** CSV 文件（CSV 是旧的错误数据）：

| Mode | TPOP | Throughput | cumul_kv |
|------|------|------------|---------|
| baseline | 0.1664 s/tok | 129.62 tok/s | **2,127,598** |
| private_default | 0.1892 s/tok | 99.32 tok/s | **4,246,400** |
| private_detector | 0.2245 s/tok | 98.21 tok/s | **4,246,400** ← 同 private_default（问题！） |
| full_safekv | 进行中 | ~96 tok/s | ~4,246,400（预计，仍为问题） |

- baseline/private_default 比率 = **2.0×** ✓
- phi4 full_safekv 已在 23:07 启动，预计 23:30 完成（若不关机）

#### EXP1（未开始）
EXP1 在 EXP2 的所有 3 个模型完成后自动启动，目前还未运行。

---

### 11.5 重要发现：旧 CSV vs 新 CSV 区分

`throughput_ablation_phi4_20260418_023951.csv` 是**早期错误数据**（04:14 AM 写入），全模式 cumul_kv ≈ 4.3M，**应忽略**。

真实 phi4 数据在 `master_20260418_152420.log` 第 484 行起。

---

### 11.6 核心未解问题：64-chunk 修复为何未改善 cumul_kv？

**现象**：private_detector 和 full_safekv 的 cumul_kv 与 private_default 相同（无共享前缀复用）

**两种可能根因**（尚未确认哪个）：

**根因 A：异步检测速度 vs 请求并发率不匹配**
- 24 个并发 worker，每个请求约 30s
- 请求 1 完成 → 插入 64 个节点 → 异步 PII 检测排队
- 在检测完成之前，请求 2-24 已经开始 prefill（全部 cache miss）
- 即使检测完成后非 PII 节点升级，后续请求来的时候可能仍看到 private_tag=1

**根因 B：共享前缀文本被 PII 检测器误判为 PII（假阳性）**
- 共享前缀来自 ShareGPT，可能包含人名、邮件等 PII 样本
- Tier-1 规则检测命中 → 共享前缀块被标记 private_tag=1，永不升级
- 即使 64-chunk 正确传递了文本，检测器仍然拒绝升级

**如何诊断**（明天早上）：
```python
# 在 private_judge_client 的 _response_task 中添加日志
# 打印前 5 个节点的检测结果
if chunk_idx < 5:
    print(f"[DEBUG] chunk_idx={chunk_idx} result={result} text_preview={context[:50]!r}")
```
或者：直接对共享前缀文本运行检测器：
```python
from sglang.srt.managers.private_service.private_service import check_pii
result = check_pii(shared_context_text[:500])
print(result)
```

**如果是根因 B**：需要换用确保非 PII 的共享前缀（如技术文档、代码片段）  
**如果是根因 A**：需要增加同步检测（在 cache_finished_req 中同步等待检测结果）或预热策略

---

### 11.7 TODO 清单（按优先级）

1. **【明天必做】诊断 64-chunk 不生效的根因**（见 §11.6 诊断方法）

2. **重跑 qwen30b EXP2**：所有代码修改已就位，直接运行：
   ```bash
   cd /home/kec23008/InferShield/ndss_scripts
   /home/kec23008/.venv/bin/python3 test_throughput_ablation.py --model qwen30b
   ```

3. **重跑 qwen32b private_detector + full_safekv**（baseline 和 private_default 结果有效，只需重跑后两个模式）

4. **如果诊断出根因并修复**：重跑所有模型所有模式

5. **运行 EXP1（Time-to-Protection）**：
   ```bash
   /home/kec23008/.venv/bin/python3 test_time_to_protection.py --model qwen32b
   # 类似 qwen30b / phi4
   ```

6. **生成论文图表**：使用 `results/` 目录中的已有绘图代码

---

### 11.8 关机前进程状态（2026-04-18 23:17 EDT）

- PID 2251317: `run_exp2_then_exp1.sh`（master 脚本，包含 phi4 full_safekv + EXP1）
- PID 2277557: `test_throughput_ablation.py --model phi4`（phi4 EXP2 测试脚本）
- PID 2288328: phi4 SGLang server（full_safekv 模式，运行中）

**关机后上述进程全部终止**，明天需要重新启动。

---

## 10. 下次 session 快速上手（早期 ablation）

```bash
# 1. 确认服务器状态
lsof -ti:8090   # 是否有进程
# 查看最近日志
tail -f /home/kec23008/InferShield/ndss_scripts/logs/qwen32b.log

# 2. 启动服务器（如需）
cd /home/kec23008/InferShield/ndss_scripts
./launch_model.sh qwen32b &

# 3. 运行功能测试
/home/kec23008/.venv/bin/python3 test_safekv_functional.py --model qwen32b

# 4. 运行 ablation（全模式）
/home/kec23008/.venv/bin/python3 test_safekv_ablation.py --model qwen32b

# 5. 查看 ablation 结果
cat logs/ablation_qwen32b_*.csv | column -t -s,
```

---

## 12. 明天早上恢复指南（2026-04-19）

### 第一步：确认机器状态

```bash
# 检查有无残留进程
ps aux | grep -E "sglang|test_throughput|test_time|run_exp"
lsof -ti:8090  # 检查端口
```

### 第二步：诊断 64-chunk 是否生效（最高优先级）

在 `private_service.py` 或 `private_client.py` 添加日志，确认检测结果：

```bash
# 快速测试：对共享前缀文本运行 PII 检测
# 启动一个服务器，然后检查前几个请求的检测结果
grep "DEBUG\|promote\|private_tag" /home/kec23008/InferShield/ndss_scripts/logs/throughput_phi4_private_detector.log | head -20
```

也可以查看 private_service 日志（如果有的话）。

### 第三步：按需重跑

```bash
# qwen30b 全部模式（必须重跑）
cd /home/kec23008/InferShield/ndss_scripts
/home/kec23008/.venv/bin/python3 test_throughput_ablation.py --model qwen30b

# phi4 仅 full_safekv（如果昨晚被打断）
/home/kec23008/.venv/bin/python3 test_throughput_ablation.py --model phi4

# qwen32b 仅后两个模式（如果诊断后发现需要重跑）
/home/kec23008/.venv/bin/python3 test_throughput_ablation.py --model qwen32b
```

### 第四步：运行 EXP1

```bash
# 三个模型串行
for model in qwen32b qwen30b phi4; do
  /home/kec23008/.venv/bin/python3 test_time_to_protection.py --model $model
  sleep 90
done
```

### 关键文件路径

| 文件 | 说明 |
|------|------|
| `logs/master_20260418_152420.log` | 今天所有实验的完整日志 |
| `logs/throughput_ablation_qwen32b_*.csv` | qwen32b EXP2 结果（baseline/private_default 有效） |
| `logs/throughput_ablation_qwen30b_*.csv` | qwen30b EXP2 结果（**全部无效，需重跑**） |
| `logs/throughput_ablation_phi4_20260418_023951.csv` | **旧数据，忽略** |
| `.venv/.../radix_cache.py` | 已修改：64-chunk 插入 + prompt 传递 |
| `.venv/.../cache_finished_req()` | 已修改：传入 origin_input_text |

---

## 13. EXP1 根本缺陷诊断与修复（2026-04-19）

### 13.1 根本原因：Cold Prompt 长度失配（测试设计缺陷）

**现象**：所有 3 个模型、所有 3 种模式（baseline/full_safekv/private_default）的 AUC 均为 0.97-1.0，甚至 private_default AUC > baseline AUC。

**诊断过程**：
1. 检查了完整的代码链：`user_id` 正确从 HTTP payload 传入 → `Req.user_id` → `_match_prefix_helper()` 的隐私检查
2. 服务器日志确认 `safekv_private_only=True` 正确传入
3. `_split_node()` 已正确继承 `creator_id` 等字段（第 721-729 行）
4. 代码逻辑正确，SafeKV 隐私检查没有 bug

**真正的根因**：
```
Cold prompt = 500 个 8 位随机数 = **4526 Qwen3 tokens**
Victim prompt = 1500 tiktoken tokens ≈ **663 Qwen3 tokens**（含 chat template）
比例 = 4526 / 663 ≈ **6.8×**
```

Cold prompt 比 victim prompt 长 **6.8 倍**！即使没有任何缓存命中：
- Victim probe (663 tokens prefill) 远快于 Cold probe (4526 tokens prefill)
- 所有模式下 AUC=1.0 是因为 victim 短，而非 SafeKV 失效

**phi4 佐证**：phi4 baseline AUC=0.70，而 private_default AUC=0.83。理论上 baseline 应更高（有 cache hit），但 phi4 DP=2 稀释了 cache 命中率，使得 length 效应主导 → 所有模式 AUC 相近。

### 13.2 修复方案（已应用到 `test_time_to_protection.py`）

1. **新增常量** `N_COLD_POOL = 200`
2. **新函数** `build_prompts(n_victims, n_cold)` 替换原 `build_victim_prompts(n)`：
   - 从同一 PII 数据集加载 entries，shuffle 后
   - 前 n_victims 个建立 victim prompts（同 1500 tiktoken tokens）
   - 后续 n_cold 个建立 cold pool prompts（**同样 1500 tiktoken tokens 级别**）
   - victim 和 cold 使用**不同 PII 条目**，cold 从未被 warmup
3. **`run_experiment()`** 新增 `victim_prompts, cold_prompts` 参数
4. **`probe_cold()`** 从 `cold_prompts` 随机选取，替换原 `gen_cold_prompt()`

**修复后预期结果**：
- baseline: AUC ≈ 0.9-1.0（cache 命中 → victim TTFT 低 → 仍可检测）
- private_default: AUC ≈ 0.5（无 cache → victim/cold TTFT 相同分布 → 无法区分）
- full_safekv: AUC 从高（未保护时）逐渐降至 ≈ 0.5（随 PII 检测完成 + budget 耗尽）

### 13.3 已完成的所有实验结果汇总

#### EXP2（吞吐量 Ablation）

| 模型 | 文件 | 状态 |
|------|------|------|
| qwen32b | `throughput_ablation_qwen32b_20260418_152421.csv` | baseline/private_default 有效；private_detector/full_safekv 需重跑 |
| qwen30b | `throughput_ablation_qwen30b_20260418_213144.csv` | **全部无效**（thinking mode + pii_task 修复前），需重跑 |
| phi4 | `throughput_ablation_phi4_20260418_220325.csv` | cumul_kv metric 因 DP=2 不可靠，TTFT 数据有效 |

#### EXP1（保护延迟）

| 模型 | 文件 | 状态 |
|------|------|------|
| qwen32b | `time_to_protection_qwen32b_20260418_233215.csv` | **全部无效**（cold prompt 长度缺陷），需重跑 |
| qwen30b | `time_to_protection_qwen30b_20260418_234915.csv` | **全部无效**（同上），需重跑 |
| phi4 | `time_to_protection_phi4_20260419_000454.csv` | **全部无效**（同上），需重跑 |

### 13.4 下一步

1. **立即重跑 EXP1（所有模型）** — 修复已应用，可直接运行：
   ```bash
   for model in qwen32b qwen30b phi4; do
     /home/kec23008/.venv/bin/python3 test_time_to_protection.py --model $model
     sleep 60
   done
   ```

2. **EXP2 重跑（优先 qwen30b，其次 qwen32b private_detector/full_safekv）** — 这些结果比 EXP1 更容易重跑（无长度缺陷，只需修复 thinking/pii_task）

3. **注意**：EXP1 重跑时，如果 AUC ≈ 0.5 for private_default → SafeKV 保护正确工作。如果仍然高 → 需要进一步调查（可能存在真正的隐私泄露 bug）。

---

## 14. 关键 Bug 发现与修复（2026-04-20 → 2026-04-21）

### 14.1 SafeKV 完全失效的根本原因（已修复）

**问题**：所有 PII ratio sweep 实验中，SafeKV 的 TTFT/吞吐量与 SGLang 几乎相同，而非介于 SGLang 和 cache_partition 之间。

**根本原因追踪路径**：

```
serving_chat.py（chat completions API）
  ↓ 非多模态请求走 input_ids 路径：
  prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
  → GenerateReqInput.text = None

tokenizer_manager.py
  ↓ input_text = obj.text = None（有 input_ids 则不重新分词）

TokenizedGenerateReqInput.input_text = None

scheduler.py
  ↓ origin_input_text = recv_req.input_text = None → Req.origin_input_text = None

radix_cache.py → cache_finished_req()
  ↓ prompt = getattr(req, "origin_input_text", "") or ""
  → None or "" → ""（空字符串）

private_client.py → update_privacy()
  ↓ prompt_ = "".strip() = ""
  ↓ not prompt_ → True → 触发哨兵！
  prompt_ = "hello, you are an simple assistant"

private_client.py → _response_task()
  ↓ task.prompt == "hello, you are an simple assistant" → task.privacy = False
  → node.private_tag = 0（所有节点变为公开！）
```

**结果**：所有 KV cache 节点都被标记为公开，SafeKV 完全退化成 SGLang。

### 14.2 两个修复（已应用）

#### 修复 1：`radix_cache.py`（`cache_finished_req` 和 `cache_unfinished_req`）

路径：`/home/kec23008/.venv/lib/python3.10/site-packages/sglang/srt/mem_cache/radix_cache.py`

当 `origin_input_text` 为空时，通过 tokenizer decode token IDs 获取真实文本：

```python
_prompt = getattr(req, "origin_input_text", "") or ""
if not _prompt:
    _tok = getattr(req, "tokenizer", None)
    _ids = getattr(req, "origin_input_ids_unpadded", None) or getattr(req, "origin_input_ids", None)
    if _tok is not None and _ids is not None:
        try:
            _prompt = _tok.decode(list(_ids))
        except Exception:
            pass
# 传入 insert()
```

两处：`cache_finished_req`（变量名 `_prompt`）和 `cache_unfinished_req`（变量名 `_prompt2`）。

#### 修复 2：`private_service.py`（`_process_second_level_tasks`）

路径：`/home/kec23008/.venv/lib/python3.10/site-packages/sglang/srt/managers/private_service/private_service.py`

原来：Tier-2 不可用时保守标记为 `privacy=True`（所有非 PII 也变为私有 → SafeKV ≈ cache_partition）

现在：Tier-2 不可用时信任 Tier-1（`privacy=False`），非 PII 内容（Tier-1 未命中）标记为可共享：

```python
# 旧：'privacy': True（保守）
# 新：'privacy': False（信任 Tier-1）
```

**意义**：
- 非 PII 内容（ShareGPT）：Tier-1 不命中 → Tier-2 不可用 → `privacy=False` → `private_tag=0` → 可共享 ✓
- PII 内容（含 email/SSN/电话号码）：Tier-1 命中 → `privacy=True` → `private_tag=1` → 隔离 ✓

### 14.3 正在运行的测试（2026-04-21 凌晨）

**进程**：PID 2730623，运行 `test_pii_ratio_sweep.py --model qwen32b`

**日志**：`/tmp/pii_retest_qwen32b.log`

**当前进度**（睡觉时的状态）：

SGLang 阶段基本完成：
| R | TTFT_mean | TPS |
|---|-----------|-----|
| 0% | 1509ms | 51.9 |
| 5% | 1372ms | 52.2 |
| 10% | 1385ms | 52.2 |
| 20% | 1553ms | 51.7 |
| 50% | 1387ms | 52.2 |
| 100% | 正在 warmup... | — |

接下来会运行 cache_partition（约 30 分钟），然后 SafeKV（约 30 分钟）。

**明天早上检查**：
```bash
# 查看是否完成
cat /tmp/pii_retest_qwen32b.log | grep -E "✓ TTFT|System=|csv"

# 或查看最新的 CSV 结果
ls -lt ndss_scripts/logs/pii_ratio_sweep_qwen32b_*.csv | head -3
cat <最新CSV>
```

### 14.4 预期验证结果（修复后）

SafeKV 应呈现**优雅降级**（R=0 时接近 SGLang，R=100 时接近 cache_partition）：

| R% | SGLang | cache_partition | SafeKV（预期） |
|----|--------|-----------------|----------------|
| 0 | ~1400ms | ~9700ms | ~1400ms（全非 PII → 全共享） |
| 50 | ~1400ms | ~9700ms | ~5500ms（半 PII → 半隔离） |
| 100 | ~1400ms | ~9700ms | ~9700ms（全 PII → 全隔离） |

如果 SafeKV 结果仍然全部 ~1400ms（与 SGLang 相同）→ 修复未生效，需进一步调查。

### 14.5 后续实验顺序

1. 等待当前 qwen32b 测试完成，验证修复效果
2. 如修复有效，依次运行 qwen30b 和 phi4：
   ```bash
   cd /home/kec23008/InferShield
   /home/kec23008/.venv/bin/python3 ndss_scripts/test_pii_ratio_sweep.py --model qwen30b
   /home/kec23008/.venv/bin/python3 ndss_scripts/test_pii_ratio_sweep.py --model phi4
   ```
3. 收集三模型全部数据后生成论文图表
