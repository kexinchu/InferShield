## Sglang

- Overview of Sglang
<img src="./pictures/Sglang-0_4_6-overview.png" wdith=600>

- Memory Pool Architecture

|Pool Type|Purpose|Allocation Strategy|
|--|--|--|
|ReqToTokenPool|Request-to-token mapping|Fixed-size|
|TokenToKVPoolAllocator|KV cache allocation|Page-based allocation (PagedAttention)|
|RadixCache|Prefix sharing|Tree-based cache with LRU eviction|

```shell
  requests -> ReqToTokenPool(转换为token id lists) -> TokenToKVPoolAllocator(为每个token分配KV Cache空间) -> RadixCache(将KV-Cache管理在RadixTree中，方便share)
```

## 进度：06/08/25-06/14/25
### request dataflow
- 用户通过 /v1/chat/completions 提交request
```shell
HTTP Request (/v1/chat/completions)
    ↓
v1_chat_completions (python/sglang/srt/entrypoints/http_server.py)
    ↓
v1_chat_completions (async; python/sglang/srt/openai_api/adapter.py)
    ↓
v1_chat_generate_request (adapter.py)
    ↓
generate_stream_resp (只考虑stream方法; adapter.py)
    ↓
tokenizer_manager.generate_request (prompts -> token list; go python/sglang/srt/managers/tokenizer_manager.py)
    ↓
tokenizer_manager._send_one_request (request send to scheduler, through ZMQ / scheduler运行在独立进程中)
    ↓
Scheduler.event_loop_normal (scheduler 主事件循环)
    ↓
Scheduler.handle_generate_request
    ↓
SchedulerBatch.init_next_round_input (call radixCache.match_prefix 读取可以reuse的KV-Cache)
    ↓
RadixCache/HiRadixCache
```

- **<font color="red">增加的修改</font>**：
  - **<font color="red">向下透传user_id/为用户输入增加一个参数：User_id</font>**
  - 入参: 
    - $ChatCompletionRequest$(python/sglang/srt/openai_api/protocol.py) 中增加 user_id 字段; 
    - 在tokenizer中放到sampling_params中; 
    - scheduler.py 中透传给 req=Req(), 向下游透传
    - scheduler_batch.py 中透传给 match_prefix 函数 (进入radixCache)
    - radixCache创建node： cache_finished_req, cache_unfinished_req 入参是 Req, 在第三步已经加上了

### RadixCache / HiRadixCache(Hierarchical)
```shell
python/sglang/srt/managers/cache_controller.py
```

|function|功能|
|--|--|
|CacheOperation|以 [node_id] list 管理KV-Cache|
|TransferBuffer|提供 buffer.put 和 buffer.get 方法管理 data-transfer 任务|
|HiCacheController|多级存储，处理主机内存和设备内存之间的数据迁移;提供load/write功能|

```shell
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/mem_cache/hiradix_cache.py
```
- RadixCache
```shell
TreeNode: 
  __init__: 初始化树节点，包含子节点、父节点、键值、锁引用计数等属性
  evicted: 属性，检查节点是否被驱逐
  backuped: 属性，检查节点是否有备份

RadixCache :
  __init__: 初始化基数树缓存，设置页面大小、内存池等
  reset: 重置缓存状态
  match_prefix: 在基数树中查找匹配的前缀
  insert: 插入新的键值对到基数树中
  cache_finished_req: 缓存已完成的请求
  cache_unfinished_req: 缓存未完成的请求
  evict: 驱逐指定数量的token
  inc_lock_ref: 增加节点的锁引用计数
  dec_lock_ref: 减少节点的锁引用计数
  evictable_size: 获取可驱逐的缓存大小
  protected_size: 获取受保护的缓存大小
  all_values_flatten: 获取所有值的扁平列表
```
- HiRadixCache - 处理主机内存和设备内存之间的数据迁移
```shell
HiRadixCache ( RadixCache):
  __init__: 初始化分层基数树缓存，设置缓存比例、大小和写入策略
  reset: 重置缓存状态
  get_height: 获取节点在树中的高度
  write_backup: 将节点数据写入备份
  inc_hit_count: 增加节点的命中计数
  writing_check: 检查写入操作状态
  loading_check: 检查加载操作状态
  evict: 驱逐指定数量的token
  evict_host: 从主机内存中驱逐数据
  load_back: 从主机内存加载数据回设备
  init_load_back: 初始化加载回操作
  ready_to_load_cache: 检查缓存是否准备好加载
  match_prefix: 重写父类的match_prefix方法，支持包含已驱逐节点

辅助函数:
  _evict_backuped: 驱逐已备份的节点
  _evict_regular: 驱逐普通节点
  _collect_leaves_device: 收集设备上的叶子节点
  _match_prefix_helper: 辅助函数，用于匹配前缀
  _split_node: 分割节点
  _insert_helper: 辅助函数，用于插入操作
```

- **<font color="red">增加的修改</font>**：
  - **<font color="red">为node增加private/public, owner_id, hit_cur, u_cnt_l, hit_pre, u_pre</font>**
  - **<font color="red">时间窗口: global epoch + self.epoch for Node</font>**
    - 注意：这里全局参数 epoch 需要实验确认


### Privacy Detection (Async)
- **<font color="red">增加的修改</font>**：
  - **<font color="red">增加 private_judge_client 异步提交任务，供RadixCache/HiRadixCache调用</font>**：
  - **<font color="red">增加 private_judge_service 后台处理privacy detection任务</font>**：
  - 后续拓展：
    - pattern-aware detection
    - light-weight model detection
    - LLM detection


## 进度：06/15/25-06/21/25
### RadixCache / HiRadixCache
- **<font color="red">增加的修改</font>**：
  - **<font color="red">private_sub_tree merge</font>**
    - merge when insert
    - search
    - evict
  - **<font color="red">logical free for LRU</font>**
  - **<font color="red">LRU 基于epoch</font>**
  - **<font color="red">计算分布熵</font>**
    - 注意：这里全局 hit_cur_threshold 需要实验确认


### Privacy Detection (Async)
- **<font color="red">增加的修改</font>**：
  - **<font color="red">修改为完全异步处理逻辑：三级Queue + batch处理</font>**：
  - **<font color="red">pattern-aware detection</font>**：
    - 继续找一些pattern
  - **<font color="red">light-weight model detection</font>**：
    - distilBert
  - 后续拓展：
    - LLM detection; 应该直接复用后台的模型，将request合并的scheduler一起去调度，并且识别后返回
      - 模型size + QPS => 硬件开销大
      - 回答：为什么使用这么模型？
      => 直接使用后台运行的那个LLM model
      => 是不是需要有一个比较： “横轴models - 纵轴 detect accuracy” ？
      => 问题：这样做，对正常请求的queuing delay/end-2-end latency的 比较： 横轴 x% 的detection req -> latency
      => 整体的QPS + 第三个阶段的x%