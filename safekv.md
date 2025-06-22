# LLM & Security

## SafeKV
#### 解决的问题：
- 在 LLM 多用户环境中，共享 KV-Cache 可以大幅提升推理性能
- Risk: 当多用户共享KV缓存时，如果一位攻击者的请求与另一用户的请求有相同的前缀，系统可能直接重用之前计算的KV缓存，从而加速响应。然而，这种优化会产生时间侧信道：攻击者可以通过测量首次令牌延迟（TTFT）或响应时间的细微差异来推断缓存是否命中，从而猜测其他用户提示中的内容.
- 一些隐私数据follow固定的pattern；大大减少了Attackers获取这部分数据的难易程度

#### Motivation
- Existing work: USTC的一篇工作通过使用KV-Cache Partitioning来保护隐私：为每个用户存储独立的prefix-tree，避免cross user的KV-Cache Sharing  =>  影响性能，因为：1，会存储冗余KV-Cache，浪费HBM/DRAM/SSD存储资源；2，在一些场景下，通过cross user sharing才能取得收益
- 选择性共享来平衡隐私和性能：将非敏感的缓存条目在用户之间共享，而将敏感内容对应的缓存隔离在私有范围

#### Challenges
- 如何实现快速，低成本的隐私block检测
    - 除了存在固定pattern的隐私信息，还包括复杂/上下文关联的隐私信息
    - light-weight + real-time
    - 支持根据场景定制privacy 类型(例如：企业内部数据,特定领域术语等)；适应复杂也现实场景
- 针对检测失败的兜底方法
    - 针对KV-Cache Reuse的侧信道攻击依赖多次对 KV-Cache block的 "试探"；会导致单一"用户"对某些KV-Cache block的命中大幅增加； -> 最简单的方法是通针对用户行为的单点检测来识别attacks
    - 现实中，并不能假定单个攻击者，attackers可能会控制多个账号来协同攻击，以绕过单点检测。如：Attackers 通过多个账号分别低频率探查不同部分，使任何单账户行为看似正常，却在整体上实现了高频覆盖。
- 如何管理 private/public KV-Cache
    - 避免数据的重复存储：优化 HBM/DRAM 资源占用
    - 支持快速的 prefix-prompt Search；避免给LLM inference造成penalty
        - 支持自适应node 合并，降低search深度
    - 支持快速的 数据插入 (private/public insert) 和 自适应的数据删除
        - 自适应删除：private 数据因为其使用频率较低，需要避免一次性删除一整个节点 => 避免用户激活requests时出现 long prefix prompt's KV-Cache Miss

#### Design
- 0， 舍弃 SafeKV-固定分块 思路的原因
    - 固定分块存在局限性：对复杂和跨chunk的private pattern可能无能为力(敏感信息的长度和位置各异， 可能恰好落在分块边界处，导致逃过detect)；
    - 敏感性往往取决于上下文：某些信息片段在上下文中才具有敏感意义，但ChunkGuard逐块独立判别，可能无法识别这类组合敏感模式。例如“她去年得了流感”单看并非典型敏感信息，但如果上文提到某人身份，这句话可能属于健康隐私。固定块长也忽略了语义边界，可能将一句话拆开，丢失了完整语义，导致误判率上升。
    - 缺乏深入的语义和上下文感知能力

- 1. Adaptive-Detection：上下文感知的自适应隐私检测
    - 目标：在KV-cache存储时，实现light-weight的实时检测(快速，低成本，可定制化)
    - 思路：多级检测(Hybrid)
        - 一级(规则/正则方法)：感知"显示敏感pattern"
            - 维护一份可插拔的“敏感模式库”（如邮箱、身份证号、银行账号、内部文档、专有名词词典等） => 支持User按照业务场景自定义 + 扩充
            - 对于文本进行扫描, 使用Trie匹配或者正则，快速截获敏感字段
            - 对于邮箱/电话/SSN等pattern可以并入Trie匹配中, 降低扫描成本；对于无法合并(不定长)的pattern，使用滑动窗口+正则匹配
            - 检测到privacy block，将当前KV-Cache block标记为 private
        - 二级(light-weight transformer-based model)
            - 使用一支蒸馏版Transformer（如TinyBERT、DistilBERT微调），为每个语义块打分（敏感概率：0~1）。
            - (阈值根据实验进行调整)将"低敏(p<0.3)"块的KV-Cache Node标记为public，将"高敏(p>0.7)"块的KV-Cache Node标记为private
            - 写作的时候需要提及：支持用户自定义微调，使用混合标签数据集(一部分公共敏感概念（PII、法律法规要求保护的数据），一部分客户自定义词典（企业内部项目代号、专利编号等）) 
        - 三级(跨block/对话上下文综合校验)
            - 针对二级检测中处于灰度区间(0.3 < p < 0.7)的block，并不立即标记public，而是引入 "cross block 关联检测"
            - 增加本次请求的上下文，使用LLM评估是否包含隐私信息 (构造prompt)
        - 实验： 1，2，3分别的比例；真实的数据集，构造一个trace数据集(3的开销较大的情况下，整体性能也是正向的)
        - 仅在创建 KV-Cache 缓存node时使用，一次分类，多次使用，减少分类频率和开销
        - 为避免分类过程影响当前的 KV-Cache Sharing; 使用异步pipeline的方式执行：
            - 默认KV-Cache node创建之后，标记为private
            - 异步触发 hybrid detection；使用batch方法合并多个请求，提高detection效率；
            - 逐步需要可reuse/sharing的public block状态
            - 当node确定private状态时，其子节点直接被标记为private
            - pipeline 在第一层的prefill执行之后就可以检测 (40GB -> 10GB; sglang的address逻辑)

- 2. Cache management(public/private)
    - 目标：动态管理private/public缓存，避免冗余存储，同时实现快速的prefix-prompt search，插入/删除
    - 思路：
        - 统一管理private/public prefix tree.
        - private node因为只被同一个user使用，不会出现"分叉"；为实现快速search，"逻辑上合并sub-tree"
        - 回收时，考虑到单个用户的request重启 + private reuse周期长的情况，在eviction时并不一次性删除全部子树，而是从leaf node逐级删除 (渐进式eviction)

```shell
        ...
    |private-1 tag=merge ([cache-0, cache-1, cache-2])|
    # |node-1 tag=invalid (private-1)|
    # |node-2 tag=invalid (private-1)|
```

- 3. 针对检测失败的兜底与攻击识别: 
    - 目标：基于模型的private评估可能存在漏网之鱼，需要从机制上兜底隐私保护；在检测到可疑Attacker时，即使止损，阻断privacy泄露的可能
    - 难点：1，需要兼容多账号协同攻击的场景；2，基于频率的变化并不能有效反映攻击，因为request本身的访问pattern就是变化的
    - 思路：基于时间窗口的Monitor + 分布熵判断
        - 即使使用多账号协同攻击，也会造成单个账号的访问频率增加
        - 对于每一个KV-Cache block，记录：hit_cur, u_cnt, hit_pre, u_pre
        - 60s
        - 100次 hit  10  0.1；
        -            50  0.5
        - 每个时间窗口计算：entropy = usr_cnt/hit_cur, entropy低/高表示少数账号占多次访问，高entropy表示访问比较分散
        - 通过直接判断 entropy 判断 少量找好直接攻击的情况； 通过hit_cur >> hit_pre + entropy_now > entropy_pre的方式来判断 多账号协同攻击
        - 根据pre的情况来判断是否需要将当前block升级成private
            - if u_pre 较大，说明当前block被多个user使用，认定为公共前缀，不做修改
            - if u_pre 较小(=1)，说明当前block仅被单一user使用，认定可能包含隐私信息，降级为private

## 修改建议 20250610
- threat model 最后两段不要写，只写攻击者怎么攻击 + 攻击者是什么？
    - 对手的cpacity
    - 对手的目的
    - 对手可能的攻击行为
- related work + motivation 放在一起
    - 两段放在motivation
    - 有一些攻击的工作，已经在攻击了
    - 分析一下现有的防守方式；
- challenges 放到design的第一部分
- design 后面的部分和challenge一一对应
- 可以将challenge 中security部分放到一起，再去讲性能
- NDSS -> 主要强调security
- attack -> access rate