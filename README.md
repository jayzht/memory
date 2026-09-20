# 三层记忆 LLM Agent（批量评测版）

面向 **批量数据集评测与基线对照实验** 的 LLM Agent 三层记忆系统。不是 demo：
所有实验产物（逐样本预测、逐 episode 日志、汇总指标、断点信息）都会落盘，可复现、可断点续跑。

硬性约束（已贯彻到代码）：

* **禁止** 向量 / embedding / 相似度检索 / RAG。记忆只能通过 **ID 精确查询**。
* `MemoryManager` 业务逻辑与存储完全解耦：`InMemoryStore` ↔ `RedisSQLiteHybridStore` 切换不触碰任何压缩 / OVERRIDES 逻辑。
* 每个 episode 独立：`reset()` 清空该 episode 热点数据，**绝不跨样本污染**；评测关键。
* Redis 只做热点缓存，**不持久化**实验数据；归档摘要与全部原始对话永久落 SQLite。

---

## 0. 审查后的重要更正（读实验结果前必读）

一次独立的代码审查发现，**若干已归档的真实实验数字是在有缺陷的代码上跑出来的**。
修复清单见 `REVIEW_FINDINGS.md`；下面是最影响结论的几条：

| 问题 | 影响 |
|---|---|
| **Redis 序列化丢字段**：`_load_summary` 丢 `fact_keys`/`index_id`，`loads_index` 丢 `previews`/`child_index_ids`；SQLite 的 `index_entries` 表根本没有后两列 | 所有真实运行都走 `redis_sqlite_hybrid`。索引标题的 `属性=当前值` digest **恒为空**；索引行回退渲染**整条成员摘要**（实测索引层 474→317 tok/轮，约 1/3 成本虚高）；"顶层不可达"的主因是这个 bug，不是架构 |
| **`_fact_digest` 顺序反了** | 标题把**最旧**的值标成"当前值"，历史也倒着写（`工位=1楼[3楼→2楼→1楼]`）。索引层一直在顶层显示**错误的当前值** |
| **索引条目在覆盖后不变** | `size` 虚高、`expand_index` 返回条数对不上、预览与 digest 仍显示已被取代的旧值 |
| **基线被灌输了它们没有的工具** | `memgpt_style`/`naive_chain` 的 executor 为 None，却收到"必须先调用工具"的提示与工具手册 → 模型输出工具调用，无法执行，**该字符串直接成为它的答案**并被判错。三项基线被系统性压低 |
| **`--system all` 直接崩溃** | `memgpt_style` 的 chain token 是 `None`（故意表示 n/a），进度行却用 `:.0f` 格式化 → 崩溃，且**不产出任何结果文件** |
| **多系统 `predictions.csv` 重复** | 累积行被反复 append（4 系统 × n 题 → 10n 行，权重 4:3:2:1），`analyze_results` 读的就是这个文件 |
| **resume 丢失指标** | 已完成的 episode 被跳过却不回读，`metrics.csv` 只覆盖本次进程跑的部分；首次 flush 还会把 `predictions.csv` 截断 |
| **跨 run 归档污染** | 归档命名空间是 `system/episode`，不含 run id，且 `reset` 保留归档 → 换 run-id 重跑会继承上一次的死数据（实测某 40 轮 cap=0 的 episode 报告 `archived_final=101`） |
| **成本无法归因** | 摘要/裁判与回答共用同一个 client，`summarizer_stats` 是同一个计数器的第二次读取（490/490、token 完全相同） |
| **自行写记忆的块会泄漏进答案** | "回复只有该块"时返回的是**原始带标签文本**；多块时只剥第一个 |
| **合成数据集 History 标准答案错误** | 一个属性变两次会生成两条**措辞相同、gold 不同**的探针，其中一条必然判错 |
| **LongMemEval 两处** | `--lme-types` 配默认 `--limit 50` 返回 **0 条**（先截断扫描再过滤）；轮级截断是纯尾部截断，会把证据会话切掉，而 meta 仍声称保留了 |

**结论**：`results/final_ab`、`results/final_merge` 等**修复前**产物的绝对数字不可直接引用；
结构性结论（"摘要+归档+精确回查"成立、索引优于容量合并、成本需要按 session 粒度优化）方向仍然成立，
但需要重跑才能得到可引用的数字。修复后 `index` 与 `merge` 的对比需要在同一份新代码上重做。

---

## 1. 快速开始

```bash
pip install -r requirements.txt          # 必需：requests；可选：redis, tiktoken, openai, datasets

# ⓪ 用 DeepSeek 跑真实实验（.env 里放 DEEPSEEK_API_KEY 即可，无需其它配置）
python evaluation.py --synthetic --limit 30 --llm-backend deepseek \
    --model deepseek-flash --system all --store redis_sqlite_hybrid \
    --run-id exp_deepseek
python analyze_results.py --results results/main        # 生成对照表 + 图 + 错误拆解

# ① 零依赖冒烟跑（内置合成数据 + 规则离线 LLM + 内存存储，不联网、不需要 Redis）
python evaluation.py --synthetic --dataset-format synthetic \
    --llm-backend heuristic --system all --limit 4 --out-dir results/smoke

# ② 真实实验（Ollama，四套系统，SQLite 断点续跑）
python evaluation.py --dataset data/sample_episodes.json --llm-backend ollama \
    --model qwen2.5:7b --store redis_sqlite_hybrid --system all --run-id exp1

# ③ 崩溃后续跑（同一个 --run-id，已完成的 episode 自动跳过）
python evaluation.py --dataset data/sample_episodes.json --llm-backend ollama \
    --system three_layer --run-id exp1

# ④ 查看某个陌生数据集的 schema（尤其是 MEME 的不同版本）
python evaluation.py --dataset meme_export.json --inspect

# ⑤ 人工调试单条 episode（不需要 Redis）
python chat_debug.py --store memory --llm-backend heuristic
```

---

## 2. 架构

```
                    ┌─────────────────────────── Prompt 拼接顺序 ───────────────────────────┐
                    │ ① 近期原始对话滑动窗口(K 轮逐字)  →  ② 活跃事件摘要链  →  ③ 系统提示词+工具 │
                    └───────────────────────────────────────────────────────────────────────┘
                                      ▲                              ▲
                                      │                              │
┌─────────────────────────────────────┴──────┐        ┌──────────────┴───────────────────────┐
│ ④ 近期滑动窗口  RecentSlidingWindow         │        │ ① 活跃事件链  Active Chain           │
│   最近 K 轮完整原始对话（默认 4）            │        │   每条摘要带 [OVERRIDES: id_list]     │
│   超长淘汰最老一条（原文仍在层③）            │        │   Redis 热数据 + SQLite 镜像          │
└────────────────────────────────────────────┘        └──────────────┬───────────────────────┘
                                                                     │ 事件驱动覆盖 / 容量压缩
┌────────────────────────────────────────────┐        ┌──────────────▼───────────────────────┐
│ ③ 原始详情库  RawDialogRecord               │◀───────│ ② 归档摘要库  ArchivedSummary         │
│   永久保存完整原始对话（SQLite）             │ raw_ref│   被覆盖 / 被容量淘汰的摘要，永不删除   │
│   只被 reference_id 精确读取                │        │   只被 summary_id 精确读取（SQLite）  │
└────────────────────────────────────────────┘        └──────────────────────────────────────┘
```

* **层① 活跃事件链**：进 LLM 上下文。每条摘要强制携带 `[OVERRIDES: id1,id2]` / `[OVERRIDES: none]`，
  并可携带机器可读的 `[FACTS: 属性=取值; ...]` 行（用于"每个属性的最新事实不被容量压缩吃掉"，见 §7.2-1）。
* **层② 归档摘要库**：`is_overridden=True` 永久保存，按 `summary_id` 精确读取（SQLite）。
* **层③ 原始详情库**：完整原始对话永久保存；摘要只存 `raw_ref_id` 引用（SQLite）。
* **层④ 近期滑动窗口**：最近 `recent_window_turns`（默认 4）轮逐字对话，拼在 prompt 最前端，减少不必要的工具调用；被淘汰的记录仍在层③。

### 2.1 两套压缩逻辑（互不混淆，可分别归因）

| | 触发 | 动作 | OVERRIDES | 归档标记 | 统计字段 |
|---|---|---|---|---|---|
| **事件驱动更新** | 本轮新事实与活跃链旧事实**冲突/更新**（同对象同属性取值变化） | 旧摘要移出活跃链进归档，新摘要留在活跃链并列出被覆盖 id | `[OVERRIDES: id_list]` | `is_overridden=True`, `superseded_by=<新id>`, `archive_reason="overridden"` | `overrides_events`, `overridden_summaries` |
| **容量驱动压缩** | 活跃链 token > `ACTIVE_CHAIN_TOKEN_LIMIT` | 选取**早期无冲突**多条摘要合并为高阶摘要放入活跃链，原摘要进归档 | **不生成 OVERRIDES 标签** | `is_overridden=False`, `superseded_by=None`, `archive_reason="capacity"` | `capacity_compressions` |

实现约束（保证指标可归因）：

* 两种压缩机制在 CSV 里用**独立字段**统计（`overrides_events` / `capacity_compressions`），且容量合并永不产生 OVERRIDES 标签。**但两者并非互斥**：一轮里覆盖了旧摘要、同时又把别的摘要归档以压回预算，是完全正常的，该轮的 `event_triggered` 与 `capacity_triggered` 会同时为真。（早期文档声称有"覆盖当轮不再压缩"的护栏，代码里从来没有；补上它只会让指标更难解释。）
* 容量压缩 **不会合并本轮刚产生的新摘要** → 最新事实始终可单独读取。
* 摘要器幻觉出的 id（不在活跃链中）被 **丢弃并计数**（`invalid_override_ids`），绝不污染归档图谱。

### 2.2 两个 Agent 工具（纯文本 function call）

```
get_archived_summary(summary_id="<id>")                          # 按 id 精确读取归档摘要（SQLite）
get_raw_record(reference_id="<ref>")                             # 按 id 精确懒加载完整原始对话（SQLite）
get_predecessor_summary(fact_key="<属性>", exclude_value="<值>")   # 按属性精确取"最近的旧值"
```

`get_predecessor_summary` 是为"被覆盖的历史值"专门加的**精确属性查询**：它在归档里按时间倒序找该属性的最近一条，
也就是当前值的直接前驱。没有它，模型只能在几十条归档 id 里猜哪条是前驱——长 episode 实测历史准确率因此从 0.043 提到 0.565。
它仍然是 ID/属性精确匹配，不涉及向量或相似度排序。

解析器（`memory3l/tools.py`）支持 `key="v"` / `key=v` / `key: 'v'` / JSON / **DeepSeek DSML 标记** 形式与多种参数别名；
解析失败、id 不存在、工具异常都会被结构化记录，直接进入 `Tool_Call_Success_Rate`。
`get_archived_summary` 还会**沿 `superseded_by` 链前向回溯**，把"现在被谁取代、当前值是什么"一并返回——
否则模型只能用旧值回答"原值"问题，History_Fact_Acc 无法测量。

### 2.3 数据模型

```python
ActiveSummary(summary_id, text, override_ids: list[str], timestamp, raw_ref_id)
ArchivedSummary(summary_id, text, override_ids, timestamp, raw_ref_id, is_overridden)
RawDialogRecord(reference_id, user_msg, agent_msg, timestamp)
```

在保持题设字段的前提下补充了少量可审计字段（均有默认值，不改变语义）：
`superseded_by`（覆盖链前向指针）、`raw_ref_ids`（容量合并需要多引用）、
`episode_id / seq / origin / merged_from / archive_reason`（评测与调试所需的溯源信息）。

---

## 3. 存储层

| 实现 | 用途 | 说明 |
|---|---|---|
| `InMemoryStore` | 快速调试单条 episode | 零依赖，`chat_debug.py` 默认 |
| `RedisSQLiteHybridStore` | 正式实验 | Redis 热 + SQLite 冷 |
| `RedisHotStore` | Redis 热点封装 | 命名空间、SCAN 清理、可降级 |
| `SQLiteColdStore` | SQLite 冷持久化 | 自动建表、WAL、断点信息 |

本机启动 Redis（任选其一）：

```bash
docker run -d --rm -p 6379:6379 --name mem-redis redis:7-alpine   # 最省事
sudo apt install redis-server && redis-server --daemonize yes     # 系统包
```

若当前 Python 环境的 site-packages 只读，可把可选依赖装到项目内目录（`config.py` 会自动加入
`sys.path`，无需改环境变量）：

```bash
pip install --target ./.pylibs redis        # 之后 hybrid 存储会走真实 Redis 热点层
```

**Redis 只存当前 episode 的热数据**（key 前缀 `episode:{episode_id}:*`）：

```
episode:{episode_id}:active_ids   LIST  summary_id 顺序（新→尾）
episode:{episode_id}:active       HASH  summary_id -> JSON(ActiveSummary)
episode:{episode_id}:window       LIST  JSON(RawDialogRecord)
episode:{episode_id}:meta         HASH  轮次等簿记
```

**SQLite 六张表**（自动建表）：

| 表 | 内容 |
|---|---|
| `raw_records` | 全部原始对话（永久） |
| `archived_summaries` | 全部归档摘要（永久，含 override_ids / superseded_by / archive_reason） |
| `active_summaries` | 活跃链**镜像**，仅用于崩溃恢复 |
| `sliding_window` | 滑动窗口**镜像**，仅用于崩溃恢复 |
| `episode_progress` | 批实验断点：`(run_id, episode_id, status)` |
| `experiment_runs` | 每个 (run, system, episode) 的结果行 |

关键语义：

* **`reset(episode_id)` 是不对称的**：删除该 episode 的 Redis 全部 key + 清空活跃链/窗口镜像，
  但 **保留** 该 episode 的归档摘要与原始记录。否则 agent 查不到自己被覆盖的历史，History_Fact_Acc 无从测量。
* **Redis 掉线不丢数据**：所有写先落 SQLite；Redis 不可用时自动降级（`redis-py` 未安装/服务未启动也照跑），
  并可从 SQLite 镜像 `rebuild_hot_state()` 恢复热点。诊断计数写入结果文件（`store_redis_available` / `store_fallbacks` / `store_resyncs`）。
* 清理用 **SCAN + UNLINK**，绝不使用 `KEYS`。

---

## 4. 评测：四套系统对照

| system | 说明 |
|---|---|
| `three_layer` | **本方案**：三层 + OVERRIDES + 滑动窗口 + Redis/SQLite，具备两个归档工具 |
| `full_context` | Baseline 1：Plain Full Context，全部原始对话直接进 prompt（超预算从最老开始截断） |
| `memgpt_style` | Baseline 2：MemGPT 风格**破坏性压缩**，滚动摘要替换并**删除**旧原文，无归档、无工具 |
| `naive_chain` | Baseline 3：朴素时序摘要链，只追加、超预算丢弃最老摘要，无归档、无 OVERRIDES、无工具 |

公平性处理（写进报告前请先读）：

* 四套系统共用同一个 LLM、同一个回答契约、同一个工具循环框架（`agents/base_agent.py`），差异只来自记忆机制。
* 四套系统都保留同样的近期窗口；只有 `three_layer` 拥有归档工具——这正是对照点：基线设计上就没有可查的地方。
* `Tool_Call_Success_Rate` 对无工具系统记为 `n/a`（而不是 0 成功），避免把"设计上没有工具"算成失败。
* baseline 的回答质量在 **离线 heuristic 后端下不代表真实能力**（它是规则匹配，不擅长跨语言/语义对齐）。
  heuristic 的用途是验证**管道**：覆盖是否发生、归档是否可查、工具是否被正确调用、指标是否正确统计。

### 4.1 指标定义

| 指标 | 定义 |
|---|---|
| `Current_Fact_Acc` | `current_fact` 探针正确率（问当前最新值） |
| `History_Fact_Acc` | `history_fact` 探针正确率（问被覆盖的历史值） |
| `Avg_Active_Chain_Tokens` | 每 episode "记忆链" token 均值。**口径**：`three_layer` 只计入活跃摘要**正文** token（id/标签/raw_ref 不计），即题设指标的严格定义；`full_context` 记全部原始上下文，`memgpt_style` 记滚动摘要，`naive_chain` 记摘要链文本——三者的"记忆"本身就是这些内容。同时输出 `Avg_Active_Chain_Tokens_rendered`（`three_layer` = 活跃链渲染行 + 滑动窗口的实际上下文开销），便于做**同口径上下文成本**对照。解析器用 `tiktoken:cl100k_base`，无 tiktoken 时回退内置字符类估算器 `heuristic:v1`，口径记录在 `run_metadata_*.json` |
| `Tool_Call_Success_Rate` | 成功执行且解析到目标 id 的工具调用 / 工具调用尝试次数（另有 `Tool_Call_Parse_Rate` 区分"格式对"与"id 对"） |

### 4.2 数据集

1. **自制 JSON**（主用，便于受控实验）——`data/sample_episodes.json` 即为示例：
   ```json
   {"episodes": [{
     "episode_id": "demo_zh_1",
     "dialogues": [{"user": "我的工位在3楼。", "agent": "好的"},
                   {"user": "我的工位改到7楼了。", "agent": "已更新"}],
     "facts": {"office floor": [["0", "3楼"], ["3", "7楼"]]},
     "probes": [
       {"type": "current_fact", "fact_key": "office floor", "question": "我现在工位在几楼？", "answer": "7楼"},
       {"type": "history_fact", "fact_key": "office floor", "question": "在改成7楼之前呢？", "answer": "3楼"}
     ]
   }]}
   ```
   逐轮对话也接受 `["user","agent"]` 二元组形式，probes 可省略（只跑记忆行为，不做 QA 计分）。
2. **MEME**（`meme-benchmark/MEME`）：字段别名表见 `memory3l/dataset.py::MEME_ALIASES`，加载器对字段命名宽容；
   本地导出直接 `--dataset path.json`，或用 `--dataset-name meme-benchmark/MEME`（需 `datasets` 包与网络）。
   **务必先 `--inspect`** 确认 schema 与探针类型识别结果。
3. **内置合成集**（`--synthetic`）：受控事实更新，自带"改前值"标注与答案，
   是唯一能让 Current / History 两类指标**可分离**的数据构造，也是单元测试的基础。

### 4.3 动态 memory-point 探针（让工具指标非空）

数据集往往不写具体 summary_id，于是归档工具永远不会被调用、`Tool_Call_Success_Rate` 恒为空。
`--memory-point-probes N`（默认 1）在对话灌入之后动态生成探针：

* 支持工具的系统：指出某条**已离开活跃链**的 `summary_id`，问它写了什么，gold 即其正文 —— 模型无法猜，必须调工具；
* 不支持工具的系统（如破坏性基线）：指出某条原始 `reference_id`，问那轮原话 —— 记录已被删除就无法回答，这是真实测量而非惩罚。

### 4.4 输出产物

一次实测（合成集 20 episode，离线 heuristic 后端，SQLite 混合存储；**这是管道验证，不是模型能力结论**）：

```
system          CurAcc  HisAcc  ChainTok   CtxTok  Arch  Ovrd  ToolOK
three_layer       0.85  0.8219      74.1    373.6  3.65    73     1.0
full_context       0.0     0.0     306.4    306.4     0       0     n/a
memgpt_style    0.3667     0.0       0.0     89.1     0       0     n/a
naive_chain     0.3667     0.0     216.0    204.5     0       0     n/a
```

```
results/
├── predictions.csv            # 逐 (system, episode) 指标行
├── probe_predictions.csv      # 逐样本预测（问题/gold/预测/判定方式/工具数/耗时）
├── metrics.csv                # 汇总指标表（每系统一行）
├── episode_logs.jsonl         # 每个 episode 的完整日志：逐轮 stats + 逐探针结果 + 记忆终态
├── run_metadata_<run_id>.json # 本次运行的完整配置快照 + LLM/裁判统计
└── logs/evaluation.log
```

断点信息写入 SQLite（`episode_progress`），`--run-id` 相同即续跑；`--force` 忽略断点重跑。
存储层用 `<system>/<episode_id>` 命名空间隔离，四套系统共用同一个 SQLite 也不会互相读串归档
（摘要 id 是确定性的，不加前缀会撞车——这一点对评测正确性很关键）。

---

## 5. 配置

全部配置在 `config.py`，且都可用环境变量或 `.env` 覆盖（`--flag` 优先级最高）。常用项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `STORE_BACKEND` | `memory` | `memory` / `redis_sqlite_hybrid` |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `127.0.0.1` / `6379` / `0` | Redis 热点层 |
| `REDIS_KEY_PREFIX` | `episode:{episode_id}` | 必须含 `{episode_id}` 占位符 |
| `SQLITE_PATH` | `./exp_memory.db` | 冷持久化 + 断点 |
| `RECENT_WINDOW_TURNS` | `4` | 滑动窗口轮数 |
| `ACTIVE_CHAIN_TOKEN_LIMIT` | `800` | 活跃链容量压缩阈值 |
| `CAPACITY_MIN_MERGE` / `CAPACITY_MAX_MERGE` | `2` / `4` | 容量合并批量 |
| `RAW_CONTEXT_TOKEN_LIMIT` | `4000` | 原始上下文预算（baseline1 截断点 / baseline2 压缩触发） |
| `MAX_TOOL_ITERATIONS` | `4` | 单题工具调用上限 |
| `LLM_BACKEND` / `MODEL_NAME` / `TEMPERATURE` | 自动探测 / 见下 / `0.0` | 回答模型。设了 `DEEPSEEK_API_KEY` 就默认 `deepseek`，否则有 `OPENAI_API_KEY` 用 `openai`，都没有才回退 `ollama` |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | `.env` / `https://api.deepseek.com` / `deepseek-flash` | DeepSeek（OpenAI 兼容，仅用 requests，无需 SDK）。`deepseek-v4-pro` 更强但更慢更贵 |
| `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` | `3` / `2.0` | 429/5xx/断连的指数退避重试，避免限流丢样本 |
| `SUMMARIZER_FORMAT_RETRY` | `true` | 摘要器漏写 `[OVERRIDES]` 标签时追加一次格式提醒（是否触发计入 `format_retries`） |
| `CAPACITY_MIN_GAIN` | `0.15` | 容量合并必须让活跃链至少缩小 15%，否则拒绝合并（计入 `capacity_merges_rejected`）。真实模型确实会产生"越合并越大"的输出 |
| `STRICT_OVERRIDE_IDS` | `true` | 丢弃不在活跃链中的（幻觉）override id |
| `SELF_WRITE_MEMORY` | `true` | 让**回答的那次调用**顺便写本轮摘要（`<MEMORY_UPDATE>` 块），一轮只花 1 次 LLM 调用；块缺失时自动回退到专用摘要器。见 §5.5 |
| `JUDGE_MODEL_NAME` | 空（复用回答模型） | 裁判模型，可用更强模型单独判分 |
| `SUMMARIZER_BACKEND` / `SUMMARIZER_MODEL_NAME` | 空 | 摘要可用更小的模型 |
| `DATASET_PATH` / `OUTPUT_CSV_PATH` | 见文件 | 数据与输出 |
| `JUDGE_BACKEND` | `auto` | `auto` / `llm` / `string` |
| `LOG_LEVEL` | `INFO` | 批量跑保持 INFO，`DEBUG` 逐轮日志 IO 很重 |

---

## 5.1 可视化页面：看见每一次提问走了哪条路径

```bash
python web_ui.py --llm-backend deepseek --model deepseek-flash --num-episodes 3 --synthetic-turns 30
# 打开 http://127.0.0.1:8010
```

页面围绕一个 JSON 契约（`GET /api/state`、`POST /api/load`、`POST /api/ask`）把**检索路径**摊开：

* **四层记忆面板**：活跃链（含 `[OVERRIDES]`、`[FACTS]`、raw_ref）、滑动窗口、归档库（标注
  "被覆盖 / 容量淘汰 / 被谁取代 / 合并自几条"）、原文库，逐条可展开。
* **检索路径追踪**：每一步显示模型原始输出、解析出的工具调用、每个工具的 `path`
  （例如 `anchor=7楼 -> recorder s017 -> it overrode s002`），以及**候选表**——
  工具考虑过哪些摘要、各自取值、哪个被选中/落选、理由是什么。这一步是回答"为什么找错"的关键。
* **标准答案核对（本地、不需模型）**：点数据集探针会自动带上 gold 答案，页面直接算出
  gold 值在四层里各出现在哪里，并给出结论："值存在于记忆中——答错就是检索路径问题，不是信息丢失"。
* **本轮日志**：灌 episode 时逐轮的覆盖/压缩事件；提问后显示 LLM 调用数、工具命中率、token、耗时。

写这个页面的过程中它就抓到了两个真 bug：`get_predecessor_summary` 的锚点方向写反
（永远返回最新值）、以及 `_slot_value` 在自然语言摘要上把整句当取值。修复后同一个问题
从"2 次工具调用、第 1 次选错"变成"1 次调用直接命中"。

## 5.2 层级设计（索引层）：实现与实测

这是"目录式记忆"的落地：摘要不再被改写合并，而是在其上**加一层标题/索引**。

```
⓪ 索引层 INDEX_LAYER      - <index_id> [INDEX turn 5-8, 3 entries] 会议时间9点→下午2点；团队人数6人
                              └ 预览 1 条下级目录项
① 活跃摘要链（未收纳）     - <summary_id> [OVERRIDES: ...] (raw_ref: ...) 正文
①b 已收进索引的下级摘要    仍然按 summary_id 可读（工具 expand_index / get_archived_summary）
② 归档摘要库               被覆盖的旧摘要，永久
③ 原始对话库               全部原文，按 reference_id 精确取
```

* 检索路径 = **标题 → `expand_index(index_id)` 拿到目录项 → `get_archived_summary` / `get_raw_record` 钻到原文**；
  索引只是"指过去"，**从不替换或删除下级摘要**。
* 建索引时机：链条渲染 token 超预算且未收纳摘要 > `INDEX_KEEP_RECENT` 时，取最老的 `INDEX_GROUP_SIZE` 条建一个目录项。
* 开关：`CHAIN_STRATEGY=index`（默认）/ `merge`（旧的"多条合并成一条"，保留用于对照）；
  `--chain-strategy index|merge`、`--index-keep-recent`、`--index-group-size`。
* 页面已在"记忆四层"里显示 ⓪ 索引层与 ①b 已收纳摘要。

### 实测对照（40 轮中文长 episode ×5，DeepSeek-flash，活跃链预算 400 token）

| 指标 | `index`（你的层级） | `merge`（旧合并） |
|---|---|---|
| **Current_Fact_Acc** | **0.925** (37/40) | 0.800 (32/40) |
| **History_Fact_Acc** | **0.970** (32/33) | 0.909 (30/33) |
| 渲染链 token / 轮 | 2619 | **476** |
| 若不分层、全平铺 / 轮 | 8824 | 1072 |
| 归档摘要 / 轮 | 24.2 | 74.4 |
| 容量合并 / 轮 | 0 | 24.6 |

**诚实的解读**（我之前的预测是"层级能提升历史召回且更省 token"，只对了一半）：

* ✅ 精度两个指标都更好：当前事实 +12.5 个点，历史事实 +6.1 个点。理由是合并会把一个属性的多段
  历史揉成一条混合摘要，而索引让每段历史都保持独立可取。
* ❌ **并没有更省 token，反而贵 5.5 倍**。原因是：被收进索引的下级摘要**仍然算作活跃摘要**，
  继续占用预算（`active_chain_tokens()` 目前把"已收纳但未展开"的条目也算在内），于是系统被推着
  不断建新索引。**当前实现的价值在精度，不在成本；"标题化省 token"还没兑现。**

要让"省 token"也兑现，需要三件事（尚未做）：

1. **按实际渲染物计预算**：被收纳的下级摘要不计入 `active_chain_tokens()`，只在 `expand_index` 时才付费。
2. **懒展开渲染**：链条涨大后，prompt 只给索引层 + 最近 `KEEP_RECENT` 条，其余全部靠工具按需展开。
3. **递归索引**：索引条目数超阈值时再往上加一层，形成真正的多级目录。

### 5.3 懒展开（v2：已修正；它本身不省 token，省 token 的是"挂到标题下"）

前两版的问题是"**收纳了但没省**"：被标题收录的摘要仍然逐条渲染，于是链条超预算 → 索引继续建 →
标题和预览反而变成新的开销（最坏一次跑出 800 个索引条目、1 万多 token 的 prompt）。

> **⚠️ 更正（代码审查后）。** 本节原先称懒展开是"真正把成本压下去的那一步"。审查发现：
> 1. `chain_summaries()` 从来没有排除 `LAZY_INDEX_ID`，所以"懒存"的摘要**一直照常渲染**——契约没实现；
> 2. 判定"两跳成员"的逻辑读的是**已经被折叠删除**的子目录 id，恒为空集；
> 3. 即使修好，它对渲染也**没有增量贡献**：`chain_summaries()` 按 live 目录项的 `members` 过滤，
>    而超级索引的 `members` 是传递闭包，两跳成员本来就不渲染。
>
> 也就是说：**已归档实验结果里 `lazy_summary_count` 全部为 0，懒展开从未生效过**，
> 那段 A/B 里 `index` 比 `merge` 省下的 token，全部来自"挂到标题下 + 超级索引折叠"。
> 现在 `chain_summaries()` 已排除 `LAZY_INDEX_ID`，covered 判定改为读 live 超级索引自身，
> 并把"把**未归档**摘要转成懒存"这条分支删掉了——那种摘要没有任何指针，藏起来就是真的不可达。
> 懒展开现在的定位是**两跳成员的可观测簿记**，不是成本机制。

现在的渲染契约（三分法，**每一层都可按 id 取回**）：

| 状态 | 是否进 prompt | 怎么取回 |
|---|---|---|
| 被标题引用（`index_id` 指向某个目录项） | **不渲染**（只在目录里占一行标题） | `expand_index(index_id)` 展开该目录，或直接 `get_archived_summary(summary_id)` |
| 未进任何目录 | 渲染（这就是"活跃链"的本义） | 直接可见 |
| 懒存（`index_id = __LAZY_SUMMARIES__`，即所属标题已被上层折叠） | 不渲染（v2 起才真正成立） | 按 `summary_id` 精确取回 |

配套的稳定性护栏（上一轮跑飞的直接教训）：

* `INDEX_MAX_PER_TURN`（默认 2）：**每轮最多新建 2 个目录项**，硬上限，杜绝失控循环；
* 标题硬上限（`IndexEntry.MAX_TITLE_CHARS`，默认 60 字）：标题是"指路牌"不是内容；
* **成本 gate**：索引条目成本 ≥ 被替换摘要成本就拒绝（`INDEX_MAX_PER_TURN` + `index_rejections` 可观测）；
* **token 单调性检查**：收纳后渲染成本没下降就停手并告警（`no token progress`），而不是继续堆；
* 超级索引折叠后**重指成员归属**（否则成员会因"旧 index_id 已不存在"而重新变成未收纳，成本反弹）。

离线实测（40 轮中文 episode ×4，`heuristic` 后端，预算 400 token）：

| 指标 | 数值 |
|---|---|
| Current_Fact_Acc | 0.188（离线规则后端对中文问句的抽槽能力有限，**不代表架构**） |
| **History_Fact_Acc** | **0.962**（26 题 25 对） |
| 渲染链成本 | **167 tok/轮**（不做分层要 2378 tok） |
| 上下文成本 | 502 tok/轮 |
| 目录项 / 懒存 | 2.0 / 0.5 条每轮 |
| 契约：全部摘要按 id 可取回 | 28/28 ✅ |

> 注：`Current_Fact_Acc` 在离线后端下偏低是**评测器的限制**（它靠正则抽属性名，中文自然语言问句
> "我现在的工位楼层是什么？" 抽出的槽位与摘要里的规范槽位对不上）。真实模型下该指标此前测得 0.958。
> 两个后端的用途不同：离线跑得快、用来验证机制与成本；真实模型用来出结论。

### 5.4 真实验证时发现的两个膨胀问题（一个已修，一个待修）

懒展开在离线环境已验证（见 5.3），但**真实模型下暴露了两个失控问题**，必须记录清楚：

**问题 A（已修）：摘要自我复述**
真实模型会把"看到的整条活跃链"复述进新摘要，逐轮自我叠加。跑到第 21 轮时，
摘要器的请求达到 **184 万 token**，超过模型上限（1M）持续报 400，而流程没有停。

两道防线（互相独立）：

* `MAX_SUMMARY_OUTPUT_TOKENS`（默认 220）：单条摘要硬截断，超长计数 `summary_truncations`；
* `SUMMARIZER_MAX_INPUT_TOKENS`（默认 3000）：喂给摘要器的链条容量上限，超了只给最新几条
  （覆盖判定仍用**全量**链条，所以裁剪只影响"展示"不影响"正确性"）。

修复后 `maximum context length` 报错降为 **0**。

**问题 B（已修）：Redis 写放大导致"内存链条"虚高**

现象：同一轮内预算循环反复迭代，日志出现 `chain is 4,354,084 tokens, showing the newest
37/56868 entries`，而 SQLite 里只有 21 行摘要——内存看到的规模与库里数据不一致。

定位过程（这次靠新加的完整性断言一次抓准）：

```
ERROR integrity at fill_counts: store returned 610129 summaries
      but only 39 distinct ids (610090 duplicates)
```

根因：**`RedisHotStore.add_active_summary` 用的是无条件 `RPUSH`**。而管理器会**合法地重写**
同一条摘要（懒存扫描要改它的 `index_id`），于是 Redis 的 id 列表里同一条被反复追加；
读取时 `HMGET` 按重复 id 逐条取回，**读放大**成 61 万条。

> 一个关键教训：之前一直以为"Redis 不可用、已降级到 SQLite"。实际上 `config.py` 会把 `.pylibs`
> 加进 `sys.path`，**Redis 从那一刻起就是通的**——所以 `--store redis_sqlite_hybrid` 一直在走
> 真实 Redis 路径，这个写放大 bug 也因此一直被真实地触发，只是被误判成"循环失控"。

修复（两处，写入治本 + 读取兜底）：

1. `RedisHotStore.add_active_summary` 改为**幂等 upsert**：先 `LREM` 再 `RPUSH`，
   保证"一个 summary_id 占一个槽位"，同时保持顺序（新→尾）；
2. `RedisHotStore.list_active_summaries` 读取时**按 id 去重**并告警，历史遗留的重复列表
   不会再被放大；
3. 新增 `STRICT_INTEGRITY` 开关 + `MemoryManager._integrity_check()`：
   O(n) 校验"管理器视图 == store 视图"，发现重复或单条超限时**打印调用栈**。
   这类静默不一致以后不会再靠猜。

**修复后验证（真实 DeepSeek，40 轮中文 episode ×1）**：

| 项 | 结果 |
|---|---|
| 处理轮数 | 40/40（此前在同一场景失控） |
| `maximum context length` 报错 | **0** |
| 完整性告警 / 循环停止 / 摘要截断 | **0 / 0 / 0** |
| Current_Fact_Acc / History_Fact_Acc | **8/8 · 8/8** |
| 渲染链 token | **141.6**（预算 400 内） |
| 工具调用 | 22/24 命中 |
| 耗时 | 111.7 秒/ episode |

**结论**：问题 A（摘要自膨胀）+ 问题 B（Redis 写放大）修复后，真实模型下长 episode 已经稳定，
可以重新用于批量评测。

### 5.5 一次调用同时"回答 + 写摘要"（self-write）

**问题**：原来一轮对话要花 **两次** LLM 调用——一次回答用户，一次专门让摘要器总结这一轮。
摘要器调用占了这个设计的主要成本（494 轮的 LongMemEval haystack = 494 次额外生成）。

**做法**：让**回答的那一次调用顺便把摘要写了**。回答调用本来就已经同时拥有"这一轮新对话"
和"当前活跃链"，它完全有条件输出同样的摘要语法；只要在 system 里加一段自写指令，要求它在
**本轮最后一次回复**的末尾附上一个块：

```
<MEMORY_UPDATE>
一句话摘要
[FACTS: 属性=值; 属性2=值2]
[OVERRIDES: s003]
</MEMORY_UPDATE>
```

解析用**同一个** `parse_summary_response`，落库走**同一条** `_apply_generated` 链路
（override 解析 → 归档 → 索引维护 → token 记账）。唯一的差别是 `_finalise_generated()` 这个
共享尾巴：两个来源（专用摘要器 / 自写块）在这里被归一成同一种"生成结果"，所以 override 图
**不关心摘要从哪来**。

三个关键设计点：

1. **块在 system role 里声明**：工具循环会追加消息，指令必须活到最后一轮，不能放在 user 里。
2. **块与答案分离**：`split_selfwrite_reply()` 把块从可见答案里剥掉——用户、judge、CSV 看到的
   只有答案正文，机器可读载荷不会污染答案（这是最容易被忽略、但会直接毁掉评测的一点）。
3. **机会式收割 + 回退**：`_apply_generated` 之前先校验块（`harvest_selfwritten()`），
   **校验通过才写 raw record**。块缺失/不可解析 → 自动退回原来的专用摘要器调用；
   所以"模型不听话"只会多花一次调用，**不会丢一次记忆更新**，也不会写重复的 raw record。

**真实 DeepSeek 验证**（`deepseek-chat`，内存存储，2 轮）：

| 项 | 结果 |
|---|---|
| 两轮对话 LLM 调用数 | **2**（自写路径）vs 4（原路径） |
| 第 1 轮 | `self_written=True`，`[FACTS: 居住城市=上海; 宠物猫名字=毛毛]` |
| 第 2 轮（改城市） | 模型自写 `[OVERRIDES: s001]` → 短 id 正确映射为规范 id，s001 归档、s002 上链 |
| 答案泄漏 `MEMORY_UPDATE` | 无 |
| 回退路径单测 | 块缺失 → 仍调用摘要器，活跃链/归档完全一致，raw record **恰好 1 条** |

适用边界要说清楚：**批量评测集（LongMemEval / MEME）里每一轮的 assistant 回复本来就是数据集
给的**，没有"回答调用"可以搭载，所以这条路省不了 ingestion 的钱——它省的是**真实交互**
（`chat_debug.py` / 可视化页面的 `/api/turn`）的那一半调用。批量评测要降本得从别处下手
（按 session 粒度摘要、只对含事实的轮次调模型，见 §8）。

开关：`config.SELF_WRITE_MEMORY`（环境变量同名，默认开）；`chat_debug.py --self-write/--no-self-write`。
指标里新增 `self_written_turns` / `self_write_fallbacks`，回退率是可观测的。

### 5.6 当前值登记表（`<CURRENT_VALUES>`，新增）

索引标题本来是"当前值"的唯一顶层出口，但它会截断、按组重算、还会在成员被覆盖后过期——
真实数据里的决定性失效模式（值只存在于某个索引成员里）正是这么来的。

登记表把"当前值"从标题的职责里摘出来：从**活跃摘要已有的 `fact_keys`** 派生
"每个槽位的最新值"，固定渲染在滑动窗口之后、索引层之前：

```
<CURRENT_VALUES>
工位楼层=12楼; 会议时间=下午2点
</CURRENT_VALUES>
```

* **零额外 LLM 调用**：`fact_keys` 本来就存在摘要上；
* **不会过期**：被覆盖/被合并的摘要自然不再贡献（它是派生的，不是增量维护的第二份真相）；
* **成本**：约 10~15 tok/槽位，实测 2 个槽位 10 tok；上限 `CURRENT_VALUES_MAX_SLOTS`（默认 12），
  超出时保留**最近更新**的槽位；
* **开关**：`CURRENT_VALUES_ENABLED`（默认开）；成本单独记在 `current_values_tokens` /
  `avg_current_values_tokens`，也计入渲染口径（`Avg_Active_Chain_Tokens_rendered`）与预算触发。

它同时也是"容量合并/索引归档不得吃掉最新事实"这条规则的结构性保障：
即使某条摘要被归档或折叠，它的**当前值**仍然在顶层可见。

### 5.7 指标口径（v2 修正）

* `Avg_Active_Chain_Tokens`：四个系统现在都上报**每轮均值**，且都是**实际渲染**的量。
  旧版把 `three_layer` 的每轮均值、`naive_chain` 的终态总量、`full_context` 的**未截断**存储量
  混在同一列里比较。
* `Avg_Active_Chain_Tokens_rendered`：渲染口径（three_layer = 链 + 索引 + 登记表 + 滑动窗口）。
* `Memory_Point_Acc` / `_n`（新增）：`MEMORY_POINT` 探针的答题正确率。此前只累计到每 episode 的
  CSV，从未聚合，"工具调用成功但归档内容答错"在总表上没有任何体现。
* `Tool_Call_Unparsable` 的 per-probe 列此前恒为 0（日志条目没有 `parsed` 键）。

---

## 6. 项目结构

```
config.py                     # 全部可配置项（环境变量/.env 可覆盖）
evaluation.py                 # ★ 主实验入口：批量跑数据集、多基线、指标、CSV、断点续跑
chat_debug.py                 # 辅助：命令行交互式调试（默认内存存储，不需要 Redis）
web_ui.py + web_ui.html       # 可视化：索引层/四层记忆 + 每次提问的检索路径追踪 + gold 值核对
memory3l/
├── models.py                 # 数据模型 + ToolCall/ToolResult/MemoryTurnStats
├── memory_manager.py         # ★ MemoryManager：三层维护、事件覆盖、容量压缩
├── prompts.py                # 摘要/合并/Agent/裁判 prompt，上下文拼接顺序
├── llm.py                    # LLM 后端：ollama / openai / heuristic(离线) / scripted(测试)
├── tools.py                  # 文本 function call 解析 + 两个归档工具执行器
├── token_utils.py            # token 估算（tiktoken 优先，字符类估算回退）
├── dataset.py                # 数据集加载：自制 JSON / MEME / 合成 + schema 诊断
├── agents/
│   ├── base_agent.py         # 统一工具循环、指标采集、Agent 工厂
│   ├── three_layer.py        # 本方案（MemoryManager 适配器，逻辑全在 manager）
│   └── baselines.py          # 三个基线系统
└── store/
    ├── base.py               # BaseMemoryStore 抽象 + InMemoryStore
    ├── redis_store.py        # Redis 热点层（命名空间/SCAN 清理/可降级）
    ├── sqlite_store.py       # SQLite 冷层（六张表、断点、结果落库）
    └── hybrid_store.py       # 混合存储（SQLite 为真源，Redis 可丢）
tests/
├── test_core.py              # 解析、覆盖语义、容量压缩、窗口、工具循环
└── test_store.py             # episode 隔离、Redis 命名空间(假客户端)、SQLite 持久化
data/sample_episodes.json     # 手写示例数据集（中英、含更新与无更新对照）
```

运行测试（无需网络 / Redis / 模型）：

```bash
python -m unittest discover -s tests -v     # 67 tests
```

---

## 7. 设计评审：原方案的 6 个坑 & 本实现的处理

这些是真跑起来才会暴露的问题，都会直接影响评测结论，建议在技术报告里显式记录：

1. **`get_archived_summary(A)` 只返回旧值，会把模型带到沟里。**
   多跳覆盖 A→B→C 时，问"原来是什么"得到 A 的值没问题，但问"现在是什么"若误查 A 就答错。
   → 归档摘要在覆盖时写入 `superseded_by`，工具返回时沿链前向回溯并附注"当前值来自哪条"。

2. **`ArchivedSummary.raw_ref_id` 单字段撑不住容量合并。**
   合并 N 条摘要必须引用 N 条原文。→ 增加 `raw_ref_ids`，`raw_ref_id` 保留为首个引用（兼容题设字段）。

3. **容量压缩的输入选择没定义，且会误伤最新事实。**
   → 规则：优先选"早期、未携带覆盖关系"的摘要，从不吞并本轮新摘要；`CAPACITY_MIN/MAX_MERGE` 控制批量；
   若模型仍输出 `[OVERRIDES]`，强制剥离（容量合并按定义不产生覆盖标签）。

4. **`reset()` 语义冲突：清热点 vs 工具查历史。**
   若 `reset` 连归档一起删，History_Fact_Acc 直接不可测。
   → `reset(episode_id)` 只清该 episode 的**热点**（Redis + 活跃链/窗口镜像），归档与原文在 SQLite 永久保留。

5. **多系统共用一个数据库会互相读串归档。**
   摘要 id 是确定性的（episode+序号），四套系统会生成同名 id。
   → 存储命名空间 `<system>/<episode_id>`，`reset` 与 `get_archived_summary` 均按该前缀隔离（写入 CSV 的 `episode_id` 仍是原始 id）。

6. **激活链的 token 监控口径不明，且 tokenizer 不稳定。**
   → 双指标：`active_chain_tokens`（渲染行，含 id/标签/raw_ref，用于**触发**压缩）与
   `active_chain_text_tokens`（仅正文，用于**上报** `Avg_Active_Chain_Tokens`）；tokenizer 名称写入元数据；
   字符类估算器保证单调性，压缩触发稳定可复现。

### 7.1 DeepSeek 实测：两个数据集、两种结论

用 `deepseek-flash` 跑了两组数据（Redis 热点 + SQLite 冷存 + 断点续跑）：

**A. 短 episode（12 轮 × 10 个，无损区间）** — 没有任何系统需要压缩，因此：

| system | Current_Fact_Acc | History_Fact_Acc | 记忆 token | 说明 |
|---|---|---|---|---|
| three_layer | 1.00 | 0.857 | 161 | 与全上下文同水平精度，**1/1.9 的 token** |
| full_context | 1.00 | 0.914 | 302 | 原始上下文本来就装得下 |
| memgpt_style | 0.50 | 0.114 | n/a（滚动摘要 199） | 破坏性压缩把旧事实真的删掉了 |
| naive_chain | 1.00 | 0.914 | 334 | 链没超预算，等于没有压缩 |

结论：**episode 太短时该架构无差别**——这是评测设计的坑，不是系统的优点。

**B. 长 episode（40 轮 × 6 个，真实多会话量级，中文）** — 历史必须经过归档检索。
下表 `run-id=final_v1`（修复后的最终代码，四套系统同一次运行）：

| system | Current_Fact_Acc | History_Fact_Acc | 活跃链 token | 上下文 token | Tool_Call_Success | 事件覆盖 | 容量合并 |
|---|---|---|---|---|---|---|---|
| **three_layer** | **0.958** | 0.457 | **348** | 863 | **0.988** | 108 | 340 |
| full_context | 1.000 | **0.848** | n/a（无摘要链） | 2065 | n/a | 0 | 0 |
| memgpt_style | 0.229 | 0.000 | n/a（滚动摘要） | 200 | n/a | 0 | 0 |
| naive_chain | 0.646 | 0.000 | 770 | 807 | n/a | 0 | 141 |

`three_layer` 逐 episode：当前事实 8/8、8/8、8/8、7/8、8/8、7/8；历史事实 4/9、4/9、3/6、2/6、3/9、5/7。

修复前后对（同一数据集、同一 seed=777、同一模型，仅代码不同）：
`CurAcc 0.479 → 0.958`，`HisAcc 0.043 → 0.457`，`ToolOK 0.764 → 0.988`。
历史准确率在不同随机性下会波动（独立复跑一次得到 0.565），**报告时请连同置信区间/方差一起写**，
6 个 episode 的样本量不足以给出小数点后三位的结论。

结论（**这是本项目最该写进报告的一段**）：

* 本方案用 **约 1/6 的上下文 token**（863 vs 2065，活跃链本体 348）换到 **0.958** 的当前事实准确率
  （全上下文 1.000），并把两个摘要型基线（0.229 / 0.646）远远甩开——
  "摘要 + 归档 + 精确回查"这套组合是成立的。
* 被覆盖历史的准确率只有 **0.457 < 全上下文 0.848**：压缩本身是有损的，归档里存的也是
  **摘要**而不是原文。也就是说：全上下文在你的上下文窗口装得下时，仍然是最强的记忆；
  本方案的价值在**上下文成本**和**不可装下时**，不是在绝对精度。
* 这是**当前实现的真实差距，不是评测 bug**（工具成功率 0.994、覆盖链 `superseded_by` 完整可查）。
  想进一步缩小它只有两条路：① 归档存原文而不是摘要（token 成本上升，等于走向全上下文）；
  ② 继续提高摘要与合并的事实保真度（本轮已加 `[FACTS: 属性=取值]`、"最新事实不被合并"
  与"合并必须缩小 15%"三道约束，历史准确率已从 0.043 提到 0.565）。

### 7.2 长 episode 首轮暴露的 5 个真实缺陷（均已修复）

第一次跑长 episode 时本方案只有 `cur=0.479 / hist=0.043`——比全上下文差得离谱。
逐条定位后修掉，最终 `cur=0.958 / hist=0.565`（同一数据集、同一 seed、同一模型）：

| # | 缺陷 | 现象 | 修复 |
|---|---|---|---|
| 1 | **容量合并丢事实** | 活跃链只剩 2 条"杂项"摘要，最新属性值整批消失 → 当前事实准确率崩到 0.48 | 摘要器输出 `[FACTS: 属性=取值]`；压缩时**每个属性的最新摘要受保护**，永不进入合并批次 |
| 2 | **合并反而变大** | 日志出现 `550 tokens before, 1887 after`，token 预算越压越大 | `CAPACITY_MIN_GAIN`：合并至少缩小 15%，否则拒绝并保留原摘要（计入 `capacity_merges_rejected`） |
| 3 | **历史值检索手段不可靠** | 只能靠 `get_archived_summary(id)`，模型在几十条归档里挑错前驱 | 新增精确属性查询 `get_predecessor_summary(fact_key, exclude_value)`：按时间倒序返回该属性的直接前驱（仍是精确匹配，不是相似度检索） |
| 4 | **合并后的归档是混合体** | 查到一条合并摘要，里面混着多个属性，模型给出错误取值 | 合并摘要在归档时保留 `merged_from` 血缘，工具提示回查原始条目；`_slot_value` 修掉跨行取值 |
| 5 | **DSML / 引用结果被误判** | 模型输出 `<|DSML|invoke name="...">` 被当成普通答案（工具成功率虚低）；引用工具结果被当成新调用 | 解析器兼容 DSML 标记（含全角 `｜`）；工具回显文本在解析前剥离，但不再整行删除，以免误杀 DSML 调用 |

另有三处工程性加固：幻觉 override id 丢弃并计数；每轮写入 SQLite 镜像以支持崩溃续跑时重建热点；
工具回显（`[TOOL RESULT ...]`）在解析前剥离，避免"模型引用工具结果"被误判成新的工具调用而死循环。

---

## 8. 已知限制

* **离线 heuristic 后端不是实验结论。** 它无法做跨语言/语义判断（例如中文归档摘要回答英文问题）。
  跑真实结论请用 `--llm-backend ollama|openai`；结果文件会记录 `llm_backend`，便于甄别。
* `Avg_Active_Chain_Tokens` 在无 `tiktoken` 时是估算值（±10% 量级），跨系统比较不受影响（同一口径），
  与论文数值对齐时建议 `pip install tiktoken`。
* MEME 各版本字段命名有差异，加载器用别名表兜底；若 `--inspect` 显示 `num_probes_total` 为 0，
  需在 `MEME_ALIASES` / `_coerce_episode` 里补一个字段名（改动很小）。
* 数据集中若没有 `history_fact` 标注，`History_Fact_Acc` 会是 `n/a`；用 `--synthetic` 或
  按 §4.2 的 `facts` 格式补标注，才能度量历史回溯能力。
* 容量压缩只做"摘要合并"，不做语义去重（禁止 embedding），因此压缩率取决于摘要器质量。
* **self-write 的槽位命名漂移。** 自写块由回答模型生成，实测同一属性可能被写成不同名字
  （第 1 轮 `居住城市=上海`，第 2 轮 `居住地=北京`）。事件式 override 不受影响（模型直接引用了
  `s001`），但依赖**槽位名相等**的两处会受影响：容量压缩的"按槽位保最新值"和索引标题的
  `_fact_digest`。缓解方向是在提示里固定一小张属性名表（或用 `SELF_WRITE_MEMORY=0` 走专用
  摘要器，它对槽位命名更稳定）。这一项尚未实现。
