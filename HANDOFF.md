# 交接文档：三层记忆系统（Memory）

> 写给**下一个对话的自己**。目标是让新会话在 10 分钟内接手，不重复踩坑。
> 最后更新：本轮会话结束（self-write 功能刚落地并通过真实 DeepSeek 验证）。

---

## 0. 一句话目标与硬约束

**目标**：给 LLM agent 做一个三层记忆系统，用于**批量数据集评测**并与主流记忆方法对比。
明确"不是面向演示 demo"。

**硬约束（不要违反，不要"顺手优化掉"）**：

1. **禁止向量 RAG、embedding、相似度检索。记忆只通过 ID 精确查询。** 这是本设计跟
   Mem0/Zep 之类方案的区别点，也是唯一的技术卖点。任何"加个向量检索会更好"的建议都已经被
   讨论过并否决。
2. **原始对话永不删除**（`three_layer` 系统内）。只有 `memgpt_style` 基线会删。
3. **Redis 不是真相来源**：先写 SQLite，再更新热点层。
4. `reset(episode_id)` 默认只清热点；`purge=True` 才是真新会话（可视化页面用）。
5. **提示顺序固定**：滑动窗口 → 活跃链 → system prompt/工具说明。
6. **容量压缩不得输出 `[OVERRIDES: ...]`**（只有事件式覆盖才允许），否则两套压缩逻辑无法归因。
7. 工具调用**保持纯文本解析**（不依赖 OpenAI 原生 `tools` 字段），因为本地模型/vLLM 不一定支持。

---

## 1. 十分钟上手

```bash
cd /home/ubuntu/Memory

# 1) 单测（43 个，0 依赖，不需要网络/Redis/LLM）
python3 -m unittest discover -s tests

# 2) 离线跑一遍完整评测（heuristic 后端，不花钱）
python3 evaluation.py --synthetic-long --synthetic-turns 40 --limit 3 \
    --systems three_layer,full_context,memgpt_style,naive_chain \
    --llm-backend heuristic --store memory --run-id smoke --out-dir results/smoke

# 3) 真实模型小规模 A/B（花钱，约 1 元）
python3 evaluation.py --synthetic-long --synthetic-turns 40 --limit 5 \
    --system three_layer --chain-strategy index --seed 777 \
    --llm-backend deepseek --model deepseek-flash \
    --store redis_sqlite_hybrid --run-id ab_check --out-dir results/ab_check

# 4) 可视化页面（默认端口 8010，需 ?token=mem2024）
python3 web_ui.py --llm-backend deepseek --model deepseek-flash --num-episodes 3 --synthetic-turns 30
#   -> http://127.0.0.1:8010/?token=mem2024
```

`.env` 里已有 `DEEPSEEK_API_KEY`，`config.py` 自动加载。**不要把 key 打印到日志或回答里。**

---

## 2. 代码地图

| 文件 | 职责 | 备注 |
|---|---|---|
| `evaluation.py` | ★ 主实验入口：批量跑数据集、多基线、指标、CSV、断点续跑 | ~1150 行 |
| `memory3l/memory_manager.py` | ★ 核心：摘要生成、覆盖解析、容量压缩、索引层、懒展开、self-write | ~1480 行 |
| `memory3l/models.py` | `ActiveSummary` / `ArchivedSummary` / `IndexEntry` / `RawDialogRecord` / `MemoryTurnStats` | 纯数据 |
| `memory3l/tools.py` | 文本工具调用解析 + 摘要语法解析 + `ToolExecutor`（4 个工具） | 三套语法 |
| `memory3l/prompts.py` | 全部提示词 + 上下文拼装（`build_memory_block` / `build_agent_messages`） | |
| `memory3l/store/` | `BaseMemoryStore` / `InMemoryStore` / `SQLiteColdStore` / `RedisHotStore` / `RedisSQLiteHybridStore` | |
| `memory3l/agents/` | `base_agent`（工具循环）+ `three_layer`（本方案）+ `baselines`（3 个基线） | |
| `memory3l/dataset.py` | 合成数据集生成 + MEME 别名兼容 + `inspect_dataset` | |
| `memory3l/longmemeval.py` | LongMemEval 适配器（ijson 流式读 277MB） | |
| `memory3l/llm.py` | DeepSeek / Ollama / OpenAI(vLLM) / Heuristic / Scripted 五种后端 | |
| `chat_debug.py` | 命令行交互调试（辅助，非主路径） | |
| `web_ui.py` + `web_ui.html` | 可视化检视页面（看每条检索路径） | |
| `analyze_results.py` | 读 `results/` 出的 CSV 做汇总/错误分解 | 存在 |
| `README.md` | 设计文档（**已经写了 5.1–5.5 节，含实测数据**） | 43KB |

---

## 3. 架构要点（接手必读）

### 3.1 三层

- **L1 活跃摘要链**：一轮一条摘要，带 `[OVERRIDES: id_list]`。
- **L2 归档摘要库**：永久保存，被覆盖的旧摘要进这里（`is_overridden=True`，有 `superseded_by` 前向指针）。
- **L3 原始对话库**：永久保存，摘要只持有 `reference_id`。

提示顺序：`滑动窗口(4轮) → 活跃链 → system prompt + 工具`。

### 3.2 两套压缩（必须可分别归因）

| | 触发 | 是否发 OVERRIDES | 指标 |
|---|---|---|---|
| 事件式覆盖 | 新事实与旧事实**同属性不同值** | ✅ | `overrides_events` |
| 容量压缩 | 活跃链超 `ACTIVE_CHAIN_TOKEN_LIMIT` | ❌ | `capacity_compressions` / `capacity_merges_rejected` |

容量压缩有 `CAPACITY_MIN_GAIN=0.15`：合并后必须至少缩小 15%，否则**拒绝**（真实模型确实会产出
"越合并越大"的结果）。

### 3.3 索引层（用户自己提的"目录/索引"架构）

- 叶子摘要可以归档到 **index entry**（标题 = `attribute=value[history]`）；
- 条目太多时递归折叠成 **super index**（标题的标题），逻辑在 `_maybe_fold_indexes`；
- **懒展开**：被索引引用过的摘要**不再在活跃链里渲染**（`LAZY_INDEX_ID = "__LAZY_SUMMARIES__"` 哨兵），
  但**仍可按 id 精确读取**。这是真正把 token 压下来的那一步。
- 检索入口先看索引标题 → `expand_index(index_id)` 展开 → `get_archived_summary(id)` / `get_raw_record(id)`。

### 3.4 self-write（本轮刚加，**已实现并验证**）

让**回答的那次调用顺便把本轮摘要写了**，一轮只花 1 次 LLM 调用（原来 2 次）。
模型在最终回复末尾附：

```
<MEMORY_UPDATE>
一句话摘要
[FACTS: 属性=值; 属性2=值2]
[OVERRIDES: s003]
</MEMORY_UPDATE>
```

- `tools.split_selfwrite_reply()` 剥块（**块绝不能进答案**，否则评测就废了）；
- 语法与摘要器完全相同 → 同一个 `parse_summary_response` → 同一条 `_apply_generated` 链路；
- 两个来源在 `_finalise_generated()` 处归一，**override 图不关心摘要谁写的**；
- 块校验通过**才**写 raw record；块缺失/不可解析 → 自动回退专用摘要器，
  **只多花一次调用，不会丢更新、不会重复写 raw**；
- 开关 `config.SELF_WRITE_MEMORY`（默认开）；指标 `self_written_turns` / `self_write_fallbacks`。

⚠️ **它省不了批量评测的钱**（评测集里 assistant 回复是现成的，没有回答调用可搭载）。
详见 §6 第一条。

---

## 4. 当前已验证的结果（可以放进论文/汇报）

### 4.1 A/B：索引层 vs 容量合并（真实 DeepSeek-flash）

配置：5 episodes × 40 轮中文合成 = 200 轮，78 probes，seed 777，链预算 400，
`redis_sqlite_hybrid`，温度 0。产物在 `results/final_ab/`（index）和 `results/final_merge/`（merge）。

| 指标 | `index` | `merge` |
|---|---|---|
| Current_Fact_Acc | **0.925** (40) | 0.925 (40) |
| History_Fact_Acc | **0.879** (33) | 0.788 (33) |
| Avg_Active_Chain_Tokens（正文） | **158.4** | 261.4 |
| Avg_Active_Chain_Tokens_rendered | **521.0** | 828.7 |
| Avg_Active_Chain_Size | 2.6 | 3.4 |
| Avg_Archived | 28.4 | 77.8 |
| overrides_events | 50 | 16 |
| capacity_compressions | 0 | 132 |
| Tool_Call_Success_Rate | 0.956 | 0.956 |
| Wall_Time_s | **657.8** | 1410.2 |

结论：同样的准确率，索引层把历史回溯拉高 9 个点、token 降 39%、时间减半。

### 4.2 LongMemEval 单题（真实 DeepSeek-flash）

题：`lme_9ea5eabc` — "Where did I go on my most recent family trip?"
（gold `Paris`，更早是 `Hawaii`）；截断协议 `max_turns=120` → 实际 ingest 60 轮。

| 项 | 结果 |
|---|---|
| ingestion 耗时 | 416 s |
| 覆盖事件 / 归档 / 索引条目 | 8 / 8 / 4 |
| 活跃链末态 | 3 条 / 885 tok |
| 回答 | **Paris**，附"已被取代"式依据 |
| 字符串匹配 / LLM 裁判 | ✅ / ✅ |
| 工具调用 | 1/1 |
| 成本 | **1.26 元**（in 328,368 / out 75,714 tok） |

**关键的诊断结论**：`Paris` 只存在于**索引成员**里（不在活跃链、不在索引标题），
`Hawaii` 在保留记忆中根本不存在。→ 真实数据的决定性失效模式是**顶层不可达**
（值藏在索引成员里），**不是模型能力**（DeepSeek 能调工具后立刻答对）。

全量协议（537 轮）尝试跑过，在 91 轮时被工具 600s 超时杀掉，外推 **~11 元 / ~63 分钟每题**。

⚠️ 这次单题运行的产物**没有落盘**（是内联脚本跑的），只有上面的数字。
重跑请走 `evaluation.py --dataset-format longmemeval --lme-max-turns 120`。

### 4.3 其他

- 离线 heuristic 长上下文：History **21/21 = 1.000**，渲染 ~165 tok/轮。
- 修复前同 seed 串行跑：Current 0.958 / History 0.457（后面修了几个 bug 才到 4.1 的数）。
- `results/lme_pilot/logs/evaluation.log` 是一次 ollama `deepseek-r1:14b` 的 LongMemEval 试跑
  （4 episodes），**被中断，没有结果文件**，只有日志。

---

## 5. 项目现状清单

**已完成且测试覆盖**：三层读写、事件式覆盖（含 `superseded_by` 链回溯）、容量压缩（含拒绝逻辑）、
索引层 + 超级索引 + 懒展开、4 个精确 id 工具、4 套系统（3 基线）、指标与 CSV/断点续跑、
Redis+SQLite 混合存储、LongMemEval 适配器、可视化页面、self-write。

**测试**：`python3 -m unittest discover -s tests` → **43 passed**
（`tests/test_core.py` + `tests/test_store.py`）。
**改动后一定要重跑**，这是唯一的安全网。

**外部服务现状（本会话结束时刻）**：
- Redis **在跑**（`127.0.0.1:6379`，78 组 `episode:*` 键）。`redis-cli` 没装，但 `.pylibs` 里的 redis-py 可用。
- Ollama **在跑**（`0.0.0.0:11434`，9 个模型 ~63GB）。
- **vLLM 没在跑**（8100 无响应）。`start_vllm.sh` 已写好（TP=2、port 8100、`--max-model-len 16384`）。
- DSH Web GUI：`127.0.0.1:3080`（返回 401 属正常，需要 token）。

---

## 6. 待办（按优先级）

### P0 — 直接决定"能不能做大规模评测"

**① 按 session 粒度摘要（评测降本 10×）**
现在 494 轮 haystack = 494 次摘要生成，这是 ~11 元/题的全部来源。
LongMemEval 每题约 50 个 session（每个 ~10 轮）：**一轮一次调用 → 一个 session 一次调用**，
494 → 约 50 次。而且 session 摘要天然就是第二层目录项，**正好落在既有索引层架构里**，不是权宜之计。
实现位置：`add_dialog_turns` / `_generate_only` 支持"一次多轮 → 多块输出"。

**② 重跑 LongMemEval 小样本得到真实分数**
用户当时正在选：**10 题（~12 元、~1 小时）** 还是 **20 题（~25 元、~2 小时）**（截断协议）。
建议顺序：先做 ①，再跑 20 题，否则 500 题全量要 5500 元。

**③ 修成本归因**（改动小，但会误导结论）
`evaluation.py:1118-1119` 里当摘要模型和回答模型同一个 client 时
`summarizer_llm = agent_llm`，两个 `stats()` 读的是**同一个计数器**
（`results/final_ab/run_metadata_final_index.json` 里 `llm_stats.calls == summarizer_stats.calls == 490`，
不是 980）。**现在无法把"摘要成本"和"回答成本"分开**——而成本正是当前的核心争议点。
修法：给每个角色单独包一层计数器。

### P1 — 已知的功能缺陷

**④ 并发 ingestion 语义不等价**
`--ingest-concurrency > 1` 时，串行跑出 24 次 override 事件，并发跑 **0 次**。
实现方式是把 raw record 全部先写、再并行生成摘要，导致每条 prompt 的 `<ACTIVE_CHAIN>` 都是空的。
默认已锁 1。修完才能靠并发把 416s/题压下来。

**⑤ 顶层不可达（真实数据的主要失效模式）**
索引标题会截断（`MAX_TITLE_CHARS=60`，`_fact_digest(max_attrs=6)`，`ManifestFactDigest.merge(max_attrs=8)`），
被截掉的属性值就只剩"某个索引成员里有"这一条路。
建议方案（**未实现**）：加一个**属性登记表 / always-visible 当前值区**，让当前有效事实永远在顶层可见。
这是目前性价比最高的架构改动。

**⑥ self-write 的槽位命名漂移**
实测同一属性第 1 轮写成 `居住城市=上海`，第 2 轮写成 `居住地=北京`。
事件式覆盖不受影响（模型直接引用了 `s001`），但依赖**槽位名相等**的两处会受影响：
容量压缩的"按槽位保最新值"、索引标题的 `_fact_digest`。
缓解：提示里固定一张属性名表；或 `SELF_WRITE_MEMORY=0` 走专用摘要器（命名更稳）。

### P2 — 其他成本优化（都未实现）

- **⑦ 只对含事实的轮次调模型**：预估省 50–60%。
- **⑧ 摘要器输入用规则挑选**：现在每次都把整条链喂进去，
  日志里 `summariser input capped (17 entries / ~3074 tokens vs budget 3000)` 一直在刷。
- **⑨ 索引层只存抽出来的事实 + raw id**。

### P3 — 环境杂务

- **⑩ Ollama 清理**：`/mnt/sda/ollama_data/models` 约 63GB。用户之前问过"你把 ollama 上的清理了吗"，
  给了 A 全清 / B 保守 / C 只删没用 三个选项，**用户没选，所以没动**。
  建议保留 `nomic-embed-text`（虽然本项目禁 embedding）和 `deepseek-r1:1.5b`。

---

## 7. 环境与操作坑（会重复踩的）

1. **每次 bash 调用是独立 PID namespace**（`bwrap --unshare-pid`）：
   杀不掉别的 shell 留下的孤儿进程；`nohup` 起的子进程会随 shell 死。
   后台服务要用受管后台任务（`run_in_background`）。
2. **`/tmp` 不可写**。文件操作限 `/home/ubuntu/Memory`（workspace-write）。
3. **GPU 在沙箱里不可见**（`torch.cuda.is_available()` False，无 `/dev/nvidia*`）。
   机器是双 RTX 4090 D 24GB + i9-14900K + 125GB RAM，驱动 590.48.01 / CUDA 12.8。
   **vLLM / Ollama 必须在宿主机上起**，不能从会话里起。
4. **本地模型**：`/home/ubuntu/models/Qwen2.5-7B-Instruct` 正常（有 chat_template）；
   `Qwen2.5-7B-Instruct-IDS2017FT` **没有 chat_template，聊天/工具全是坏的，不要用**。
5. **vLLM 起服务时没加 `--enable-auto-tool-choice`**，原生 OpenAI `tools` 会 400；
   本项目的纯文本工具调用**不受影响**（已验证模型会输出 `get_raw_record(reference_id="raw008")`）。
6. **端口**：DSH GUI 3080（仅 localhost）；用户业务服务占 8000/8002/8008；Ollama 11434；
   检视页面用过 8010→8011→8012→8013（**起之前先确认端口空着**）。
7. 可视化页面默认 `--host 0.0.0.0`，但**必须在用户侧做端口转发**才能访问；会话内建不了路由。
   页面需要 `?token=mem2024`（`--token` 可改），否则 403。
8. `redis_sqlite_hybrid` 在 blank 模式下曾 500（episode 列表读了 `redis_store`），已修；
   若再遇到类似 AttributeError，先怀疑 hot/cold 两个后端的接口不对称。
9. **`_finalise_generated()` / `_apply_generated()` / `_chain_for_summariser()` 是覆盖图的
   唯一权威**。要改摘要行为，改这里，不要在调用方各写一份——历史上
   `add_dialog_turn` 和 `add_dialog_turns` 分叉过，导致单轮路径静默用了长 id、覆盖功能在
   一个入口下生效、另一个下不生效。
10. 长 id 陷阱：规范 id 形如 `episode/s001@ab12cd`。模型会把它截成尾部 hash，
    导致精确 id 校验失败、覆盖静默失效。**给模型看的永远是短 id（`s001`），落库前映射回规范 id。**

---

## 8. 对外汇报可以引用的基线数字

同数据集上公开系统的量级（第三方转述，注意标注来源）：
AXME 89.20% / Mastra 84.23–94.87% / Supermemory 85.40% / Zep 71.20% / Mem0 49.00%（第三方）·
66.88%（LoCoMo）。长上下文基线普遍掉 30–60%。**"读策略"错误可占 10 分**——
这条正好对应 §6 的 ⑤ 顶层不可达。

成本对照：公开系统的 500 题全量约 $20（AXME ~9K tok/题）、$22（Mem0 ~15K）、
$75（Zep ~50K）、$150（Mastra ~100K）；本设计**全量协议 ~11 元/题 ≈ 5500 元/500 题**。
**成本大头是"每写一轮就调一次 LLM 做摘要"，必须先解决 §6 ①。**

DeepSeek 官方价（元/M token）：flash 输入 2（峰）/1（谷），输出 8/4；pro 9/4.5 与 27/13.5。
高峰 = 北京时间周一至周五 09:00–12:00、14:00–18:00。

---

## 9. 本轮最后的改动（self-write）文件清单

| 文件 | 改了什么 |
|---|---|
| `memory3l/tools.py` | 新增 `split_selfwrite_reply()` + `_MEMORY_UPDATE_RE` / `_MEMORY_UPDATE_FENCE_RE`；docstring 改成"三套语法" |
| `memory3l/prompts.py` | 新增 `SELF_WRITE_ZH/EN` + `self_write_instruction()`；`build_agent_messages(..., selfwrite=False)` 把指令加到 **system role**（工具循环里必须活到最后一轮） |
| `memory3l/memory_manager.py` | 抽出共享的 `_finalise_generated()`；新增 `harvest_selfwritten()`（先校验后写盘）与 `add_dialog_turn_selfwritten()` |
| `memory3l/models.py` | `MemoryTurnStats` 加 `self_written` / `self_write_fallback` |
| `memory3l/agents/base_agent.py` | 新增 `context_messages()` 钩子；`answer(..., selfwrite=False)` 剥块并记录 `memory_block` |
| `memory3l/agents/three_layer.py` | 新增 `context_messages()` 与 `answer_and_remember()`；`finalize_episode` 加两个指标 |
| `chat_debug.py` / `web_ui.py` | 交互路径切到 `answer_and_remember`；`--self-write/--no-self-write` |
| `config.py` | `SELF_WRITE_MEMORY`（默认 true） |
| `tests/test_core.py` | `TestSelfWrittenMemory` 5 个测试（一轮一次调用、override 映射、回退不重复写 raw、剥块变体） |
| `README.md` | 新增 §5.5（设计 + 实测表 + 适用边界）、§5 配置表加一行、§8 加槽位漂移限制 |

**验证状态**：43 单测全过；真实 DeepSeek 两轮对话验证 —— 2 轮 = **2 次 LLM 调用**（原 4 次），
第 2 轮模型自写 `[OVERRIDES: s001]` 正确映射为规范 id，s001 归档、s002 上链，答案无块泄漏。

---

## 10. 新会话建议的第一个动作

1. 读 `README.md` §5.1–5.5（尤其 §5.5 和 §8）。
2. `python3 -m unittest discover -s tests` 确认 43 过。
3. 问用户：**先做 §6 ①（按 session 粒度摘要，评测降本 10×），还是先跑 10/20 题小样本拿真实分数？**
   —— 上一轮我建议前者，用户说要开新对话，还没答复。
