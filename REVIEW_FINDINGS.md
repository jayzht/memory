# 架构审查报告与修复记录

> 一次独立的、从零重读代码的审查。所有结论都**先用实验复现**再修，每条修复都有对应的回归测试。
> 基线 commit `77d6b13`（审查开始时 43 个单测通过）；当前 **67 个单测通过**。

审查范围：`evaluation.py`、`memory3l/`（manager / models / prompts / tools / llm / dataset / longmemeval /
store / agents）、`web_ui.py`、`chat_debug.py`、`config.py`、`analyze_results.py`。

---

## 0. 一句话结论

代码质量比交接文档描述的更好，但**存在 1 个必崩的 P0 和 3 处"文档宣称的机制实际未生效"**，
其中一处（Redis 序列化丢字段）足以让**所有已归档的真实实验数字不可直接引用**。

---

## 1. P0 — 会直接毁掉实验

### 1.1 Redis 序列化丢 4 个字段（最严重）

`memory3l/store/redis_store.py:_load_summary` 丢 `fact_keys` / `index_id`；
`loads_index` 丢 `previews` / `child_index_ids`；`sqlite_store.index_entries` 表**根本没有后两列**。

复现（连本机 Redis 的实测）：

```
HOT  read: fact_keys=[]                index_id=''
COLD read: fact_keys=['工位=3楼']       index_id='review_ser_001/idx001@zz'
HOT  index: previews=[]  child_index_ids=[]
```

所有真实运行都走 `redis_sqlite_hybrid`，于是：

* **索引标题的 `属性=当前值` digest 恒为空** → 索引层从不显示当前值，这是"顶层不可达"的主因；
* `previews` 丢失 → `IndexEntry.render(preview=1)` 回退渲染**整条成员摘要**（~100 tok），
  实测索引层 474 → 317 tok/轮（约 1/3 虚高）；
* `index_id` 丢失 → 懒展开扫描每轮重写所有摘要（写放大），折叠后的成员重指向失效；
* 同一对象 hot/cold 读出来不一样 → 指标依赖缓存热度，memory 与 hybrid 不可比。

**修复**：补齐字段 + SQLite 建表/迁移加列（JSON 编码，避免 `|` 分隔符冲突）+ 全字段 round-trip 测试。
commit `7f14f98`。

### 1.2 `--system all` 必崩且不产出任何结果文件

`evaluation.py:1054` 用 `f"{result.active_chain_text_tokens_mean:.0f}"` 格式化，而 `memgpt_style`
的该字段被**故意**设为 `None`（"n/a 不能伪装成 0"的设计本身是对的）→ `TypeError`，
且崩溃点在 `try/except` **之外**，整个进程死掉。

复现：`--systems memgpt_style --limit 1 --synthetic-turns 12` → 崩溃，`results/` 下只有空的 `logs/`。
README §1 ⓪ 与 HANDOFF §1 ② 推荐的 `--system all` / 四系统对照**跑不完**。

**修复**：`n/a` 安全格式化。commit `57283ad`。

### 1.3 多系统 `predictions.csv` 静默重复

`all_rows` 是累积的，却用 `append=first_write_done` 逐系统追加：系统 k 追加全部 k·n 行。
4 系统 × n → **10n 行，权重 4:3:2:1**，而 `analyze_results` 默认读的就是这个文件。
`evaluation.aggregate` 用的是内存里的 `all_rows`，所以**跑的时候看不出问题**。

实测：3 系统 × 2 episodes → 12 行（应为 6）。

**修复**：flush 改为整体覆盖。commit `57283ad`。

### 1.4 resume 丢失指标并截断 CSV

已完成的 episode 被 `continue` 跳过却从不回读，`aggregate(all_rows)` 只覆盖本次进程跑的部分；
首次 flush 还会把已有的 `predictions.csv` 覆盖成这个子集；全部跳完时直接 `return`，一行指标都不输出。

实测（修复后）：全跳过的第二次运行打印出与第一次**完全相同**的指标，`predictions.csv` 保留全部 6 行。

**修复**：把 checkpoint payload（现在含 probe 行）回填进本次运行。commit `57283ad`。

### 1.5 基线被灌输了它们没有的工具（公平性）

`build_agent_messages` 无条件追加 `TOOL_DESCRIPTIONS`，而 `AGENT_SYSTEM_*` 明确要求"必须先调用工具"。
`memgpt_style` / `naive_chain` 的 `executor is None` → `allow_tools` 恒为 False →
模型输出的工具调用**无法执行**，直接落到 `final_text = text`，**该字符串成为它的答案**并被判错。
而且 `Tool_Calls` 记 0，指标上看不出原因。

复现：

```
baseline supports_tools: False
baseline answer: 'get_raw_record(reference_id="e1/raw000@abcdef")'
tool_attempted: 0
```

**修复**：无 executor 的系统改用**中立回答契约**（不描述它没有的层、也不提工具），
并把 `tools_available` 从 `supports_tools()` 显式传入。同时修掉两个基线把摘要块**插在
`<RECENT_RAW_DIALOGUE>` 之前**（违反固定顺序、也违反它们自己的注释）。commit `3ac2786`。

---

## 2. P0/P1 — 答案完整性

### 2.1 工具调用会被当成答案

* 迭代上限（默认 4）那一次：`final_text = text` 没有检查，循环的模型输出被直接采信；
* 无工具的系统发出调用时同样；
* 连 `_force_answer` 的"禁止调用工具"重答也可能仍是调用。

**修复**：最终答案统一消毒（`is_tool_call` / `_looks_like_tool_intent`），丢弃并记为
`tool_call_as_answer`；空答案按答错处理（诚实）。commit `3ac2786`。

### 2.2 self-write 块会泄漏进答案

`split_selfwrite_reply` 在"回复只有该块"时返回**原始带标签文本**
（`<MEMORY_UPDATE>...` 进入答案、judge、CSV，并被写进 raw record 反复出现在后续滑动窗口）；
多块时只剥第一个；未闭合的 fenced 块完全泄漏。

**修复**：两种定界形式**循环剥离全部块**；一旦找到块就绝不返回原文（必要时用块内的摘要正文充当答案）；
未闭合的标签/围栏**只在剩余内容确实带有 `[OVERRIDES]`/`[FACTS]` 语法时才当成块**，
避免把"这句话里提到了 `<MEMORY_UPDATE>`"误判成块而删掉真正的答案。commit `3ac2786`。

### 2.3 工具调用解析器

* **全角标点**（`（）：＝`，对中文提示很常见）完全不解析：不仅没算作尝试，工具调用文本还会成为答案。
  → 解析前 NFKC 归一化；
* `strip_tool_echoes` 写了却没接上：模型引用 `[TOOL RESULT ...]` 行会被**重新执行**。
  → 接入 `parse_tool_calls`；
* `normalise_tool_args` 是**扁平别名表**，`reference` 命中 `reference_id` 分支，
  于是 `get_predecessor_summary(reference="12楼")` 把锚点丢到错误的槽位，
  历史查询退化成"取最新旧值"——正是锚点模式要修的问题。→ 改为**按工具**解析歧义别名。

commit `3ac2786`。

---

## 3. P1 — 记忆架构

### 3.1 `_fact_digest` 把最旧的值当成"当前值"

```
members（旧→新）: 工位=1楼, 2楼, 3楼
_fact_digest -> '工位=1楼[3楼→2楼→1楼]'      ← 错
expected     -> '工位=3楼[1楼→2楼→3楼]'
```

原因：为了"属性顺序优先考虑最近变更"，值也按最新→最旧累积，而 `seq[-1]` 被当作当前值。
**索引层一直在顶层显示错误的当前值。** `ManagerFactDigest.merge` 有同样的反向问题。

**修复**：属性顺序与值顺序分开——属性按最近出现优先（利于截断），值按旧→新累积。
commit `2568b05`。

### 3.2 覆盖后索引条目残留

把已归档进索引的摘要覆盖掉之后，条目：

* `size` 不变（`N entries` 虚高），而 `expand_index` 返回的条数更少；
* `members` 仍含死成员；
* 预览与 digest 仍显示**已被取代的旧值**。

**修复**：`_detach_from_indexes()` —— 覆盖/容量合并后剪掉成员、重算 digest（**无需 LLM**，
用存好的 `fact_keys`）、刷新预览；成员少于 2 的条目解散并让幸存者重新渲染。
新增 `index_updates` 指标。commit `2568b05`。

### 3.3 索引标题的"构建 220 字 / 渲染 60 字"

`_file_one_index_group` 造 title 上限 220 字，`IndexEntry.render()` 却按 `MAX_TITLE_CHARS=60` 截断
→ 约 **73% 的 digest 被丢弃**；`_fact_digest(max_attrs=6)` 想要的 6 个属性在 60 字里只装得下 1~2 个。
叠加超级索引前缀累积，实测标题形如：

```
[2 组] [2 组] [2 组] 工位=1楼[2楼→1楼] … 工位=9楼[10楼→9楼] ｜ 第14轮：工位是
```

**修复**：`INDEX_TITLE_CHARS` 成为配置项（默认 120）；`[N 组]` 前缀不再写进标题，
改由 `child_index_ids` 在 `render()` 里渲染组数。commit `2568b05`。

### 3.4 懒展开整层从未生效（且它本来就不省 token）

证据：

1. `chain_summaries()` 从未排除 `LAZY_INDEX_ID` → "懒存"的摘要**照常渲染**，契约没实现；
2. covered 判定读的是**已被折叠删除**的子目录 id，实测算出 0（正确值应为 22）；
3. **即使修好也没有增量收益**：`chain_summaries()` 按 live 目录项的 `members` 过滤，
   而超级索引的 `members` 是传递闭包，"两跳成员"本来就不渲染（实测 overlap = 0）。
4. 所有已归档运行 `lazy_summary_count = 0`。

即：**那段 A/B 里 `index` 省下的 token 全部来自"挂到标题下 + 折叠"，与懒展开无关。**

**修复**：`chain_summaries()` 排除 LAZY；covered 判定改读 live 超级索引自身；
**删除"把未归档摘要转成懒存"的分支**——那种摘要没有任何指针，藏起来就是真的不可达。
懒展开现在的定位是**两跳成员的可观测簿记**。README §5.3 已加更正。commit `2568b05`。

### 3.5 新增：当前值登记表 `<CURRENT_VALUES>`

把"当前值"从索引标题的职责里摘出来。从活跃摘要已有的 `fact_keys` 派生"每槽位最新值"，
固定渲染在滑动窗口之后、索引层之前：

* 零额外 LLM 调用；派生的，因此**不会过期**；
* 实测 2 个槽位 = 10 tok（约 10~15 tok/槽位），上限 `CURRENT_VALUES_MAX_SLOTS=12`；
* 成本单独记 `current_values_tokens` / `avg_current_values_tokens`，并计入渲染口径与预算触发。

这是对真实数据失效模式（值只在索引成员里）的**结构性修复**：即使摘要被归档/折叠，
当前值仍在顶层可见。commit `2568b05`。

---

## 4. P1 — 评测方法学

| 问题 | 证据 / 修复 |
|---|---|
| **成本无法归因** | 摘要/裁判 client 与回答 client 是同一个对象，`summarizer_stats` 是同一计数器的第二次读取（`final_ab` 两者都是 490 calls、token 完全相同），而且还混入了建索引的调用。→ 每个角色独立 client。commit `57283ad` |
| **跨 run 归档污染** | 命名空间 `system/episode` 不含 run id，`reset` 又保留归档；`exp_final_ab.db` 里某 40 轮、`capacity_compressions=0` 的 episode 报告 `archived_final=101`。→ 新增 `store.clear_archive()`（**绝不动 raw**），每个 episode 开始时调用。commit `57283ad` |
| **并发 ingestion 指标恒为 0** | 并发分支丢弃 `_apply_generated` 的返回值，从不设置 `event_triggered`。实测 concurrency=2：`overrides_events=0` 但 `overridden_summaries=1`、归档里确有 1 条。→ 记账搬进 `_apply_generated`（三条路径共享），并按**实际归档**计数。同时 `ingest_concurrency` 补默认值（此前不先 `set_ingest_concurrency` 就 `AttributeError`），并更正"与串行等价"的过度声明（窗口内共享链快照，原理上不可能等价）。commit `4e4f7c1` |
| **token 口径混用** | 同一列里三个系统分别填：每轮均值 / 终态总量 / **未截断**的存储量。→ 四系统统一为"实际渲染量的每轮均值"。commit `585e49c` |
| **`string_match` 与自己的文档矛盾** | 只要 gold 出现在不超过 4 倍长度的预测里就算对 → 一个"先说错值、再引用 gold"的啰嗦回答会被判对，且**绕过 LLM 裁判**。→ 改为精确匹配或"预测以 gold 开头"（剥掉 `答案是` 之类引导语）。commit `585e49c` |
| **`MEMORY_POINT` 正确率从不聚合** | 每 episode 累计却不出现在总表 → "工具调用成功但归档内容答错"没有任何体现。→ 新增 `Memory_Point_Acc/_n`。commit `585e49c` |
| **per-probe `tool_unparsable` 恒为 0** | 日志条目没有 `parsed` 键，`.get("parsed", True)` 永远为真（78 行全 0，而总表是 1）。→ 补键。commit `585e49c` |
| **run metadata 不记录 CLI 覆盖** | `ConfigSnapshot` 只显式传 9 个字段，`--raw-context-token-limit`、`--max-tool-iterations`、`--sqlite-path`、各种 capacity/index 旋钮全部记成默认值。→ `ConfigSnapshot.from_args(args)`。commit `a90929a` |
| **部分失败的 episode 被标记 done** | 之后再 resume 会跳过它，坏数据被静默保留。→ 错误时 `mark_episode_failed`，且 payload 先于 done 落库。commit `585e49c` |

---

## 5. P1/P2 — 数据集与 LongMemEval

* **合成数据集 History 标准答案是错的**：一个属性变两次会生成两条**措辞相同、gold 不同**的探针
  （`在改成<最终值>之前` → gold 分别是 1楼 和 2楼），其中一条必然判错。
  → 每条探针锚定**该属性时间线上的下一个值**（长上下文生成器早已这么做）。commit `585e49c`
* **探针类型推断**：子串匹配包含 `"was"`，于是 `"What was the meeting time?"`（问当前值）
  被判成历史题；无法匹配时默认 `OTHER`，而 `OTHER` **不在任何一个头条指标里**，等于静默消失。
  → 改为词边界标记 + 当前值标记 + 有 gold 时默认当前值。commit `a90929a`
* **LongMemEval `--lme-types` 配默认 `--limit 50` 返回 0 条**：先按记录数截断扫描再按类型过滤，
  而真实文件前 50 条全是 `single-session-user`。→ `limit` 现在计**匹配数**，`--limit 0` 表示全部，
  扫描上限另设并在用尽时告警。实测 `types=['knowledge-update'], limit=3` 返回 3 条。
* **轮级截断会切掉证据**：会话选择保护了证据会话，最后的 `dialogues[-max_turns:]` 是纯尾部截断，
  仍会删掉长证据会话的前半段，而 `answer_session_included`（截断前算的）仍称"已保留"。
  → 显式保护证据轮，meta 由**幸存轮**计算。实测 `max_turns=10` 时证据仍在。
* **预算单位不一致**：会话选择按**消息数**扣预算、最终按**轮数**截断，导致实际只用了约一半预算
  （30 → 15 轮）。→ 统一按轮数。修复后 `kept_turns == max_turns`。
* `full_turns` 记消息数、`kept_turns` 记轮数。→ 统一。

commit `a90929a`。

---

## 6. P2 — 存储健壮性

* **Redis pipeline 错误未包装**：只有 `_call` 被包成 `RedisUnavailable`，9 处 pipeline 写直接
  `pipe.execute()`，连接错误会**逃过 hybrid 的 `except RedisUnavailable`** 而中止 episode
  （尽管 SQLite 已提交同一笔写）。→ 全部走 `_pipeline()` 守卫。commit `f5454ce`
* **`append_window_record` 是唯一的 Redis-first 写**：失败会让崩溃恢复镜像落后，
  而 `rebuild_hot_state` 只从该镜像恢复窗口。→ 改为 cold-first。commit `f5454ce`
* **`InMemoryStore.add_raw_record` 不幂等**（SQLite 是 upsert）：重复添加确定性 id 会让
  `count_raw_records` 与 full_context 的 token 总量翻倍。→ 加 membership 检查。commit `f5454ce`
* **`count_active_summaries` 用 Redis `LLEN`**：会数进列表路径已去重的历史重复项，
  热键缺失时返回 0 而列表路径会回落 SQLite。→ 改走读路径。commit `f5454ce`

---

## 7. 清理

* 删除死代码：`_summarise_turn()`（104 行，正是交接文档警告过的"第二套摘要路径"，
  已与真正路径分叉）、`build_prompt_context()`（从未调用且渲染错误的东西）、
  `TokenCounter`、`render_archived_summary()`、`token_utils.active_chain_tokens()`。
  删了 105 行 + 独立函数。
* 修正与实现不符的注释：模块 docstring 的"覆盖当轮不再压缩"（从来没实现，改为如实描述）、
  `_previews_on` 声称会恢复（**补上了实现**，70% 预算滞回）、`INGEST_CONCURRENCY` 自相矛盾的两段说明、
  `INDEX_MAX_PER_TURN` 的"硬上限 2"与循环实际的 3× 预算。
* README 新增 §0（审查更正清单）、§5.6（登记表）、§5.7（指标口径），更正 §2.1、§5.3，测试数 34 → 67。

---

## 8. 尚未修复（有意留给你决定）

| 项 | 说明 | 为什么没动 |
|---|---|---|
| 并发 ingestion 的语义 | 即使指标修好了，窗口内共享链快照仍会漏掉窗口内覆盖。默认锁 1。 | 要真正等价只能把窗口降到 1；并发值不值得，取决于你愿不愿意用近似换墙钟时间 |
| Redis `degraded` 标志无人消费 | 部分失败后 `available` 仍为 True，读会返回**过期的热数据**。 | 需要按 episode 的 dirty 标志 + 重建策略，改动面较大，且与"Redis 只是缓存"的定位需要一起设计 |
| `bind_episode(resume=True)` 恢复不了东西 | 它先 `reset`（清掉镜像）再 `rebuild_hot_state`（从镜像恢复）。生产路径没用到。 | 同上；正确做法是 resume 时跳过镜像清空 |
| `flush_hot_keys` 参数 | 到处接受、从未被使用。 | 删除需要改多处签名；留着只是一个无声的旋钮 |
| `_slot_value` 是子串包含而非精确属性名 | `_slot_value("Facts: 工位楼层=12楼", "楼层")` 返回 `12楼`。 | 仍是无向量的确定性词法匹配，但严格来说不是"精确属性名"。需要你决定是补文档还是收紧 |
| `fact_keys` 双表示 | 解析器产出 `属性=值`，词法回退产出裸 `属性`；`_maybe_capacity_compress` 用原始串比较。 | 只影响"正文里恰好写了 `x=y` 但没写 `[FACTS]`"的边角情况，失败方向是过度保护（成本），不是丢事实 |
| MEME 别名表 `MEME_ALIASES` 从未被引用 | 文档承诺的 schema 容错没实现。 | 本地没有 MEME 导出可验证 |
| `--seed` 不控制生成 | 只 seed 了数据集构造，从未传给 LLM；真实模型下不可复现。 | 需要模型侧支持 `seed` 参数，且温度 0 也不保证 |
| 成本大头：按 session 粒度摘要 | `494 轮 haystack = 494 次摘要调用` → 约 11 元/题。 | 这是**下一步的研究工作**，不是 bug（见下） |

---

## 9. 建议的下一步

1. **重跑对照实验**。修复前的 `results/final_ab` / `final_merge` 绝对数字不可引用。
   建议：同一份新代码、同 seed、同预算，重跑 `index` vs `merge`，并**记录
   `lazy_summary_count` / `index_updates` / `current_values_tokens`** 以确认新机制真的在动。
2. **按 session 粒度摘要**（评测降本 ~10×）。LongMemEval 每题约 50 个 session
   → 494 次调用降到约 50 次；而且 session 摘要天然就是第二层目录项，**落在既有索引层架构里**。
   实现位置：`add_dialog_turns` / `_generate_only` 支持"一次多轮 → 多块输出"。
3. **验证登记表对真实模型的收益**。它是针对"顶层不可达"的结构性修复，
   建议在 LongMemEval 小样本上先跑 10 题看 History/Current 准确率与 token 的变化。
4. **补 evaluation.py 的测试**。`tests/` 里没有任何评测层测试，本次修的 5 个评测 bug
   全靠手工复现；至少该有一个"resume 后指标不变"和"多系统 CSV 不重复"的测试。

---

## 10. 复现命令

```bash
cd /home/ubuntu/Memory

# 单测（67 个，零依赖、不需要网络/Redis/LLM）
python3 -m unittest discover -s tests

# 四系统离线对照（不花钱）—— 修复前这条命令会崩溃
python3 evaluation.py --synthetic-long --synthetic-turns 40 --limit 3 \
    --systems three_layer,full_context,memgpt_style,naive_chain \
    --llm-backend heuristic --store memory --run-id smoke --out-dir results/smoke

# resume 语义验证：第二条命令应打印与第一条完全相同的指标
python3 evaluation.py --synthetic-long --synthetic-turns 20 --limit 3 \
    --systems three_layer,naive_chain --llm-backend heuristic \
    --store redis_sqlite_hybrid --sqlite-path exp_resume.db \
    --run-id resume_check --out-dir results/resume_check
python3 evaluation.py --synthetic-long --synthetic-turns 20 --limit 3 \
    --systems three_layer,naive_chain --llm-backend heuristic \
    --store redis_sqlite_hybrid --sqlite-path exp_resume.db \
    --run-id resume_check --out-dir results/resume_check

# LongMemEval：类型过滤现在真的能返回结果
python3 evaluation.py --dataset-format longmemeval --lme-types knowledge-update \
    --limit 10 --lme-max-turns 120 --systems three_layer \
    --llm-backend heuristic --store memory --run-id lme_types_check \
    --out-dir results/lme_types_check
```

---

## 11. 本次审查的 commit 序列

| commit | 内容 |
|---|---|
| `77d6b13` | 审查基线（43 单测） |
| `7f14f98` | Redis/SQLite 序列化补齐字段 |
| `4e4f7c1` | 并发记账路径无关 + `ingest_concurrency` 默认值 |
| `57283ad` | 评测层：崩溃 / CSV 重复 / resume / 成本归因 / 跨 run 污染 |
| `3ac2786` | 基线公平性 + 答案完整性 + 工具解析器 |
| `2568b05` | 登记表 + 索引陈旧 + digest 反向 + 懒展开语义 |
| `585e49c` | 合成 gold / 严格 judge / token 口径 / memory-point / 失败 episode |
| `a90929a` | config 快照 / 探针分类 / LongMemEval 截断与过滤 |
| `f5454ce` | 存储健壮性：pipeline 错误、窗口顺序、幂等、计数 |

（另有死代码清理与文档更正，见 `git log`。）
