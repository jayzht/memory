# 把记忆系统接进你自己的 agent

这份文档回答一件事：**你写了一个 agent，怎么让它用上这套记忆。**
可直接运行的例子在 [`examples/agent_with_memory.py`](examples/agent_with_memory.py)。

---

## 一、两条路，选一条

同一套东西有两种用法，都能让 agent 有记忆，区别是**你要不要改代码**：

| | 怎么用 | 要改代码吗 | 自动程度 |
|---|---|---|---|
| **A. 库**（`memory3l`，第二节） | 你的循环里调四次 | 要，约四行 | 完全自动 |
| **B. MCP 插件**（`memory3l-mcp`，第五节） | 起个 MCP 服务，agent 每轮调两个工具 | 不要，改配置 | 靠 agent 自觉调用 |
| **C. 审计面**（`memory3l-mcp` 只读模式） | 指向一个已存在的台账文件 | 不要 | — |

**先讲清楚一个容易混的地方**：C 和 B 是**同一个服务**的两种模式。默认只读（C）时它只能**查**记忆，不能产生记忆；加上 `--allow-write`（B）它才暴露 `remember`/`memory_context`，agent 才真正能**有**记忆。

要"验证记忆可信 / 让别人审我的记忆"，用 C 就够；要"让另一个 agent 用上这套记忆"，得用 B。

---

## 二、路线 A：库，四次调用

整个接入就是一个循环：

```python
from memory3l.llm import build_llm
from memory3l.memory_manager import MemoryManager
from memory3l.prompts import build_agent_messages
from memory3l.store.sqlite_store import SQLiteColdStore

# ── 一次性：装配三件东西 ────────────────────────────────────────────────
store = SQLiteColdStore("agent_memory.db")     # 不需要 Redis、不需要 broker
agent_llm = build_llm("deepseek")              # 你回答用户用的模型
summarizer = build_llm("deepseek")             # 记忆维护用的模型（可以更便宜）
manager = MemoryManager(
    store, summarizer,
    episode_id="user-42",          # 一个 episode = 一段独立对话
    recent_window_turns=4,
    active_chain_token_limit=800,
    reset_on_bind=True,
)

# ── 每一轮 ──────────────────────────────────────────────────────────────
def turn(user_input: str) -> str:
    # 1) 从「记忆」而不是从原始聊天记录构造 prompt
    messages = build_agent_messages(
        manager.get_window(),                    # 第 3 层：最近原文
        manager.chain_summaries(),               # 第 1 层：活跃摘要链
        user_input,
        recent_window_turns=4,
        indexes=manager.rendered_index_entries(),
        current_values=manager.render_current_values(),   # 当前值登记表
    )
    # 2) 调你自己的模型
    answer = agent_llm.generate(messages).text
    # 3) 记录这一轮 —— 摘要、覆盖解析、台账写入都在这里发生
    manager.add_dialog_turn(user_input, answer)
    return answer
```

就这四步。**第 1 步和第 3 步是契约**，其余都是配置。

### 三个必须知道的事实

**① 默认每轮多一次模型调用。** 第 3 步会调 `summarizer`。想要一次调用，用自写模式：让 agent 在自己的回复里输出 `<MEMORY_UPDATE>` 块，然后调 `manager.add_dialog_turn_selfwritten(user, answer, block)` —— 下游的覆盖解析、归档、索引维护是**同一份代码**，只有摘要来源不同。

**② prompt 必须由 `build_agent_messages` 拼。** 顺序（`[滑动窗口][摘要链][system/工具]`）是设计的一部分，不是随便排的。你不能把原始聊天历史直接丢给模型、再"顺便"用一下记忆——那等于没用。

**③ 检索是精确 id 的，没有向量。** 没有 embedding、没有相似度搜索。事实进入 prompt 靠的是**摘要链 + 当前值登记表**，不是"检索相关记忆"。这是这个项目的硬约束，也是它和主流方案最大的区别。

---

## 三、让它能回答"以前是什么"

记忆本体自带历史查询，不需要额外组件：

```python
manager.current("工位楼层")      # 现在是什么，从哪一轮来，证据是什么
manager.history("工位楼层")      # 这个槽位历来所有值（旧→新，含被谁取代）
manager.explain_fact(fact_id)    # 这条事实为什么离开了工作集
manager.evidence(fact_id)        # 回到它来源的原始对话
```

`history` 返回的每一行都带 `from_turn` / `to_turn` / `reason` / `superseded_by`，
这就是"在改成 7 楼之前是什么"能答对的原因。

---

## 四、审计：证明它没丢东西

```python
from memory3l.audit import audit_episode, audit_all

report = audit_episode(store, "user-42").to_dict()
print(report["ok"], report["violations"])
```

五条不变式检查的是 I1 守恒、I2 顶层可达、I3 溯源可解、I4 抽取完整性、I5 删除可验证。

**它不检查抽取是否正确。** 一条错误的事实在守恒、可达、可溯源三个维度上都是"合格"的，所以会通过审计。`I4` 是唯一看原始对话的那条，它抓的是**漏掉**的事实，不是**编造**的事实。这一点在 `examples/agent_with_memory.py` 的离线输出里能直接看到示范。

---

## 五、不写代码的用法：让任何 MCP agent 有记忆（MCP）

第二节那四行是**库**的用法，需要你改自己的循环。如果你不想动代码，同一个 SQLite 文件也可以用 MCP 服务起来，让 agent 通过工具调用来记忆——**这才是"装个插件就能用"的那条路**：

```bash
uvx memory3l-mcp --db agent_memory.db --allow-write
```

在 DSH 的 `$DSH_HOME/profiles/<name>/cordis.patch.yml` 里加一行：

```yaml
- insert:
    - id: mcp-memory
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: memory
        transport: stdio
        command: uvx
        args: ['memory3l-mcp', '--db', '/absolute/path/to/agent_memory.db', '--allow-write']
        env:
          DEEPSEEK_API_KEY: !!js process.env.DEEPSEEK_API_KEY   # 摘要器要用的模型
```

工具会以 `mcp__memory__remember`、`mcp__memory__memory_context`、`mcp__memory__audit` 等名字出现。

### 然后 agent 每轮调两次

```
答之前：memory_context(episode_id)                  → 取回记忆块，放进它自己的 prompt
答之后：remember(episode_id, user_message, answer)   → 记录这一轮
```

**必须说清楚的一点：MCP 没有钩子，所以没有任何东西是自动的。** 你不调 `memory_context`，agent 就是在无记忆的情况下回答问题——哪怕 `remember` 已经记下了前面所有轮次。让 agent 知道要去调这两个工具，靠的是技能：

```bash
memory3l-mcp-install-skill      # → ~/.agents/skills/memory3l
```

技能教它：回合循环的顺序、哪个问题该用哪个工具、以及**如何不过度声称**（"没有违规"不等于"没丢数据"）。

### 三个操作要点

| | |
|---|---|
| **摘要器每轮都要调模型** | 这是 `remember` 里的开销，`--summarizer` 指个小模型 |
| **重启不失忆** | 一个 `episode_id` = 一段对话；链和轮次计数都从 SQLite 读回，新的 server 进程接着往下走 |
| **没有模型也能跑** | `--summarizer none` 时原话进链，记忆仍可用，但没有事实抽取 → 登记表为空、审计无内容。`store_info` 会**明确报告**当前是哪种模式 |

`--allow-write` 是必须的：不加它就是纯审计面（只读），拿不到 `remember`。这是刻意的默认——写记忆是特权操作。

**库 vs MCP，怎么选：**

| | 库（第二节） | MCP（这一节） |
|---|---|---|
| 要改代码 | 是，四行包住你的循环 | 否，改配置 + 让 agent 调工具 |
| 自动程度 | 完全自动 | 每轮两次工具调用，靠 agent 自觉 |
| 跨 agent | 只服务你写的那一个 | 任何 MCP client 共用一份记忆 |

---

## 六、边界：它不给你什么

诚实地讲，以下都是**不做**的：

| 不做 | 意味着 |
|---|---|
| 向量检索 / 语义搜索 | 不能"找找跟这个问题相关的记忆"。进 prompt 的是摘要链和当前值表 |
| **MCP 上的自动接入** | MCP 没有钩子，agent 必须**每轮主动调** `memory_context`/`remember`。要完全自动只能走路线 A（库） |
| 抽取正确性保证 | 抽错了它不知道；只有"漏抽"被 I4 覆盖 |
| 多租户 / 配额 | 做 SaaS 才需要，现在没有 |
| 按主体加密删除 | 现在是"删原文 + 墓碑"，对静态磁盘的历史备份无效 |

另外两条工程注意：

- **摘要器建议用比 agent 更小/更便宜的模型。** 它每轮都跑。
- **`episode_id` 决定隔离边界。** 一个用户一段对话就用一个 id；换 id 且 `reset_on_bind=True` 就会开一段干净的对话（归档与原文永久保留，不删）。

---

## 七、先跑起来看看

```bash
python3 examples/agent_with_memory.py                       # 离线，不需要 key
python3 examples/agent_with_memory.py --backend deepseek    # 真实行为
python3 examples/agent_with_memory.py --show-prompt         # 看 agent 实际收到什么
```

离线模式用规则抽取器，它只认显式句式（`我的X是Y` / `我的X改成Y了`），而且会把问句
`我的X是什么？` 也抽成事实 `X=什么`。那是这个 stub 的毛病，不是记忆系统的；换成真实模型即可。
