# 可审计记忆层组件

> **定位**：不做通用记忆系统，做一个**能被验证的记忆层组件**。
> 你卖的不是代码——`remember()`/`current()` 谁都能写；你卖的是**那套他们不会想到去写的检查**。
>
> 状态：M1（不变式 + 台账）、M2（落盘 + sidecar）、M3（抽取完整性 / 合规删除 / 时态外包 / 事实写入 / 聚合审计）**全部完成**。
> `python3 -m unittest discover -s tests` → **101 passed**。

---

## 一、组件是什么

一个**按精确 id / 槽位检索、不引入向量**的记忆层，保证：

> 每一个被记住的事实都**可溯源**、**有版本**、**可验证地没有静默丢失**；而每一个**被删除**的事实都**可验证地真的没了**。

---

## 二、五条不变式

| | 保证 | 可测形式 | 需要 gold？ |
|---|---|---|---|
| **I1** | 事实守恒：台账里每个事实都还能解析到一条摘要（活跃或归档） | `missing == 0` | 否 |
| **I2** | 顶层可达：每个槽位的最新值在顶层可见（登记表 / 渲染链 / 索引标题） | `unreachable == 0` | 否 |
| **I3** | 溯源可解析：每个（未删除）事实的 `raw_ref_id` 都能取到原文 | `unresolved == 0` | 否 |
| **I4** | 抽取完整性：说出口的事实是否进了记忆 | `strict_recall` + `gaps` | 否（独立词法检测器） |
| **I5** | 删除可验证：被删除的事实真的读不到了 | `value_still_present == 0` 且 `evidence_still_resolvable == 0` | 否 |
| — | 抽取真值 | `silent_loss_rate` | **是**（仅评测期） |

**I1–I5 都是运行时可查的**（生产可跑，无需标注）；`silent_loss_rate` 需要 gold，是评测期指标。

> **为什么需要 I4**：I1–I3 只审计**存储路径**。一个**从来没被抽取出来**的事实，对台账根本不存在——内部任何检查都看不见它。M1 第一次运行正是这个情形：`silent_loss_rate = 0.2`，而三条存储不变式全绿。
>
> **为什么需要 I5**：I1 证明"什么都没丢"，I5 证明"该删的真的删了"。两者是镜像。

---

## 三、接口

```python
# ── 写 ────────────────────────────────────────────────────────────────
remember(user_text, agent_text) -> FactDelta        # 抽取 + 判定新版本/更新/忽略

# ── 读（精确，无向量）────────────────────────────────────────────────
current(slot)                  -> {value, fact_id, since_turn, evidence}
history(slot, upto_turn=None)  -> [{value, fact_id, from_turn, to_turn, superseded_by, reason, erased}]
evidence(fact_id)              -> {turn_index, raw_ref_id, span_text}

# ── 审计（差异化全部在这里）──────────────────────────────────────────
verify(gold_facts=None, extraction_min_recall=None) -> AuditReport
explain_fact(fact_id)          -> 它为什么离开工作集、被谁取代、依据是什么
erase_facts(fact_ids, reason)  -> 合规删除 + 残余暴露报告
```

---

## 四、审计产物

```
=== audit: audit/long_0000 | OK ===
  [ok ] I1_fact_conservation               ledger=20, active=28, archived=12, missing=0
  [ok ] I2_top_level_current_value_reachable slots=8, unreachable=0
  [ok ] I3_provenance_resolvable           checked=20, erased_exempt=0, unresolved_evidence=0
  [ok ] I4_extraction_completeness         strict_candidates=9, strict_captured=9, strict_recall=1.0,
                                           all_recall=0.79, gaps=<0 item(s)>, min_recall=0.0
  [ok ] I5_deletion_verifiable             erased_facts=0, value_still_present=0,
                                           evidence_still_resolvable=0, residual_prose_mentions=0
  counters: {'ledger_facts': 20, 'ledger_slots': 8, 'ledger_archived': 12, ...}
  gold    : {'gold_facts': 20, 'captured_and_resolvable': 20, 'silent_loss_rate': 0.0}
```

`explain_fact()` 的输出就是"审计员问的那个问题"的答案：

```python
{'fact_id': 'audit/long_0000/s002@6ecae5#工位楼层', 'state': 'archived',
 'reason': 'overridden', 'observed_turn': 1,
 'superseded_by': 'audit/long_0000/s027@aa3053',
 'evidence': ['audit/long_0000/raw001@323d0e']}
```

---

## 五、实测结果（离线，零成本）

| 项 | 结果 |
|---|---|
| `silent_loss_rate` | **0.0000**（55/55 gold 事实被捕获且证据可解析） |
| I4 strict recall | **0.91–1.00**（合成集，3 episodes） |
| 台账守恒 | 全部 episode `missing = 0` |
| 聚合审计（4 episodes） | clean=4, with_violations=0, ledger_facts=48 |
| 时态投影等价性 | `current` + `history`（多个 as-of 点）与台账**完全一致**；干净运行 0 异常 |

### 四个**必须失败**的对照

一个不会失败的检查等于没有检查。`audit_check.py` 故意破坏系统四次，断言对应不变式会失败（若未失败则退出码非 0）：

| 对照 | 破坏方式 | 必须失败 |
|---|---|---|
| **A** | 从 store 里静默删掉一条摘要（不归档） | **I1** ✅ |
| **B** | 复现历史序列化 bug（读路径丢 `fact_keys`/`index_id`） | **I2** ✅ |
| **C** | 让抽取器完全失效 | **I4** ✅（且 I1–I3 **仍然全绿**——这就是盲区） |
| **D** | 只打墓碑、不删原文 | **I5** ✅ |

**对照 C 最有说服力**：抽取器彻底死掉时，I1–I3 报告一个"健康"的系统，只有 I4 发现它什么都没记住。

---

## 六、合规删除

`erase_facts(fact_ids, reason)` 做四件事，然后**检查自己**：

1. 在清空值**之前**先量残余暴露；
2. 把该事实从活跃摘要的 `fact_keys` 里摘掉 → 它从派生登记表消失（**还会被渲染给模型的值，不算被删除**）；
3. 销毁原文记录，但**只销毁没有任何幸存事实还需要的那一些**；
4. 台账行变成墓碑：值清空、**证据指针保留**，这样 I5 能**证明**数据没了，而不是相信一个标志。

**残余散文如实上报**：LLM 写的摘要不能安全改写，所以那一部分作为 `residual_prose_mentions` 暴露，而不是假装删干净。

---

## 七、时态存储外包契约

组件**不拥有存储引擎**。"这个事实在第 16 轮变了"正是时态表（SQL:2011 系统版本化 / XTDB / Dolt / `valid_from`+`valid_to`）擅长的事。客户需要的是**契约**而不是实现：

> 如果你的时态存储能以下面的形状回答 `current(slot)` 与 `history(slot, upto)`，本组件就可以用它替代自己的表。

```sql
CREATE TABLE fact_history (
    episode_id TEXT NOT NULL, slot TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',              -- 合规删除后为 ''
    fact_id TEXT NOT NULL,
    valid_from_turn INTEGER NOT NULL,
    valid_to_turn   INTEGER,                     -- NULL = 仍然是当前值
    superseded_by TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '', erased INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (fact_id)
);
CREATE UNIQUE INDEX idx_open ON fact_history(episode_id, slot) WHERE valid_to_turn IS NULL;
```

`TemporalFactTable.anomalies()` 报告版本化表的**六种坏法**：区间重叠、区间有洞、同一槽位两个开区间、没有开区间、已删除行仍带值、`valid_to` 早于 `valid_from`。**当存储是别人的时候，这就是你要告警的东西。**

---

## 八、sidecar（只读服务 + 一个受审计的写入口）

不需要 Redis（一切从 SQLite 推导），agent 进程退出后照样工作：

```bash
python3 audit_server.py --sqlite-path exp_lm.db --port 8020 --token mem2024
```

| 端点 | 用途 |
|---|---|
| `GET /health` | 存活 + 用的是哪个 db |
| `GET /episodes` | 有台账的 episode 列表 |
| `GET /audit?episode=<id>` | AuditReport（I1–I5 + counters） |
| `GET /audit/summary` | **跨 episode 聚合**（单个违规在均值里看不见） |
| `GET /current?episode=<id>` | 顶层当前值 |
| `GET /history?episode=<id>&slot=<s>[&upto_turn=N]` | 该槽位的取值历史 |
| `GET /temporal?episode=<id>` | 时态投影 + 一致性检查 + DDL |
| `GET /tombstones?episode=<id>` | 合规删除记录（什么没了） |
| `GET /fact?fact_id=<id>` | 这个事实为什么离开工作集 |
| `GET /fact/evidence?fact_id=<id>` | 它的原文出处 |
| `POST /facts?episode=<id>` | **幂等**的事实写入（写入后立刻审计） |

**两个真实的集成陷阱（都已在代码里处理）**：

1. **`fact_id` 含 `#`**（`<episode>/<kind><seq>@<hash>#<slot>`）——URL 里 `#` 之后是 fragment，客户端把 id 拼进路径会**静默失去 slot**。规范形式是 `?fact_id=`，路径形式也做百分号解码。
2. **事实写入不能让值变可见**：登记表是从摘要的 `fact_keys` 派生的，注入一个事实并不会让它出现在顶层。所以 `POST /facts` **先跑审计再回答**——调用方立刻知道它破坏了什么（I1 悬空引用 / I2 顶层不可见）。

---

## 九、边界：组件**不做**什么

| 不做 | 为什么 |
|---|---|
| ❌ 向量检索 | 这是身份，不是妥协 |
| ❌ 自己的存储引擎 | 外包给系统版本化表（见第七节契约） |
| ❌ orchestration / agent loop | 客户的 |
| ❌ 多租户 / 鉴权 | 客户的；做 SaaS 才需要 |
| ❌ 重写 LLM 写的散文 | 不能安全改写，所以**上报残余**而不是假装干净 |

---

## 十、尚未完成（诚实清单）

| # | 项 | 说明 |
|---|---|---|
| 1 | **生产环境的 I4 替代口径** | `strict_recall` 依赖一个词法检测器，而它在真实对话上判别力有限（见 `GATE_EXPERIMENT.md`）。生产需要一个语义口径的"候选事实召回"——**这是目前最大的空缺** |
| 2 | **按主体加密（crypto-shredding）** | 现在是"删原文 + 墓碑"，对静态磁盘上的历史备份无效。真正的删除权需要按主体加密、删密钥 |
| 3 | **真实时态适配器** | 契约与等价性已验证，但还没有一个真的把 `fact_ledger` 同步到 Postgres/XTDB 的 adapter |
| 4 | **多租户隔离与配额** | 做 SaaS 才需要 |
| 5 | **散文重写** | 残余暴露目前只上报；真正"改写到不含该值"需要 LLM 重写 + 二次验证 |

---

## 十一、复现

```bash
python3 -m unittest discover -s tests                        # 109 tests
python3 audit_check.py --episodes 3 --turns 40               # 审计 + 四个必须失败的对照
python3 audit_check.py --aggregate --episodes 6              # 附带跨 episode 聚合
python3 gate_sweep.py --episodes 3 --turns 40                # 摘要器门控前沿（另一条线）
python3 gate_sweep.py --dataset longmemeval --lme-records 25 # 真实数据上的门控失效诊断

# sidecar
python3 audit_server.py --sqlite-path exp_lm.db --port 8020 --token mem2024
curl 'http://127.0.0.1:8020/audit?episode=three_layer%2Flong_0000&token=mem2024'

# MCP（需要 mcp>=2；用独立 venv 以免污染主环境）
python3 memory3l-mcp/scripts/seed_demo.py --db demo_ledger.db --episodes 2 --turns 30
MEMORY3L_ROOT=$PWD PYTHONPATH=$PWD/memory3l-mcp/src \
  .venv-mcp/bin/python -m unittest discover -s memory3l-mcp/tests   # 19 tests
```

---

## 十二、文件清单

| 文件 | 内容 |
|---|---|
| `memory3l/audit.py` | `FactLedger`（append-only + 墓碑）、`AuditReport`、I1–I5、`extraction_report`、`audit_episode`、`audit_all`、`derive_current_values` |
| `memory3l/temporal.py` | `TemporalFactTable`：SCD-2 投影、契约查询、异常检测、DDL |
| `memory3l/gate.py` | 摘要器内容门控（4 档），含真实数据失效证据 |
| `memory3l/memory_manager.py` | `verify` / `explain_fact` / `evidence` / `current` / `history` / `erase_facts`；台账写入点 |
| `memory3l/store/*` | `fact_ledger` 表（append-only + 墓碑列）、三套后端实现、冷存储读侧便利方法 |
| `audit_server.py` | 只读审计 sidecar + 幂等事实写入 |
| `audit_check.py` | 审计运行器 + **四个必须失败的对照** + 聚合模式 |
| `gate_sweep.py` | 门控前沿扫描 + 真实数据证据轮诊断 |
| `tests/test_core.py` | 109 个测试，含每个对照与每个已修 bug 的回归钉 |

---

## 十三、MCP 服务与跨 agent 分发

不把审计能力锁在 sidecar 的 HTTP 里：同一个 `AuditService` 也以 MCP 工具暴露，于是任何 MCP client（Claude Code / Codex / Cursor / DSH / Copilot）都能直接调用。

| 文件 | 内容 |
|---|---|
| `memory3l/service.py` | `AuditService`：审计面的唯一实现，sidecar 与 MCP **共用**，避免两种传输给出不同结论 |
| `memory3l-mcp/` | PyPI 包：`MCPServer`（stdio）、自管存储路径、10 个工具 |
| `memory3l-mcp/skills/memory3l/SKILL.md` | 跨 agent 技能：记忆的回合循环（`remember`/`memory_context`）+ 审计纪律 |
| `memory3l-mcp/server.json` | MCP Registry 元数据（`registryType: pypi`、`runtimeHint: uvx`） |

**已发布**（PyPI + MCP Registry）：

| 位置 | 内容 |
|---|---|
| PyPI | `memory3l 1.0.0`、`memory3l-mcp 0.1.0` |
| MCP Registry | `io.github.jayzht/memory3l-mcp`（`registryType: pypi`、`runtimeHint: uvx`、stdio） |

```bash
uvx memory3l-mcp                                  # 无需仓库、无需 MEMORY3L_ROOT
memory3l-mcp-install-skill                        # → ~/.agents/skills/memory3l
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.jayzht/memory3l-mcp"
```

Registry 的所有权校验靠 **PyPI 上的 README** 含 `mcp-name:` 标记并与 `server.json` 的 `name` 完全一致——所以**改了 README 里的名字就必须同步改 `server.json`**，否则下一次发布会被拒。

验证方式：从 PyPI 全新装（干净 venv，仓库不在 `sys.path`）→ stdio 握手 10 个工具 → `store_info` / `list_episodes` / `audit` 全部返回正确结果；Registry 条目另经公开 API 检索复核。发布流程是 `mcp-publisher login github`（交互式 OAuth）→ `publish`，事后已 `logout` 清除本地凭据。

```bash
uvx memory3l-mcp                                  # 无需仓库、无需 MEMORY3L_ROOT
memory3l-mcp-install-skill                        # → ~/.agents/skills/memory3l
```

验证方式：从 PyPI 全新装（干净 venv，仓库不在 `sys.path`）→ stdio 握手 10 个工具 → `store_info` / `list_episodes` / `audit` 全部返回正确结果。

为此做了一处必要的重构：`config.py` 原本在仓库根、被 4 个包内模块 `import config`，**装成 wheel 后 `import memory3l` 会直接失败**。现在配置移入 `memory3l/config.py`，包内改用相对导入，仓库根 `config.py` 保留为**模块别名**（`sys.modules[__name__] = memory3l.config`）——因为根脚本（`evaluation.py` 等）不只读它，还会**写**它（CLI flag 回写），若用 `import *` 会变成两个对象、flag 静默失效。


设计取舍：

* **默认只读**。10 个工具里 9 个只读；唯一的写入口 `append_facts` 仅在 `--allow-write` / `MEMORY3L_ALLOW_WRITE=1` 时注册。
* **空库可服务**。`SQLiteColdStore` 自动建目录与建表，因此新装机器上 server 能启动并回答 `episodes: []`，而不是崩溃——"装了没反应"是即插即用最常见的死法。
* **未知 episode 是错误，不是空结果**。零事实的 episode 平凡满足全部不变式，所以对拼错的 id 返回 `ok: true` 会是一张**虚假的健康证明**。工具显式报 `unknown episode` 并列出真实 id。
* **不需要 `bind_episode` 的冷存储**（本次补上）：见下。

### 顺带修掉的一个潜伏契约违规（由 I1 抓出）

`SQLiteColdStore.remove_active_summary` 原本只执行 DELETE 并返回 `None`，而 `BaseMemoryStore` 的契约是返回被移除的 `ActiveSummary`。`MemoryManager` 的 override 路径正是靠这个返回值决定是否归档：

```python
old = self.store.remove_active_summary(overridden_id, episode_id=self.episode_id)
if old is None:
    continue                                   # ← 永远命中，归档被跳过
self.store.add_archived_summary(ArchivedSummary.from_active(old, ...))
```

后果是**每一次 override 都静默丢弃被覆盖的摘要**：`archived=0`、`missing=9`。

为什么以前没暴露：hybrid 路径先自己 `get_active_summary` 取值、再忽略冷库返回值，把它掩盖了。本次给冷库补上 episode 绑定 / 轮次记账 / 窗口追加，使它**能单独驱动 manager**，才把这个潜伏违规暴露出来。

证据链（可复现）：

| 行为 | 回归测试 | I1 实测 |
|---|---|---|
| 修复前（monkeypatch 还原） | **必须失败**（已确认） | `ok=false, ledger=15, archived=0, missing=9` |
| 修复后 | 通过 | `ok=true, ledger=15, archived=9, missing=0` |

这是"可审计"这一主张的正面证据：审计面不是装饰，它在真实代码库里抓到了一个其它测试全绿时看不见的数据丢失。
