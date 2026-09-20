# 可审计记忆层组件（M1）

> 定位：**不做通用记忆系统，做一个能被验证的记忆层组件。**
> 你不卖代码——`remember()`/`current()` 谁都能写；你卖的是**那套他们不会想到去写的检查**。

---

## 一、组件是什么

一个**按精确 id / 槽位检索、不引入向量**的记忆层，保证：

> 每一个被记住的事实都**可溯源**、**有版本**、且**可验证地没有静默丢失**。

三条性质互相独立，各自可测：

| | 保证 | 可测形式 | 需要 gold？ |
|---|---|---|---|
| **I1** | 事实守恒：台账里每个事实都还能解析到一条摘要（活跃或归档） | `missing == 0` | 否（运行时可查） |
| **I2** | 顶层可达：每个槽位的**最新值**在顶层可见（登记表 / 渲染链 / 索引标题） | `unreachable == 0` | 否（运行时可查） |
| **I3** | 溯源可解析：每个事实的 `raw_ref_id` 都能取到原文 | `unresolved == 0` | 否（运行时可查） |
| — | 抽取完整性 | `silent_loss_rate`（对 gold） | **是**（仅评测可查） |

**I1–I3 是运行时不变式**（生产可跑，无需标注）；**`silent_loss_rate` 是评测期指标**。这个区分很重要，见第五节。

---

## 二、接口

```python
# ── 写（唯一入口；被跳过的轮次也会明确说明"什么都没记"）────────
remember(user_text, agent_text) -> FactDelta

# ── 读（精确，无向量）─────────────────────────────────────────
current(slot)                  -> {value, fact_id, since_turn, evidence}
history(slot, upto_turn=None)  -> [{value, fact_id, from_turn, to_turn, superseded_by, reason}]
evidence(fact_id)              -> {turn_index, raw_ref_id, span_text}

# ── 审计（差异化全部在这里）───────────────────────────────────
verify(gold_facts=None) -> AuditReport
explain(fact_id)        -> 它为什么离开工作集、被谁取代、依据是什么
```

M1 已实现 `verify()` / `explain_fact()` / `evidence()`，以及内部的 append-only 事实台账。

---

## 三、审计产物（客户/审计员实际看的东西）

```
=== audit: audit/long_0000 | OK ===
  [ok ] I1_fact_conservation               ledger=20, active=28, archived=12, missing=0
  [ok ] I2_top_level_current_value_reachable slots=8, unreachable=0
  [ok ] I3_provenance_resolvable           checked=20, unresolved_evidence=0, missing_evidence=0
  counters: {'ledger_facts': 20, 'ledger_slots': 8, 'ledger_archived': 12,
             'facts_left_overridden': 12, 'ledger_duplicate_observations': 0}
  gold    : {'gold_facts': 20, 'captured_and_resolvable': 20, 'silent_loss_rate': 0.0}
```

`explain_fact()` 的输出就是"审计员问的那个问题"的答案：

```python
{'fact_id': 'audit_smoke/s002@6ecae5#工位楼层', 'state': 'archived',
 'reason': 'overridden', 'observed_turn': 1,
 'superseded_by': 'audit_smoke/s027@aa3053',
 'evidence': ['audit_smoke/raw001@323d0e']}
```

---

## 四、M1 结果（离线，零成本）

```bash
python3 audit_check.py --episodes 3 --turns 40
```

| 项 | 结果 |
|---|---|
| 三条不变式 | 全部通过（逐 episode） |
| `silent_loss_rate` | **0.0000**（55/55 gold 事实被捕获且证据可解析） |
| 违规数 | 0 |
| 对照 A（静默删除一条摘要，不归档） | ✅ **I1 按预期失败** |
| 对照 B（复现历史序列化 bug：读路径丢 `fact_keys`/`index_id`） | ✅ **I2 按预期失败** |

**对照 B 是这一节的重点**：它就是那个让所有真实 hybrid 运行失效的缺陷的复现——数值写进了 SQLite，但读路径重建对象时丢了字段，于是索引 digest 恒空、登记表恒空。**I2 会在第一时间抓到它。**

---

## 五、第一次运行就抓到了一个真 bug（这是组件价值的最好证明）

M1 的**第一次**运行就报出 `silent_loss_rate = 0.2`——三条存储不变式全 OK，但 20% 的 gold 事实对不上。顺着台账查下去，全部来自离线后端的词法抽取：

| 输入 | 抽出 | 根因 |
|---|---|---|
| `alpha` | `lpha` | `_VALUE_LEADING` 里的 `a\|an` 缺词边界，把单词首字母当冠词删了（`theatre`→`atre` 同理） |
| `最喜欢的饮料` | `饮料` | `clean_slot` 把 `最` 当槽位噪声删掉，导致同一槽位前后对不上、覆盖检测失效 |
| `我的所在城市是上海` | `所` = `城市是上海` | 中文模式的槽位是**非贪婪**的，而 `在` 同时是状态动词——`所在城市` 被切成 `所` + `在` |

**三个都修了**（`memory3l/llm.py`），修完 `silent_loss_rate` 从 0.2 → **0.0**。

**这里有个必须写进说明的结论**：

> I1–I3 只审计**存储路径**。一个**从来没被抽取出来**的事实，对台账是不存在的——内部任何检查都看不见它。
> 所以运行时不变式是**必要但不充分**的，`silent_loss_rate`（对 gold）必须作为组件的一部分存在。

这也正好印证了那个邻居发现：**"静默丢掉关键事实"是被独立承认的失败模式**，而它发生在抽取与保存两段，必须分别度量。

---

## 六、边界：组件**不做**什么

| 不做 | 为什么 |
|---|---|
| ❌ 向量检索 | 这是身份，不是妥协 |
| ❌ 自己的存储引擎 | 时态存储外包（Postgres 系统版本化表 / XTDB / Dolt） |
| ❌ orchestration / agent loop | 客户的 |
| ❌ 多租户 / 鉴权 | 客户的；做 SaaS 才需要 |
| ❌ 决定保留策略 | 但**必须暴露矛盾**：永不删除 ↔ 合规删除，需提供墓碑 / 密钥擦除钩子 |

---

## 七、还没做（M2 候选，按价值排序）

| # | 项 | 说明 |
|---|---|---|
| 1 | **台账持久化** | M1 的台账是**内存内、单 episode**。生产必须落盘（append-only 表），否则重启即失去审计能力。设计是 store-agnostic 的，加一个适配器即可 |
| 2 | **sidecar HTTP 形态** | 审计员要的是**一个端点**，不是一个 Python 对象。`GET /audit`、`GET /fact/{id}/evidence` 是关键接口 |
| 3 | **抽取完整性度量** | `silent_loss_rate` 需要一个 gold 来源；生产上没有 gold 时，需要"候选事实召回率"之类的替代口径 |
| 4 | **合规删除** | 墓碑 + 按主体加密（删密钥），保留"变更史"但满足删除权 |
| 5 | **时态存储适配器** | 把 `archive_reason`/`superseded_by` 映射到系统版本化表，验证"外包存储"这条路真的走得通 |

---

## 八、复现

```bash
python3 -m unittest discover -s tests          # 86 tests
python3 audit_check.py --episodes 3 --turns 40 # 审计 + 两个对照
python3 gate_sweep.py --episodes 3 --turns 40  # 摘要器门控前沿（另一条线）
```

| 文件 | 内容 |
|---|---|
| `memory3l/audit.py` | `FactLedger` + `AuditReport` + 三条不变式的实现 |
| `memory3l/memory_manager.py` | `verify()` / `explain_fact()` / `evidence()`；台账写入点 |
| `audit_check.py` | 审计 + **两个必须失败的对照** |
| `tests/test_core.py` | `TestFactLedgerAudit`（含两个对照）、`TestHeuristicExtraction`（三个 bug 的回归钉） |
