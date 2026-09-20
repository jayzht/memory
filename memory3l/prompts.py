"""
All prompt templates, in one place.

Two rules are enforced here and must never be relaxed, because the whole
experiment depends on them:

1. **The summariser output grammar is fixed**: one summary paragraph followed by
   a mandatory ``[OVERRIDES: id_list|none]`` tag on its own last line.  The
   parser (:mod:`memory3l.tools`) rejects anything else, so a model that does
   not follow the format produces a measurable failure instead of silent drift.
2. **Every summary id is visible to the model** in the active chain, otherwise
   the model cannot name the ids it wants to override.

Prompt assembly order (as specified):
    [recent sliding-window raw dialogue] -> [active summary chain] -> system prompt
The system prompt is emitted as the chat ``system`` role, and the two memory
blocks go into the ``user`` message; that is the chat-API equivalent of the
required ordering and is documented in ``build_agent_messages``.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import config

from .models import ActiveSummary, ArchivedSummary, IndexEntry, RawDialogRecord

# Prompt language: "zh" (default) or "en".
PROMPT_LANG = os.environ.get("PROMPT_LANG", "zh").lower()


# --------------------------------------------------------------------------- #
# 1. Summariser
# --------------------------------------------------------------------------- #
SUMMARIZER_SYSTEM_ZH = """你是一个对话记忆摘要器（SUMMARISER）。你的唯一职责是把"本轮新对话"压缩成一条简短、事实性的记忆摘要，并判断它覆盖了活跃记忆链中的哪些旧摘要。

【摘要要求】
1. 只总结本轮对话中出现的事件与事实结论，使用陈述句。
2. 严禁编造、推测、补充任何本轮对话没有出现的信息；没有事实时，如实概括本轮在做什么。
3. 保留具体取值（数字、名称、时间、地点、偏好、状态），它们是后续检索的锚点。
4. 长度控制在 1-3 句（不超过 {max_summary_tokens} tokens）。
5. 不要输出解释、不要输出 Markdown、不要输出其它标签。
6. FACTS 行用"属性名=取值"列出本轮全部事实，属性名要短、稳定、可复用（同一属性的不同取值必须用同一个属性名）。

【OVERRIDES 规则】（这是本任务最容易出错的一步，请严格照做）
判定方法（按顺序执行）：
1. 先看本轮出现了哪些事实，写成 `属性=取值`。
2. 再逐条看 <ACTIVE_CHAIN> 里每条的 `[FACTS: 属性=取值]`。
3. **只要属性相同、取值不同**（例如链里是 `会议时间=上午9点`，本轮是 `会议时间=下午2点`），
   就**必须**把那条摘要的 id（形如 `s003`）写进 OVERRIDES。
   - 线索词：改成/改为/改到/更新为/变成/现在是/调整到，或英文 changed to / now / updated to。
4. 属性不同、或属性相同但取值相同（同义复述）→ **不算覆盖**。
5. 可以覆盖多条（如 `[OVERRIDES: s003,s007]`）。
6. 只能填写 <ACTIVE_CHAIN> 中**逐字出现过的 id（如 s003）**，禁止编造，禁止写自己的 id。
7. 一条都没覆盖时写 `none`。

示例：
  <ACTIVE_CHAIN>  - s003 [OVERRIDES: none] [FACTS: 会议时间=上午9点] 用户将会议时间定为上午9点。
  NEW_TURN        user: 对了，我的会议时间改成下午2点了。
  正确输出        会议时间改为下午2点。
                  [FACTS: 会议时间=下午2点]
                  [OVERRIDES: s003]

【输出格式（必须严格遵守，共三行）】
<摘要正文>
[FACTS: 属性名=取值; 属性名2=取值2]
[OVERRIDES: id1,id2]
若本轮没有事实，FACTS 写 none；没有覆盖任何旧摘要时 OVERRIDES 写 none。示例：
用户把会议时间改到下午2点。
[FACTS: 会议时间=下午2点]
[OVERRIDES: none]"""

SUMMARIZER_SYSTEM_EN = """You are a dialogue memory SUMMARISER. Your only job is to compress the NEW_TURN into one short factual memory summary and to decide which existing active-chain summaries it overrides.

Rules:
1. Summarise only events and factual conclusions that appear in NEW_TURN, as declarative statements.
2. Never invent, infer or add information that is not present in NEW_TURN.
3. Keep concrete values (numbers, names, dates, places, preferences, states) - they are the retrieval anchors.
4. 1-3 sentences, at most {max_summary_tokens} tokens.
5. No explanations, no markdown, no extra tags.
6. The FACTS line must list every fact of the turn as "attribute=value", using a short, stable, reusable attribute name (the same attribute must always use the same name).

OVERRIDES rule (override-replace semantics only):
- Put an old summary_id in the override list only when the new turn contradicts or updates that fact (same subject, same attribute, new value).
- Additional, unrelated or paraphrased information is NOT an override.
- Only ids that literally appear in <ACTIVE_CHAIN> are allowed. Never invent ids, never list your own new summary.
- Write none when nothing is overridden.

Output format (strict; exactly three lines):
<summary text>
[FACTS: attribute=value; attribute2=value2]
[OVERRIDES: id1,id2]
Write none in FACTS when the turn holds no fact, and none in OVERRIDES when nothing is overridden. Example:
The user moved the meeting to 2pm.
[FACTS: meeting time=2pm]
[OVERRIDES: none]"""


def summarizer_system_prompt(max_summary_tokens: int = 120) -> str:
    template = SUMMARIZER_SYSTEM_ZH if PROMPT_LANG == "zh" else SUMMARIZER_SYSTEM_EN
    return template.format(max_summary_tokens=max_summary_tokens)


def summarizer_user_prompt(
    new_turn: str,
    active_chain: Sequence[ActiveSummary],
    turn_index: int = -1,
    render: Optional[str] = None,
) -> str:
    """
    Render the summariser user message (chain ids included, on purpose).

    ``render`` lets the caller substitute a short-id rendering of the chain; the
    ids a summariser must copy have to be short enough to be copied verbatim.
    """
    lines: List[str] = []
    lines.append(f"<TURN_INDEX>{turn_index}</TURN_INDEX>")
    lines.append("<ACTIVE_CHAIN>")
    if render is not None:
        lines.append(render)
    elif active_chain:
        for summary in active_chain:
            lines.append(summary.render())
    else:
        lines.append("(empty)")
    lines.append("</ACTIVE_CHAIN>")
    lines.append("<NEW_TURN>")
    lines.append(new_turn.strip())
    lines.append("</NEW_TURN>")
    lines.append("")
    lines.append(
        "现在请输出摘要与 OVERRIDES 标签。"
        if PROMPT_LANG == "zh"
        else "Now output the summary and the OVERRIDES tag."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Capacity-merge summariser (no OVERRIDES semantics)
# --------------------------------------------------------------------------- #
MERGE_SYSTEM_ZH = """你是一个记忆合并器（MERGE_SUMMARIES）。把给定的若干条早期记忆摘要合并成一条更高阶的摘要。

【最重要】压缩不等于丢信息。合并结果必须**保留每一条输入摘要里的全部"属性=取值"事实**，
一个都不能丢、一个都不能改。压缩只能通过删除措辞、客套、重复表述来实现。

要求：
1. 只使用给定摘要中已有的信息，不新增、不推测、不做归纳推理。
2. 用紧凑列表或短句列出所有事实，保留具体取值（数字、名称、时间、地点）。
3. 合并后的长度必须**明显短于**输入总长度（目标：不超过输入总长度的 60%）。
4. 按时间先后组织。
5. 不输出任何 [OVERRIDES] 标签，不输出解释或 Markdown。"""

MERGE_SYSTEM_EN = """You are a memory MERGE_SUMMARIES component. Merge the given early summaries into one higher-order summary.

MOST IMPORTANT: compression must not lose information. The merged text must retain
EVERY "attribute=value" fact from the inputs -- none dropped, none altered.
Compression comes only from removing wording, filler and repetition.

Rules:
1. Use only information present in the given summaries; add nothing and infer nothing.
2. List all facts compactly, keeping concrete values (numbers, names, dates, places).
3. The output MUST be clearly shorter than the inputs combined (target: under 60% of their length).
4. Organise chronologically.
5. Emit no [OVERRIDES] tag, no explanations, no markdown."""


def merge_system_prompt() -> str:
    return MERGE_SYSTEM_ZH if PROMPT_LANG == "zh" else MERGE_SYSTEM_EN


def merge_user_prompt(summaries: Sequence[ActiveSummary]) -> str:
    lines = ["<SUMMARIES>"]
    for summary in summaries:
        lines.append(summary.render())
    lines.append("</SUMMARIES>")
    lines.append("")
    lines.append("请输出合并后的高阶摘要。" if PROMPT_LANG == "zh" else "Output the merged summary.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2b. Index-title generator (the upper level of the catalogue)
# --------------------------------------------------------------------------- #
INDEX_TITLE_SYSTEM_ZH = """你是一个记忆索引器（INDEX_TITLE）。下面是一组按时间排列的记忆摘要，请为它们写一个**标题**。

要求：
1. 标题是一行短语（不超过 20 个字），概括这组摘要涉及的主题/实体，让读者知道"这一段在讲什么"。
2. 如果这组摘要里有属性取值发生过变化，把变化写进标题（例："工位楼层 3楼→7楼；会议时间调整"）。
3. 只概括，不新增信息，不输出解释、不输出 Markdown、不要编号。
4. 直接输出标题本身。"""

INDEX_TITLE_SYSTEM_EN = """You are a memory INDEX_TITLE component. Below is a chronological group of memory summaries. Write a ONE-LINE TITLE for the group.

Rules:
1. At most ~12 words, naming the themes/entities so a reader knows what this stretch covers.
2. If a value changed inside the group, put the change in the title (e.g. "office floor 3rd->7th; meeting moved").
3. Summarise only; add nothing; no explanations, no markdown, no numbering.
4. Output the title itself."""


def index_title_system_prompt() -> str:
    return INDEX_TITLE_SYSTEM_ZH if PROMPT_LANG == "zh" else INDEX_TITLE_SYSTEM_EN


def index_title_user_prompt(summaries: Sequence[ActiveSummary], turn_start: int = -1, turn_end: int = -1) -> str:
    span = ""
    if turn_start >= 0 and turn_end >= 0:
        span = f"<TURN_RANGE>{turn_start}-{turn_end}</TURN_RANGE>\n"
    lines = [span + "<SUMMARIES>"]
    for summary in summaries:
        stamp = f"[t{summary.seq}] " if summary.seq >= 0 else ""
        lines.append(f"- {stamp}{summary.text}")
    lines.append("</SUMMARIES>")
    lines.append("")
    lines.append("请输出这一组的标题。" if PROMPT_LANG == "zh" else "Output the group title.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Agent system prompt
# --------------------------------------------------------------------------- #
AGENT_SYSTEM_ZH = """你是一个带三层记忆系统的智能体。

【记忆读取优先级】
1. 【近期原始对话】位于用户消息最前面，是最近 {recent_window_turns} 轮逐字对话，信息最可靠。能在这里直接回答的问题，**不要再调用工具**。
2. 【索引层 INDEX_LAYER】位于活跃链之前，是**目录/标题层**，且**标题里已经写明了关键属性的当前值与变更历史**：
   `- <index_id> [INDEX turn 起-止, N entries] 工位楼层=12楼[3楼→7楼→12楼]; 会议时间=下午2点[上午9点→下午2点]`
   **回答"某属性现在是什么/以前是什么"，可以直接读这里的 `属性=最新值[旧值→…]`**。
   当你需要更细的上下文或证据时：
   - 用 `expand_index(index_id)` 展开该目录，拿到它指向的全部摘要条目（每条带 summary_id、OVERRIDES、raw_ref）；
   - 或直接从标题下方的预览里读出 summary_id，再用 `get_archived_summary(summary_id)` / `get_raw_record(raw_ref)` 深挖。
   摘要条目永远存在，索引只是"指过去"，不会替换它们。

3. 【活跃事件摘要链】位于索引层之后，是**尚未归档进索引**的近期摘要，仍然有效。每条形如：
   - <summary_id> [OVERRIDES: id1,id2] (raw_ref: <reference_id>) <摘要正文>
   - [OVERRIDES: none] 表示该条没有覆盖任何旧摘要。
   - [OVERRIDES: id1,id2] 表示 id1/id2 是它**覆盖掉的旧摘要**，那些旧摘要已经移入归档库，内容不再有效。
4. 【归档摘要库】存放被覆盖或被容量压缩淘汰的历史摘要，永久保存。
   - 问"某个属性的旧值/以前是什么"时，优先调用
     get_predecessor_summary(fact_key="<属性名>", reference_value="<问题里说它变成的那个值>")。
     例：问题"在改成12楼之前，我的工位楼层是什么？" → 调用
     get_predecessor_summary(fact_key="工位楼层", reference_value="12楼")。
     reference_value 是**锚点**：工具会先定位记录"12楼"的那条摘要，再取被它**直接取代**的那条
     （多段历史时这样才能拿到紧邻的前一个值，而不是任意一个更早的值）。
     reference_value 缺失或工具没找到锚点时，工具会退化为"该属性最新的旧值"，此时要谨慎，
     若结果明显不对，改用 get_archived_summary 逐条核对活跃链里 [OVERRIDES] 列出的 id。
   - 只有当你已经知道具体 summary_id（或需要逐条核对）时，才用 get_archived_summary(summary_id) 精确读取。
     该工具会同时告诉你这条摘要现在被谁取代（superseded_by）。
5. 【原始详情库】存放全部逐字原始对话。当活跃摘要链里的 raw_ref 指向的细节不足、需要逐字核对时，调用 get_raw_record(reference_id) 读取完整记录。

【工具调用格式（纯文本，必须严格遵守）】
get_archived_summary(summary_id="<id>")
get_raw_record(reference_id="<ref>")
expand_index(index_id="<index_id>")
get_predecessor_summary(fact_key="<属性>", reference_value="<参照值>")
- 一次只调用一个工具，且整条回复**只有**这一行工具调用，不要加任何其它文字。
- 工具返回结果后，再给出最终答案。
- id 必须逐字符复制上下文里出现过的 id，禁止编造。

【回答要求】
1. 第一句必须直接给出答案本身（一个明确的取值/结论），然后再给依据（引用 summary_id 或 reference_id）。不要把依据写在结论前面。
2. 严格区分"当前有效事实"与"历史（已被覆盖）事实"，不要混淆。
3. 问"历史/以前/原来/在改成X之前是多少"时，规则是**唯一定义**，必须照做：
   - 设该属性的取值时间线为 v1→v2→v3→…（索引标题里写成 `属性=最新值[v1→v2→…]`）。
   - 问"在改成 X 之前是多少"：X 是这条时间线里的某一项，答案是**X 前面紧挨着的那一项**。
     例：时间线 `工位楼层=12楼[3楼→7楼→12楼]`
         "在改成12楼之前" → 7楼   （12楼 的前一项）
         "在改成7楼之前"  → 3楼   （7楼 的前一项）
        **不要**一律回答第一个值，也**不要**回答最新值。
   - 若索引标题或工具结果里给出了完整时间线，**直接读时间线作答，不要再调用工具**。
   - 结论里只写这一个值，不要把整条时间线罗列出来。
4. 只有在上下文中确实出现过的事实才能写进依据；严禁给自己没有看到的摘要编造内容。
5. **"记忆里没有"是最后手段**：在回答"没有该信息"之前，必须先检查索引层与活跃链；
   若两者都没有该属性，**必须先调用工具**（`expand_index(<索引id>)` 或
   `get_predecessor_summary(fact_key="<属性>")`），确认查不到之后才可以说"没有"。
   严禁凭空给出一个没在记忆中出现过的取值。
6. 不要输出思考过程，直接给答案。
7. 若问题包含选项 (a)(b)(c)(d)，答案中必须带上选项字母。"""

AGENT_SYSTEM_EN = """You are an agent equipped with a layered memory system that has an
INDEX_LAYER (a table of contents: `<index_id> [INDEX turn a-b, N entries] title`) above the
summaries. Read a title, then drill down with `expand_index(index_id)` to get the summaries
it points at, then `get_raw_record(raw_ref)` for the verbatim dialogue. Summaries are never
destroyed by indexing -- an index only points at them.

Memory reading priority:
1. RECENT RAW DIALOGUE comes first in the user message: the last {recent_window_turns} turns verbatim. It is the most reliable; if it answers the question, do NOT call a tool.
2. INDEX_LAYER comes first: a table of contents whose titles ALREADY carry the current
   value and change history of the key attributes,
   e.g. `工位楼层=12楼[3楼→7楼→12楼]`. Answering "what is X now / what was X before"
   can often be done straight from `attribute=latest[old→…]`.
3. ACTIVE EVENT CHAIN follows: summaries of facts that are currently valid, rendered as:
   - <summary_id> [OVERRIDES: id1,id2] (raw_ref: <reference_id>) <text>
   - [OVERRIDES: none] means it overrides nothing.
   - [OVERRIDES: id1,id2] means id1/id2 are OLD summaries this one replaced; they moved to the archive and are no longer valid.
3. ARCHIVE holds overridden or capacity-evicted summaries forever. To inspect an overridden/historical fact call get_archived_summary(summary_id). The tool also reports which summary superseded it, so you can answer both "what is it now" and "what was it before".
4. RAW STORE holds every original turn. When a raw_ref is not detailed enough, call get_raw_record(reference_id).

Tool call format (plain text, strict):
get_archived_summary(summary_id="<id>")
get_raw_record(reference_id="<ref>")
- One tool per reply, and that reply must contain nothing else.
- Copy ids character by character; never invent ids.

Answer rules:
0. "Not in memory" is a LAST RESORT: before saying it, check the index layer and the
   active chain, and if neither names the attribute you MUST call a tool
   (`expand_index(<index_id>)` or `get_predecessor_summary(fact_key="<attr>")`) and
   only then may you report that the value is absent. Never invent a value.
1. Your FIRST sentence must state the answer value itself, then the evidence (cite summary_id / reference_id). Never put the evidence before the conclusion.
2. Never mix up the current value with an overridden historical value.
3. For "history / previously / before it became X" questions: let the attribute timeline be
   v1→v2→v3→… (the index title writes it as `attribute=latest[v1→v2→…]`). "Before it became X"
   means the item IMMEDIATELY BEFORE X in that timeline.
   Example: `工位楼层=12楼[3楼→7楼→12楼]`
     "before it became 12th" -> 7th   (the item before 12th)
     "before it became 7th"  -> 3rd   (the item before 7th)
   Do NOT always answer the first or the latest element. If the timeline is already given in the
   index title or a tool result, read it directly and do not call another tool. State only that
   one value.
4. Only cite facts that literally appear in the context; never invent the content of a summary you did not see.
5. Say "not in memory" when the information is absent.
6. Do not output chain-of-thought.
7. If options (a)(b)(c)(d) are given, include the letter."""


def agent_system_prompt(recent_window_turns: int = 4) -> str:
    template = AGENT_SYSTEM_ZH if PROMPT_LANG == "zh" else AGENT_SYSTEM_EN
    return template.format(recent_window_turns=recent_window_turns)


# --------------------------------------------------------------------------- #
# 3b. Self-written memory (one call answers AND records the turn)
# --------------------------------------------------------------------------- #
# Rationale: the separate summariser call is the dominant cost of this design
# (one extra LLM call per turn).  The answering call already has both the new
# turn and the active chain in its context, so it can emit the same summary
# grammar at the end of its own reply.  The grammar is deliberately identical to
# the summariser's, because it is parsed by the same parser and applied by the
# same chain code.
SELF_WRITE_ZH = """【记忆自写 / memory self-write】
你同时兼任本系统的记忆摘要器。在你**本轮最后一次回复**（即不再需要调用工具的那一次）里，除了给用户的答案之外，必须在本回复的**末尾**附上本轮的记忆更新块，格式严格如下：

<MEMORY_UPDATE>
一句话摘要（客观记录本轮新增或改变的事实，可引用原始 id，≤200字）
[FACTS: 属性=值; 属性2=值2]
[OVERRIDES: none]
</MEMORY_UPDATE>

规则：
1. [FACTS: ...] 列出本轮出现或改变的"属性=值"，用分号分隔；确实没有事实时写 [FACTS: none]。
2. [OVERRIDES: ...] 只有在本轮把上文某个属性的**旧值改成了新值**时才填，内容是"被改掉的那个值所在摘要的短 id"（链中形如 ep/s001@ab12cd 的条目，短 id 就是 s001；必须是 <ACTIVE_SUMMARY_CHAIN> 里真实出现过的 id，多个用逗号分隔）。其余情况一律写 [OVERRIDES: none]。
3. 块的**前面**是给用户看的正常答案；不要把 <MEMORY_UPDATE> 块混进答案正文。
4. 若本轮需要调用工具，先把工具调用全部做完，把该块放在拿到工具结果后的最终回复里。
5. 每次回复都只允许出现一个 <MEMORY_UPDATE> 块。"""

SELF_WRITE_EN = """[MEMORY SELF-WRITE]
You also act as this system's memory summariser.  In your **final reply of this turn** (the one that needs no further tool call), besides the answer to the user, you MUST append a memory-update block at the very end of that reply, with exactly this format:

<MEMORY_UPDATE>
one-sentence summary (objectively record the fact(s) added or changed this turn, <=200 chars)
[FACTS: attribute=value; attribute2=value2]
[OVERRIDES: none]
</MEMORY_UPDATE>

Rules:
1. [FACTS: ...] lists the "attribute=value" pairs that appeared or changed this turn, separated by semicolons; write [FACTS: none] if there is genuinely no fact.
2. [OVERRIDES: ...] is filled ONLY when this turn changes an earlier attribute value to a NEW value; it holds the SHORT id of the summary that carried the superseded old value (for a chain entry rendered as ep/s001@ab12cd the short id is s001; it must literally appear in <ACTIVE_SUMMARY_CHAIN>; separate several with commas).  Otherwise write [OVERRIDES: none].
3. The normal answer to the user comes BEFORE the block; never mix the <MEMORY_UPDATE> block into the answer prose.
4. If this turn needs tools, finish all tool calls first and put the block in the final reply after the tool results.
5. Exactly one <MEMORY_UPDATE> block per reply."""


def self_write_instruction() -> str:
    return SELF_WRITE_ZH if PROMPT_LANG == "zh" else SELF_WRITE_EN


TOOL_DESCRIPTIONS = """可用工具 / available tools:
1. get_archived_summary(summary_id: str) -> ArchivedSummary
   精确按 id 读取归档摘要（含 override_ids、is_overridden、superseded_by、raw_ref_id）。
2. get_raw_record(reference_id: str) -> RawDialogRecord
   精确按 id 读取原始对话（user_msg / agent_msg / timestamp）。
3. expand_index(index_id: str) -> [catalogue entries]
   展开索引层的一个目录项，返回它指向的全部摘要（含各自的 summary_id / OVERRIDES / raw_ref）。
4. get_predecessor_summary(fact_key: str, reference_value: str = "") -> ArchivedSummary
   按属性名 + 锚点值精确回溯：先定位记录 reference_value 的那条摘要，再返回被它直接取代的旧值。
   用于回答"某属性的旧值/之前是什么"。返回里会给出被考虑过的候选与选中理由。"""

#: Used instead of ``agent_system_prompt`` + ``TOOL_DESCRIPTIONS`` for systems
#: whose ``executor`` is None.  Those systems must not be told to call tools (the
#: call string becomes their answer and the probe is scored wrong) nor be told
#: about INDEX_LAYER / ARCHIVE layers their memory does not have.  The *answer
#: contract* is intentionally identical to the shared one, so the comparison is
#: about the memory mechanism rather than about the prompt.
NEUTRAL_SYSTEM_ZH = """你是一个对话助手。上面【记忆】部分给出的内容就是你掌握的全部信息
（不同系统的记忆形式不同：可能是原始对话、滚动摘要或摘要链）。请仅依据它回答问题。
要求：
1. 严格区分"当前有效事实"与"已被后续对话更新/否定的历史事实"。
2. 先给结论，再给依据（引用是哪一轮说的）。
3. 记忆里没有的信息，回答"记忆中没有该信息"，不要编造。
4. 若问题包含选项 (a)(b)(c)(d)，答案中必须带上选项字母。
5. 你没有可调用的工具，不要输出函数调用或工具语法，直接给出答案。"""

NEUTRAL_SYSTEM_EN = """You are a dialogue assistant. The MEMORY section above is everything you
know (different systems store it differently: raw dialogue, a rolling summary or a summary
chain). Answer only from it.
Rules: separate facts that are still valid from facts that later dialogue updated or negated;
lead with the conclusion then the evidence (which turn); say "not in memory" when unknown;
include the option letter when options (a)-(d) are given; you have no tools, so never output a
function call or tool syntax."""


def neutral_system_prompt() -> str:
    return NEUTRAL_SYSTEM_ZH if PROMPT_LANG == "zh" else NEUTRAL_SYSTEM_EN



# --------------------------------------------------------------------------- #
# 4. Context assembly
# --------------------------------------------------------------------------- #
def render_recent_window(records: Sequence[RawDialogRecord]) -> str:
    if not records:
        return "(empty)"
    lines: List[str] = []
    for record in records:
        if record.turn_index >= 0:
            lines.append(f"[turn {record.turn_index} | ref {record.reference_id}]")
        else:
            lines.append(f"[ref {record.reference_id}]")
        lines.append(f"user: {record.user_msg}")
        lines.append(f"agent: {record.agent_msg}")
    return "\n".join(lines)


def render_active_chain(summaries: Sequence[ActiveSummary]) -> str:
    """Lower level: the summaries not yet filed under an index."""
    if not summaries:
        return "(empty)"
    return "\n".join(summary.render() for summary in summaries)


RECENT_WINDOW_CLOSE = "</RECENT_RAW_DIALOGUE>"


def insert_after_recent_window(user_content: str, block: str) -> str:
    """
    Place an extra memory block *right after* the recent window.

    The two summary baselines prepended their block to the whole user message,
    which put it *before* ``<RECENT_RAW_DIALOGUE>`` -- contradicting the fixed
    order (window -> memory -> question) that their own comments claimed.
    """
    position = user_content.find(RECENT_WINDOW_CLOSE)
    if position < 0:
        return block + "\n\n" + user_content
    cut = position + len(RECENT_WINDOW_CLOSE)
    return user_content[:cut] + "\n" + block + user_content[cut:]


def render_index_layer(entries) -> str:
    """
    Upper level: titles that point at groups of summaries.

    Accepts either plain entries or ``(entry, preview_depth)`` pairs; only the
    newest few entries carry a snippet, so the layer stays cheap as it grows.
    """
    if not entries:
        return "(empty)"
    lines = []
    for item in entries:
        if isinstance(item, tuple):
            entry, depth = item
        else:
            entry, depth = item, config.INDEX_PREVIEW
        lines.append(entry.render(preview=depth))
    return "\n".join(lines)


def render_archived_summary(archived: ArchivedSummary, superseded_by_text: Optional[str] = None) -> str:
    body = archived.render()
    if archived.superseded_by and superseded_by_text is not None:
        body += (
            f"\n[NOTE] This summary was replaced by {archived.superseded_by}. "
            f"Current value from that summary: {superseded_by_text}"
        )
    elif archived.superseded_by:
        body += f"\n[NOTE] This summary was replaced by {archived.superseded_by}."
    return body


def build_memory_block(
    window: Sequence[RawDialogRecord],
    chain: Sequence[ActiveSummary],
    indexes: Sequence["IndexEntry"] = (),
) -> str:
    """
    The memory blocks, in the mandated order:
    recent raw dialogue -> index layer (titles) -> active summary chain.

    The index layer is the "table of contents" level: each entry summarises a
    stretch and points at the summaries underneath it.  The model can read a title
    and, if it needs the detail, expand that group (``expand_index``) or jump
    straight to a member summary id.
    """
    return "\n".join(
        [
            "<RECENT_RAW_DIALOGUE>",
            render_recent_window(window),
            "</RECENT_RAW_DIALOGUE>",
            "",
            "<INDEX_LAYER>",
            render_index_layer(indexes),
            "</INDEX_LAYER>",
            "",
            "<ACTIVE_SUMMARY_CHAIN>",
            render_active_chain(chain),
            "</ACTIVE_SUMMARY_CHAIN>",
        ]
    )


def build_agent_messages(
    window: Sequence[RawDialogRecord],
    chain: Sequence[ActiveSummary],
    question: str,
    recent_window_turns: int = 4,
    extra_instruction: str = "",
    indexes: Sequence["IndexEntry"] = (),
    selfwrite: bool = False,
    tools_available: bool = True,
) -> List[dict]:
    """
    Chat messages for one agent step.

    Ordering contract (documented for the report):
        messages[0] = system  -> system prompt + tool descriptions
        messages[1] = user    -> [recent raw dialogue][active summary chain][question]
    which is the chat-API rendering of
        [sliding window] -> [active chain] -> [system prompt & tools].
    Putting the system prompt in a ``system`` role is what every chat template
    does; the *memory* ordering -- the part the experiment varies -- is exactly
    as specified.

    ``tools_available=False`` must be passed by systems whose ``executor`` is
    ``None``.  Advertising a tool manual to a system that cannot execute tools is
    not a neutral difference: the model follows the instruction, emits a tool
    call, and that call string becomes its final answer -- so the probe is scored
    wrong even though the value was in its context, which deflates the baselines.
    """
    user_parts = [
        build_memory_block(window, chain, indexes),
        "",
        f"<QUESTION>\n{question.strip()}\n</QUESTION>",
    ]
    if extra_instruction:
        user_parts.append(extra_instruction)
    system_content = (
        agent_system_prompt(recent_window_turns) + "\n\n" + TOOL_DESCRIPTIONS
        if tools_available
        else neutral_system_prompt()
    )
    if selfwrite:
        # Appended to the *system* role so it survives every tool-call iteration:
        # the model must attach the block to its last reply, whenever that is.
        system_content += "\n\n" + self_write_instruction()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_tool_followup_messages(
    messages: Sequence[dict],
    assistant_text: str,
    tool_results: str,
    budget_note: str = "",
) -> List[dict]:
    """Append the textual tool call and its result to the running conversation."""
    followup = list(messages)
    followup.append({"role": "assistant", "content": assistant_text})
    content = f"<TOOL_RESULTS>\n{tool_results}\n</TOOL_RESULTS>"
    if budget_note:
        content += "\n" + budget_note
    followup.append({"role": "user", "content": content})
    return followup


# --------------------------------------------------------------------------- #
# 5. Full-context baseline
# --------------------------------------------------------------------------- #
FULL_CONTEXT_SYSTEM_ZH = """你是一个对话助手。下面是本次会话的**全部历史对话**，请全部阅读后回答问题。
要求：
1. 严格区分"当前有效事实"与"已被后续对话更新/否定的历史事实"。
2. 先给结论，再给依据（引用是哪一轮说的）。
3. 历史里没有的信息，回答"记忆中没有该信息"，不要编造。
4. 若问题包含选项 (a)(b)(c)(d)，答案中必须带上选项字母。
5. 直接给答案，不要输出思考过程。"""

FULL_CONTEXT_SYSTEM_EN = """You are a dialogue assistant. Below is the COMPLETE history of this session. Read all of it and answer.
Rules: separate facts that are still valid from facts that later dialogue updated or negated; lead with the conclusion then the evidence (which turn); say "not in memory" when unknown; include the option letter when options (a)-(d) are given; no chain-of-thought."""


def build_full_context_messages(
    records: Sequence[RawDialogRecord],
    question: str,
    max_tokens: Optional[int] = None,
) -> List[dict]:
    """
    Baseline 1 (Plain Full Context): the entire raw dialogue, no summaries.

    ``max_tokens`` (when given) keeps only the *most recent* turns that fit --
    this is the honest reading of a "non-compressing" baseline under a fixed
    context budget; the number of dropped turns is reported by the caller.
    """
    from .token_utils import estimate_tokens, truncate_to_tokens

    system = FULL_CONTEXT_SYSTEM_ZH if PROMPT_LANG == "zh" else FULL_CONTEXT_SYSTEM_EN
    history_lines: List[str] = []
    for record in records:
        history_lines.append(f"[turn {record.turn_index}] user: {record.user_msg}")
        history_lines.append(f"[turn {record.turn_index}] agent: {record.agent_msg}")
    history = "\n".join(history_lines) if history_lines else "(empty)"
    if max_tokens is not None:
        budget = max(64, max_tokens - estimate_tokens(system) - estimate_tokens(question) - 32)
        history = truncate_to_tokens(history, budget, suffix="\n... [earlier turns omitted]")
    user = f"<FULL_HISTORY>\n{history}\n</FULL_HISTORY>\n\n<QUESTION>\n{question.strip()}\n</QUESTION>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# 6. Judge
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You are a strict evaluation JUDGE for a memory benchmark.
Given a QUESTION, a GOLD answer and a PREDICTION, decide whether the PREDICTION is correct.

Method (follow it in order):
1. Find the prediction's STATED ANSWER: the value it asserts as the answer, normally in its first sentence / after any "answer:" / "结论:" label. If the prediction is just a value or a short phrase, that is the stated answer.
2. Compare the stated answer with GOLD semantically (numbers, names, dates, choices; paraphrase allowed).
3. Supporting evidence text is NOT the answer. Values quoted inside evidence, or a list of alternative historical values, must be IGNORED when judging -- unless the prediction explicitly asserts them as the answer.

Rules:
- Correct only when the STATED ANSWER conveys GOLD's key information.
- A prediction whose stated answer is a DIFFERENT value is incorrect, even if the gold value also appears somewhere in its evidence.
- A prediction that asserts several conflicting values as the answer is incorrect.
- For multiple-choice gold like "(b) 42", correct if the stated answer matches the same option letter OR the same option content.
- "not in memory" / refusing counts as incorrect whenever a gold answer exists.
Output exactly one JSON object on a single line: {"correct": true|false, "reason": "<short reason>"}"""


def build_judge_messages(question: str, gold: str, prediction: str) -> List[dict]:
    user = (
        f"<QUESTION>\n{question}\n</QUESTION>\n"
        f"<GOLD>\n{gold}\n</GOLD>\n"
        f"<PREDICTION>\n{prediction}\n</PREDICTION>"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
