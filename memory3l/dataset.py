"""
Dataset loading for batch evaluation.

Supported inputs
----------------
1. **Custom evaluation JSON** (the primary format for controlled experiments):
   ``{"episodes": [ ... ]}`` or a bare ``[ ... ]`` list, with any of these
   episode shapes::

       {
         "episode_id": "ep_0001",
         "dialogues": [{"user": "...", "agent": "..."}, ...],
         "probes":    [{"question": "...", "answer": "...", "type": "current_fact",
                        "fact_key": "food", "history_ref": "s001"}, ...],
         "facts":     {"food": [["2024-01-01", "pizza"], ["2024-02-01", "sushi"]]}
       }

   A flat ``[{"user": ..., "agent": ...}, ...]`` episode (no probes) is also
   accepted -- it is useful for memory-behaviour logging without QA scoring.

2. **MEME dataset** (``meme-benchmark/MEME``).  MEME is multi-session and
   multi-entity; the exact parquet/json field names have changed between
   revisions, so the loader is schema-tolerant and maps any combination of the
   known aliases onto :class:`Episode`.  Use ``python evaluation.py --inspect``
   to print the detected schema of a local MEME export before a big run.

3. **Synthetic episodes** (``build_synthetic_episodes``): fully controlled
   fact-update episodes with known overwrite chains.  These are what makes
   Current_Fact_Acc vs History_Fact_Acc separable, so they are the recommended
   debugging/validation set and the basis of the unit tests.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CURRENT_FACT = "current_fact"
HISTORY_FACT = "history_fact"
#: Probe that names a concrete memory point (summary id or raw reference id) and
#: asks what it contains.  Requires the archival/raw tools, so it is how
#: ``Tool_Call_Success_Rate`` is measured on a synthetic set.  Probes of this type
#: may be written by hand (``"type": "memory_point"``) or generated at run time by
#: ``evaluation.py`` (``--memory-point-probes``).
MEMORY_POINT = "memory_point"
OTHER = "other"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Probe:
    """One QA probe over an episode's memory."""

    question: str
    answer: str
    probe_type: str = CURRENT_FACT
    fact_key: str = ""
    history_ref: str = ""
    expected_summary_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "probe_type": self.probe_type,
            "fact_key": self.fact_key,
            "history_ref": self.history_ref,
            "expected_summary_id": self.expected_summary_id,
        }


@dataclass
class Episode:
    """One independent dataset sample (memory must be isolated per episode)."""

    episode_id: str
    dialogues: List[Tuple[str, str]] = field(default_factory=list)
    probes: List[Probe] = field(default_factory=list)
    facts: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.dialogues)

    def current_probes(self) -> List[Probe]:
        return [p for p in self.probes if p.probe_type == CURRENT_FACT]

    def history_probes(self) -> List[Probe]:
        return [p for p in self.probes if p.probe_type == HISTORY_FACT]

    def memory_point_probes(self) -> List[Probe]:
        return [p for p in self.probes if p.probe_type == MEMORY_POINT]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "num_turns": self.num_turns,
            "probes": [p.to_dict() for p in self.probes],
            "facts": {k: list(v) for k, v in self.facts.items()},
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Custom JSON loader
# --------------------------------------------------------------------------- #
_DIALOG_KEYS = (
    ("user", "agent"),
    ("user_msg", "agent_msg"),
    ("user_input", "agent_output"),
    ("question", "answer"),
    ("human", "assistant"),
    ("input", "output"),
    ("role_user", "role_assistant"),
)


def _coerce_dialogue(item: Any) -> Optional[Tuple[str, str]]:
    """Best-effort conversion of one dataset row into a (user, agent) pair."""
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    if not isinstance(item, dict):
        return None
    for user_key, agent_key in _DIALOG_KEYS:
        if user_key in item and agent_key in item:
            return str(item[user_key]), str(item[agent_key])
    # OpenAI-style message list
    if "messages" in item and isinstance(item["messages"], list):
        user_parts: List[str] = []
        agent_parts: List[str] = []
        for message in item["messages"]:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role in ("user", "human"):
                user_parts.append(content)
            elif role in ("assistant", "agent", "ai"):
                agent_parts.append(content)
        if user_parts or agent_parts:
            return "\n".join(user_parts), "\n".join(agent_parts)
    # Fallback: concatenate every string field, split in half (last resort).
    for key in ("text", "content", "dialogue", "session"):
        if key in item and isinstance(item[key], str):
            return item[key], ""
    return None


def _normalise_probe_type(raw: Any, question: str = "") -> str:
    text = str(raw or "").strip().lower()
    if text in ("current_fact", "current", "currentfact", "now", "latest"):
        return CURRENT_FACT
    if text in ("history_fact", "history", "historyfact", "archived", "overridden", "past"):
        return HISTORY_FACT
    if text in ("memory_point", "memorypoint", "mem_point", "recall", "lookup", "retrieval"):
        return MEMORY_POINT
    if text:
        return text
    # Infer from the question text when the dataset does not label probes.
    lowered = question.lower()
    if any(token in lowered for token in ("before", "previously", "used to", "originally", "was", "earlier", "以前", "之前", "原来")):
        return HISTORY_FACT
    return OTHER


def _coerce_probe(item: Any, index: int) -> Optional[Probe]:
    if isinstance(item, str):
        return Probe(question=item, answer="", probe_type=OTHER)
    if not isinstance(item, dict):
        return None
    question = (
        item.get("question")
        or item.get("query")
        or item.get("prompt")
        or item.get("q")
        or ""
    )
    answer = (
        item.get("answer")
        or item.get("gold")
        or item.get("target")
        or item.get("expected")
        or item.get("label")
        or ""
    )
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    probe_type = _normalise_probe_type(
        item.get("type") or item.get("probe_type") or item.get("category"), str(question)
    )
    return Probe(
        question=str(question),
        answer=str(answer),
        probe_type=probe_type,
        fact_key=str(item.get("fact_key") or item.get("key") or item.get("entity") or ""),
        history_ref=str(item.get("history_ref") or item.get("reference") or item.get("summary_id") or ""),
        expected_summary_id=str(item.get("expected_summary_id") or ""),
        meta={k: v for k, v in item.items() if k not in ("question", "answer", "gold", "type")},
    )


def _coerce_facts(raw: Any) -> Dict[str, List[Tuple[str, str]]]:
    """Normalise the many ways a dataset can describe fact timelines."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, list):
                pairs: List[Tuple[str, str]] = []
                for entry in value:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        pairs.append((str(entry[0]), str(entry[1])))
                    elif isinstance(entry, dict):
                        pairs.append(
                            (
                                str(entry.get("timestamp") or entry.get("time") or entry.get("turn") or len(pairs)),
                                str(entry.get("value") or entry.get("fact") or entry.get("text") or ""),
                            )
                        )
                out[str(key)] = pairs
            elif isinstance(value, (str, int, float)):
                out[str(key)] = [("", str(value))]
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                key = str(entry.get("key") or entry.get("fact_key") or entry.get("entity") or "")
                value = str(entry.get("value") or entry.get("fact") or entry.get("text") or "")
                stamp = str(entry.get("timestamp") or entry.get("time") or entry.get("turn") or "")
                if key:
                    out.setdefault(key, []).append((stamp, value))
    return out


def _coerce_episode(item: Any, index: int) -> Episode:
    prefix = f"ep_{index:05d}"
    if isinstance(item, list):
        dialogues = [d for d in (_coerce_dialogue(x) for x in item) if d]
        return Episode(episode_id=prefix, dialogues=dialogues)
    if not isinstance(item, dict):
        return Episode(episode_id=prefix)

    episode_id = str(
        item.get("episode_id")
        or item.get("id")
        or item.get("sample_id")
        or item.get("session_id")
        or item.get("uid")
        or prefix
    )
    dialogues: List[Tuple[str, str]] = []
    for key in ("dialogues", "dialogue", "conversation", "conversations", "turns", "sessions", "history", "messages"):
        raw = item.get(key)
        if isinstance(raw, list) and raw:
            dialogues = [d for d in (_coerce_dialogue(x) for x in raw) if d]
            if dialogues:
                break

    probes: List[Probe] = []
    for key in ("probes", "questions", "qas", "qa", "test_questions", "queries", "evaluation"):
        raw = item.get(key)
        if isinstance(raw, list):
            for probe_index, entry in enumerate(raw):
                probe = _coerce_probe(entry, probe_index)
                if probe and probe.question:
                    probes.append(probe)
            if probes:
                break
    if not probes and item.get("question"):
        probe = _coerce_probe(item, 0)
        if probe:
            probes.append(probe)

    facts = _coerce_facts(item.get("facts") or item.get("fact_timeline") or item.get("ground_truth") or {})
    meta = {
        k: v
        for k, v in item.items()
        if k
        not in (
            "episode_id", "id", "sample_id", "dialogues", "dialogue", "conversation",
            "conversations", "turns", "sessions", "history", "messages", "probes",
            "questions", "qas", "qa", "test_questions", "queries", "evaluation", "facts",
            "fact_timeline", "ground_truth",
        )
    }
    return Episode(episode_id=episode_id, dialogues=dialogues, probes=probes, facts=facts, meta=meta)


# --------------------------------------------------------------------------- #
# MEME loader
# --------------------------------------------------------------------------- #
#: Alias table for MEME-like field names observed across revisions.
MEME_ALIASES = {
    "episode": ("episode", "episodes", "sample", "samples", "instance", "instances"),
    "dialogue": ("dialogue", "dialogues", "conversation", "conversations", "sessions", "session", "messages", "turns"),
    "question": ("question", "questions", "query", "probe", "probes", "test", "eval_question"),
    "answer": ("answer", "answers", "gold", "label", "target", "expected"),
}


def load_meme_dataset(path: str = None, name: str = "meme-benchmark/MEME", split: str = "train") -> List[Episode]:
    """
    Load MEME.

    ``path`` may be a local JSON/JSONL export.  When it is omitted we try the
    HuggingFace ``datasets`` library; if that (or network access) is unavailable
    we raise a clear, actionable error instead of silently returning nothing.
    """
    if path and os.path.exists(path):
        return load_json_dataset(path)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MEME is not available locally and the `datasets` package is missing.\n"
            "Fix with either:\n"
            "  pip install datasets   # then: python evaluation.py --dataset-name meme-benchmark/MEME\n"
            "  or download a JSON/JSONL export and pass --dataset /path/to/meme.json\n"
            "Use `python evaluation.py --inspect --dataset <file>` to check the detected schema."
        ) from exc

    logger.info("loading MEME from HuggingFace: %s [%s]", name, split)
    dataset = load_dataset(name, split=split)
    rows = [dict(row) for row in dataset]
    return episodes_from_records(rows, source="meme")


def episodes_from_records(records: Sequence[Any], source: str = "generic") -> List[Episode]:
    """Map a list of raw records (HF rows, JSON objects) onto :class:`Episode`."""
    episodes: List[Episode] = []
    for index, record in enumerate(records):
        episode = _coerce_episode(record, index)
        episode.meta.setdefault("source", source)
        if episode.dialogues or episode.probes:
            episodes.append(episode)
    logger.info("loaded %d episodes from %s", len(episodes), source)
    return episodes


# --------------------------------------------------------------------------- #
# JSON / JSONL loader
# --------------------------------------------------------------------------- #
def load_json_dataset(path: str) -> List[Episode]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        records: List[Any] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return episodes_from_records(records, source=os.path.basename(path))

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("episodes", "data", "samples", "instances", "records"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"unsupported dataset root type: {type(payload)}")
    return episodes_from_records(records, source=os.path.basename(path))


# --------------------------------------------------------------------------- #
# Synthetic episodes (controlled fact updates)
# --------------------------------------------------------------------------- #
SYNTH_TEMPLATES_EN = {
    "announce": "Hi, let me tell you something: my {attr} is {value}.",
    "update": "By the way, my {attr} has changed - it is {value} now.",
    "ask_agent": "Please remember that my {attr} is {value}, ok?",
}
SYNTH_TEMPLATES_ZH = {
    "announce": "你好，跟你说一下，我的{attr}是{value}。",
    "update": "对了，我的{attr}改成{value}了。",
    "ask_agent": "请记住，我的{attr}现在是{value}。",
}
SYNTH_ATTRS = [
    ("favourite food", "pizza", "sushi", "ramen", "curry"),
    ("home city", "Beijing", "Shanghai", "Chengdu", "Hangzhou"),
    ("office floor", "3rd", "7th", "12th", "2nd"),
    ("phone number", "13800000001", "13800000002", "13800000003", "13800000004"),
    ("project codename", "falcon", "otter", "lynx", "bison"),
    ("meeting time", "9am", "11am", "2pm", "4pm"),
    ("team size", "4", "6", "9", "12"),
    ("favourite drink", "coffee", "tea", "juice", "water"),
]


def build_synthetic_episodes(
    num_episodes: int = 8,
    turns_per_episode: int = 12,
    seed: int = 20240501,
    language: str = "en",
    noise_ratio: float = 0.35,
) -> List[Episode]:
    """
    Deterministic fact-update episodes.

    Each episode picks a few attributes and changes their values over time.  For
    every changed attribute the episode carries:

    * a ``current_fact`` probe whose gold answer is the *latest* value;
    * one ``history_fact`` probe per superseded value, whose gold answer is the
      value that was valid *before* the change.

    This is the only construction that lets History_Fact_Acc be measured without
    an LLM judge guessing which historical state the question refers to.
    """
    rng = random.Random(seed)
    templates = SYNTH_TEMPLATES_ZH if language == "zh" else SYNTH_TEMPLATES_EN
    episodes: List[Episode] = []

    for episode_index in range(num_episodes):
        attrs = rng.sample(SYNTH_ATTRS, k=min(3, len(SYNTH_ATTRS)))
        dialogues: List[Tuple[str, str]] = []
        script: List[Tuple[int, str, str]] = []  # (turn, attr, value)

        # Every attribute gets 2-3 *distinct* values over the episode.  Values
        # must not repeat: with a timeline like 3rd -> 7th -> 12th -> 7th the probe
        # "before it became 7th, what was it?" has two correct answers, and a
        # perfectly good retrieval then scores as wrong.  Distinct values keep
        # each historical probe unambiguous.
        pending: List[Tuple[str, str]] = []
        for attr, *values in attrs:
            count = rng.choice([2, 2, 3])
            chosen_values = list(values[:count])
            assert len(set(chosen_values)) == len(chosen_values)
            for value in chosen_values:
                pending.append((attr, value))
        rng.shuffle(pending)
        total_fact_turns = min(len(pending), max(1, int(turns_per_episode * (1 - noise_ratio))))
        chosen = pending[:total_fact_turns]

        # Pre-assign each fact to a turn so the template can depend on whether the
        # attribute was already mentioned (announcement vs. update).
        slots = sorted(rng.sample(range(turns_per_episode), k=len(chosen))) if chosen else []
        assignment: Dict[int, Tuple[str, str]] = {turn: pair for turn, pair in zip(slots, chosen)}
        seen_attrs: set = set()

        for turn in range(turns_per_episode):
            if turn in assignment:
                attr, value = assignment[turn]
                style = "announce" if attr not in seen_attrs else "update"
                seen_attrs.add(attr)
                user_text = templates[style].format(attr=attr, value=value)
                script.append((turn, attr, value))
            else:
                user_text = rng.choice(
                    [
                        "How is the weather today?",
                        "Can you remind me what we discussed?",
                        "Tell me a fun fact about databases.",
                        "Let's take a short break.",
                        "Thanks, that helps.",
                    ]
                )
            agent_text = "Got it, I will remember that." if any(t[0] == turn for t in script) else "Sure."
            dialogues.append((user_text, agent_text))

        # Facts with their concrete turn stamps.
        facts: Dict[str, List[Tuple[str, str]]] = {}
        stamps: Dict[str, List[int]] = {}
        for turn, attr, value in script:
            stamps.setdefault(attr, []).append(turn)
        for attr, turns in stamps.items():
            value_of_turn = {t: v for t, a, v in script if a == attr}
            facts[attr] = [(str(t), value_of_turn[t]) for t in turns]

        probes: List[Probe] = []
        for attr, entries in facts.items():
            if not entries:
                continue
            latest_value = entries[-1][1]
            if language == "zh":
                probes.append(
                    Probe(
                        question=f"我现在的{attr}是什么？",
                        answer=latest_value,
                        probe_type=CURRENT_FACT,
                        fact_key=attr,
                    )
                )
            else:
                probes.append(
                    Probe(
                        question=f"What is my {attr} now?",
                        answer=latest_value,
                        probe_type=CURRENT_FACT,
                        fact_key=attr,
                    )
                )
            for index in range(len(entries) - 1):
                old_value = entries[index][1]
                # Anchor on the *next* value in this attribute's own timeline, not on
                # the episode-final value.  Anchoring on the final value emitted one
                # probe per transition with identical wording but different gold
                # ("before it became 12th -> 3rd" AND "... -> 7th"), so the question
                # was unanswerable and a correct lookup was scored wrong.
                next_value = entries[index + 1][1]
                if language == "zh":
                    probes.append(
                        Probe(
                            question=f"在改成{next_value}之前，我的{attr}原来是什么？",
                            answer=old_value,
                            probe_type=HISTORY_FACT,
                            fact_key=attr,
                            history_ref=f"{attr}#{index}",
                        )
                    )
                else:
                    probes.append(
                        Probe(
                            question=f"Before it became {next_value}, what was my {attr}?",
                            answer=old_value,
                            probe_type=HISTORY_FACT,
                            fact_key=attr,
                            history_ref=f"{attr}#{index}",
                        )
                    )
        episodes.append(
            Episode(
                episode_id=f"synth_{episode_index:04d}",
                dialogues=dialogues,
                probes=probes,
                facts=facts,
                meta={"source": "synthetic", "language": language, "seed": seed},
            )
        )
    return episodes


# --------------------------------------------------------------------------- #
# Long-context synthetic episodes (the discriminating regime)
# --------------------------------------------------------------------------- #
# The short generator above produces ~300-token episodes, which the full-context
# baseline can hold entirely: every system scores the same and the comparison says
# nothing.  Real multi-session memory benchmarks are long.  This generator builds
# episodes whose verbatim history is large (multi-thousand tokens), so:
#
#   * the sliding window cannot contain the superseded facts any more;
#   * a no-compression baseline must either blow the context budget (and lose the
#     oldest turns) or pay a large per-turn token cost;
#   * the archive becomes the only way to answer "what was it before", which is
#     exactly the capability the three-layer design claims.
#
# History probes are anchored on updates that happened well before the episode
# end, so they cannot be answered from the recent window.
LONG_ATTRS_EN = [
    ("favourite food", "pizza", "sushi", "ramen", "curry", "tacos"),
    ("home city", "Beijing", "Shanghai", "Chengdu", "Hangzhou", "Shenzhen"),
    ("office floor", "3rd", "7th", "12th", "2nd", "9th"),
    ("phone number", "13800000001", "13800000002", "13800000003", "13800000004"),
    ("project codename", "falcon", "otter", "lynx", "bison", "heron"),
    ("meeting time", "9am", "11am", "2pm", "4pm", "6pm"),
    ("team size", "4", "6", "9", "12", "15"),
    ("favourite drink", "coffee", "tea", "juice", "water", "soda"),
    ("gym schedule", "Monday", "Tuesday", "Thursday", "Saturday"),
    ("backup server", "alpha", "beta", "gamma", "delta", "epsilon"),
]
LONG_ATTRS_ZH = [
    ("最喜欢的食物", "披萨", "寿司", "拉面", "咖喱", "塔可"),
    ("所在城市", "北京", "上海", "成都", "杭州", "深圳"),
    ("工位楼层", "3楼", "7楼", "12楼", "2楼", "9楼"),
    ("手机号码", "13800000001", "13800000002", "13800000003", "13800000004"),
    ("项目代号", "猎鹰", "水獭", "猞猁", "野牛", "苍鹭"),
    ("会议时间", "上午9点", "上午11点", "下午2点", "下午4点", "下午6点"),
    ("团队人数", "4人", "6人", "9人", "12人", "15人"),
    ("最喜欢的饮料", "咖啡", "茶", "果汁", "水", "苏打水"),
    ("健身安排", "周一", "周二", "周四", "周六"),
    ("备份服务器", "alpha", "beta", "gamma", "delta", "epsilon"),
]
# Filler turns are deliberately substantive: a one-word filler would make the
# full-context baseline cheap and hide the context-cost difference.
LONG_FILLER_EN = [
    ("Could you go over what we changed in the deployment pipeline last week?",
     "We reordered the stages so the integration tests run before the container build, added a cache warm-up step, and pinned the base image digest. Rollback is still a single tag revert."),
    ("I am trying to decide between two database engines for the reporting service.",
     "If the workload is mostly analytical, a columnar engine will scan far less data; if it is point lookups, a row store with a good index is simpler. The team already runs both, so operational cost is similar."),
    ("What is the difference between optimistic and pessimistic locking here?",
     "Optimistic locking checks a version column at write time and retries on conflict, which is cheap under low contention. Pessimistic locking takes the row lock up front, which is safer under high contention but can serialise workers."),
    ("Remind me how the retry budget interacts with the circuit breaker.",
     "Retries happen per request with exponential backoff, while the breaker counts failures across requests. Once the breaker opens, retries are rejected immediately, so the budget is only consumed while the breaker is closed."),
    ("I keep forgetting which environment uses the sandbox credentials.",
     "The staging environment uses the sandbox tenant and its keys rotate every thirty days; production uses the live tenant with keys managed by the secret store. Never point a staging job at the live tenant."),
    ("Can we estimate the cost of the nightly batch job?",
     "The batch reads about two hundred gigabytes and writes twenty, so storage egress dominates. Switching the intermediate format to columnar cut the previous run's time by a third."),
    ("How should we handle schema migrations during a rolling deploy?",
     "Additive changes go first, so old and new code can both read the table. Destructive changes wait one full release after the code that stopped using the column is deployed everywhere."),
    ("What did the incident review conclude about the queue backlog?",
     "The consumer was using a fixed thread pool while the producer scaled with traffic, so the backlog grew until the pool was resized and a dead-letter route was added. Alerting now fires on queue age rather than depth."),
]
LONG_FILLER_ZH = [
    ("你能再讲一下上周我们在部署流程里改了什么吗？",
     "我们把集成测试挪到镜像构建之前，加了一个缓存预热步骤，并把基础镜像的摘要固定下来了。回滚依然只是一个标签回退操作。"),
    ("我在给报表服务选数据库引擎，有点犹豫。",
     "如果负载主要是分析型查询，列式引擎扫描的数据量会小很多；如果是点查，带好索引的行存更简单。团队两种都在跑，运维成本差不多。"),
    ("乐观锁和悲观锁在这里到底有什么区别？",
     "乐观锁在写入时校验版本号，冲突就重试，低竞争下开销很小；悲观锁提前持有行锁，高竞争下更安全，但会让工作线程排队。"),
    ("再提醒我一下重试预算和熔断器是怎么配合的。",
     "重试是按请求做指数退避，熔断器统计的是跨请求的失败率。熔断打开后重试会被直接拒绝，所以预算只在熔断关闭时被消耗。"),
    ("我总是记不清哪个环境用的是沙箱凭据。",
     "预发环境用沙箱租户，密钥每三十天轮换一次；生产环境用正式租户，密钥由密钥管理服务托管。绝不要让预发任务连到正式租户。"),
    ("能估算一下每晚批处理任务的开销吗？",
     "批处理大约读取两百 GB、写入二十 GB，所以存储出口流量占大头。把中间格式换成列式之后，上一轮运行时间少了三分之一。"),
    ("滚动发布期间数据库迁移应该怎么处理？",
     "先做增量变更，让新旧代码都能读这张表。破坏性变更要等新代码全量上线后的下一个版本再执行。"),
    ("事故复盘对队列积压的结论是什么？",
     "消费者用的是固定线程池，生产者却随流量扩容，于是积压持续增长，直到扩大线程池并补上死信路由。现在告警看的是队列中最老消息的年龄，而不是队列深度。"),
]


def build_long_context_episodes(
    num_episodes: int = 6,
    turns_per_episode: int = 40,
    seed: int = 20240501,
    language: str = "zh",
    min_history_gap: int = 8,
) -> List[Episode]:
    """
    Long multi-session episodes with unambiguous fact-update timelines.

    Guarantees that make the resulting numbers interpretable:

    * every attribute's values are distinct, so "before it became X" has exactly
      one correct answer;
    * history probes only ask about updates at least ``min_history_gap`` turns
      before the end, so the sliding window alone cannot answer them;
    * filler turns carry real content, so the verbatim history is genuinely large.
    """
    rng = random.Random(seed)
    is_zh = language == "zh"
    attrs = LONG_ATTRS_ZH if is_zh else LONG_ATTRS_EN
    filler = LONG_FILLER_ZH if is_zh else LONG_FILLER_EN
    announce = SYNTH_TEMPLATES_ZH["announce"] if is_zh else SYNTH_TEMPLATES_EN["announce"]
    update = SYNTH_TEMPLATES_ZH["update"] if is_zh else SYNTH_TEMPLATES_EN["update"]
    episodes: List[Episode] = []

    for episode_index in range(num_episodes):
        chosen_attrs = rng.sample(attrs, k=min(8, len(attrs)))
        # 2-3 distinct values per attribute, interleaved so updates are spread out.
        pending: List[Tuple[str, str]] = []
        for attr, *values in chosen_attrs:
            count = rng.choice([2, 2, 3])
            for value in list(values)[:count]:
                pending.append((attr, value))
        rng.shuffle(pending)

        fact_turns = max(len(chosen_attrs), min(len(pending), int(turns_per_episode * 0.5)))
        slots = sorted(rng.sample(range(turns_per_episode), k=fact_turns))
        assignment = {turn: pair for turn, pair in zip(slots, pending[:fact_turns])}

        dialogues: List[Tuple[str, str]] = []
        script: List[Tuple[int, str, str]] = []
        seen: set = set()
        filler_index = 0
        for turn in range(turns_per_episode):
            if turn in assignment:
                attr, value = assignment[turn]
                style = "announce" if attr not in seen else "update"
                seen.add(attr)
                user_text = (announce if style == "announce" else update).format(
                    attr=attr, value=value
                )
                agent_text = (
                    "好的，我记住了。" if is_zh else "Got it, I will remember that."
                )
                script.append((turn, attr, value))
            else:
                user_text, agent_text = filler[filler_index % len(filler)]
                filler_index += 1
            dialogues.append((user_text, agent_text))

        # Fact timelines with concrete turn stamps.
        facts: Dict[str, List[Tuple[str, str]]] = {}
        for turn, attr, value in script:
            facts.setdefault(attr, []).append((str(turn), value))

        probes: List[Probe] = []
        for attr, entries in facts.items():
            latest_turn, latest_value = int(entries[-1][0]), entries[-1][1]
            if is_zh:
                probes.append(
                    Probe(question=f"我现在的{attr}是什么？", answer=latest_value,
                          probe_type=CURRENT_FACT, fact_key=attr)
                )
            else:
                probes.append(
                    Probe(question=f"What is my {attr} now?", answer=latest_value,
                          probe_type=CURRENT_FACT, fact_key=attr)
                )
            # Anchor each probe on the *next* value in that attribute's own
            # timeline, not on the episode-final value.  Using the final value for
            # every step produced several probes with identical wording but
            # different gold answers ("before it became 12th -> 3rd" AND "... -> 7th"),
            # which makes the question unanswerable and misjudges a correct lookup.
            for index in range(len(entries) - 1):
                old_turn, old_value = int(entries[index][0]), entries[index][1]
                next_turn, next_value = int(entries[index + 1][0]), entries[index + 1][1]
                if next_turn - old_turn < min_history_gap:
                    continue
                if is_zh:
                    probes.append(
                        Probe(
                            question=f"在改成{next_value}之前，我的{attr}是什么？",
                            answer=old_value,
                            probe_type=HISTORY_FACT,
                            fact_key=attr,
                            history_ref=f"{attr}#turn{old_turn}->{next_turn}",
                        )
                    )
                else:
                    probes.append(
                        Probe(
                            question=f"Before it became {next_value}, what was my {attr}?",
                            answer=old_value,
                            probe_type=HISTORY_FACT,
                            fact_key=attr,
                            history_ref=f"{attr}#turn{old_turn}->{next_turn}",
                        )
                    )
        episodes.append(
            Episode(
                episode_id=f"long_{episode_index:04d}",
                dialogues=dialogues,
                probes=probes,
                facts=facts,
                meta={
                    "source": "synthetic_long",
                    "language": language,
                    "seed": seed,
                    "turns": turns_per_episode,
                },
            )
        )
    return episodes


# --------------------------------------------------------------------------- #
# Inspection helper
# --------------------------------------------------------------------------- #
def inspect_dataset(path: str = None, episodes: Sequence[Episode] = None) -> Dict[str, Any]:
    """
    Diagnose a dataset before a long run: schema, counts, probe types.

    Printed by ``python evaluation.py --inspect``.  The first raw record is
    included verbatim so an unfamiliar MEME revision can be mapped in seconds.
    """
    raw_preview: Dict[str, Any] = {}
    if path and os.path.exists(path) and not path.endswith((".jsonl", ".ndjson")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            record = payload["episodes"][0] if isinstance(payload, dict) and payload.get("episodes") else (
                payload[0] if isinstance(payload, list) and payload else payload
            )
            if isinstance(record, dict):
                raw_preview = {k: _shorten(v) for k, v in record.items()}
        except Exception as exc:  # noqa: BLE001
            raw_preview = {"<error>": str(exc)}

    episodes = list(episodes or [])
    type_counts: Dict[str, int] = {}
    for episode in episodes:
        for probe in episode.probes:
            type_counts[probe.probe_type] = type_counts.get(probe.probe_type, 0) + 1
    return {
        "num_episodes": len(episodes),
        "num_turns_total": sum(e.num_turns for e in episodes),
        "num_turns_mean": round(sum(e.num_turns for e in episodes) / len(episodes), 2) if episodes else 0,
        "num_probes_total": sum(len(e.probes) for e in episodes),
        "probe_types": type_counts,
        "episodes_without_probes": sum(1 for e in episodes if not e.probes),
        "episodes_without_dialogues": sum(1 for e in episodes if not e.dialogues),
        "first_episode": episodes[0].to_dict() if episodes else {},
        "raw_first_record_keys": list(raw_preview.keys()),
        "raw_first_record_preview": raw_preview,
    }


def _shorten(value: Any, limit: int = 300) -> Any:
    if isinstance(value, str):
        return value[:limit] + ("..." if len(value) > limit else "")
    if isinstance(value, list):
        return [_shorten(v, limit) for v in value[:3]] + (["..."] if len(value) > 3 else [])
    if isinstance(value, dict):
        return {k: _shorten(v, limit) for k, v in list(value.items())[:12]}
    return value


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def load_dataset(
    path: str = None,
    dataset_name: str = None,
    dataset_format: str = "auto",
    limit: int = 0,
    seed: int = 20240501,
) -> List[Episode]:
    """
    Single entry point used by ``evaluation.py``.

    ``dataset_format``: ``auto`` | ``json`` | ``meme`` | ``synthetic``.
    """
    dataset_format = (dataset_format or "auto").lower()
    episodes: List[Episode]

    if dataset_format in ("synthetic_long", "long") or path == "__synthetic_long__":
        episodes = build_long_context_episodes(num_episodes=limit or 6, seed=seed)
    elif dataset_format == "synthetic" or (dataset_format == "auto" and path == "__synthetic__"):
        episodes = build_synthetic_episodes(num_episodes=limit or 8, seed=seed)
    elif dataset_format == "meme" or (dataset_format == "auto" and dataset_name):
        episodes = load_meme_dataset(path=path, name=dataset_name or "meme-benchmark/MEME")
    elif path:
        episodes = load_json_dataset(path)
    else:
        raise ValueError("no dataset given: pass --dataset <file>, --dataset-name <hf-name> or --synthetic")

    if limit and len(episodes) > limit:
        episodes = episodes[:limit]
    logger.info("dataset ready: %d episodes", len(episodes))
    return episodes
