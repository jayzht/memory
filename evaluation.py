#!/usr/bin/env python3
"""
evaluation.py -- batch dataset evaluation entry point.

What it does
------------
1. Loads a dataset (custom JSON, MEME, or the built-in synthetic fact-update set).
2. Runs every requested system over every episode, with **strict per-episode
   isolation** (``store.reset(episode_id)`` before each sample, plus a
   ``<system>/<episode>`` namespace so systems never share archived memory).
3. Produces, per (system, episode):
   * ``Current_Fact_Acc``  -- accuracy on probes about the *latest* value
   * ``History_Fact_Acc``  -- accuracy on probes about *overridden* values
   * ``Avg_Active_Chain_Tokens`` -- mean active-chain tokens per episode
   * ``Tool_Call_Success_Rate``  -- successful tool executions / attempts
4. Writes per-episode logs, per-probe predictions, an aggregate metrics table
   and a flat summary CSV, and checkpoints progress into SQLite so an
   interrupted batch resumes without re-running finished episodes.

Examples
--------
    # zero-dependency smoke run: synthetic data, rule-based LLM, in-memory store
    python evaluation.py --synthetic --dataset-format synthetic \
        --llm-backend heuristic --system all --limit 4

    # real experiment (Ollama), all four systems, checkpointed in SQLite
    python evaluation.py --dataset data/episodes.json --llm-backend ollama \
        --model qwen2.5:7b --store redis_sqlite_hybrid --system all --run-id exp1

    # resume after a crash (same --run-id): finished episodes are skipped
    python evaluation.py --dataset data/episodes.json --llm-backend ollama \
        --system three_layer --run-id exp1

    # inspect an unfamiliar MEME export before spending GPU time
    python evaluation.py --dataset meme.json --inspect
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
from memory3l.agents import BaseAgent, build_agent
from memory3l.dataset import (
    CURRENT_FACT,
    HISTORY_FACT,
    MEMORY_POINT,
    Episode,
    Probe,
    build_long_context_episodes,
    build_synthetic_episodes,
    inspect_dataset,
    load_dataset,
)
from memory3l.llm import BaseLLM, LLMError, build_llm
from memory3l.prompts import build_judge_messages
from memory3l.store import InMemoryStore, SQLiteColdStore
from memory3l.store.base import BaseMemoryStore
from memory3l.store.hybrid_store import RedisSQLiteHybridStore
from memory3l.token_utils import TOKENIZER_NAME

logger = logging.getLogger("evaluation")

SYSTEMS = ("three_layer", "full_context", "memgpt_style", "naive_chain")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """INFO for batch runs (DEBUG per-turn logs are opt-in and IO-heavy)."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


# --------------------------------------------------------------------------- #
# Answer scoring
# --------------------------------------------------------------------------- #
import re
import unicodedata

_PUNCT_RE = re.compile(r"[\s\u3000]+")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:答案|answer|结论)\s*[:：]\s*", re.IGNORECASE)


def normalise_answer(text: str) -> str:
    """Aggressive normalisation used by the string scorer."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = _ANSWER_PREFIX_RE.sub("", value)
    value = re.sub(r"[\s\u3000]+", "", value)
    value = re.sub(r"[。，,.;；:：!！?？\"'“”‘’()（）\[\]【】]", "", value)
    return value


#: Short lead-ins a model puts in front of its answer.  Stripping them lets the
#: gold be tested as a *prefix* without accepting it anywhere in a verbose reply.
_ANSWER_LEAD_RE = re.compile(
    r"^(?:the\s+answer\s+is|final\s+answer|answer\s*[:：]|答案是?|答案\s*[:：]|it\s+is|是)\s*[:：]?\s*",
    re.IGNORECASE,
)


def _answer_head(text: str) -> str:
    previous = None
    current = (text or "").strip()
    while current != previous:
        previous = current
        current = _ANSWER_LEAD_RE.sub("", current).strip()
    return current


def string_match(prediction: str, gold: str) -> bool:
    """
    Exact match, or the prediction *leads with* the gold, after normalisation.

    Strict on purpose: a wrong value that merely mentions the right entity must not
    be a match, because that is exactly the failure mode History_Fact_Acc exists to
    catch (answering the current value to a question about the old one).  The
    previous version accepted the gold anywhere inside a prediction up to 4x its
    length, which let a verbose answer that asserted a different value pass without
    ever reaching the LLM judge.
    """
    pred = normalise_answer(prediction)
    target = normalise_answer(gold)
    if not target or not pred:
        return False
    if pred == target:
        return True

    pred_head = _answer_head(pred)
    target_head = _answer_head(target)
    if pred_head == target_head:
        return True

    # Multiple choice: gold "(b) sushi" vs prediction "(b)" or "sushi".
    target_body = re.sub(r"^\(?[a-d]\)?\s*", "", target_head).strip()
    pred_body = re.sub(r"^\(?[a-d]\)?\s*", "", pred_head).strip()
    if target_body and pred_body:
        if target_body == pred_body:
            return True
        if pred_body.startswith(target_body):
            return True

    # The answer must come first; trailing prose is fine, leading prose is not.
    if len(target) >= 2 and pred_head.startswith(target):
        return True
    if len(target_head) >= 2 and pred_head.startswith(target_head):
        return True
    return False


@dataclass
class JudgeStats:
    llm_calls: int = 0
    llm_failures: int = 0
    string_fallbacks: int = 0


class AnswerJudge:
    """LLM-as-judge with an automatic string-match fallback."""

    def __init__(self, llm: Optional[BaseLLM] = None, backend: str = "auto"):
        self.llm = llm
        self.backend = (backend or "auto").lower()
        self.stats = JudgeStats()

    def judge(self, question: str, gold: str, prediction: str) -> Tuple[bool, str, str]:
        """Return ``(correct, method, reason)``."""
        string_hit = string_match(prediction, gold)
        if string_hit:
            # A exact/containment hit needs no model call -- cheaper and stable.
            return True, "string", "normalised match"
        if self.backend == "string" or self.llm is None:
            return False, "string", "no normalised match"
        try:
            response = self.llm.generate(build_judge_messages(question, gold, prediction), json_mode=True)
            self.stats.llm_calls += 1
        except LLMError as exc:
            self.stats.llm_failures += 1
            logger.debug("judge LLM failed: %s", exc)
            return False, "string", f"judge failed: {exc}"
        verdict, reason = _parse_judge_output(response.text)
        if verdict is None:
            self.stats.string_fallbacks += 1
            return False, "string", f"unparsable judge output: {response.text[:80]}"
        return verdict, "llm", reason


def _parse_judge_output(text: str) -> Tuple[Optional[bool], str]:
    if not text:
        return None, ""
    try:
        payload = json.loads(text.strip())
        if isinstance(payload, dict) and "correct" in payload:
            value = payload["correct"]
            if isinstance(value, str):
                value = value.strip().lower() in ("true", "yes", "1", "correct")
            return bool(value), str(payload.get("reason", ""))
    except (ValueError, TypeError):
        pass
    lowered = text.lower()
    match = re.search(r'"correct"\s*:\s*(true|false|yes|no)', lowered)
    if match:
        return match.group(1) in ("true", "yes"), text[:120]
    if re.search(r"\b(correct|yes|对的|正确)\b", lowered):
        return True, text[:120]
    if re.search(r"\b(incorrect|wrong|no|错误|不正确)\b", lowered):
        return False, text[:120]
    return None, ""


# --------------------------------------------------------------------------- #
# Store / LLM construction
# --------------------------------------------------------------------------- #
def build_store(backend: str, args: argparse.Namespace, strict_redis: bool = False) -> BaseMemoryStore:
    """Instantiate the configured store, degrading gracefully when asked to."""
    backend = (backend or config.STORE_BACKEND).lower()
    if backend == "memory":
        store: BaseMemoryStore = InMemoryStore(recent_window_turns=args.recent_window_turns)
    elif backend in ("redis_sqlite_hybrid", "hybrid", "redis_sqlite", "redis"):
        store = RedisSQLiteHybridStore(
            sqlite_path=args.sqlite_path or config.SQLITE_PATH,
            recent_window_turns=args.recent_window_turns,
            strict_redis=strict_redis,
        )
        if not store.hot.available and not strict_redis:
            logger.warning(
                "Redis is unavailable: the hybrid store keeps working with SQLite only "
                "(hot layer degraded). Start a Redis server for the real configuration."
            )
    else:
        raise ValueError(f"unknown store backend: {backend!r} (use memory | redis_sqlite_hybrid)")
    store.recent_window_turns = args.recent_window_turns
    return store


def store_progress_handle(store: BaseMemoryStore) -> Optional[SQLiteColdStore]:
    """The SQLite handle used for checkpointing (None for the memory backend)."""
    return getattr(store, "cold", None)


def build_llms(args: argparse.Namespace) -> Tuple[BaseLLM, BaseLLM, Optional[BaseLLM]]:
    """
    Build ``(agent_llm, summarizer_llm, judge_llm)``.

    ``SUMMARIZER_BACKEND``/``SUMMARIZER_MODEL_NAME`` let the summary model differ
    from the answering model -- a common experimental setting (e.g. a small local
    model for compression, a strong one for answering).
    """
    backend = args.llm_backend or config.LLM_BACKEND
    model = args.model or config.MODEL_NAME
    agent_llm = build_llm(backend, model, args.temperature)

    summarizer_backend = config.SUMMARIZER_BACKEND or backend
    summarizer_model = config.SUMMARIZER_MODEL_NAME or model
    # Deliberately a *separate client* even when the configuration is identical.
    # Sharing one object made ``stats()`` a single counter read twice, so the
    # metadata reported summarizer calls == total calls (490/490 with equal
    # token counts) and the summariser/answer split -- the whole point of the
    # cost discussion -- was unrecoverable.
    summarizer_llm = build_llm(summarizer_backend, summarizer_model, args.temperature)

    # Judge: reuse the agent *model* by default (same-family judging is the usual
    # protocol in memory benchmarks); JUDGE_MODEL_NAME/JUDGE_BACKEND can override.
    # A separate client keeps judge cost out of the answer counters.
    judge_llm: Optional[BaseLLM] = None
    if args.judge == "llm" or (args.judge == "auto" and backend != "heuristic"):
        judge_llm = build_llm(backend, config.JUDGE_MODEL_NAME or model, 0.0)
    return agent_llm, summarizer_llm, judge_llm


# --------------------------------------------------------------------------- #
# Episode-scoped naming
# --------------------------------------------------------------------------- #
def scoped_episode_id(system: str, episode_id: str) -> str:
    """
    Namespace an episode per system inside the shared archive.

    Without this, four systems evaluated into the same SQLite database would
    write identical summary ids (they are deterministic) and one system's
    archival tool could read another system's history.  The store-level reset
    handles isolation *within* a system; this handles isolation *between* systems.
    """
    return f"{system}/{episode_id}"


# --------------------------------------------------------------------------- #
# Per-episode execution
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    run_id: str
    system: str
    episode_id: str
    scoped_id: str
    store_backend: str
    llm_backend: str
    model: str
    num_turns: int = 0
    num_probes: int = 0
    current_total: int = 0
    current_correct: int = 0
    history_total: int = 0
    history_correct: int = 0
    other_total: int = 0
    other_correct: int = 0
    tool_attempted: int = 0
    tool_parsed: int = 0
    tool_resolved: int = 0
    tool_failed: int = 0
    active_chain_tokens_mean: float = 0.0
    active_chain_text_tokens_mean: Optional[float] = None
    active_chain_tokens_final: int = 0
    active_chain_size_final: int = 0
    archived_final: int = 0
    overrides_events: int = 0
    capacity_compressions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = {
            "run_id": self.run_id,
            "system": self.system,
            "episode_id": self.episode_id,
            "num_turns": self.num_turns,
            "num_probes": self.num_probes,
            "current_total": self.current_total,
            "current_correct": self.current_correct,
            "current_acc": round(self.current_correct / self.current_total, 4) if self.current_total else "",
            "history_total": self.history_total,
            "history_correct": self.history_correct,
            "history_acc": round(self.history_correct / self.history_total, 4) if self.history_total else "",
            "other_total": self.other_total,
            "other_correct": self.other_correct,
            "tool_attempted": self.tool_attempted,
            "tool_parsed": self.tool_parsed,
            "tool_resolved": self.tool_resolved,
            "tool_failed": self.tool_failed,
            "tool_success_rate": round(self.tool_resolved / self.tool_attempted, 4) if self.tool_attempted else "",
            "active_chain_tokens_mean": round(self.active_chain_tokens_mean, 2),
            "active_chain_text_tokens_mean": (
                "" if self.active_chain_text_tokens_mean is None
                else round(self.active_chain_text_tokens_mean, 2)
            ),
            "active_chain_tokens_final": self.active_chain_tokens_final,
            "active_chain_size_final": self.active_chain_size_final,
            "archived_final": self.archived_final,
            "overrides_events": self.overrides_events,
            "capacity_compressions": self.capacity_compressions,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_time_s": round(self.wall_time_s, 2),
            "store_backend": self.store_backend,
            "llm_backend": self.llm_backend,
            "model": self.model,
            "error": self.error[:300],
        }
        for key, value in (self.extra or {}).items():
            if key not in row:
                row[key] = value
        return row

    @classmethod
    def from_payload(cls, row: Dict[str, Any]) -> "EpisodeResult":
        """
        Rebuild a result row from a checkpoint payload.

        A resumed run must aggregate the *whole* run, not just the episodes this
        process happened to execute -- otherwise the printed table and
        ``metrics.csv`` quietly describe a subset (and the first flush truncates
        ``predictions.csv`` down to that subset).
        """
        def _num(key: str, cast=float, default=0):
            value = row.get(key, default)
            if value in ("", None):
                return default
            try:
                return cast(value)
            except (TypeError, ValueError):
                return default

        text_mean = row.get("active_chain_text_tokens_mean")
        result = cls(
            run_id=str(row.get("run_id", "")),
            system=str(row.get("system", "")),
            episode_id=str(row.get("episode_id", "")),
            scoped_id=str(row.get("scoped_id", "")),
            store_backend=str(row.get("store_backend", "")),
            llm_backend=str(row.get("llm_backend", "")),
            model=str(row.get("model", "")),
            num_turns=_num("num_turns", int),
            num_probes=_num("num_probes", int),
            current_total=_num("current_total", int),
            current_correct=_num("current_correct", int),
            history_total=_num("history_total", int),
            history_correct=_num("history_correct", int),
            other_total=_num("other_total", int),
            other_correct=_num("other_correct", int),
            tool_attempted=_num("tool_attempted", int),
            tool_parsed=_num("tool_parsed", int),
            tool_resolved=_num("tool_resolved", int),
            tool_failed=_num("tool_failed", int),
            active_chain_tokens_mean=_num("active_chain_tokens_mean", float),
            active_chain_text_tokens_mean=(
                None if text_mean in ("", None) else float(text_mean)
            ),
            active_chain_tokens_final=_num("active_chain_tokens_final", int),
            active_chain_size_final=_num("active_chain_size_final", int),
            archived_final=_num("archived_final", int),
            overrides_events=_num("overrides_events", int),
            capacity_compressions=_num("capacity_compressions", int),
            prompt_tokens=_num("prompt_tokens", int),
            completion_tokens=_num("completion_tokens", int),
            wall_time_s=_num("wall_time_s", float),
            error=str(row.get("error", "") or ""),
        )
        # Anything the row carries beyond the modelled fields (e.g.
        # summarizer_failures, index_count) is preserved so the resumed
        # predictions.csv keeps the same columns.
        base_keys = set(result.to_row())
        result.extra = {k: v for k, v in row.items() if k not in base_keys}
        return result


@dataclass
class ProbeRecord:
    run_id: str
    system: str
    episode_id: str
    probe_index: int
    probe_type: str
    fact_key: str
    question: str
    gold: str
    prediction: str
    correct: bool
    judge_method: str
    judge_reason: str
    tool_attempted: int
    tool_resolved: int
    tool_unparsable: int
    context_tokens: int
    answer_chars: int
    latency_ms: float
    error: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "system": self.system,
            "episode_id": self.episode_id,
            "probe_index": self.probe_index,
            "probe_type": self.probe_type,
            "fact_key": self.fact_key,
            "question": self.question,
            "gold": self.gold,
            "prediction": self.prediction.replace("\n", " ")[:500],
            "correct": int(self.correct),
            "judge_method": self.judge_method,
            "judge_reason": self.judge_reason[:200],
            "tool_attempted": self.tool_attempted,
            "tool_resolved": self.tool_resolved,
            "tool_unparsable": self.tool_unparsable,
            "context_tokens": self.context_tokens,
            "answer_chars": self.answer_chars,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error[:200],
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ProbeRecord":
        """Inverse of :meth:`to_row`, so a resumed run keeps its probe rows."""
        return cls(
            run_id=str(row.get("run_id", "")),
            system=str(row.get("system", "")),
            episode_id=str(row.get("episode_id", "")),
            probe_index=int(row.get("probe_index", 0) or 0),
            probe_type=str(row.get("probe_type", "")),
            fact_key=str(row.get("fact_key", "")),
            question=str(row.get("question", "")),
            gold=str(row.get("gold", "")),
            prediction=str(row.get("prediction", "")),
            correct=bool(row.get("correct", 0)),
            judge_method=str(row.get("judge_method", "")),
            judge_reason=str(row.get("judge_reason", "")),
            tool_attempted=int(row.get("tool_attempted", 0) or 0),
            tool_resolved=int(row.get("tool_resolved", 0) or 0),
            tool_unparsable=int(row.get("tool_unparsable", 0) or 0),
            context_tokens=int(row.get("context_tokens", 0) or 0),
            answer_chars=int(row.get("answer_chars", 0) or 0),
            latency_ms=float(row.get("latency_ms", 0.0) or 0.0),
            error=str(row.get("error", "") or ""),
        )


# --------------------------------------------------------------------------- #
# Dynamic memory-point probes
# --------------------------------------------------------------------------- #
def build_memory_point_probes(agent: BaseAgent, episode: Episode, count: int = 1) -> List[Probe]:
    """
    Ask about concrete memory points by id, so the archival path is measured.

    Two flavours, chosen by what the system actually keeps:

    * **archived summary** (only systems with the archival tool): ask what a
      summary that has left the active chain says; gold = its text.  The model
      cannot guess it -- it must call ``get_archived_summary``.
    * **raw record** (systems without tools, e.g. the destructive baseline):
      ask what a raw turn said; gold = the turn text.  A system that dropped the
      record can no longer answer, which is a real measurement, not a penalty.

    These probes are generated after the dialogue has been ingested, when the ids
    exist.  They are scored exactly like any other probe and are what makes
    ``Tool_Call_Success_Rate`` non-vacuous on datasets without hand-written ids.
    """
    if count <= 0:
        return []
    probes: List[Probe] = []
    store = agent.store
    episode_id = agent.episode_id or episode.episode_id

    if agent.supports_tools():
        archived = [a for a in store.list_archived_summaries(episode_id) if a.is_overridden]
        for summary in _sample_spread(archived, count):
            probes.append(
                Probe(
                    question=(
                        f"What does the archived memory entry {summary.summary_id} say? "
                        "Quote its content."
                    ),
                    answer=summary.text,
                    probe_type=MEMORY_POINT,
                    fact_key="archived_summary",
                    history_ref=summary.summary_id,
                )
            )
    else:
        records = store.list_raw_records(episode_id)
        for record in _sample_spread(records, count):
            probes.append(
                Probe(
                    question=(
                        f"What exactly did turn {record.turn_index} say? "
                        f"Quote the original dialogue (reference {record.reference_id})."
                    ),
                    answer=record.user_msg,
                    probe_type=MEMORY_POINT,
                    fact_key="raw_record",
                    history_ref=record.reference_id,
                )
            )
    return probes


def _sample_spread(items: Sequence[Any], count: int) -> List[Any]:
    """Pick ``count`` items spread across ``items`` (deterministic, no RNG)."""
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / float(count)
    return [items[min(len(items) - 1, int(index * step))] for index in range(count)]


def run_episode(
    agent: BaseAgent,
    episode: Episode,
    *,
    run_id: str,
    system: str,
    judge: AnswerJudge,
    store_backend: str,
    memory_point_probes: int = 0,
) -> Tuple[EpisodeResult, List[ProbeRecord], Dict[str, Any]]:
    """Run one episode end to end: dialogue ingestion, then probe answering."""
    scoped = scoped_episode_id(system, episode.episode_id)
    started = time.perf_counter()
    result = EpisodeResult(
        run_id=run_id,
        system=system,
        episode_id=episode.episode_id,
        scoped_id=scoped,
        store_backend=store_backend,
        llm_backend=getattr(agent.llm, "name", ""),
        model=getattr(agent.llm, "model_name", ""),
        num_turns=episode.num_turns,
        num_probes=len(episode.probes),
    )

    # --- isolation: wipe this episode's hot memory before anything else ------ #
    agent.reset_episode(scoped)
    # The namespace is ``system/episode`` and carries no run id, so a re-run
    # (--force, or a new run id over the same SQLite file) would otherwise inherit
    # archived summaries from the previous attempt.  Raw records are never touched.
    clear_archive = getattr(getattr(agent, "store", None), "clear_archive", None)
    if callable(clear_archive):
        try:
            stale = clear_archive(scoped)
        except Exception as exc:  # noqa: BLE001 - isolation must not kill the episode
            logger.warning("could not clear the archive for %s: %s", scoped, exc)
        else:
            if stale:
                logger.info("cleared %d stale archived summary(ies) for %s", stale, scoped)

    turn_stats: List[Dict[str, Any]] = []
    try:
        # One batch call: the manager decides whether to summarise sequentially or
        # concurrently, and applies results in order either way.
        batch_stats = agent.add_turns(episode.dialogues)
    except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the run
        logger.exception("episode %s ingestion failed", episode.episode_id)
        result.error = f"ingest: {exc}"
        batch_stats = []
    for stats in batch_stats or []:
        payload = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats or {})
        turn_stats.append(payload)

    # --- probes ------------------------------------------------------------- #
    probes: List[ProbeRecord] = []
    probe_set: List[Probe] = list(episode.probes) + build_memory_point_probes(
        agent, episode, memory_point_probes
    )
    result.num_probes = len(probe_set)
    chain_tokens: List[int] = []
    chain_text_tokens: List[int] = []
    for probe_index, probe in enumerate(probe_set):
        probe_record, agent_result = _answer_probe(agent, probe, probe_index, run_id, system, episode.episode_id, judge)
        probes.append(probe_record)
        if probe.probe_type == CURRENT_FACT:
            result.current_total += 1
            result.current_correct += int(probe_record.correct)
        elif probe.probe_type == HISTORY_FACT:
            result.history_total += 1
            result.history_correct += int(probe_record.correct)
        else:
            result.other_total += 1
            result.other_correct += int(probe_record.correct)
        result.tool_attempted += agent_result.tool_calls_attempted
        result.tool_parsed += agent_result.tool_calls_parsed
        result.tool_resolved += agent_result.tool_calls_resolved
        result.tool_failed += agent_result.tool_calls_failed
        result.prompt_tokens += agent_result.prompt_tokens
        result.completion_tokens += agent_result.completion_tokens
        if agent_result.error and not result.error:
            result.error = agent_result.error

    # --- per-episode memory statistics -------------------------------------- #
    for payload in turn_stats:
        chain_tokens.append(int(payload.get("active_chain_tokens", 0) or 0))
        # Only systems with a discrete summary chain report a chain-token figure.
        # Absence must stay "not applicable", never a 0.0 that reads as free memory.
        if "active_chain_text_tokens" in payload:
            chain_text_tokens.append(int(payload.get("active_chain_text_tokens") or 0))
    result.active_chain_tokens_mean = statistics.fmean(chain_tokens) if chain_tokens else 0.0
    result.active_chain_text_tokens_mean = (
        statistics.fmean(chain_text_tokens) if chain_text_tokens else None
    )

    manager = getattr(agent, "manager", None)
    if manager is not None:
        metrics = manager.episode_metrics()
        result.active_chain_tokens_final = int(metrics.get("active_chain_tokens", 0))
        result.active_chain_size_final = int(metrics.get("active_chain_size", 0))
        result.archived_final = int(metrics.get("archived_count", 0))
        result.overrides_events = int(metrics.get("overrides_events", 0))
        result.capacity_compressions = int(metrics.get("capacity_compressions", 0))
        result.extra.update({
            "summarizer_failures": metrics.get("summarizer_failures", 0),
            "index_count": metrics.get("index_count", 0),
            "filed_summary_count": metrics.get("filed_summary_count", 0),
            "all_rendered_tokens": metrics.get("all_rendered_tokens", 0),
        })
    finalize = agent.finalize_episode() or {}
    for key, value in finalize.items():
        if key not in result.extra and isinstance(value, (int, float, str, bool)):
            result.extra[key] = value
    # Systems without a MemoryManager report their compression/chain statistics
    # through finalize_episode(); prefer those values so every system is measured
    # through one code path (the metric definition is documented in the README).
    raw_chain_mean = finalize.get("avg_active_chain_tokens", finalize.get("active_chain_tokens"))
    # A baseline without a discrete summary chain reports None -> stays n/a
    # instead of a misleading 0.0 tokens.
    chain_mean: Optional[float] = None if raw_chain_mean in (None, "") else float(raw_chain_mean)
    context_mean = float(
        finalize.get("avg_context_tokens", finalize.get("avg_active_chain_tokens_rendered", chain_mean))
        or 0.0
    )
    if chain_mean is not None:
        result.active_chain_text_tokens_mean = chain_mean
    if context_mean:
        result.active_chain_tokens_mean = context_mean
    if not result.capacity_compressions:
        result.capacity_compressions = int(finalize.get("capacity_compressions", 0) or 0)
    if not result.active_chain_size_final:
        result.active_chain_size_final = int(finalize.get("active_chain_size", 0) or 0)
    if not result.active_chain_tokens_final:
        result.active_chain_tokens_final = int(finalize.get("active_chain_tokens", 0) or 0)
    if not result.archived_final:
        result.archived_final = int(finalize.get("archived_count", 0) or 0)
    if not result.overrides_events:
        result.overrides_events = int(finalize.get("overrides_events", 0) or 0)
    result.wall_time_s = time.perf_counter() - started

    episode_log = {
        "run_id": run_id,
        "system": system,
        "episode_id": episode.episode_id,
        "scoped_id": scoped,
        "facts": {k: list(v) for k, v in episode.facts.items()},
        "turns": turn_stats,
        "probes": [p.to_row() for p in probes],
        "episode_metrics": {**result.to_row(), **{f"finalize_{k}": v for k, v in finalize.items()}},
    }
    return result, probes, episode_log


def _answer_probe(
    agent: BaseAgent,
    probe: Probe,
    probe_index: int,
    run_id: str,
    system: str,
    episode_id: str,
    judge: AnswerJudge,
):
    try:
        run_result = agent.answer(probe.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("probe failed on %s: %s", episode_id, probe.question[:60])
        from memory3l.agents.base_agent import AgentRunResult

        run_result = AgentRunResult(answer="", error=str(exc))
    correct, method, reason = judge.judge(probe.question, probe.answer, run_result.answer)
    record = ProbeRecord(
        run_id=run_id,
        system=system,
        episode_id=episode_id,
        probe_index=probe_index,
        probe_type=probe.probe_type,
        fact_key=probe.fact_key,
        question=probe.question,
        gold=probe.answer,
        prediction=run_result.answer,
        correct=bool(correct),
        judge_method=method,
        judge_reason=reason,
        tool_attempted=run_result.tool_calls_attempted,
        tool_resolved=run_result.tool_calls_resolved,
        tool_unparsable=sum(1 for entry in run_result.tool_call_log if not entry.get("parsed", True)),
        context_tokens=run_result.context_tokens,
        answer_chars=len(run_result.answer or ""),
        latency_ms=run_result.latency_ms,
        error=run_result.error,
    )
    return record, run_result


# --------------------------------------------------------------------------- #
# Aggregation + output
# --------------------------------------------------------------------------- #
def aggregate(rows: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    """One metrics row per system (plus a per-episode breakdown in the CSV)."""
    grouped: "OrderedDict[str, List[EpisodeResult]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(row.system, []).append(row)

    out: List[Dict[str, Any]] = []
    for system, items in grouped.items():
        current_total = sum(i.current_total for i in items)
        current_correct = sum(i.current_correct for i in items)
        history_total = sum(i.history_total for i in items)
        history_correct = sum(i.history_correct for i in items)
        # MEMORY_POINT probes (retrieved-content questions) were accumulated per
        # episode but never aggregated, so a system that executed every tool call and
        # still answered the archived-content question wrongly lost nothing in the
        # headline table.
        other_total = sum(i.other_total for i in items)
        other_correct = sum(i.other_correct for i in items)
        tool_attempted = sum(i.tool_attempted for i in items)
        tool_resolved = sum(i.tool_resolved for i in items)
        tool_parsed = sum(i.tool_parsed for i in items)
        chain_tokens = [
            i.active_chain_text_tokens_mean for i in items
            if i.num_turns and i.active_chain_text_tokens_mean is not None
        ]
        chain_tokens_rendered = [i.active_chain_tokens_mean for i in items if i.num_turns]
        out.append(
            {
                "system": system,
                "episodes": len(items),
                "turns": sum(i.num_turns for i in items),
                "probes": sum(i.num_probes for i in items),
                "Current_Fact_Acc": round(current_correct / current_total, 4) if current_total else "",
                "Current_Fact_n": current_total,
                "History_Fact_Acc": round(history_correct / history_total, 4) if history_total else "",
                "History_Fact_n": history_total,
                "Memory_Point_Acc": round(other_correct / other_total, 4) if other_total else "",
                "Memory_Point_n": other_total,
                "Avg_Active_Chain_Tokens": round(statistics.fmean(chain_tokens), 2) if chain_tokens else "",
                "Avg_Active_Chain_Tokens_rendered": (
                    round(statistics.fmean(chain_tokens_rendered), 2) if chain_tokens_rendered else 0.0
                ),
                "Avg_Active_Chain_Size": round(
                    statistics.fmean([i.active_chain_size_final for i in items]), 2
                ) if items else 0.0,
                "Avg_Archived": round(statistics.fmean([i.archived_final for i in items]), 2) if items else 0.0,
                "overrides_events": sum(i.overrides_events for i in items),
                "capacity_compressions": sum(i.capacity_compressions for i in items),
                "Tool_Calls": tool_attempted,
                "Tool_Call_Parse_Rate": round(tool_parsed / tool_attempted, 4) if tool_attempted else "",
                "Tool_Call_Unparsable": tool_attempted - tool_parsed,
                "Tool_Call_Success_Rate": round(tool_resolved / tool_attempted, 4) if tool_attempted else "",
                "Prompt_Tokens": sum(i.prompt_tokens for i in items),
                "Completion_Tokens": sum(i.completion_tokens for i in items),
                "Wall_Time_s": round(sum(i.wall_time_s for i in items), 1),
                "Errors": sum(1 for i in items if i.error),
            }
        )
    return out


def write_csv(path: str, rows: Sequence[Dict[str, Any]], append: bool = False) -> None:
    """Write (or append to) a CSV, keeping the union of all keys as columns."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    exists = os.path.exists(path)
    mode = "a" if (append and exists) else "w"
    with open(path, mode, newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w" or not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def print_metrics_table(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    header = (
        f"{'system':<14} {'CurAcc':>7} {'HisAcc':>7} {'ChainTok':>9} "
        f"{'ToolOK':>7} {'Tools':>6} {'Arch':>5} {'Ovrd':>5} {'CapCmp':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        def _fmt(value, spec=">7"):
            if value == "" or value is None:
                return f"{'n/a':>{spec[1:]}}"
            return f"{value:{spec}}"

        print(
            f"{row['system']:<14} {_fmt(row.get('Current_Fact_Acc'))} {_fmt(row.get('History_Fact_Acc'))} "
            f"{_fmt(row.get('Avg_Active_Chain_Tokens'), '>9')} {_fmt(row.get('Tool_Call_Success_Rate'))} "
            f"{row.get('Tool_Calls', 0):>6} {row.get('Avg_Archived', 0):>5} "
            f"{row.get('overrides_events', 0):>5} {row.get('capacity_compressions', 0):>6}"
        )
    print("=" * len(header) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three-layer memory agent: batch dataset evaluation vs baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # dataset
    parser.add_argument("--dataset", default=None, help="path to a JSON/JSONL dataset file")
    parser.add_argument("--dataset-name", default=None, help="HuggingFace dataset name (e.g. meme-benchmark/MEME)")
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "json", "meme", "synthetic", "synthetic_long", "longmemeval"],
        help="loader to use",
    )
    parser.add_argument("--synthetic", action="store_true", help="use the built-in synthetic fact-update set")
    parser.add_argument(
        "--synthetic-long",
        action="store_true",
        help="use the long-context synthetic set (saturates the window; separates the systems)",
    )
    parser.add_argument("--lme-path", default=None,
                        help="path to longmemeval_s_cleaned.json (default: data/longmemeval/...)")
    parser.add_argument("--lme-max-turns", type=int, default=0,
                        help="turn budget per LongMemEval episode (0 = full haystack, the official protocol). "
                             "Truncation keeps the evidence session, but changes task difficulty.")
    parser.add_argument("--lme-types", default=None,
                        help="comma-separated LongMemEval question types, e.g. knowledge-update,multi-session")
    parser.add_argument("--synthetic-turns", type=int, default=40,
                        help="turns per episode for the long synthetic set")
    parser.add_argument("--synthetic-language", default="zh", choices=["zh", "en"],
                        help="language of generated synthetic dialogue")
    parser.add_argument("--limit", type=int, default=0, help="max episodes (0 = all)")
    parser.add_argument("--seed", type=int, default=config.SEED, help="synthetic data / sampling seed")
    parser.add_argument("--inspect", action="store_true", help="print the dataset schema and exit")
    # systems
    parser.add_argument(
        "--system",
        default="three_layer",
        help="three_layer | full_context | memgpt_style | naive_chain | all",
    )
    parser.add_argument("--systems", default=None, help="comma-separated list of systems (overrides --system)")
    # models
    parser.add_argument("--llm-backend", default=None,
                        choices=["deepseek", "ollama", "openai", "heuristic"])
    parser.add_argument("--model", default=None, help="model name for the agent")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--judge", default=config.JUDGE_BACKEND, choices=["auto", "llm", "string"])
    # memory
    parser.add_argument("--store", default=None, choices=["memory", "redis_sqlite_hybrid"])
    parser.add_argument("--recent-window-turns", type=int, default=config.RECENT_WINDOW_TURNS)
    parser.add_argument("--active-chain-token-limit", type=int, default=config.ACTIVE_CHAIN_TOKEN_LIMIT)
    parser.add_argument("--raw-context-token-limit", type=int, default=config.RAW_CONTEXT_TOKEN_LIMIT)
    parser.add_argument("--max-tool-iterations", type=int, default=config.MAX_TOOL_ITERATIONS)
    parser.add_argument("--ingest-concurrency", type=int, default=config.INGEST_CONCURRENCY,
                        help="concurrent summariser calls while ingesting a dialogue "
                             "(1 = sequential and semantics-preserving; >1 is an "
                             "approximation: turns inside one window share a chain "
                             "snapshot, so overrides within a window are missed)")
    parser.add_argument(
        "--chain-strategy",
        default=config.CHAIN_STRATEGY,
        choices=["index", "merge"],
        help="how the summary chain is kept under budget: "
             "'index' = add a title level over groups (keeps every summary readable); "
             "'merge' = rewrite several summaries into one (lossy)",
    )
    parser.add_argument("--index-keep-recent", type=int, default=config.INDEX_KEEP_RECENT)
    parser.add_argument("--index-group-size", type=int, default=config.INDEX_GROUP_SIZE)
    parser.add_argument("--sqlite-path", default=None, help="SQLite file (default: config.SQLITE_PATH)")
    parser.add_argument("--strict-redis", action="store_true", help="fail instead of degrading when Redis is down")
    # output / checkpointing
    parser.add_argument("--run-id", default=None, help="checkpoint key; reuse it to resume a crashed batch")
    parser.add_argument("--out-dir", default="results", help="directory for CSV/JSONL outputs")
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--episode-log", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-level", default=config.LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--force", action="store_true", help="re-run episodes already marked done for this run-id")
    parser.add_argument("--no-checkpoint", action="store_true", help="do not write/read the SQLite checkpoint")
    parser.add_argument("--debug-agent", action="store_true", help="log the agent's replies per step")
    parser.add_argument(
        "--memory-point-probes",
        type=int,
        default=1,
        help="per-episode dynamically generated probes that name a concrete summary/raw id "
        "(0 disables; these are what make Tool_Call_Success_Rate measurable)",
    )
    return parser


def resolve_systems(args: argparse.Namespace) -> List[str]:
    if args.systems:
        requested = [s.strip() for s in args.systems.split(",") if s.strip()]
    elif (args.system or "").lower() == "all":
        requested = list(SYSTEMS)
    else:
        requested = [args.system]
    aliases = {
        "ours": "three_layer",
        "baseline1": "full_context",
        "baseline2": "memgpt_style",
        "baseline3": "naive_chain",
        "memgpt": "memgpt_style",
        "naive": "naive_chain",
    }
    resolved = []
    for name in requested:
        canonical = aliases.get(name.lower(), name.lower())
        if canonical not in SYSTEMS:
            raise SystemExit(f"unknown system {name!r}; choose from {', '.join(SYSTEMS)}, all")
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def resolve_dataset(args: argparse.Namespace) -> List[Episode]:
    if args.dataset_format == "longmemeval" or args.dataset == "longmemeval":
        from memory3l.longmemeval import load_longmemeval

        types = [t.strip() for t in (args.lme_types or "").split(",") if t.strip()] or None
        return load_longmemeval(
            path=args.lme_path,
            limit=args.limit or 50,
            max_turns=args.lme_max_turns or None,
            types=types,
        )
    if args.synthetic_long or args.dataset_format == "synthetic_long":
        return build_long_context_episodes(
            num_episodes=args.limit or 6,
            turns_per_episode=args.synthetic_turns,
            seed=args.seed,
            language=args.synthetic_language,
        )
    if args.synthetic:
        return build_synthetic_episodes(
            num_episodes=args.limit or 8,
            seed=args.seed,
            turns_per_episode=12,
            language=args.synthetic_language,
        )
    return load_dataset(
        path=args.dataset,
        dataset_name=args.dataset_name,
        dataset_format=args.dataset_format,
        limit=args.limit,
        seed=args.seed,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    default_log = os.path.join(args.out_dir, "logs", "evaluation.log")
    setup_logging(args.log_level, args.log_file or default_log)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- dataset ----------------------------------------------------------- #
    try:
        episodes = resolve_dataset(args)
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to load dataset: %s", exc)
        return 2

    if args.inspect:
        if args.dataset_format == "longmemeval" or args.dataset == "longmemeval":
            from memory3l.longmemeval import describe

            report = {"source": "longmemeval", **describe(episodes)}
        else:
            report = inspect_dataset(path=args.dataset, episodes=episodes)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if not episodes:
        logger.error("dataset contains no usable episodes")
        return 2

    systems = resolve_systems(args)
    store_backend = (args.store or config.STORE_BACKEND).lower()
    run_id = args.run_id or f"{store_backend}-{args.llm_backend or config.LLM_BACKEND}-{int(time.time())}"

    predictions_csv = args.predictions_csv or os.path.join(args.out_dir, "predictions.csv")
    metrics_csv = args.metrics_csv or os.path.join(args.out_dir, "metrics.csv")
    episode_log_path = args.episode_log or os.path.join(args.out_dir, "episode_logs.jsonl")

    logger.info(
        "run_id=%s | systems=%s | episodes=%d | store=%s | llm=%s/%s | window=%d | chain_limit=%d | tokenizer=%s",
        run_id, systems, len(episodes), store_backend, args.llm_backend or config.LLM_BACKEND,
        args.model or config.MODEL_NAME, args.recent_window_turns, args.active_chain_token_limit,
        TOKENIZER_NAME,
    )
    if store_backend == "memory":
        logger.info("in-memory store: no Redis/SQLite persistence, no checkpointing")
    if store_backend == "redis_sqlite_hybrid":
        logger.info("hybrid store: Redis=hot (episode:{episode_id}:*) SQLite=%s cold+checkpoint", args.sqlite_path or config.SQLITE_PATH)

    # --- LLMs -------------------------------------------------------------- #
    agent_llm, summarizer_llm, judge_llm = build_llms(args)
    judge = AnswerJudge(judge_llm, backend=args.judge)
    logger.info("LLM backend=%s model=%s summarizer=%s", agent_llm.name, agent_llm.model_name, summarizer_llm.model_name)

    # --- storage ----------------------------------------------------------- #
    try:
        store = build_store(store_backend, args, strict_redis=args.strict_redis)
    except Exception as exc:  # noqa: BLE001
        logger.error("could not build store: %s", exc)
        return 2

    cold = store_progress_handle(store)
    if cold is not None:
        cold.execute(
            """CREATE TABLE IF NOT EXISTS episode_events (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   run_id TEXT, system TEXT, episode_id TEXT, event TEXT, payload TEXT, ts REAL)"""
        )

    all_rows: List[EpisodeResult] = []
    all_probes: List[ProbeRecord] = []
    episode_logs: List[Dict[str, Any]] = []
    skipped = 0
    failed = 0

    try:
        for system in systems:
            done = set()
            checkpoints: Dict[str, Dict[str, Any]] = {}
            if cold is not None and not args.no_checkpoint and not args.force:
                done = cold.get_done_episodes(run_id=f"{run_id}::{system}")
                if done:
                    logger.info("[%s] resuming: %d episode(s) already done, skipping", system, len(done))
                    # Load their stored payloads so aggregation still covers the
                    # whole run instead of only the episodes this process ran.
                    for checkpoint in cold.list_run_payloads(run_id=run_id):
                        if checkpoint["system_name"] == system:
                            checkpoints[checkpoint["episode_id"]] = checkpoint["payload"]

            agent = build_agent(
                system,
                store,
                agent_llm,
                summarizer_llm,
                recent_window_turns=args.recent_window_turns,
                active_chain_token_limit=args.active_chain_token_limit,
                raw_context_token_limit=args.raw_context_token_limit,
                max_tool_iterations=args.max_tool_iterations,
                chain_strategy=args.chain_strategy,
                index_keep_recent=args.index_keep_recent,
                index_group_size=args.index_group_size,
            )
            agent.debug = args.debug_agent
            agent.set_ingest_concurrency(args.ingest_concurrency)
            if hasattr(agent, "reset_episode"):
                # The executor (and therefore tool support) only exists inside an
                # episode, so report it once the first episode has been bound.
                pass
            logger.info(
                "[%s] system ready (tool-capable=%s)",
                system,
                system == "three_layer",
            )

            for index, episode in enumerate(episodes):
                scoped = scoped_episode_id(system, episode.episode_id)
                if scoped in done:
                    skipped += 1
                    payload = checkpoints.get(scoped)
                    if payload:
                        all_rows.append(EpisodeResult.from_payload(payload))
                        for probe_row in payload.get("_probes") or []:
                            try:
                                all_probes.append(ProbeRecord.from_row(probe_row))
                            except Exception:  # noqa: BLE001 - a bad stored row is not fatal
                                logger.debug("skipping unreadable stored probe row in %s", scoped)
                    else:
                        logger.warning(
                            "%s is checkpointed as done but has no stored payload; it will "
                            "be missing from this run's aggregate", scoped,
                        )
                    continue
                log_prefix = f"[{system}] episode {index + 1}/{len(episodes)} {episode.episode_id}"
                logger.info("%s (%d turns, %d probes)", log_prefix, episode.num_turns, len(episode.probes))
                try:
                    result, probes, episode_log = run_episode(
                        agent,
                        episode,
                        run_id=run_id,
                        system=system,
                        judge=judge,
                        store_backend=store_backend,
                        memory_point_probes=args.memory_point_probes,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.exception("%s CRASHED: %s", log_prefix, exc)
                    if cold is not None and not args.no_checkpoint:
                        cold.mark_episode_failed(scoped, run_id=f"{run_id}::{system}", detail=str(exc))
                    continue

                all_rows.append(result)
                all_probes.extend(probes)
                episode_logs.append(episode_log)
                chain_display = (
                    "n/a" if result.active_chain_text_tokens_mean in (None, "")
                    else f"{result.active_chain_text_tokens_mean:.0f}"
                )
                print(
                    f"  {log_prefix}: cur={result.current_correct}/{result.current_total} "
                    f"hist={result.history_correct}/{result.history_total} "
                    f"chain={chain_display}tok arch={result.archived_final} "
                    f"tools={result.tool_resolved}/{result.tool_attempted} {result.wall_time_s:.1f}s"
                )
                if cold is not None and not args.no_checkpoint:
                    # Payload first, done-flag second: a crash in between would
                    # otherwise leave an episode marked done with no payload, and a
                    # resume would aggregate a run that silently misses it.
                    cold.save_run_payload(
                        run_id, system, scoped,
                        {**result.to_row(), "_probes": [p.to_row() for p in probes]},
                    )
                    if result.error:
                        # Partially ingested: it is scored here (the data may still be
                        # usable) but must NOT be checkpointed as done, or a resume
                        # would skip it and the run would silently keep a bad row.
                        failed += 1
                        cold.mark_episode_failed(
                            scoped, run_id=f"{run_id}::{system}", detail=result.error,
                        )
                        logger.warning(
                            "%s finished with an error and will be retried on resume: %s",
                            log_prefix, result.error,
                        )
                    else:
                        cold.mark_episode_done(
                            scoped,
                            run_id=f"{run_id}::{system}",
                            num_turns=result.num_turns,
                            detail=json.dumps(
                                {
                                    "current_acc": result.current_correct / result.current_total if result.current_total else None,
                                    "history_acc": result.history_correct / result.history_total if result.history_total else None,
                                    "active_chain_tokens_mean": result.active_chain_text_tokens_mean,
                                },
                                ensure_ascii=False,
                            ),
                        )

            # flush after each system so an interruption keeps completed work.
            # ``append=False``: ``all_rows`` is cumulative, so appending it again on
            # the next system duplicated every earlier row (4 systems x n episodes
            # produced 10n rows with weights 4:3:2:1), and the analysis then read a
            # differently-weighted mixture from predictions.csv.
            if all_rows:
                write_csv(predictions_csv, [r.to_row() for r in all_rows])
                write_csv(metrics_csv, aggregate(all_rows))
                with open(episode_log_path, "w", encoding="utf-8") as handle:
                    for entry in episode_logs:
                        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    finally:
        store.close()

    if not all_rows:
        logger.warning("no episodes were run (all skipped or failed)")
        return 1 if failed else 0

    # --- outputs ----------------------------------------------------------- #
    write_probe_csv(os.path.join(args.out_dir, "probe_predictions.csv"), all_probes)
    metrics = aggregate(all_rows)
    write_csv(metrics_csv, metrics)

    metadata = {
        **config.ConfigSnapshot(
            store_backend=store_backend,
            recent_window_turns=args.recent_window_turns,
            active_chain_token_limit=args.active_chain_token_limit,
            llm_backend=args.llm_backend or config.LLM_BACKEND,
            model_name=args.model or config.MODEL_NAME,
            temperature=args.temperature,
            system_name=",".join(systems),
            judge_backend=args.judge,
            seed=args.seed,
        ).as_dict(),
        "run_id": run_id,
        "dataset": args.dataset or args.dataset_name or ("synthetic" if args.synthetic else ""),
        "num_episodes": len(episodes),
        "episodes_skipped": skipped,
        "episodes_failed": failed,
        "tokenizer": TOKENIZER_NAME,
        "systems": systems,
        "chain_strategy": args.chain_strategy,
        "index_keep_recent": args.index_keep_recent,
        "index_group_size": args.index_group_size,
        "judge_stats": judge.stats.__dict__,
        "judge_model": getattr(judge_llm, "model_name", "") if judge_llm else "",
        "summarizer_model": getattr(summarizer_llm, "model_name", ""),
        "llm_stats": agent_llm.stats(),
        "summarizer_stats": summarizer_llm.stats(),
        "llm_max_retries": config.LLM_MAX_RETRIES,
        "summarizer_format_retry": config.SUMMARIZER_FORMAT_RETRY,
        "strict_override_ids": config.STRICT_OVERRIDE_IDS,
    }
    metadata_path = os.path.join(args.out_dir, f"run_metadata_{run_id.replace('/', '_')}.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)

    print_metrics_table(metrics)
    print(
        f"results -> {predictions_csv}\n"
        f"           {os.path.join(args.out_dir, 'probe_predictions.csv')}\n"
        f"           {metrics_csv}\n"
        f"           {episode_log_path}\n"
        f"           {metadata_path}"
        + (f"\n(skipped {skipped} finished episode(s), {failed} failed)" if skipped or failed else "")
    )
    return 0


def write_probe_csv(path: str, probes: Sequence[ProbeRecord]) -> None:
    write_csv(path, [p.to_row() for p in probes])


if __name__ == "__main__":
    raise SystemExit(main())
