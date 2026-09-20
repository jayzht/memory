"""
LongMemEval adapter.

LongMemEval (Wu et al., ICLR 2025) is the de-facto public benchmark for chat
memory: 500 curated questions over ~50-session haystacks (~494 dialogue turns per
question in the ``_s`` configuration).

What this module does
---------------------
* streams the 277 MB cleaned JSON with ``ijson`` (the file does not fit in memory
  comfortably and we only need the first N questions);
* maps one LongMemEval question into one :class:`~memory3l.dataset.Episode`:
  the haystack becomes the episode's dialogue, the question becomes a probe;
* maps the benchmark's question types onto the project's probe types so the
  existing metrics (Current_Fact_Acc / History_Fact_Acc) apply:

  ==========================  ===========================  ==================
  LongMemEval type            project probe type           metric column
  ==========================  ===========================  ==================
  knowledge-update            current_fact                 Current_Fact_Acc
  single-session-user         current_fact                 Current_Fact_Acc
  single-session-assistant    current_fact                 Current_Fact_Acc
  single-session-preference   current_fact                 Current_Fact_Acc
  multi-session               other (reasoning/aggregate)  other
  temporal-reasoning          other (reasoning/time)       other
  abstention                  other (refusal expected)     other
  ==========================  ===========================  ==================

  ``knowledge-update`` is the close analogue of this project's "overridden
  history" case: the answer must be the *updated* value, so it is reported under
  Current_Fact_Acc and, when the dataset provides the superseded value, a
  ``history_fact`` probe is added from ``answer_session_ids`` ordering.

Cost note
---------
Ingesting a full haystack means ~494 summariser calls per question.  That is the
dominant cost of any benchmark run (the probe itself is ~1% of it), so
``--sample-turns`` exists to bound a pilot.  It changes the task difficulty
(fewer distractors), so the value used must be reported alongside any number.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .dataset import CURRENT_FACT, HISTORY_FACT, OTHER, Episode, Probe

logger = logging.getLogger(__name__)

#: LongMemEval question type -> project probe type.
QUESTION_TYPE_MAP = {
    "knowledge-update": CURRENT_FACT,
    "single-session-user": CURRENT_FACT,
    "single-session-assistant": CURRENT_FACT,
    "single-session-preference": CURRENT_FACT,
    "multi-session": OTHER,
    "temporal-reasoning": OTHER,
    "abstention": OTHER,
}

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "longmemeval", "longmemeval_s_cleaned.json",
)


def _iter_raw(path: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Stream the cleaned JSON array without loading it whole."""
    try:
        import ijson  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "LongMemEval needs the `ijson` streaming parser: "
            "pip install --target ./.pylibs ijson"
        ) from exc
    with open(path, "rb") as handle:
        for index, record in enumerate(ijson.items(handle, "item")):
            if limit is not None and index >= limit:
                break
            yield record


def sessions_to_dialogues(
    sessions: Sequence[Sequence[Dict[str, Any]]],
    session_dates: Optional[Sequence[str]] = None,
    keep_roles: bool = True,
) -> List[tuple]:
    """
    Flatten LongMemEval sessions into ``(user, assistant)`` turn pairs.

    A session is a list of ``{"role", "content"}`` messages that normally
    alternates user/assistant.  The project's memory model is turn-based, so pairs
    are formed sequentially; a trailing user with no assistant gets an empty reply
    so no content is dropped.
    """
    turns: List[tuple] = []
    for session_index, session in enumerate(sessions):
        pending_user: Optional[str] = None
        for message in session:
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "") or "")
            if role in ("user", "human"):
                if pending_user is not None:
                    turns.append((pending_user, ""))
                pending_user = content
            elif role in ("assistant", "agent", "ai"):
                turns.append((pending_user if pending_user is not None else "", content))
                pending_user = None
            # system / tool messages are not part of the user-visible dialogue
        if pending_user is not None:
            turns.append((pending_user, ""))
    return turns


def record_to_episode(
    record: Dict[str, Any],
    index: int,
    max_turns: Optional[int] = None,
    include_answer_session: bool = True,
) -> Episode:
    """
    Convert one raw LongMemEval record into an :class:`Episode`.

    ``max_turns`` bounds the ingested dialogue (pilot mode).  When it truncates,
    the answer session is kept (it holds the evidence), otherwise the question
    would become unanswerable for a reason that has nothing to do with memory.
    """
    sessions = record.get("haystack_sessions") or []
    session_ids = record.get("haystack_session_ids") or []
    dates = record.get("haystack_dates") or []
    answer_sessions = set(record.get("answer_session_ids") or [])

    # ---- session selection ------------------------------------------------ #
    # ``max_turns`` is a *turn* budget, not a session count.  The evidence session
    # is always kept (otherwise the question becomes unanswerable for reasons that
    # have nothing to do with memory), then the most recent sessions fill the rest,
    # because recency is the strongest distractor signal.
    selected_indices: List[int] = []
    if max_turns is not None and max_turns > 0:
        answer_positions = [i for i, sid in enumerate(session_ids) if sid in answer_sessions]
        if not answer_positions:
            answer_positions = [len(sessions) - 1] if sessions else []
        budget = max_turns
        used: set = set()
        # The budget is in *turns* (what the memory system ingests), so cost each
        # session by the number of turns it produces, not by its message count.  The
        # old message-based accounting under-spent the budget by ~2x.
        turn_cost = [len(sessions_to_dialogues([session])) for session in sessions]
        # 1) evidence first
        for pos in answer_positions:
            if pos in used or pos >= len(sessions):
                continue
            cost = turn_cost[pos]
            if cost > budget and used:
                continue
            selected_indices.append(pos); used.add(pos); budget -= cost
        # 2) then the newest remaining sessions until the budget is spent
        for pos in range(len(sessions) - 1, -1, -1):
            if budget <= 0:
                break
            if pos in used:
                continue
            cost = turn_cost[pos]
            if cost > budget and selected_indices:
                continue
            selected_indices.append(pos); used.add(pos); budget -= cost
        selected_indices.sort()
    else:
        selected_indices = list(range(len(sessions)))

    selected = [sessions[i] for i in selected_indices]
    # Tag every turn with whether it came from an evidence session *before*
    # flattening: the turn-level cut below must protect the gold, and a plain
    # suffix cut cannot tell which turns those are.
    per_session_turns = [sessions_to_dialogues([session]) for session in selected]
    evidence_flags = [
        session_ids[pos] in answer_sessions if pos < len(session_ids) else False
        for pos in selected_indices
    ]
    dialogues = [turn for turns in per_session_turns for turn in turns]
    turn_is_evidence = [
        flag for turns, flag in zip(per_session_turns, evidence_flags) for _ in turns
    ]
    if max_turns is not None and max_turns > 0 and len(dialogues) > max_turns:
        # Keep every evidence turn, then fill the remaining budget with the newest
        # other turns.  The previous `dialogues[-max_turns:]` cut the *newest* turns
        # only, which silently deleted earlier turns of a long evidence session while
        # ``answer_session_included`` (computed before the cut) still claimed it was
        # kept -- the question became unanswerable for a reason unrelated to memory.
        kept = [i for i, is_evidence in enumerate(turn_is_evidence) if is_evidence]
        keep_set = set(kept)
        for i in range(len(dialogues) - 1, -1, -1):
            if len(kept) >= max_turns:
                break
            if i not in keep_set:
                kept.append(i)
                keep_set.add(i)
        ordered = sorted(keep_set)
        if len(ordered) > max_turns:
            # Evidence alone exceeds the budget: keep its newest turns.
            ordered = ordered[-max_turns:]
        dialogues = [dialogues[i] for i in ordered]
        kept_evidence = [i for i in ordered if turn_is_evidence[i]]
    else:
        kept_evidence = [i for i, is_evidence in enumerate(turn_is_evidence) if is_evidence]

    question = str(record.get("question", "") or "")
    answer = record.get("answer")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    probe_type = QUESTION_TYPE_MAP.get(str(record.get("question_type", "")), OTHER)

    probes: List[Probe] = []
    if question:
        probes.append(
            Probe(
                question=question,
                answer=str(answer or ""),
                probe_type=probe_type,
                fact_key=str(record.get("question_type", "")),
                history_ref=",".join(sorted(answer_sessions)),
                meta={
                    "question_id": record.get("question_id"),
                    "question_type": record.get("question_type"),
                    "question_date": record.get("question_date"),
                    "answer_session_ids": sorted(answer_sessions),
                },
            )
        )

    return Episode(
        episode_id=f"lme_{record.get('question_id', index)}",
        dialogues=dialogues,
        probes=probes,
        facts={},
        meta={
            "source": "longmemeval",
            "question_type": record.get("question_type"),
            "num_sessions": len(sessions),
            "num_sessions_kept": len(selected),
            # Turns, not messages: ``kept_turns`` counts turn pairs, so mixing units
            # here made "kept vs full" incomparable.
            "full_turns": len(sessions_to_dialogues(sessions)),
            "kept_turns": len(dialogues),
            "truncated": max_turns is not None and len(selected) < len(sessions),
            # Computed from the turns that actually survived the cut, not from the
            # pre-cut session selection.
            "answer_session_included": (
                bool(kept_evidence) if answer_sessions else None
            ),
        },
    )


def load_longmemeval(
    path: str = None,
    limit: Optional[int] = None,
    max_turns: Optional[int] = None,
    types: Optional[Sequence[str]] = None,
) -> List[Episode]:
    """
    Load LongMemEval questions as episodes.

    ``types`` filters question types (e.g. ``["knowledge-update"]``) and ``limit``
    counts *matching episodes*, not scanned records.  The old behaviour capped the
    scan first and filtered afterwards, so ``--lme-types knowledge-update --limit
    50`` returned ZERO episodes (the first 50 records are all
    ``single-session-user``) -- silently, since a small dataset looks normal.
    """
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LongMemEval not found at {path}. Download it with:\n"
            "  mkdir -p data/longmemeval && curl -L -o data/longmemeval/"
            "longmemeval_s_cleaned.json \\\n"
            "  https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
            "longmemeval_s_cleaned.json"
        )
    wanted = set(types) if types else None
    # A filter has to scan past non-matching records, so the record cap is derived
    # from the episode cap (and is generous) rather than equal to it.
    scan_cap = None if not limit else max(limit * 50, 500)
    episodes: List[Episode] = []
    type_counts: Counter = Counter()
    scanned = 0
    for index, record in enumerate(_iter_raw(path, scan_cap)):
        scanned = index + 1
        qtype = str(record.get("question_type", ""))
        type_counts[qtype] += 1
        if wanted and qtype not in wanted:
            continue
        episodes.append(record_to_episode(record, index, max_turns=max_turns))
        if limit is not None and len(episodes) >= limit:
            break
    if limit is not None and len(episodes) < limit and scan_cap is not None and scanned >= scan_cap:
        logger.warning(
            "LongMemEval: scanned the %d-record cap and found only %d matching "
            "episode(s) (sought %d); raise --limit or drop --lme-types",
            scanned, len(episodes), limit,
        )
    logger.info(
        "LongMemEval: %d episodes (of %d scanned) | types: %s | max_turns=%s",
        len(episodes), sum(type_counts.values()), dict(type_counts), max_turns,
    )
    return episodes


def describe(episodes: Sequence[Episode]) -> Dict[str, Any]:
    """Small summary used by ``evaluation.py --inspect``."""
    if not episodes:
        return {"episodes": 0}
    turns = [e.num_turns for e in episodes]
    import statistics as st

    return {
        "episodes": len(episodes),
        "turns_mean": round(st.mean(turns), 1),
        "turns_min": min(turns),
        "turns_max": max(turns),
        "probe_types": dict(Counter(p.probe_type for e in episodes for p in e.probes)),
        "question_types": dict(Counter(e.meta.get("question_type") for e in episodes)),
        "truncated_episodes": sum(1 for e in episodes if e.meta.get("truncated")),
    }
