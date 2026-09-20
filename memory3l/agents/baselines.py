"""
The three baselines.

Each baseline is implemented with the *same* LLM, the same answering contract and
the same tool-loop harness, so differences in the metrics are attributable to the
memory mechanism.  Baseline-specific behaviour is documented per class.

Fairness notes (recorded because they matter for the report):

* Every baseline gets the same recent-window of verbatim turns.  ``full_context``
  additionally gets everything older; ``memgpt_style`` and ``naive_chain`` get a
  rolling summary instead of the raw text they dropped.
* Only ``three_layer`` has archival tools.  That is the point of the comparison:
  a baseline cannot look up what it deleted because its design has no place to
  look it up in.  The tool-call metrics therefore report per-system rates, and
  ``full_context`` (no tools) is excluded from ``Tool_Call_Success_Rate``
  averaging rather than counted as 0 successes.
* ``full_context`` is the honest upper bound on the raw-history baselines under a
  fixed context budget: when the history exceeds the budget, the *oldest* turns
  are dropped (they cannot be summarised, that is the whole point of the
  baseline).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from ..llm import LLMError
from ..models import RawDialogRecord, make_id, now_ts
from ..prompts import build_agent_messages, build_full_context_messages, insert_after_recent_window
from ..token_utils import count_message_tokens, estimate_tokens
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Baseline 1: Plain Full Context
# --------------------------------------------------------------------------- #
class FullContextAgent(BaseAgent):
    """No memory mechanism: every raw turn is kept and stuffed into the prompt."""

    system_name = "full_context"

    def reset_episode(self, episode_id: str, *, resume: bool = False) -> None:
        self.episode_id = episode_id
        self.turn_index = -1
        self.turn_tokens: List[int] = []
        self.store.bind_episode(
            episode_id, recent_window_turns=self.recent_window_turns, reset=True
        )
        self.executor = None  # no tools by design

    def add_turn(self, user_input: str, agent_output: str) -> Dict[str, Any]:
        self.turn_index = self.store.next_turn_index()
        raw_ref_id = make_id(self.episode_id, "raw", self.turn_index, salt=f"{user_input}|{agent_output}")
        record = RawDialogRecord(
            reference_id=raw_ref_id,
            user_msg=user_input,
            agent_msg=agent_output,
            timestamp=now_ts(),
            episode_id=self.episode_id,
            turn_index=self.turn_index,
        )
        self.store.add_raw_record(record, episode_id=self.episode_id)
        self.store.append_window_record(record, episode_id=self.episode_id)
        # Context size this system actually *renders* for the history so far, which
        # is the truncated one.  Reporting the untruncated stored total (as before)
        # over-stated the baseline's cost on every episode past the budget, which is
        # exactly the regime where the cost comparison matters.
        rendered = build_full_context_messages(
            self.store.list_raw_records(self.episode_id), "", max_tokens=self.raw_context_token_limit
        )
        self.turn_tokens.append(count_message_tokens(rendered))
        return {"turn_index": self.turn_index, "raw_ref_id": raw_ref_id}

    def build_messages(self, question: str) -> List[Dict[str, str]]:
        records = self.store.list_raw_records(self.episode_id)
        return build_full_context_messages(records, question, max_tokens=self.raw_context_token_limit)

    def finalize_episode(self) -> Dict[str, Any]:
        records = self.store.list_raw_records(self.episode_id)
        stored_tokens = sum(
            estimate_tokens(r.user_msg) + estimate_tokens(r.agent_msg) for r in records
        )
        # Per-turn mean of the context actually rendered, so this column means the
        # same thing for every system (see README "metric definitions").
        rendered_mean = (
            sum(self.turn_tokens) / len(self.turn_tokens) if self.turn_tokens else 0.0
        )
        return {
            "system": self.system_name,
            "num_turns": self.turn_index + 1,
            "context_raw_tokens": stored_tokens,
            # The baseline's "memory" IS the raw context, so the comparable chain
            # figure is the rendered context (post-truncation), averaged per turn.
            "active_chain_tokens": rendered_mean,
            "avg_active_chain_tokens": rendered_mean,
            "avg_active_chain_tokens_rendered": rendered_mean,
            "avg_context_tokens": rendered_mean,
            "archived_count": 0,
        }


# --------------------------------------------------------------------------- #
# Baseline 2: MemGPT-style destructive compression
# --------------------------------------------------------------------------- #
class MemGPTStyleAgent(BaseAgent):
    """
    Older history is replaced by a *rolling* summary and then deleted.

    Mirrors the MemGPT/Letta "recursive summarisation" pattern: there is a single
    mutable summary blob, regenerated from (old summary + evicted messages), with
    no archive, no ids, no override bookkeeping.  Facts that the blob drops are
    simply gone -- which is exactly what History_Fact_Acc is designed to expose.
    """

    system_name = "memgpt_style"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rolling_summary = ""
        self.evictions = 0
        self.evicted_records = 0
        self.turn_tokens: List[int] = []

    def reset_episode(self, episode_id: str, *, resume: bool = False) -> None:
        self.episode_id = episode_id
        self.turn_index = -1
        self._rolling_summary = ""
        self.evictions = 0
        self.evicted_records = 0
        self.turn_tokens = []
        self.store.bind_episode(
            episode_id, recent_window_turns=self.recent_window_turns, reset=True
        )
        self.executor = None

    def add_turn(self, user_input: str, agent_output: str) -> Dict[str, Any]:
        self.turn_index = self.store.next_turn_index()
        raw_ref_id = make_id(self.episode_id, "raw", self.turn_index, salt=f"{user_input}|{agent_output}")
        record = RawDialogRecord(
            reference_id=raw_ref_id,
            user_msg=user_input,
            agent_msg=agent_output,
            timestamp=now_ts(),
            episode_id=self.episode_id,
            turn_index=self.turn_index,
        )
        self.store.add_raw_record(record, episode_id=self.episode_id)
        window = self.store.append_window_record(record, episode_id=self.episode_id)
        self._maybe_evict(window)
        # What this system carries in context: rolling summary + surviving window.
        self.turn_tokens.append(
            estimate_tokens(self._rolling_summary)
            + sum(estimate_tokens(r.user_msg) + estimate_tokens(r.agent_msg) for r in window)
        )
        return {"turn_index": self.turn_index, "raw_ref_id": raw_ref_id, "evicted": self.evictions}

    def _maybe_evict(self, window: Sequence[RawDialogRecord]) -> None:
        """
        Destructive compression: when the verbatim history exceeds the budget,
        all but the newest ``recent_window_turns`` messages are summarised into
        the rolling blob and dropped.
        """
        all_records = self.store.list_raw_records(self.episode_id)
        verbatim_tokens = sum(estimate_tokens(r.user_msg) + estimate_tokens(r.agent_msg) for r in all_records)
        if verbatim_tokens <= self.raw_context_token_limit:
            return
        # Keep exactly the newest ``recent_window_turns`` records.  The old
        # ``max(1, ...)`` evicted one record even when the history was already
        # shorter than the window, contradicting the docstring.
        to_evict = all_records[: max(0, len(all_records) - self.recent_window_turns)]
        if not to_evict:
            return
        turn_text = "\n".join(f"user: {r.user_msg}\nagent: {r.agent_msg}" for r in to_evict)
        previous = self._rolling_summary
        prompt = (
            "Recursive summary update. Merge the EXISTING SUMMARY with the NEW MESSAGES into one "
            "compact summary. Keep facts and concrete values; drop conversational filler. "
            "Output only the summary text.\n\n"
            f"<EXISTING_SUMMARY>\n{previous or '(empty)'}\n</EXISTING_SUMMARY>\n"
            f"<NEW_MESSAGES>\n{turn_text}\n</NEW_MESSAGES>"
        )
        try:
            response = self.summarizer.generate(
                [
                    {"role": "system", "content": "You compress dialogue history recursively. Output only the summary."},
                    {"role": "user", "content": prompt},
                ]
            )
            self._rolling_summary = (response.text or "").strip() or previous
        except LLMError as exc:
            logger.warning("memgpt_style compression failed: %s", exc)
            self._rolling_summary = (previous + "\n" + turn_text).strip()
        # Destructive: the evicted raw records are deleted from the store entirely.
        for record in to_evict:
            try:
                self.store.delete_raw_record(record.reference_id, episode_id=self.episode_id)
            except NotImplementedError:  # pragma: no cover - store without deletion
                logger.warning("store cannot delete raw records; eviction is summary-only")
        self.evicted_records += len(to_evict)
        self.evictions += 1

    # ------------------------------------------------------------------ #
    def build_messages(self, question: str) -> List[Dict[str, str]]:
        remaining = self._remaining_records()
        window = remaining[-self.recent_window_turns :] if remaining else []
        chain_block = (
            [f"<ROLLING_SUMMARY>\n{self._rolling_summary}\n</ROLLING_SUMMARY>"]
            if self._rolling_summary
            else ["<ROLLING_SUMMARY>\n(empty)\n</ROLLING_SUMMARY>"]
        )
        messages = build_agent_messages(
            window, [], question,
            recent_window_turns=self.recent_window_turns,
            tools_available=self.supports_tools(),
        )
        # Inject the rolling summary right after the recent window (it replaces
        # the older raw turns that were destroyed).
        messages[1]["content"] = insert_after_recent_window(
            messages[1]["content"], "\n".join(chain_block)
        )
        return messages

    def _remaining_records(self) -> List[RawDialogRecord]:
        return self.store.list_raw_records(self.episode_id)

    def finalize_episode(self) -> Dict[str, Any]:
        rolling_tokens = estimate_tokens(self._rolling_summary)
        return {
            "system": self.system_name,
            "num_turns": self.turn_index + 1,
            "archived_count": 0,
            "capacity_compressions": self.evictions,
            "evicted_records": self.evicted_records,
            "active_chain_tokens": rolling_tokens,
            "avg_active_chain_tokens_rendered": (
                sum(self.turn_tokens) / len(self.turn_tokens) if self.turn_tokens else 0.0
            ),
            "avg_context_tokens": (
                sum(self.turn_tokens) / len(self.turn_tokens) if self.turn_tokens else 0.0
            ),
            "rolling_summary_tokens": rolling_tokens,
            # No discrete summary chain exists in this baseline: report it as
            # absent (None) rather than 0 tokens, which would read as "free memory".
            "avg_active_chain_tokens": None,
        }


# --------------------------------------------------------------------------- #
# Baseline 3: naive temporal summary chain
# --------------------------------------------------------------------------- #
class NaiveChainAgent(BaseAgent):
    """
    Append-only summary chain, oldest-summary eviction, no archive, no OVERRIDES.

    Facts overtaken by later turns are *never* marked -- the chain simply grows
    until the token budget forces the oldest summaries to be dropped.  There is no
    archive and no raw-detail tool, so a dropped fact is unrecoverable.
    """

    system_name = "naive_chain"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain: List[Dict[str, Any]] = []
        self.dropped = 0
        self.turn_tokens: List[int] = []
        #: chain-body tokens per turn (the column that is comparable with three_layer)
        self.turn_chain_tokens: List[int] = []

    def reset_episode(self, episode_id: str, *, resume: bool = False) -> None:
        self.episode_id = episode_id
        self.turn_index = -1
        self.chain = []
        self.dropped = 0
        self.turn_tokens = []
        self.turn_chain_tokens = []
        self.store.bind_episode(
            episode_id, recent_window_turns=self.recent_window_turns, reset=True
        )
        self.executor = None

    def add_turn(self, user_input: str, agent_output: str) -> Dict[str, Any]:
        self.turn_index = self.store.next_turn_index()
        raw_ref_id = make_id(self.episode_id, "raw", self.turn_index, salt=f"{user_input}|{agent_output}")
        record = RawDialogRecord(
            reference_id=raw_ref_id,
            user_msg=user_input,
            agent_msg=agent_output,
            timestamp=now_ts(),
            episode_id=self.episode_id,
            turn_index=self.turn_index,
        )
        self.store.add_raw_record(record, episode_id=self.episode_id)
        self.store.append_window_record(record, episode_id=self.episode_id)

        full_prompt = (
            "Summarise the NEW_TURN in one short factual sentence. Output only the summary text, "
            "with no tags and no explanation.\n\n"
            f"<NEW_TURN>\nuser: {user_input}\nagent: {agent_output}\n</NEW_TURN>"
        )
        text = ""
        try:
            response = self.summarizer.generate(
                [
                    {"role": "system", "content": "You summarise dialogue turns in one factual sentence. No tags."},
                    {"role": "user", "content": full_prompt},
                ]
            )
            text = (response.text or "").strip()
        except LLMError as exc:
            logger.warning("naive_chain summary failed: %s", exc)
        if not text:
            text = f"user: {user_input} / agent: {agent_output}"[:200]
        # A naive chain stores no OVERRIDES tag even if the model emits one.
        if "[OVERRIDES" in text.upper():
            text = text[: text.upper().index("[OVERRIDES")].strip()
        self.chain.append({"turn": self.turn_index, "text": text, "raw_ref_id": raw_ref_id})
        self._trim()
        chain_body_tokens = estimate_tokens(" ".join(c["text"] for c in self.chain))
        self.turn_chain_tokens.append(chain_body_tokens)
        self.turn_tokens.append(
            chain_body_tokens
            + sum(
                estimate_tokens(r.user_msg) + estimate_tokens(r.agent_msg)
                for r in self.store.get_window(self.episode_id)
            )
        )
        return {"turn_index": self.turn_index, "chain_size": len(self.chain)}

    def _trim(self) -> None:
        """Drop oldest summaries until the chain fits the token budget."""
        while self.chain and sum(estimate_tokens(c["text"]) for c in self.chain) > self.active_chain_token_limit:
            self.chain.pop(0)
            self.dropped += 1

    def build_messages(self, question: str) -> List[Dict[str, str]]:
        # Represent the chain as "summaries" without ids and without OVERRIDES,
        # which is precisely the naive design.
        block_lines = ["<SUMMARY_CHAIN>"]
        for item in self.chain:
            block_lines.append(f"- (turn {item['turn']}) {item['text']}")
        if not self.chain:
            block_lines.append("(empty)")
        block_lines.append("</SUMMARY_CHAIN>")
        messages = build_agent_messages(
            self.store.get_window(self.episode_id), [], question,
            recent_window_turns=self.recent_window_turns,
            tools_available=self.supports_tools(),
        )
        messages[1]["content"] = insert_after_recent_window(
            messages[1]["content"], "\n".join(block_lines)
        )
        return messages

    def finalize_episode(self) -> Dict[str, Any]:
        chain_tokens = sum(estimate_tokens(c["text"]) for c in self.chain)
        return {
            "system": self.system_name,
            "num_turns": self.turn_index + 1,
            "archived_count": 0,
            "active_chain_size": len(self.chain),
            "active_chain_tokens": chain_tokens,
            # Same definition as three_layer's column: per-turn mean of the summary
            # *bodies* the system carries.  Reporting the end-of-episode total here
            # compared a final state against a per-turn mean.
            "avg_active_chain_tokens": (
                sum(self.turn_chain_tokens) / len(self.turn_chain_tokens)
                if self.turn_chain_tokens else 0.0
            ),
            "avg_active_chain_tokens_rendered": (
                sum(self.turn_tokens) / len(self.turn_tokens) if self.turn_tokens else 0.0
            ),
            "avg_context_tokens": (
                sum(self.turn_tokens) / len(self.turn_tokens) if self.turn_tokens else 0.0
            ),
            "capacity_compressions": self.dropped,
        }
