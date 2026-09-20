"""
The proposed system: three-layer memory + OVERRIDES + sliding window.

This class is a thin adapter -- all memory logic lives in :class:`MemoryManager`,
which in turn only knows :class:`BaseMemoryStore`.  Swapping
``InMemoryStore`` for ``RedisSQLiteHybridStore`` changes nothing here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm import BaseLLM
from ..memory_manager import MemoryManager
from ..prompts import build_agent_messages
from ..store.base import BaseMemoryStore
from ..token_utils import estimate_tokens
from ..tools import ToolExecutor
from .base_agent import BaseAgent


class ThreeLayerAgent(BaseAgent):
    """Ours."""

    system_name = "three_layer"

    def __init__(
        self,
        store: BaseMemoryStore,
        llm: BaseLLM,
        summarizer: Optional[BaseLLM] = None,
        chain_strategy: Optional[str] = None,
        index_keep_recent: Optional[int] = None,
        index_group_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(store, llm, summarizer, **kwargs)
        self.manager: MemoryManager = MemoryManager(
            store,
            self.summarizer,
            episode_id="pending",
            recent_window_turns=self.recent_window_turns,
            active_chain_token_limit=self.active_chain_token_limit,
            chain_strategy=chain_strategy,
            index_keep_recent=index_keep_recent,
            index_group_size=index_group_size,
            reset_on_bind=False,   # reset_episode() owns the lifecycle
        )
        self.manager.debug = False

    # ------------------------------------------------------------------ #
    def reset_episode(self, episode_id: str, *, resume: bool = False, purge: bool = False) -> None:
        self.episode_id = episode_id
        self.manager.recent_window_turns = self.recent_window_turns
        self.manager.active_chain_token_limit = self.active_chain_token_limit
        self.manager.reset_episode(episode_id, resume=resume, purge=purge)
        self.executor = ToolExecutor(self.store, episode_id=episode_id)
        self.turn_index = -1

    def add_turn(self, user_input: str, agent_output: str) -> Any:
        stats = self.manager.add_dialog_turn(user_input, agent_output)
        self.turn_index = stats.turn_index
        return stats

    def add_turns(self, turns) -> Any:
        """Ingest a batch; the manager may summarise concurrently."""
        stats = self.manager.add_dialog_turns(list(turns))
        if stats:
            self.turn_index = stats[-1].turn_index
        return stats

    def set_ingest_concurrency(self, concurrency: int) -> None:
        self.manager.ingest_concurrency = max(0, int(concurrency))

    # ------------------------------------------------------------------ #
    def build_messages(self, question: str) -> List[Dict[str, str]]:
        return build_agent_messages(
            self.manager.get_window(),
            self.manager.chain_summaries(),      # only unfiled summaries are rendered
            question,
            recent_window_turns=self.recent_window_turns,
            indexes=self.manager.rendered_index_entries(),
        )

    def context_messages(self, question: str, selfwrite: bool = False) -> List[Dict[str, str]]:
        return build_agent_messages(
            self.manager.get_window(),
            self.manager.chain_summaries(),
            question,
            recent_window_turns=self.recent_window_turns,
            indexes=self.manager.rendered_index_entries(),
            selfwrite=selfwrite,
        )

    def answer_and_remember(self, question: str):
        """
        One LLM call that answers the user *and* records the turn.

        Returns ``(AgentRunResult, MemoryTurnStats)``.  When the model did not
        emit a usable ``<MEMORY_UPDATE>`` block, the manager transparently falls
        back to the dedicated summariser, so this is never worse in correctness
        than ``answer()`` followed by ``add_turn()`` -- only in cost.
        """
        run = self.answer(question, selfwrite=True)
        stats = self.manager.add_dialog_turn_selfwritten(
            question, run.answer, run.memory_block
        )
        self.turn_index = stats.turn_index
        return run, stats

    def finalize_episode(self) -> Dict[str, Any]:
        metrics = self.manager.episode_metrics()
        metrics["system"] = self.system_name
        # Per-turn means so the harness can compare all systems on one code path.
        if self.manager.stats:
            # "context_*": everything this system carries into the prompt
            # (rendered chain + sliding window) -- comparable with the baselines.
            # "chain_*": the active summary chain only -- this system's own metric.
            window_tokens = sum(
                estimate_tokens(r.user_msg) + estimate_tokens(r.agent_msg)
                for r in self.manager.get_window()
            )
            metrics["context_tokens_mean"] = (
                sum(s.active_chain_tokens for s in self.manager.stats) / len(self.manager.stats)
                + window_tokens
            )
            # The headline metric the spec asks for: mean tokens of the *summary
            # chain* per episode (bodies only).  ``avg_context_tokens`` is the
            # extra, comparable figure (chain + sliding window, as rendered).
            metrics["avg_active_chain_tokens"] = sum(
                s.active_chain_text_tokens for s in self.manager.stats
            ) / len(self.manager.stats)
            metrics["avg_active_chain_tokens_rendered"] = sum(
                s.active_chain_tokens for s in self.manager.stats
            ) / len(self.manager.stats)
            metrics["avg_context_tokens"] = (
                metrics["avg_active_chain_tokens_rendered"] + window_tokens
            )
            metrics["avg_all_rendered_tokens"] = sum(
                estimate_tokens(s.render()) for s in self.manager.list_active_summaries()
            )
            metrics["format_retries"] = sum(
                1 for s in self.manager.stats if s.format_retry_used
            )
            # One-call ("self-write") accounting: a self-written turn needed no
            # separate summariser call at all.
            metrics["self_written_turns"] = sum(1 for s in self.manager.stats if s.self_written)
            metrics["self_write_fallbacks"] = sum(
                1 for s in self.manager.stats if s.self_write_fallback
            )
        # Hybrid-store diagnostics (Redis availability, fallback count, resyncs).
        diagnostics = getattr(self.store, "diagnostics", None)
        if diagnostics:
            metrics.update({f"store_{k}": v for k, v in diagnostics.items()})
        return metrics

    # Convenience for chat_debug.py
    def debug_state(self) -> str:
        return self.manager.debug_state()
