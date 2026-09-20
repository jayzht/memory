"""
Agent systems under comparison.

Four interchangeable implementations, all sharing the same
``bind/add_turn/answer/reset`` interface so ``evaluation.py`` can treat them
uniformly:

===============  =========================================================
name             description
===============  =========================================================
three_layer      **ours**: active chain + `[OVERRIDES: ...]` + permanent
                 archive + raw store + sliding window (Redis+SQLite capable)
full_context     Baseline 1 -- plain full context, every raw turn, no memory
                 mechanism at all (budget-truncated from the oldest side when
                 it exceeds the context budget)
memgpt_style     Baseline 2 -- MemGPT-style *destructive* compression: when the
                 budget is exceeded, older messages are replaced by a rolling
                 summary and the originals are dropped (no archive lookup, no
                 OLD/NEW bookkeeping)
naive_chain      Baseline 3 -- naive temporal summary chain: append-only
                 summaries, compressed by dropping the oldest once over budget
                 (no archive, no OVERRIDES tags, no raw store access)
===============  =========================================================
"""

from __future__ import annotations

from .base_agent import AgentRunResult, BaseAgent, build_agent
from .three_layer import ThreeLayerAgent
from .baselines import FullContextAgent, MemGPTStyleAgent, NaiveChainAgent

__all__ = [
    "AgentRunResult",
    "BaseAgent",
    "build_agent",
    "ThreeLayerAgent",
    "FullContextAgent",
    "MemGPTStyleAgent",
    "NaiveChainAgent",
]

AGENT_REGISTRY = {
    "three_layer": ThreeLayerAgent,
    "ours": ThreeLayerAgent,
    "full_context": FullContextAgent,
    "plain_full_context": FullContextAgent,
    "baseline1": FullContextAgent,
    "memgpt_style": MemGPTStyleAgent,
    "memgpt": MemGPTStyleAgent,
    "baseline2": MemGPTStyleAgent,
    "naive_chain": NaiveChainAgent,
    "naive": NaiveChainAgent,
    "baseline3": NaiveChainAgent,
}

#: Canonical system names written into every result file.
CANONICAL_NAMES = {
    ThreeLayerAgent: "three_layer",
    FullContextAgent: "full_context",
    MemGPTStyleAgent: "memgpt_style",
    NaiveChainAgent: "naive_chain",
}
