"""
memory3l -- a three-layer memory system for LLM agents, built for batch
dataset evaluation against published memory baselines.

Layers
------
1. Active event chain   (sent to the LLM): short summaries, each tagged with the
   ids of the older summaries it *overrides*:  ``[OVERRIDES: id1,id2|none]``.
2. Archived summary library (exact-id lookup only): every summary that was
   overridden or evicted by capacity compression.  Never deleted.
3. Raw detail store (exact-id lookup only): the complete original dialogue.
   Summaries only keep a ``raw_ref_id``.

Plus a recent-turn sliding window of verbatim dialogue prepended to the prompt.

Hard constraints honoured by this package
-----------------------------------------
* No vectors, no embeddings, no RAG / similarity search.  Retrieval is by exact
  id (``summary_id`` / ``reference_id``) only.
* ``MemoryManager`` is storage agnostic: ``InMemoryStore`` and
  ``RedisSQLiteHybridStore`` are interchangeable without touching memory logic.
* ``store.reset(episode_id=...)`` is called before every dataset episode so that
  episodes cannot contaminate each other.
"""

from .models import ActiveSummary, ArchivedSummary, RawDialogRecord, ToolCall, ToolResult
from .token_utils import estimate_tokens
from .memory_manager import MemoryManager, MemoryTurnStats

__all__ = [
    "ActiveSummary",
    "ArchivedSummary",
    "RawDialogRecord",
    "ToolCall",
    "ToolResult",
    "estimate_tokens",
    "MemoryManager",
    "MemoryTurnStats",
]

__version__ = "1.0.0"
