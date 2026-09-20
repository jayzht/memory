"""
Data model for the three-layer memory system.

These dataclasses are the *only* contract between business logic
(``MemoryManager``) and any storage backend, which is what keeps the manager
storage-agnostic.

Extensions over the original sketch (documented so experiments can be traced):

* ``ActiveSummary.superseded_by`` / ``ArchivedSummary.superseded_by``
  When summary ``A`` is overridden by ``B``, ``A`` is archived with
  ``superseded_by = B``.  This lets ``get_archived_summary(A)`` walk *forward*
  to the currently-valid fact, which is what makes History_Fact_Acc measurable
  instead of returning a stale value and confusing the agent.
* ``raw_ref_ids: list[str]``
  A capacity-merge summary condenses several older summaries, so it must be able
  to reference several raw records.  ``raw_ref_id`` is kept as the convenience
  "primary" reference (first raw ref) for the single-reference case required by
  the spec, and ``raw_ref_ids`` is the complete list.
* ``episode_id`` on every record keeps ids globally unique in the shared
  archive, while Redis keys stay episode-scoped.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
REF_SEP = "|"  # separator used when packing a description list into one cell

#: Sentinel ``index_id`` meaning "kept by id only, not rendered".
#: It lives here (not in the manager) so the retrieval tools can use it without
#: importing the manager, which would create an import cycle.
LAZY_INDEX_ID = "__LAZY_SUMMARIES__"


def pack_refs(refs: Sequence[str]) -> str:
    """Pack a list of ids into a single text cell (SQLite / Redis safe)."""
    return REF_SEP.join(r for r in refs if r)


def unpack_refs(value: Optional[str]) -> List[str]:
    """Inverse of :func:`pack_refs`; tolerant to None / empty."""
    if not value:
        return []
    return [item for item in str(value).split(REF_SEP) if item]


def now_ts() -> float:
    return time.time()


def make_id(episode_id: str, kind: str, seq: int, salt: str = "") -> str:
    """
    Deterministic, human-readable, globally unique id.

    ``ep_007/s003@a1b2c3`` -- episode-scoped, sortable by sequence, collision
    resistant (short hash of the content salt).  Determinism matters: re-running
    an episode must reproduce the same ids so checkpoints and logs line up.
    """
    digest = hashlib.sha1(f"{episode_id}:{kind}:{seq}:{salt}".encode("utf-8")).hexdigest()[:6]
    return f"{episode_id}/{kind}{seq:03d}@{digest}"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class RawDialogRecord:
    """Layer 3: the complete original dialogue, kept forever (SQLite table)."""

    reference_id: str
    user_msg: str
    agent_msg: str
    timestamp: float = field(default_factory=now_ts)
    # --- bookkeeping (not part of the required core fields) ----------------- #
    episode_id: str = ""
    turn_index: int = -1          # turn ordinal inside the episode, -1 = unset
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> tuple:
        return (
            self.reference_id,
            self.episode_id,
            self.turn_index,
            self.user_msg,
            self.agent_msg,
            float(self.timestamp),
            "",  # meta kept as JSON text
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActiveSummary:
    """
    Layer 1 (lower level): one summary per dialogue turn -- a "catalogue entry"
    that points at the raw dialogue it came from.
    """

    summary_id: str
    text: str
    override_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=now_ts)
    raw_ref_id: str = ""
    # --- bookkeeping ------------------------------------------------------- #
    episode_id: str = ""
    seq: int = -1                       # monotonic creation order inside episode
    origin: str = "event"               # "event" | "capacity_merge"
    raw_ref_ids: List[str] = field(default_factory=list)
    merged_from: List[str] = field(default_factory=list)
    fact_keys: List[str] = field(default_factory=list)   # canonical slots ([FACTS: k=v])
    # --- membership in the level-2 index ------------------------------ #
    #: Index entry this summary belongs to ("" = still directly visible in the
    #: active chain).  A summary is never deleted when it is filed under an
    #: index: the index only *points* at it, exactly like a table of contents.
    index_id: str = ""

    def __post_init__(self) -> None:
        if not self.raw_ref_ids and self.raw_ref_id:
            self.raw_ref_ids = [self.raw_ref_id]
        if self.raw_ref_ids and not self.raw_ref_id:
            self.raw_ref_id = self.raw_ref_ids[0]

    @property
    def overrides_tag(self) -> str:
        """Render the mandatory ``[OVERRIDES: ...]`` tag."""
        return f"[OVERRIDES: {','.join(self.override_ids) if self.override_ids else 'none'}]"

    def render(self, turn_label: str = "") -> str:
        """Line as it appears in the LLM context (id must stay visible)."""
        when = f"[{turn_label}] " if turn_label else ""
        return (
            f"- {self.summary_id} {self.overrides_tag} "
            f"{when}(raw_ref: {','.join(self.raw_ref_ids) or 'none'}) {self.text}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexEntry:
    """
    Layer 1 upper level: a title/index entry over a *group* of summaries.

    This is the user's design: when the chain grows, you do not merge the leaves
    into one lossy blob -- you keep every leaf intact and add a level above it that
    says "these N summaries cover period t1..t2, and this is what they are about".
    Retrieval then walks title -> catalogue entries -> raw dialogue.

    ``members`` order is chronological; ``member_summaries`` caches the rendered
    member lines so a store can round-trip the entry without a join.
    """

    index_id: str
    title: str
    #: The LLM-written subject of the group, kept separately from the derived
    #: ``属性=值`` digest so the title can be *recomputed* from the surviving members
    #: (no LLM call) whenever a member is overridden or merged away.  Storing the
    #: whole title made it go stale: the index still advertised the old value.
    theme: str = ""
    members: List[str] = field(default_factory=list)
    span_start: float = 0.0
    span_end: float = 0.0
    turn_start: int = -1
    turn_end: int = -1
    fact_keys: List[str] = field(default_factory=list)
    episode_id: str = ""
    seq: int = -1
    timestamp: float = field(default_factory=now_ts)
    #: member lines as rendered at index-build time (legacy, full lines)
    member_summaries: List[str] = field(default_factory=list)
    #: short fact snippets -- what a preview should show.  A preview must be a
    #: *hint*, not a copy of the summary: full lines cost ~100 tokens each and made
    #: the index layer as expensive as the chain it was supposed to replace.
    previews: List[str] = field(default_factory=list)
    #: when this entry is itself a "super index" over older index entries
    child_index_ids: List[str] = field(default_factory=list)
    #: Hard cap on rendered title length.  A title is a pointer, not content: an
    #: unbounded title made a single index entry cost more than the whole budget
    #: (measured: 96-token title + 164-token preview), and nested super indexes then
    #: multiplied the problem.
    MAX_TITLE_CHARS: int = 60

    @property
    def size(self) -> int:
        return len(self.members)

    def render_cost(self) -> int:
        """Tokens this entry costs with a bare title (the floor for any layout)."""
        from .token_utils import estimate_tokens

        return estimate_tokens(self.capped_title()) + estimate_tokens(self.index_id) + 12

    def time_label(self) -> str:
        if self.turn_start < 0 or self.turn_end < 0:
            return "?"
        return f"turn {self.turn_start}-{self.turn_end}" if self.turn_start != self.turn_end else f"turn {self.turn_start}"

    def capped_title(self) -> str:
        # The cap is a config knob: the digest is built from several attributes, and
        # a hard 60 chars discarded most of it while the builder allowed 220.
        try:
            import config

            limit = int(getattr(config, "INDEX_TITLE_CHARS", 0) or self.MAX_TITLE_CHARS)
        except Exception:  # pragma: no cover - config is always importable in practice
            limit = self.MAX_TITLE_CHARS
        if len(self.title) <= limit:
            return self.title
        return self.title[: max(1, limit - 1)] + "…"

    def render(self, preview: int = 1) -> str:
        """
        One line in the prompt: the title, plus (optionally) a short hint of content.

        ``preview=0`` -> bare title (cheapest).  ``preview=1`` -> one *fact snippet*
        (~10 tokens), not a full summary line (~100 tokens).  The hierarchy's whole
        point is that detail is fetched on demand, so the prompt must stay cheap.
        """
        head = f"- {self.index_id} [INDEX {self.time_label()}, {self.size} entries"
        # The group count is rendered from ``child_index_ids`` rather than baked into
        # the title: prefixing each fold put "[2 组] [2 组] [2 组]" in front of the
        # digest, pushing the actual facts out of the truncated title.
        if self.child_index_ids:
            head += f", {len(self.child_index_ids)} groups"
        head += f"] {self.capped_title()}"
        snippet_source = self.previews or self.member_summaries
        if preview > 0 and snippet_source:
            body = "\n".join(f"    · {line}" for line in snippet_source[:preview])
            more = "" if self.size <= preview else f"  (+{self.size - preview} more)"
            return f"{head}\n{body}{more}"
        return head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArchivedSummary:
    """Layer 2: permanent, read by exact ``summary_id`` only."""

    summary_id: str
    text: str
    override_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=now_ts)
    raw_ref_id: str = ""
    is_overridden: bool = False
    # --- bookkeeping ------------------------------------------------------- #
    episode_id: str = ""
    seq: int = -1
    origin: str = "event"
    superseded_by: Optional[str] = None   # id of the summary that replaced this one
    archive_reason: str = "overridden"    # "overridden" | "capacity"
    raw_ref_ids: List[str] = field(default_factory=list)
    merged_from: List[str] = field(default_factory=list)
    fact_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.raw_ref_ids and self.raw_ref_id:
            self.raw_ref_ids = [self.raw_ref_id]
        if self.raw_ref_ids and not self.raw_ref_id:
            self.raw_ref_id = self.raw_ref_ids[0]

    @property
    def overrides_tag(self) -> str:
        return f"[OVERRIDES: {','.join(self.override_ids) if self.override_ids else 'none'}]"

    def render(self) -> str:
        status = "OVERRIDDEN" if self.is_overridden else "ARCHIVED"
        superseded = f" superseded_by={self.superseded_by}" if self.superseded_by else ""
        return (
            f"[{status}] {self.summary_id} {self.overrides_tag}{superseded} "
            f"(raw_ref: {','.join(self.raw_ref_ids) or 'none'}, reason={self.archive_reason}) "
            f"{self.text}"
        )

    @classmethod
    def from_active(
        cls,
        active: ActiveSummary,
        *,
        is_overridden: bool,
        superseded_by: Optional[str] = None,
        archive_reason: str = "overridden",
        merged_from: Optional[List[str]] = None,
    ) -> "ArchivedSummary":
        return cls(
            summary_id=active.summary_id,
            text=active.text,
            override_ids=list(active.override_ids),
            timestamp=active.timestamp,
            raw_ref_id=active.raw_ref_id,
            is_overridden=is_overridden,
            episode_id=active.episode_id,
            seq=active.seq,
            origin=active.origin,
            superseded_by=superseded_by,
            archive_reason=archive_reason,
            raw_ref_ids=list(active.raw_ref_ids),
            # ``merged_from`` (when given) records that this archived entry was a
            # capacity merge of those summaries.  Keeping it lets an id lookup walk
            # back to the merge inputs instead of returning a mixed fact bundle.
            merged_from=list(merged_from if merged_from is not None else active.merged_from),
            fact_keys=list(active.fact_keys),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Tool-call plumbing (text-parsed function calling)
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A parsed textual tool call emitted by the agent."""

    name: str
    args: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    def render(self) -> str:
        inner = ", ".join(f'{k}="{v}"' for k, v in self.args.items())
        return f"{self.name}({inner})"


@dataclass
class ToolResult:
    """Result of executing one :class:`ToolCall`."""

    ok: bool
    content: str
    name: str = ""
    error: str = ""
    parsed: bool = True           # False when the call text itself was malformed
    resolved: bool = False        # True when the requested id existed
    latency_ms: float = 0.0
    # --- retrieval trace (drives the inspection UI, harmless otherwise) ----- #
    #: One-line description of how the result was obtained, e.g.
    #: ``slot match -> 3 candidates -> supersede-chain walk -> s007@ab12``.
    path: str = ""
    #: Candidate list the tool evaluated, each with value / rank / outcome.
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    #: Extra structured facts for the UI (resolved id, walked chain, ...).
    details: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        if self.ok:
            return f"[TOOL RESULT: {self.name}] {self.content}"
        kind = "PARSE ERROR" if not self.parsed else "ERROR"
        return f"[TOOL RESULT: {self.name} {kind}] {self.content or self.error}"


# --------------------------------------------------------------------------- #
# Per-turn statistics
# --------------------------------------------------------------------------- #
@dataclass
class MemoryTurnStats:
    """Everything an experiment needs to know about what one turn did."""

    episode_id: str = ""
    turn_index: int = -1
    new_summary_id: str = ""
    overridden_ids: List[str] = field(default_factory=list)
    invalid_override_ids: List[str] = field(default_factory=list)
    capacity_merged_ids: List[str] = field(default_factory=list)
    index_ids: List[str] = field(default_factory=list)
    super_index_ids: List[str] = field(default_factory=list)
    fact_safety_fallbacks: int = 0
    index_rejections: int = 0
    #: index entries rewritten because a member left the active chain
    index_updates: int = 0
    lazy_moved: int = 0
    lazy_summary_count: int = 0
    index_count: int = 0
    filed_summary_count: int = 0
    active_chain_size: int = 0
    active_chain_tokens: int = 0
    active_chain_text_tokens: int = 0
    #: cost of the derived current-value registry rendered at the top of layer 1
    current_values_tokens: int = 0
    window_size: int = 0
    archived_total: int = 0
    raw_total: int = 0
    event_triggered: bool = False
    capacity_triggered: bool = False
    summarizer_raw_output: str = ""
    format_retry_used: bool = False   # summariser omitted the OVERRIDES tag once
    summary_truncated: bool = False    # summary body hit the hard token ceiling
    #: True when the turn was recorded (raw + window) but the paid summariser call
    #: was skipped by the content gate because the turn carried no fact update.
    summariser_skipped: bool = False
    #: True when this turn's summary came from the answering call itself
    #: (``<MEMORY_UPDATE>`` block) instead of a dedicated summariser call.
    self_written: bool = False
    self_write_fallback: bool = False  # block missing/unusable -> summariser called

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
