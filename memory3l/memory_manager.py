"""
``MemoryManager`` -- the storage-agnostic core.

Responsibilities
----------------
1. Keep layer 3 (raw dialogue, permanent) and the recent sliding window.
2. Generate one summary per turn (LLM) with a mandatory
   ``[OVERRIDES: id_list|none]`` tag.
3. **Event-driven update**: a new fact that conflicts with / updates an active
   fact overrides the old summary.  The old summary leaves the active chain and
   is archived with ``is_overridden=True`` and a forward ``superseded_by``
   pointer; the new summary stays active and lists the ids it replaced.
4. **Capacity-driven compression**: when the active chain exceeds
   ``ACTIVE_CHAIN_TOKEN_LIMIT`` tokens, merge the oldest *conflict-carrying-free*
   summaries into one higher-order summary and move the originals to the archive.
   Capacity merges emit **no** OVERRIDES tag (by design).

Everything here is expressed in terms of :class:`BaseMemoryStore`, so the exact
same logic runs on ``InMemoryStore`` and ``RedisSQLiteHybridStore``; only the
store is swapped by the experiment.

Robustness rules that protect the metrics
-----------------------------------------
* A hallucinated override id (not present in the active chain) is dropped and
  counted in ``invalid_override_ids`` -- it can never corrupt the archive graph.
* When a turn overrides summaries, no capacity compression runs in the same
  turn; the two mechanisms stay independently attributable in the CSV.
* If the summariser call fails, the turn is still recorded (raw + window) and the
  failure is reported in the stats; the episode does not crash.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import config

from .llm import BaseLLM, LLMError
from .models import (
    ActiveSummary,
    ArchivedSummary,
    IndexEntry,
    LAZY_INDEX_ID,
    MemoryTurnStats,
    RawDialogRecord,
    make_id,
    now_ts,
)
from .prompts import (
    index_title_system_prompt,
    index_title_user_prompt,
    merge_system_prompt,
    merge_user_prompt,
    summarizer_system_prompt,
    summarizer_user_prompt,
)
from .store.base import BaseMemoryStore
from .token_utils import estimate_tokens, extract_fact_keys, truncate_to_tokens
from .tools import SummaryParseError, parse_summary_response

logger = logging.getLogger(__name__)


def _merge_origin_lineage(summary: ActiveSummary) -> List[str]:
    """
    Lineage list for an archived merge: if the summary being archived was itself a
    merge of earlier summaries, record the *original* summary ids, not just the
    intermediate merge id.  Otherwise an id lookup would end at a node whose
    ``merged_from`` points at a summary that no longer exists.
    """
    if not summary.merged_from:
        return []
    lineage: List[str] = []
    for parent in summary.merged_from:
        lineage.append(parent)
    return lineage


_SNIPPET_STRIP = re.compile(r"\[(?:FACTS|OVERRIDES)[^\]]*\]", re.IGNORECASE)


class ManagerFactDigest:
    """Merge ``属性=值[历史]`` fragments coming from several index titles."""

    _PAIR = re.compile(r"([^=;｜|]+?)\s*=\s*([^[;｜|]+?)(?:\[([^\]]*)\])?(?=;|｜|\||$)")

    @classmethod
    def merge(cls, titles: Sequence[str], max_attrs: int = 8) -> str:
        values: "Dict[str, List[str]]" = {}
        order: List[str] = []
        for title in titles:
            for attr, latest, history in cls._PAIR.findall(title or ""):
                attr, latest = attr.strip(), latest.strip()
                if not attr or not latest:
                    continue
                seq = values.setdefault(attr, [])
                if attr not in order:
                    order.append(attr)
                for value in [h.strip() for h in (history or "").split("→") if h.strip()] + [latest]:
                    if value not in seq:
                        seq.append(value)
        parts = []
        for attr in order[:max_attrs]:
            seq = values[attr]
            history = "→".join(seq[:-1][-3:])
            parts.append(f"{attr}={seq[-1]}" + (f"[{history}→{seq[-1]}]" if history else ""))
        return "; ".join(parts)


def _preview_snippet(text: str, limit: int = 26) -> str:
    """
    A ~10-token hint of a summary, for index previews.

    Deliberately *not* the full summary line: the whole value of the hierarchy is
    that detail is pulled on demand, so a preview must never cost what the thing it
    previews costs.
    """
    cleaned = _SNIPPET_STRIP.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit] + ("…" if len(cleaned) > limit else "")


class MemoryManager:
    """Main class.  Storage-agnostic; depends only on ``BaseMemoryStore``."""

    def __init__(
        self,
        store: BaseMemoryStore,
        summarizer: Optional[BaseLLM] = None,
        *,
        episode_id: Optional[str] = None,
        recent_window_turns: int = None,
        active_chain_token_limit: int = None,
        capacity_min_merge: int = None,
        capacity_max_merge: int = None,
        max_summary_tokens: int = 120,
        chain_strategy: Optional[str] = None,
        index_keep_recent: Optional[int] = None,
        index_group_size: Optional[int] = None,
        reset_on_bind: bool = True,
        resume: bool = False,
        flush_hot_keys: bool = False,
    ):
        self.store = store
        self.summarizer = summarizer
        # Window size: an explicit argument wins, then the store's configured
        # value, then the global default.  This keeps `MemoryManager(store)`
        # consistent with how the store was constructed.
        if recent_window_turns is None:
            recent_window_turns = getattr(store, "recent_window_turns", None)
        if recent_window_turns is None:
            recent_window_turns = config.RECENT_WINDOW_TURNS
        self.recent_window_turns = recent_window_turns
        self.active_chain_token_limit = (
            config.ACTIVE_CHAIN_TOKEN_LIMIT
            if active_chain_token_limit is None
            else active_chain_token_limit
        )
        self.capacity_min_merge = config.CAPACITY_MIN_MERGE if capacity_min_merge is None else capacity_min_merge
        self.capacity_max_merge = config.CAPACITY_MAX_MERGE if capacity_max_merge is None else capacity_max_merge
        self.max_summary_tokens = max_summary_tokens
        self.episode_id = episode_id or "episode_default"
        self.turn_index = -1
        self._seq = 0
        self.stats: List[MemoryTurnStats] = []
        self.compressions = 0
        self.indexes_built = 0
        self.super_indexes = 0
        self.chain_strategy = (
            chain_strategy or getattr(config, "CHAIN_STRATEGY", "index") or "index"
        ).lower()
        self.index_keep_recent = (
            config.INDEX_KEEP_RECENT if index_keep_recent is None else index_keep_recent
        )
        self.index_group_size = (
            config.INDEX_GROUP_SIZE if index_group_size is None else index_group_size
        )
        self.index_min_visible = max(1, config.INDEX_MIN_VISIBLE)
        self.index_max_entries = max(2, config.INDEX_MAX_ENTRIES)
        self.index_max_per_turn = max(1, config.INDEX_MAX_PER_TURN)
        self.index_render_recent = max(1, config.INDEX_RENDER_RECENT)
        self.lazy_enabled = bool(config.LAZY_MODE) and self.chain_strategy == "index"
        #: run the O(n) store-vs-manager consistency check on every turn (debug aid)
        self.strict_integrity = bool(getattr(config, "STRICT_INTEGRITY", False))
        #: adaptive preview switch: turned off when previews would keep layer 1 over
        #: budget, restored once there is room again.
        self._previews_on = True
        self.overrides = 0
        self.summarizer_failures = 0
        self.summary_truncations = 0
        self.capacity_merges_rejected = 0
        #: batch-ingestion window.  1 (the default) is the only setting under which
        #: a turn is summarised against the chain produced by *every* earlier turn;
        #: >1 trades that guarantee for wall clock -- see ``add_dialog_turns``.
        self.ingest_concurrency = 1
        # The manager is the authority on the window size: make sure the store
        # enforces exactly the same bound (they are configured independently).
        self.store.recent_window_turns = self.recent_window_turns
        self._bind(self.episode_id, reset=reset_on_bind, resume=resume, flush_hot_keys=flush_hot_keys)

    # ------------------------------------------------------------------ #
    # episode lifecycle
    # ------------------------------------------------------------------ #
    def _bind(self, episode_id: str, reset: bool, resume: bool = False, flush_hot_keys: bool = False) -> None:
        bind = getattr(self.store, "bind_episode")
        try:
            bind(
                episode_id,
                recent_window_turns=self.recent_window_turns,
                reset=reset,
                resume=resume,
                flush_hot_keys=flush_hot_keys,
            )
        except TypeError:
            # InMemoryStore.bind_episode has no `resume` parameter.
            bind(
                episode_id,
                recent_window_turns=self.recent_window_turns,
                reset=reset,
                flush_hot_keys=flush_hot_keys,
            )
            if resume:
                self.store._current_turn_index = self.store.count_raw_records(episode_id) - 1
        # bind_episode may accept window overrides; the manager's value always wins.
        self.store.recent_window_turns = self.recent_window_turns

    def reset_episode(self, episode_id: str, *, resume: bool = False, purge: bool = False) -> None:
        """
        Start a fresh episode.

        ``purge=False`` (default, used by batch evaluation) clears only the *hot*
        memory: the archive and raw store of that episode stay, because a dataset
        episode may still need to look up its own overridden history.

        ``purge=True`` deletes the episode's archived summaries and raw records as
        well, so the session is genuinely empty.  The interactive inspector uses
        this for "new session"; keeping it opt-in means batch runs cannot
        accidentally lose their permanent data.
        """
        self._bind(episode_id, reset=True, resume=resume)
        if purge:
            self._purge_episode(episode_id)
        self.episode_id = episode_id
        self.turn_index = -1
        self._seq = 0
        if resume:
            # Continue numbering after whatever is already in the archive/mirror.
            for summary in self.store.list_active_summaries(episode_id):
                self._seq = max(self._seq, summary.seq + 1)
            for archived in self.store.list_archived_summaries(episode_id):
                self._seq = max(self._seq, archived.seq + 1)
            self.turn_index = self.store.count_raw_records(episode_id) - 1
            # Keep the store's own counter in sync so the next turn is numbered
            # correctly (and identically in the raw record and the window copy).
            self.store.set_turn_index(self.turn_index)
        self.stats = []
        self.compressions = 0
        self.indexes_built = 0
        self.super_indexes = 0
        # Counters reset per episode; the strategy configured at construction time
        # stays in force (reset_episode takes no strategy arguments).
        self.overrides = 0
        self.summarizer_failures = 0
        self.summary_truncations = 0
        self.capacity_merges_rejected = 0
        logger.debug(
            "episode %s reset (resume=%s): hot state cleared, next turn=%d",
            episode_id, resume, self.turn_index + 1,
        )

    def _purge_episode(self, episode_id: str) -> None:
        """Hard-delete this episode's permanent layers (opt-in, never in batch)."""
        purge = getattr(self.store, "purge_episode", None)
        if callable(purge):
            purge(episode_id)
        logger.info("episode %s purged: archive + raw store cleared", episode_id)

    def start_episode(self, episode_id: str, *, resume: bool = False, purge: bool = False) -> None:
        """Alias kept for readability at call sites in the baselines."""
        self.reset_episode(episode_id, resume=resume, purge=purge)

    # ------------------------------------------------------------------ #
    # layer accessors (used by prompts, tools and metrics)
    # ------------------------------------------------------------------ #
    def get_window(self) -> List[RawDialogRecord]:
        return self.store.get_window(self.episode_id)

    def list_active_summaries(self) -> List[ActiveSummary]:
        """Everything still in layer 1 -- filed under an index or not."""
        return self.store.list_active_summaries(self.episode_id)

    def chain_summaries(self) -> List[ActiveSummary]:
        """
        The summaries actually rendered in the prompt.

        A summary renders only when it is not referenced by any title.  Being
        referenced is what makes it safe to stop rendering: the title above it is
        the navigation handle, and its own id still resolves on demand via
        ``get_archived_summary`` / ``expand_index``.

        Whether a referenced summary also sits in the lazy store is therefore an
        implementation detail -- both cases render nothing.
        """
        title_members: set = set()
        for entry in self.list_index_entries():
            title_members.update(entry.members)
        return [
            s for s in self.store.list_active_summaries(self.episode_id)
            if s.summary_id not in title_members
        ]

    def lazy_summaries(self) -> List[ActiveSummary]:
        """Kept by id, not rendered: reachable via exact-id lookup."""
        return [s for s in self.store.list_active_summaries(self.episode_id)
                if s.index_id == LAZY_INDEX_ID]

    def reachable_summaries(self) -> List[ActiveSummary]:
        """
        Everything a title or the chain can still lead to -- the honest denominator
        for "did we lose anything?".  Rendered + filed + lazy must always equal all
        live summaries.
        """
        return self.list_active_summaries()

    def list_index_entries(self) -> List[IndexEntry]:
        return self.store.list_index_entries(self.episode_id)

    def rendered_index_entries(self):
        """
        Index entries paired with the preview depth they are rendered at.

        Preview budget is *adaptive*: only the newest few entries carry a snippet,
        and if the index layer is large enough to threaten the budget, previews are
        dropped entirely so the agent has to expand a title it actually needs.
        """
        entries = self.list_index_entries()
        if not entries:
            return []
        recent = max(0, config.INDEX_PREVIEW_RECENT)
        # Preview depth is a manager-level decision (``_previews_on``) updated by the
        # budget loop.  Deciding it here from the current cost would recurse, because
        # the cost function calls this method.
        if len(entries) > self.index_max_entries:
            recent = min(recent, 1)
        if not self._previews_on:
            recent = 0
        out = []
        for offset, entry in enumerate(reversed(entries)):
            out.append((entry, config.INDEX_PREVIEW if offset < recent else 0))
        return list(reversed(out))

    def expand_index(self, index_id: str) -> List[ActiveSummary]:
        return self.store.summaries_under_index(index_id, self.episode_id)

    def list_archived_summaries(self, limit: Optional[int] = None) -> List[ArchivedSummary]:
        return self.store.list_archived_summaries(self.episode_id, limit=limit)

    def get_archived_summary(self, summary_id: str) -> Optional[ArchivedSummary]:
        return self.store.get_archived_summary(summary_id, episode_id=None)

    def get_raw_record(self, reference_id: str) -> Optional[RawDialogRecord]:
        return self.store.get_raw_record(reference_id, episode_id=self.episode_id)

    def active_chain_tokens(self) -> int:
        """
        Tokens actually rendered into the prompt for layer 1: the index titles plus
        the unfiled summaries.  This is the quantity the budget is measured on, so
        filing a group under an index must make it go *down*.
        """
        return self.rendered_layer1_tokens()

    def rendered_layer1_tokens(self) -> int:
        """Cost of everything layer 1 contributes to the prompt."""
        chain_cost = sum(estimate_tokens(s.render()) for s in self.chain_summaries())
        index_cost = sum(
            estimate_tokens(e.render(preview=depth)) for e, depth in self.rendered_index_entries()
        )
        return chain_cost + index_cost

    def active_chain_text_tokens(self) -> int:
        """Tokens of the rendered chain bodies only (the reported cost figure)."""
        return sum(estimate_tokens(s.text) for s in self.chain_summaries()) + sum(
            estimate_tokens(e.title) for e in self.list_index_entries()
        )

    def all_summary_tokens(self) -> int:
        """Bodies of every live summary, filed or not -- the archive-free upper bound."""
        return sum(estimate_tokens(s.text) for s in self.list_active_summaries())

    def all_rendered_tokens(self) -> int:
        """
        What layer 1 would cost if *every* summary were still rendered flat.

        This is the honest baseline for "did the hierarchy actually save anything":
        comparing against it isolates the effect of filing from the per-line
        formatting overhead (ids, tags, raw_refs).
        """
        return sum(estimate_tokens(s.render()) for s in self.list_active_summaries())

    def build_prompt_context(self) -> Dict[str, str]:
        """The two memory blocks as they will be rendered into the prompt."""
        from .prompts import render_active_chain, render_recent_window

        return {
            "recent_window": render_recent_window(self.get_window()),
            "active_chain": render_active_chain(self.list_active_summaries()),
        }

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def add_dialog_turn(self, user_input: str, agent_output: str) -> MemoryTurnStats:
        """
        Record one completed turn and maintain all three layers.

        Deliberately implemented on top of the same two primitives the batch path
        uses (:meth:`_generate_only` + :meth:`_apply_generated`).  A first version kept
        a separate summariser call here, and the two paths diverged -- the single-turn
        path silently kept rendering long, uncopyable ids, so overriding worked under
        one entry point and not the other.
        """
        record, stats = self._prepare_turn(user_input, agent_output)
        gen = self._generate_only(record)
        self._apply_generated(gen, stats)
        self._after_turn(stats)
        return stats

    # ------------------------------------------------------------------ #
    # self-written memory: one call answers *and* records the turn
    # ------------------------------------------------------------------ #
    def harvest_selfwritten(self, block: str) -> Optional[Dict[str, Any]]:
        """
        Validate a ``<MEMORY_UPDATE>`` block against the *current* chain.

        Returns ``None`` when the block is absent or unusable, in which case the
        caller must fall back to the dedicated summariser -- an unparseable block
        must cost an extra call, never a lost memory update.
        """
        if not block:
            return None
        try:
            parsed = parse_summary_response(block, valid_ids=None)
        except SummaryParseError as exc:
            logger.warning("self-written memory block unparsable: %s", exc)
            return None
        if not parsed.get("valid"):
            return None
        chain = self._chain_for_summariser()
        # ``_finalise_generated`` needs a record only to echo it back; the real one
        # is substituted once the turn has been prepared for storage.
        return self._finalise_generated(None, chain, block, parsed)

    def add_dialog_turn_selfwritten(
        self, user_input: str, agent_output: str, block: str
    ) -> MemoryTurnStats:
        """
        Record one turn whose summary the answering LLM wrote itself.

        This is the one-call path: the reply that answered the user also carries
        the summary, so no second LLM call is issued.  Everything downstream
        (override resolution, archiving, index maintenance, token accounting) is
        the *same* code as the two-call path -- only the summary's origin differs.
        """
        gen = self.harvest_selfwritten(block)
        if gen is None or self.summarizer is None:
            stats = self.add_dialog_turn(user_input, agent_output)
            stats.self_write_fallback = True
            return stats
        # Prepare *after* validation: a rejected block must not leave a stray raw
        # record / window entry behind (the fallback would then double-write it).
        record, stats = self._prepare_turn(user_input, agent_output)
        gen["record"] = record
        self._apply_generated(gen, stats)
        # The block was validated before we touched the store, so the summary is
        # stored either way; ``new_summary`` is None only if the apply step failed.
        stats.self_written = bool(gen.get("text"))
        self._after_turn(stats)
        return stats

    # ------------------------------------------------------------------ #
    # short-id rendering (what the summariser is allowed to quote)
    # ------------------------------------------------------------------ #
    # A summariser must copy the id of every summary it overrides, verbatim.  The
    # canonical id (``episode/s001@ab12cd``) is long and models truncate it to the
    # trailing hash, which fails exact-id validation and silently disables
    # overriding.  Short ids (``s001``) are copyable and are mapped back before
    # anything is stored.  The ``[FACTS: attribute=value]`` part must stay visible:
    # override detection is "same attribute, new value", so the model needs both.
    @staticmethod
    def _slot_of(entry: str) -> str:
        return entry.split("=", 1)[0].strip().lower()

    @staticmethod
    def short_id(summary_id: str) -> str:
        tail = (summary_id or "").split("/")[-1]
        return tail.split("@")[0] or (summary_id or "")

    def _chain_rendered_short(self, chain: Sequence[ActiveSummary]) -> str:
        lines = []
        for item in chain:
            keys = list(item.fact_keys) or sorted(extract_fact_keys(item.text))
            facts = f" [FACTS: {'; '.join(keys)}]" if keys else ""
            lines.append(
                f"- {self.short_id(item.summary_id)} [OVERRIDES: "
                f"{','.join(self.short_id(o) for o in item.override_ids) or 'none'}]"
                f"{facts} (raw_ref: {self.short_id(item.raw_ref_id)}) {item.text}"
            )
        return "\n".join(lines)

    def _generate_only(self, record: RawDialogRecord) -> Dict[str, Any]:
        """
        Summariser half of a turn: produce text + override ids, change nothing.

        Separated from :meth:`_apply_generated` so the expensive LLM call can be
        issued concurrently for many turns while the cheap, order-sensitive chain
        bookkeeping stays strictly sequential.  Semantic equivalence between the
        concurrent and sequential paths is a tested property.
        """
        chain = self._chain_for_summariser()
        new_turn = f"user: {record.user_msg}\nagent: {record.agent_msg}"
        if self.summarizer is None:
            return {"record": record, "chain_ids": {x.summary_id for x in chain},
                    "text": new_turn, "override_ids": [], "invalid": [],
                    "raw": "", "ok": True}
        messages = [
            {"role": "system", "content": summarizer_system_prompt(self.max_summary_tokens)},
            {"role": "user", "content": summarizer_user_prompt(
                new_turn, chain, turn_index=record.turn_index,
                render=self._chain_rendered_short(chain))},
        ]
        attempts = 1 + (1 if config.SUMMARIZER_FORMAT_RETRY else 0)
        parsed = None
        raw_output = ""
        for attempt in range(attempts):
            try:
                raw_output = self.summarizer.generate(messages, json_mode=False).text
            except LLMError as exc:
                logger.warning("summariser failed on turn %d: %s", record.turn_index, exc)
                return {"record": record, "chain_ids": {x.summary_id for x in chain},
                        "text": "", "override_ids": [], "invalid": [],
                        "raw": f"<ERROR> {exc}", "ok": False}
            try:
                # No id whitelist here: the model quotes SHORT ids and the mapping
                # back to canonical ids happens below.  Filtering here would reject
                # every valid short id.
                parsed = parse_summary_response(raw_output, valid_ids=None)
            except SummaryParseError as exc:
                logger.warning("unparsable summary on turn %d: %s", record.turn_index, exc)
                return {"record": record, "chain_ids": {x.summary_id for x in chain},
                        "text": "", "override_ids": [], "invalid": [],
                        "raw": raw_output, "ok": False}
            if not parsed["missing_tag"] or attempt == attempts - 1:
                break
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": (
                    "Your reply did not contain the mandatory final line. Reply again with ONLY "
                    "the summary followed by a final line [OVERRIDES: none] or [OVERRIDES: id1,id2] "
                    "using ids that literally appear in <ACTIVE_CHAIN>.")},
            ]
        if not parsed.get("valid"):
            return {"record": record, "chain_ids": {x.summary_id for x in chain},
                    "text": "", "override_ids": [], "invalid": parsed.get("invalid_override_ids", []),
                    "raw": raw_output, "ok": False}
        return self._finalise_generated(record, chain, raw_output, parsed)

    def _finalise_generated(
        self, record: RawDialogRecord, chain: Sequence[ActiveSummary],
        raw_output: str, parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Shared tail of both summary sources (dedicated summariser / self-write).

        Maps the ids the model quoted back to canonical ids.  Both forms are
        accepted: a short id (the normal case) or a full id (some models expand it).
        Keeping this in one place is what makes the two sources interchangeable --
        the override graph must not care where a summary came from.
        """
        chain_ids = {x.summary_id for x in chain}
        by_short = {self.short_id(x.summary_id): x.summary_id for x in chain}
        mapped, invalid = [], []
        for oid in parsed["override_ids"]:
            if oid in by_short:
                mapped.append(by_short[oid])
            elif oid in chain_ids:
                mapped.append(oid)
            elif oid.split("@")[0].split("/")[-1] in by_short:
                mapped.append(by_short[oid.split("@")[0].split("/")[-1]])
            else:
                invalid.append(oid)
        return {
            "record": record,
            "chain_ids": chain_ids,
            # bounded here, at generation time, so a runaway summary can never
            # reach the next prompt
            "text": self._bound_summary_text(parsed["summary_text"]),
            "override_ids": mapped,
            "invalid": list(parsed.get("invalid_override_ids", [])) + invalid,
            "fact_keys": list(parsed.get("fact_keys") or []),
            "raw": raw_output,
            "ok": True,
        }

    def _apply_generated(
        self, gen: Dict[str, Any], stats: MemoryTurnStats, seq: Optional[int] = None
    ) -> Optional[ActiveSummary]:
        """Chain half of a turn: resolve overrides, archive, maintain the chain."""
        record = gen["record"]
        stats.turn_index = record.turn_index
        stats.summarizer_raw_output = gen.get("raw", "")
        if not gen.get("ok"):
            self.summarizer_failures += 1
            return None
        chain_ids = gen["chain_ids"]
        override_ids = [i for i in gen["override_ids"] if i in chain_ids]
        stats.invalid_override_ids = list(gen.get("invalid") or [])

        assigned = self._next_seq() if seq is None else seq
        new_summary = ActiveSummary(
            summary_id=make_id(self.episode_id, "s", assigned, salt=gen["text"]),
            text=gen["text"],
            override_ids=override_ids,
            timestamp=now_ts(),
            raw_ref_id=record.reference_id,
            episode_id=self.episode_id,
            seq=assigned,
            origin="event",
            raw_ref_ids=[record.reference_id],
            fact_keys=gen.get("fact_keys") or sorted(extract_fact_keys(gen["text"])),
        )
        self.store.add_active_summary(new_summary, episode_id=self.episode_id)
        archived_ids: List[str] = []
        for overridden_id in override_ids:
            old = self.store.remove_active_summary(overridden_id, episode_id=self.episode_id)
            if old is None:
                continue
            self.store.add_archived_summary(
                ArchivedSummary.from_active(
                    old, is_overridden=True, superseded_by=new_summary.summary_id,
                    archive_reason="overridden",
                ),
                episode_id=self.episode_id,
            )
            self.overrides += 1
            archived_ids.append(overridden_id)
        # Bookkeeping lives here, not in the callers: the sequential, batch and
        # self-write paths all funnel through this method, and when the batch path
        # kept its own copy the two metrics disagreed within a single run
        # (``overrides_events`` read 0 while ``overridden_summaries`` read 1).
        # Counting *archived* ids (not requested ones) keeps the two consistent.
        stats.new_summary_id = new_summary.summary_id
        stats.overridden_ids = archived_ids
        stats.event_triggered = bool(archived_ids)
        return new_summary

    def add_dialog_turns(self, turns: Sequence[tuple]) -> List[MemoryTurnStats]:
        """
        Ingest several turns; the manager decides between sequential and windowed
        concurrent summarisation.

        **Ordering contract**: with ``ingest_concurrency == 1`` (the default) a
        turn's summary is generated against the chain that already contains every
        *earlier* turn of this episode -- exactly the sequential semantics.  A first
        concurrent version wrote all raw records first and then generated every
        summary in parallel, leaving each prompt with an empty ``<ACTIVE_CHAIN>``;
        the model had nothing to compare against and the override mechanism
        silently never fired.

        **Concurrency > 1 is an approximation, not an equivalent speedup.**  The
        window generates turns ``[i, i+W)`` in parallel against one shared chain
        snapshot, so a turn cannot override another turn *inside its own window*.
        The result is therefore close to, but not identical with, sequential mode:
        overrides whose target was created in the same window are lost.  Window 1
        is the only semantics-preserving setting; the flag is kept for wall-clock
        experiments that accept the difference.
        """
        concurrency = self.ingest_concurrency
        if concurrency <= 1 or self.summarizer is None:
            return [self.add_dialog_turn(u, a) for u, a in turns]
        logger.warning(
            "batch ingestion with concurrency=%d: turns inside one window are "
            "summarised against a shared chain snapshot, so overrides within a "
            "window cannot be detected (results are NOT identical to sequential)",
            concurrency,
        )

        turns = list(turns)
        window = max(2, int(concurrency))
        # Future reservations keep summary ids assigned in turn order even though
        # generation happens out of order inside a window.
        reservations = {index: self.reserve_seq() for index in range(len(turns))}
        results: List[MemoryTurnStats] = []
        for start in range(0, len(turns), window):
            chunk = turns[start : start + window]
            records: List[RawDialogRecord] = []
            stats_list: List[MemoryTurnStats] = []
            for offset, (user_input, agent_output) in enumerate(chunk):
                index = start + offset
                record, stats = self._prepare_turn(user_input, agent_output)
                records.append(record)
                stats_list.append(stats)
                # Publish this turn's raw record + window entry before generating, so
                # the window reflects the dialogue so far (raw records never feed the
                # override decision, so this is safe).
            import concurrent.futures as _cf

            with _cf.ThreadPoolExecutor(max_workers=min(window, len(records))) as pool:
                generated = list(pool.map(self._generate_only, records))
            for offset, (gen, stats) in enumerate(zip(generated, stats_list)):
                self._apply_generated(gen, stats, seq=reservations[start + offset])
                self._after_turn(stats)
                results.append(stats)
        return results

    def reserve_seq(self) -> int:
        """Pre-assign a sequence number so ids follow turn order under concurrency."""
        self._next_seq()
        return self._seq

    def _prepare_turn(self, user_input: str, agent_output: str):
        """Write layer 3 + the sliding window for one turn; return record + stats."""
        self.turn_index = self.store.next_turn_index()
        raw_ref = make_id(self.episode_id, "raw", self.turn_index, salt=f"{user_input}|{agent_output}")
        record = RawDialogRecord(
            reference_id=raw_ref, user_msg=user_input, agent_msg=agent_output,
            timestamp=now_ts(), episode_id=self.episode_id, turn_index=self.turn_index,
        )
        self.store.add_raw_record(record, episode_id=self.episode_id)
        window = self.store.append_window_record(record, episode_id=self.episode_id)
        stats = MemoryTurnStats(episode_id=self.episode_id, turn_index=self.turn_index,
                                window_size=len(window))
        return record, stats

    def _after_turn(self, stats: MemoryTurnStats) -> None:
        """Chain maintenance for one applied turn (budget loop + counters)."""
        if self.chain_strategy == "merge":
            stats.capacity_triggered = self._maybe_capacity_compress(stats)
        else:
            stats.capacity_triggered = self._maybe_build_index(stats)
        self._fill_counts(stats)
        self.stats.append(stats)

    def _chain_for_summariser(self) -> List[ActiveSummary]:
        """
        The chain shown to the summariser, capped at ``SUMMARIZER_MAX_INPUT_TOKENS``.

        Defence in depth: ``_bound_summary_text`` caps what goes *into* storage, and
        this caps what gets sent even if a stored summary is unexpectedly large.  The
        newest entries are kept whole because they carry the current values.
        """
        chain = self.store.list_active_summaries(self.episode_id)
        budget = config.SUMMARIZER_MAX_INPUT_TOKENS
        costs = [estimate_tokens(x.render()) for x in chain]
        total = sum(costs)
        if total <= budget:
            return chain
        kept: List[ActiveSummary] = []
        remaining = budget
        for summary, cost in zip(reversed(chain), reversed(costs)):
            if kept and cost > remaining:
                break
            kept.append(summary)
            remaining -= cost
        # Say what is actually true: tokens, entries, and the worst single entry.
        logger.warning(
            "summariser input capped: %d entries / %d tokens (budget %d); "
            "showing the newest %d entries. Largest single entry: %d tokens.",
            len(chain), total, budget, len(kept), max(costs) if costs else 0,
        )
        return list(reversed(kept))

    def _summarise_turn(self, record: RawDialogRecord, stats: MemoryTurnStats) -> Optional[ActiveSummary]:
        chain = self._chain_for_summariser()
        new_turn = f"user: {record.user_msg}\nagent: {record.agent_msg}"

        if self.summarizer is None:
            # No LLM attached (pure storage test): store the turn verbatim as the
            # summary so downstream layers still hold a usable fact.
            summary_text, override_ids, raw_output = new_turn, [], ""
            parsed_fact_keys = []
        else:
            messages = [
                {"role": "system", "content": summarizer_system_prompt(self.max_summary_tokens)},
                {
                    "role": "user",
                    "content": summarizer_user_prompt(new_turn, chain, turn_index=stats.turn_index),
                },
            ]
            # Real models occasionally omit the mandatory tag.  One explicit
            # format reminder is issued; whether it was needed is recorded, so the
            # effect on the metrics stays auditable instead of hidden.
            attempts = 1 + (1 if config.SUMMARIZER_FORMAT_RETRY else 0)
            parsed = None
            for attempt in range(attempts):
                try:
                    response = self.summarizer.generate(messages, json_mode=False)
                    raw_output = response.text
                except LLMError as exc:
                    self.summarizer_failures += 1
                    logger.warning("summariser failed on turn %d: %s", stats.turn_index, exc)
                    stats.summarizer_raw_output = f"<ERROR> {exc}"
                    return None
                try:
                    parsed = parse_summary_response(
                        raw_output, valid_ids=[s.summary_id for s in chain]
                    )
                except SummaryParseError as exc:
                    self.summarizer_failures += 1
                    logger.warning("unparsable summary on turn %d: %s", stats.turn_index, exc)
                    stats.summarizer_raw_output = raw_output
                    return None
                stats.summarizer_raw_output = raw_output
                if not parsed["missing_tag"] or attempt == attempts - 1:
                    break
                stats.format_retry_used = True
                logger.debug(
                    "turn %d: summariser omitted the OVERRIDES tag; retrying with a reminder",
                    stats.turn_index,
                )
                messages = messages + [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "Your reply did not contain the mandatory final line. "
                            "Reply again with ONLY the summary followed by a final line "
                            "[OVERRIDES: none] or [OVERRIDES: id1,id2] using ids that literally "
                            "appear in <ACTIVE_CHAIN>."
                        ),
                    },
                ]
            summary_text = self._bound_summary_text(parsed["summary_text"], stats)
            override_ids = parsed["override_ids"]
            parsed_fact_keys = list(parsed.get("fact_keys") or [])
            stats.invalid_override_ids = parsed["invalid_override_ids"]
            if not parsed["valid"]:
                self.summarizer_failures += 1
                logger.warning("empty summary body on turn %d", stats.turn_index)
                return None

        new_summary = ActiveSummary(
            summary_id=make_id(self.episode_id, "s", self._next_seq(), salt=summary_text),
            text=summary_text,
            override_ids=override_ids,
            timestamp=now_ts(),
            raw_ref_id=record.reference_id,
            episode_id=self.episode_id,
            seq=self._seq,
            origin="event",
            raw_ref_ids=[record.reference_id],
            # Explicit keys from the [FACTS: ...] line; lexical fallback keeps the
            # fact-safety rule working when a model omits that line.
            fact_keys=parsed_fact_keys or sorted(extract_fact_keys(summary_text)),
        )
        self.store.add_active_summary(new_summary, episode_id=self.episode_id)

        # Override resolution always uses the FULL chain: capping is a display
        # concern only, never a correctness one.
        chain_ids = {s.summary_id for s in self.store.list_active_summaries(self.episode_id)}
        for overridden_id in override_ids:
            if overridden_id not in chain_ids:
                continue  # defensive: parse layer already filtered, keep it safe
            old = self.store.remove_active_summary(overridden_id, episode_id=self.episode_id)
            if old is None:
                continue
            self.store.add_archived_summary(
                ArchivedSummary.from_active(
                    old,
                    is_overridden=True,
                    superseded_by=new_summary.summary_id,
                    archive_reason="overridden",
                ),
                episode_id=self.episode_id,
            )
        return new_summary

    # ------------------------------------------------------------------ #
    # Hierarchy: build an index (title) level above the summaries
    # ------------------------------------------------------------------ #
    def _maybe_build_index(self, stats: MemoryTurnStats) -> bool:
        """
        Keep filing summaries under index entries until the rendered layer-1 cost
        fits the budget.

        Three things this version fixes over the first attempt:

        1. **It loops.** One filing used to happen per turn; if the protected
           "recent" tail alone exceeded the budget, the chain could never comply and
           the system kept building indexes on top of an over-budget chain.
        2. **It files past ``INDEX_KEEP_RECENT``** when necessary, keeping only
           ``INDEX_MIN_VISIBLE`` summaries unfiled.  Protection that ignores the
           budget is not protection, it is drift.
        3. **It folds old index entries into a super index** (title over titles)
           once there are more than ``INDEX_MAX_ENTRIES`` of them, so the index
           layer does not grow linearly with the episode either.
        """
        if self.active_chain_tokens() <= self.active_chain_token_limit:
            return False

        built: List[str] = []
        # Step 0: drop previews -- cheapest possible saving (they are hints, not content).
        if self._previews_on and len(self.list_index_entries()) > 2:
            self._previews_on = False

        # ------------------------------------------------------------------ #
        # ONE budget loop, with one set of hard stops.
        #
        # The previous version had a second, unguarded loop ("re-evaluate after
        # folding") that could spin indefinitely when filing kept succeeding without
        # ever bringing the chain under budget -- it reached hundreds of thousands of
        # iterations in one turn.  There is now exactly one loop and every stop below
        # is unconditional.
        # ------------------------------------------------------------------ #
        max_operations = max(2, self.index_max_per_turn * 3)
        operations = 0
        while self.active_chain_tokens() > self.active_chain_token_limit:
            operations += 1
            if operations > max_operations:
                logger.warning(
                    "hierarchy loop stopped at turn %d after %d operations "
                    "(rendered %d > budget %d)",
                    stats.turn_index, operations - 1,
                    self.active_chain_tokens(), self.active_chain_token_limit,
                )
                break

            # Progress is a *strict* triple: tokens must fall, live summaries must
            # fall, and the loop must be doing something new.  Any of these failing
            # means another iteration would only churn.
            before = (self.active_chain_tokens(), len(self.list_active_summaries()))

            if self._file_one_index_group(stats):
                built.append(stats.index_ids[-1])
            elif not self._maybe_fold_indexes(stats, force=True):
                # Nothing left to file and nothing left to fold.
                break

            after = (self.active_chain_tokens(), len(self.list_active_summaries()))
            if after[0] >= before[0] and after[1] >= before[1]:
                logger.warning(
                    "hierarchy made no progress at turn %d (tokens %d -> %d, summaries %d -> %d); stopping",
                    stats.turn_index, before[0], after[0], before[1], after[1],
                )
                break

        stats.index_ids = built
        # Step 2: lazy expansion.  Titles are cheap; the summaries that no title
        # points at are the remaining cost.  Keep the newest few rendered and move
        # the rest to id-only storage.  This is the step that actually lowers cost.
        if self.lazy_enabled:
            self._lazy_store_summaries(stats)
        if built:
            stats.capacity_triggered = True
            logger.info(
                "hierarchy at turn %d: %d new index entr%s, rendered chain %d tok (budget %d), "
                "%d unfiled / %d filed / %d lazy",
                stats.turn_index, len(built), "y" if len(built) == 1 else "ies",
                self.active_chain_tokens(), self.active_chain_token_limit,
                len(self.chain_summaries()),
                len(self.list_active_summaries()) - len(self.chain_summaries()),
                len(self.lazy_summaries()),
            )
        return bool(built)

    def _lazy_store_summaries(self, stats: MemoryTurnStats) -> int:
        """
        Move non-rendered summaries to the lazy store (id-only, still fully readable).

        Contract: a summary is either (a) rendered, (b) pointed at by a title, or
        (c) lazily stored by id.  Lazy storage never touches a title's members, so
        ``expand_index`` keeps returning the same set it returned before.
        """
        active = self.store.list_active_summaries(self.episode_id)
        newest_keep = {x.summary_id for x in active[-max(1, self.index_render_recent):]}
        entries = self.list_index_entries()
        # A title that has itself been folded into a super index is "covered": its
        # members are two hops from the rendered layer, so they can stop rendering.
        # This is the situation lazy storage exists for -- before a super index
        # exists, every summary is either rendered or one hop from a title.
        folded_titles = {cid for e in entries for cid in e.child_index_ids}
        covered_members: set = set()
        for entry in entries:
            if entry.index_id in folded_titles:
                covered_members.update(entry.members)
        moved = 0
        for summary in active:
            if summary.summary_id in newest_keep:
                continue
            if summary.index_id == LAZY_INDEX_ID:
                continue
            if summary.summary_id in covered_members:
                # Keep the title membership for provenance, but mark id-only so the
                # renderer and the diagnostics agree on its state.
                summary.index_id = LAZY_INDEX_ID
                self.store.add_active_summary(summary, episode_id=self.episode_id)
                moved += 1
            elif not summary.index_id:
                summary.index_id = LAZY_INDEX_ID
                self.store.add_active_summary(summary, episode_id=self.episode_id)
                moved += 1
        if moved:
            stats.lazy_moved = moved
            logger.debug("lazy store at turn %d: %d summaries moved to id-only storage",
                         stats.turn_index, moved)
        return moved

    def _file_one_index_group(self, stats: MemoryTurnStats) -> bool:
        """
        File the oldest eligible group under a new index entry (one iteration).

        Returns ``False`` when nothing can be filed, which is the loop's stop
        condition.  It must be *monotone*: every ``True`` has to remove at least two
        summaries from the rendered chain, otherwise a budget loop would spin
        forever re-filing the same group (which is exactly what a first version did).
        """
        chain = self.chain_summaries()
        min_visible = max(1, self.index_min_visible)
        if len(chain) <= min_visible:
            return False

        keep = min(self.index_keep_recent, max(0, len(chain) - min_visible))
        candidates = chain[: len(chain) - keep] if keep > 0 else list(chain)
        if len(candidates) < 2:
            return False

        # Fact-safety: prefer a group that does not contain the newest copy of a
        # fact slot.  If no such group exists we fall back to the oldest pair, but
        # note it in ``stats`` -- the budget wins over the preference, never the
        # other way round, and the choice stays auditable.
        seen: set = set()
        protected: set = set()
        for summary in reversed(chain):
            keys = {self._slot_of(k) for k in (set(summary.fact_keys) or extract_fact_keys(summary.text))}
            keeps = {k for k in keys if k not in seen}
            if keeps:
                protected.add(summary.summary_id)
                seen.update(keeps)

        group_size = max(2, self.index_group_size)
        batch = [x for x in candidates if x.summary_id not in protected][:group_size]
        fell_back = False
        if len(batch) < 2:
            batch = candidates[:group_size]
            fell_back = True
        if len(batch) < 2:
            return False
        # A file operation that does not shrink the rendered chain is forbidden:
        # it would make the caller loop without progress.
        if len({s.summary_id for s in batch}) < 2:
            return False

        turn_start = min((x.seq for x in batch), default=-1)
        turn_end = max((x.seq for x in batch), default=-1)
        digest = self._fact_digest(batch)
        theme = self._make_index_title(batch, turn_start, turn_end)
        # Facts first: they are what makes the index answerable.  The theme is kept
        # only as a short suffix for readability.
        if digest and theme:
            title = f"{digest} ｜ {theme}"[:220]
        else:
            title = (digest or theme or "; ".join((x.fact_keys or [x.text[:20]])[0] for x in batch))[:220]

        entry = IndexEntry(
            index_id=make_id(self.episode_id, "idx", self._next_seq(), salt=f"{title}|{turn_start}|{turn_end}"),
            # (title is already capped by _make_index_title)
            title=title,
            members=[x.summary_id for x in batch],
            span_start=min(x.timestamp for x in batch),
            span_end=max(x.timestamp for x in batch),
            turn_start=turn_start,
            turn_end=turn_end,
            fact_keys=sorted({k for x in batch for k in (x.fact_keys or extract_fact_keys(x.text))}),
            episode_id=self.episode_id,
            seq=self._seq,
            previews=[_preview_snippet(x.text) for x in batch],
            member_summaries=[x.render() for x in batch],
        )
        # Cost gate: filing must pay for itself.  Replacing a group with an index
        # entry whose title is nearly as expensive is not compression.
        batch_cost = sum(estimate_tokens(x.render()) for x in batch)
        if entry.render_cost() >= batch_cost:
            logger.info(
                "index rejected at turn %d: entry cost %d >= batch cost %d",
                stats.turn_index, entry.render_cost(), batch_cost,
            )
            stats.index_rejections += 1
            return False

        for summary in batch:
            summary.index_id = entry.index_id
            self.store.add_active_summary(summary, episode_id=self.episode_id)
        self.store.add_index_entry(entry, episode_id=self.episode_id)
        self.indexes_built += 1
        stats.index_ids = [entry.index_id]
        if fell_back:
            stats.fact_safety_fallbacks += 1
            logger.debug(
                "turn %d: filed a group containing a newest-fact summary (budget priority): %s",
                stats.turn_index, [s.summary_id.split("/")[-1] for s in batch],
            )
        return True

    def _maybe_fold_indexes(self, stats: MemoryTurnStats, force: bool = False) -> bool:
        """
        Recursive level: fold the oldest index entries into a "super index".

        A super index is a title over titles -- the same idea one level up.  It does
        not delete the child entries: they stay readable and ``expand_index`` on the
        super index lists them.

        ``force=True`` folds on budget pressure even when the entry count is below
        ``INDEX_MAX_ENTRIES``; the count threshold alone let the index layer grow
        until it dominated the prompt.
        """
        entries = self.list_index_entries()
        if not force and len(entries) <= self.index_max_entries:
            return False
        group = entries[: max(2, config.SUPER_INDEX_GROUP)]
        if len(group) < 2:
            return False
        # A super index aggregates its children's attribute histories, so a value
        # that moved across group boundaries is still visible at the top level.
        child_digest = ManagerFactDigest.merge([e.title for e in reversed(list(group))], max_attrs=8)
        title = (child_digest or f"{group[0].capped_title()[:24]} … {group[-1].capped_title()[:24]}")[:220]
        super_entry = IndexEntry(
            index_id=make_id(self.episode_id, "sidx", self._next_seq(), salt=title),
            title=f"[{len(group)} 组] {title}",
            members=[m for e in group for m in e.members],   # transitive, for expand
            child_index_ids=[e.index_id for e in group],
            span_start=min(e.span_start for e in group),
            span_end=max(e.span_end for e in group),
            turn_start=min(e.turn_start for e in group if e.turn_start >= 0) if group else -1,
            turn_end=max(e.turn_end for e in group),
            fact_keys=sorted({k for e in group for k in e.fact_keys}),
            episode_id=self.episode_id,
            seq=self._seq,
            previews=[e.title for e in group],
            member_summaries=[e.render(preview=0) for e in group],
        )
        children_cost = sum(estimate_tokens(e.render(preview=1)) for e in group)
        if super_entry.render_cost() >= children_cost:
            stats.index_rejections += 1
            logger.info("super index rejected at turn %d: %d >= %d",
                        stats.turn_index, super_entry.render_cost(), children_cost)
            return False
        # The children stop being rendered directly; the super index replaces them.
        self.store.add_index_entry(super_entry, episode_id=self.episode_id)
        for child in group:
            self.store.remove_index_entry(child.index_id, self.episode_id)
        # CRITICAL: re-point the members at the super index.  Leaving them pointing
        # at a removed child made them count as unfiled again, so they reappeared in
        # the rendered chain and the cost *grew* after folding.
        members = {x.summary_id: x for x in self.list_active_summaries()}
        for member_id in super_entry.members:
            summary = members.get(member_id)
            if summary is not None and summary.index_id in {c.index_id for c in group}:
                summary.index_id = super_entry.index_id
                self.store.add_active_summary(summary, episode_id=self.episode_id)
        self.super_indexes += 1
        stats.super_index_ids = list(getattr(stats, "super_index_ids", []) or []) + [super_entry.index_id]
        logger.info("super index built at turn %d: %d index entries folded",
                    stats.turn_index, len(group))
        return True

    @staticmethod
    def _fact_digest(members: Sequence[ActiveSummary], max_attrs: int = 6, max_history: int = 3) -> str:
        """
        ``属性=最新值[旧值→…]`` for the attributes covered by a group.

        The index title is what the answering model sees for everything that has been
        filed away.  A thematic title ("batch processing and desk moves") tells it
        nothing it can answer with -- an early version produced exactly that and the
        model replied "not in memory" while the value sat in the archive.  Naming the
        attribute and its current value (plus a short change history) makes the answer
        directly readable from the index layer.
        """
        order: List[str] = []
        values: Dict[str, List[str]] = {}
        # Newest members first: when the title has to be truncated, the attributes
        # that changed most recently are the ones an answer is most likely to need.
        for summary in reversed(list(members)):
            for pair in (summary.fact_keys or sorted(extract_fact_keys(summary.text))):
                if "=" not in pair:
                    continue
                attr, _, value = pair.partition("=")
                attr, value = attr.strip(), value.strip()
                if not attr or not value:
                    continue
                if attr not in values:
                    values[attr] = []
                    order.append(attr)
                if value not in values[attr]:
                    values[attr].append(value)
        parts: List[str] = []
        for attr in order[:max_attrs]:
            seq = values[attr]
            latest = seq[-1]
            history = "→".join(seq[:-1][-max_history:])
            parts.append(f"{attr}={latest}" + (f"[{history}→{latest}]" if history else ""))
        return "; ".join(parts)

    def _make_index_title(
        self, batch: Sequence[ActiveSummary], turn_start: int, turn_end: int
    ) -> str:
        if self.summarizer is None:
            return "; ".join(s.text[:24] for s in batch)[:70]
        try:
            response = self.summarizer.generate(
                [
                    {"role": "system", "content": index_title_system_prompt()},
                    {"role": "user", "content": index_title_user_prompt(batch, turn_start, turn_end)},
                ]
            )
            title = (response.text or "").strip().splitlines()[0] if response.text else ""
        except LLMError as exc:
            self.summarizer_failures += 1
            logger.warning("index title generation failed: %s", exc)
            return ""
        # Strip any tag the model may echo, keep it one line and inside the hard
        # title budget (a long "title" is content smuggled into the index layer).
        if "[OVERRIDES" in title.upper():
            title = title[: title.upper().index("[OVERRIDES")].strip()
        title = title.strip().strip("#*- ")
        title = re.split(r"[。；;\n]", title)[0].strip()   # first clause only
        return title[: IndexEntry.MAX_TITLE_CHARS]

    # ------------------------------------------------------------------ #
    # Legacy flat capacity compression (kept for A/B comparison)
    # ------------------------------------------------------------------ #
    def _maybe_capacity_compress(self, stats: MemoryTurnStats) -> bool:
        if self.active_chain_tokens() <= self.active_chain_token_limit:
            return False
        chain = self.list_active_summaries()
        if len(chain) <= self.capacity_min_merge:
            return False

        # Never merge the summary produced by this very turn: the new fact must
        # stay individually readable.
        candidates = [s for s in chain if s.summary_id != stats.new_summary_id]

        # Fact-safety rule (learned the hard way): keep the newest summary for
        # every fact slot on the chain.  Merging is lossy in practice, so if the
        # only copy of a slot were merged away, "what is X now?" would become
        # unanswerable -- the first long-context run scored 0.48 current-fact
        # accuracy for exactly that reason.  Strictly lexical (slot matching).
        protected: set = set()
        seen_keys: set = set()
        for summary in reversed(chain):
            keys = set(summary.fact_keys) or extract_fact_keys(summary.text)
            keeps = {k for k in keys if k not in seen_keys}
            if keeps:
                protected.add(summary.summary_id)
                seen_keys.update(keeps)

        eligible = [s for s in candidates if s.summary_id not in protected]
        if len(eligible) >= self.capacity_min_merge:
            candidates = eligible
        else:
            # Not enough unprotected material: fall back to the regular pool but
            # still keep the protected set out of the very first merge slot.
            candidates = eligible + [s for s in candidates if s.summary_id in protected]
            candidates = candidates[: max(self.capacity_max_merge, self.capacity_min_merge)]

        # Earliest first; prefer entries that override nothing (the ones carrying
        # conflict relations are the interesting history and stay granular).
        no_conflict = [s for s in candidates if not s.override_ids]
        ordered = no_conflict + [s for s in candidates if s.override_ids]
        batch = ordered[: self.capacity_max_merge]
        if len(batch) < self.capacity_min_merge:
            return False
        # Refuse to merge if doing so would drop a protected (newest-per-slot) entry.
        if any(s.summary_id in protected for s in batch) and len(
            [s for s in batch if s.summary_id not in protected]
        ) < self.capacity_min_merge:
            return False

        merged_text = self._merge_summaries(batch)
        if not merged_text:
            logger.warning("capacity compression skipped (empty merge result) at turn %d", stats.turn_index)
            return False
        merged_text = merged_text.strip()
        # A capacity merge must not carry an OVERRIDES tag; strip it if the model
        # added one anyway.
        if "[OVERRIDES" in merged_text.upper():
            merged_text = merged_text[: merged_text.upper().index("[OVERRIDES")].strip()

        # Guard: a merge that does not actually shrink the chain is not
        # compression.  Real models sometimes expand a merge (observed: 550 -> 1887
        # tokens), which would make the token budget diverge instead of compress.
        # Reject the merge and keep the originals; counted so it stays visible.
        before_tokens = sum(estimate_tokens(s.render()) for s in batch)
        after_tokens = estimate_tokens(merged_text)
        if after_tokens >= before_tokens * (1.0 - config.CAPACITY_MIN_GAIN):
            self.capacity_merges_rejected += 1
            logger.info(
                "capacity compression rejected at turn %d: merge would not shrink "
                "(%d tokens before, %d after; need < %d)",
                stats.turn_index, before_tokens, after_tokens,
                int(before_tokens * (1.0 - config.CAPACITY_MIN_GAIN)),
            )
            return False

        merged_refs: List[str] = []
        for summary in batch:
            merged_refs.extend(summary.raw_ref_ids or [summary.raw_ref_id])
        merged = ActiveSummary(
            summary_id=make_id(self.episode_id, "m", self._next_seq(), salt=merged_text),
            text=merged_text,
            override_ids=[],          # capacity merges never generate OVERRIDES
            timestamp=now_ts(),
            raw_ref_id=merged_refs[0] if merged_refs else "",
            episode_id=self.episode_id,
            seq=self._seq,
            origin="capacity_merge",
            raw_ref_ids=merged_refs,
            merged_from=[s.summary_id for s in batch],
            fact_keys=sorted({k for s in batch for k in (s.fact_keys or extract_fact_keys(s.text))}),
        )
        for summary in batch:
            self.store.remove_active_summary(summary.summary_id, episode_id=self.episode_id)
            self.store.add_archived_summary(
                ArchivedSummary.from_active(
                    summary,
                    is_overridden=False,
                    superseded_by=None,
                    archive_reason="capacity",
                    merged_from=_merge_origin_lineage(summary),
                ),
                episode_id=self.episode_id,
            )
        self.store.add_active_summary(merged, episode_id=self.episode_id)
        self.compressions += 1
        stats.capacity_merged_ids = [s.summary_id for s in batch]
        logger.info(
            "capacity compression at turn %d: merged %d summaries (%d tokens before, %d after)",
            stats.turn_index,
            len(batch),
            sum(estimate_tokens(s.render()) for s in batch),
            estimate_tokens(merged.render()),
        )
        return True

    def _bound_summary_text(self, text: str, stats: Optional[MemoryTurnStats] = None) -> str:
        """
        Cap a generated summary.

        Two guard levels, because they catch different failures:

        * a *suspect* length (the model copied its input) is logged loudly so the
          behaviour is visible in the run log rather than silently absorbed;
        * a hard token ceiling, applied unconditionally, so one bad generation can
          never make the next prompt too large to send.  A summary is a catalogue
          entry -- 1-3 sentences -- not a transcript.
        """
        if stats is not None and len(text) > config.SUMMARY_SUSPECT_CHARS:
            logger.warning(
                "turn %d: summariser returned %d chars (looks like it copied its input); truncating",
                stats.turn_index, len(text),
            )
        limit = config.MAX_SUMMARY_OUTPUT_TOKENS
        if estimate_tokens(text) <= limit:
            return text
        self.summary_truncations += 1
        if stats is not None:
            stats.summary_truncated = True
        return truncate_to_tokens(text, limit)

    def _merge_summaries(self, batch: Sequence[ActiveSummary]) -> str:
        if self.summarizer is None:
            return " | ".join(s.text for s in batch)
        messages = [
            {"role": "system", "content": merge_system_prompt()},
            {"role": "user", "content": merge_user_prompt(batch)},
        ]
        try:
            response = self.summarizer.generate(messages)
        except LLMError as exc:
            self.summarizer_failures += 1
            logger.warning("merge failed, falling back to concatenation: %s", exc)
            return " | ".join(s.text for s in batch)
        text = (response.text or "").strip()
        return text or " | ".join(s.text for s in batch)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _integrity_check(self, where: str) -> None:
        """
        O(n) sanity check: the manager's view of layer 1 must match the store's.

        A mismatch here is how the "chain reports 56k entries while SQLite holds 21"
        symptom shows up, so it is checked (cheaply -- these lists are tens of items)
        and reported with a stack trace instead of being silently averaged away.
        """
        if not logger.isEnabledFor(logging.DEBUG) and not self.strict_integrity:
            return
        try:
            from_store = self.store.list_active_summaries(self.episode_id)
        except Exception:  # noqa: BLE001
            return
        ids = [x.summary_id for x in from_store]
        if len(ids) != len(set(ids)):
            dupes = len(ids) - len(set(ids))
            logger.error(
                "integrity at %s: store returned %d summaries but only %d distinct ids "
                "(%d duplicates)", where, len(ids), len(set(ids)), dupes,
            )
            if self.strict_integrity:
                import traceback

                logger.error("call stack:\n%s", "".join(traceback.format_stack()[-6:]))
        # Tokens are what the budget loop consumes: report the largest single entry
        # so an outsized summary is visible immediately.
        if ids:
            costs = [estimate_tokens(x.render()) for x in from_store]
            if max(costs) > config.SUMMARIZER_MAX_INPUT_TOKENS:
                logger.error(
                    "integrity at %s: one summary costs %d tokens (ceiling %d); id=%s",
                    where, max(costs), config.SUMMARIZER_MAX_INPUT_TOKENS,
                    from_store[costs.index(max(costs))].summary_id,
                )

    def _fill_counts(self, stats: MemoryTurnStats) -> None:
        self._integrity_check("fill_counts")
        actives = self.list_active_summaries()
        stats.active_chain_size = len(self.chain_summaries())
        stats.index_count = len(self.list_index_entries())
        stats.filed_summary_count = len(actives) - stats.active_chain_size
        stats.lazy_summary_count = len(self.lazy_summaries())
        stats.active_chain_tokens = self.active_chain_tokens()
        stats.active_chain_text_tokens = self.active_chain_text_tokens()
        stats.window_size = len(self.get_window())
        stats.archived_total = self.store.count_archived_summaries(self.episode_id)
        stats.raw_total = self.store.count_raw_records(self.episode_id)

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def episode_metrics(self) -> Dict[str, Any]:
        """Per-episode aggregates for the summary CSV."""
        chain = self.list_active_summaries()
        return {
            "episode_id": self.episode_id,
            "num_turns": self.turn_index + 1,
            "active_chain_size": len(self.chain_summaries()),
            "index_count": len(self.list_index_entries()),
            "filed_summary_count": len(chain) - len(self.chain_summaries()),
            "lazy_summary_count": len(self.lazy_summaries()),
            "all_summary_tokens": self.all_summary_tokens(),
            "all_rendered_tokens": self.all_rendered_tokens(),
            "active_chain_tokens": self.active_chain_tokens(),
            "active_chain_text_tokens": self.active_chain_text_tokens(),
            "archived_count": self.store.count_archived_summaries(self.episode_id),
            "raw_count": self.store.count_raw_records(self.episode_id),
            "overrides_events": sum(1 for s in self.stats if s.event_triggered),
            "overridden_summaries": self.overrides,
            "capacity_compressions": self.compressions,
            "indexes_built": self.indexes_built,
            "summary_truncations": self.summary_truncations,
            "super_indexes": self.super_indexes,
            "chain_strategy": self.chain_strategy,
            "capacity_merges_rejected": self.capacity_merges_rejected,
            "summarizer_failures": self.summarizer_failures,
            "window_turns": len(self.get_window()),
        }

    def debug_state(self) -> str:
        """Compact human-readable dump for chat_debug.py."""
        lines = [f"=== episode {self.episode_id} | turn {self.turn_index} ==="]
        window = self.get_window()
        lines.append(f"[sliding window] {len(window)}/{self.recent_window_turns} turns")
        for record in window:
            lines.append(f"  turn {record.turn_index:>3} ({record.reference_id})")
            lines.append(f"    user : {record.user_msg}")
            lines.append(f"    agent: {record.agent_msg}")
        chain = self.list_active_summaries()
        lines.append(
            f"[active chain] {len(chain)} summaries | "
            f"~{self.active_chain_tokens()} tokens (limit {self.active_chain_token_limit}, "
            f"bodies ~{self.active_chain_text_tokens()})"
        )
        for summary in chain:
            lines.append(f"  {summary.render()}")
        archived = self.list_archived_summaries()
        lines.append(f"[archive] {len(archived)} summaries")
        for item in archived:
            lines.append(f"  {item.render()}")
        return "\n".join(lines)
