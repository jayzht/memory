"""
Storage abstraction layer.

``BaseMemoryStore`` is the single contract between :class:`MemoryManager` and
persistence, so switching ``memory`` <-> ``redis_sqlite_hybrid`` never touches a
line of memory/compression logic.

Design decisions worth recording for the report
-----------------------------------------------
* **All permanent data lives in SQLite.**  Redis holds only the current
  episode's hot data (sliding window + active chain).  Losing Redis therefore
  loses nothing that matters: the active chain is rebuildable from SQLite.
* **Episode namespacing.**  Redis keys are ``episode:{episode_id}:...``
  (``REDIS_KEY_PREFIX``), so two dataset episodes can never collide and
  ``reset(episode_id)`` can delete exactly one episode's keys via SCAN+UNLINK.
* **reset() is intentionally asymmetric.**  It drops the episode's *hot* keys but
  keeps that episode's archived summaries and raw records in SQLite forever --
  otherwise the agent could not look up its own overridden history, and
  History_Fact_Acc would be unmeasurable.
* **find_raw_record / find_archived_summary are episode-scoped in normal
  operation, globally unique as a fallback.**  Ids already contain the episode
  id, so a stray cross-episode id can still be resolved when a caller passes
  ``episode_id=None``; within an episode the fast path never scans.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

from ..models import ActiveSummary, ArchivedSummary, IndexEntry, RawDialogRecord

logger = logging.getLogger(__name__)

# Redis key suffix layout (inside the per-episode namespace).
K_ACTIVE_IDS = "active_ids"       # LIST : summary_id, creation order (newest last)
K_ACTIVE_HASH = "active"          # HASH : summary_id -> JSON(ActiveSummary)
K_WINDOW = "window"               # LIST : JSON(RawDialogRecord), oldest -> newest
K_META = "meta"                   # HASH : episode bookkeeping


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _loads(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(text)


class BaseMemoryStore(abc.ABC):
    """
    Unified storage interface.

    Every method that touches per-episode hot state takes an optional
    ``episode_id``; when omitted the store's bound episode is used, which is what
    keeps the manager's call sites short.
    """

    backend_name: str = "base"

    def __init__(self, recent_window_turns: int = 4):
        self.recent_window_turns = recent_window_turns
        self._bound_episode: Optional[str] = None
        self._current_turn_index: int = -1

    # ------------------------------------------------------------------ #
    # Episode binding
    # ------------------------------------------------------------------ #
    def bind_episode(
        self,
        episode_id: str,
        *,
        recent_window_turns: Optional[int] = None,
        reset: bool = False,
        flush_hot_keys: bool = False,
    ) -> None:
        """Select the active episode namespace."""
        if reset:
            self.reset(episode_id=episode_id, flush_hot_keys=flush_hot_keys)
        self._bound_episode = episode_id
        self._current_turn_index = -1
        if recent_window_turns:
            self.recent_window_turns = recent_window_turns

    @property
    def bound_episode(self) -> Optional[str]:
        return self._bound_episode

    def _ep(self, episode_id: Optional[str]) -> str:
        episode = episode_id or self._bound_episode
        if not episode:
            raise ValueError("no episode bound: call bind_episode() first")
        return episode

    def set_turn_index(self, turn_index: int) -> None:
        self._current_turn_index = turn_index

    def next_turn_index(self) -> int:
        """
        Monotonic turn ordinal inside the bound episode.

        Owned by the store so that raw records and window entries always agree on
        the turn numbering even after a mid-episode resume.
        """
        self._current_turn_index += 1
        return self._current_turn_index

    @property
    def current_turn_index(self) -> int:
        return self._current_turn_index

    # ------------------------------------------------------------------ #
    # Layer 3: raw dialogue records
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def add_raw_record(self, record: RawDialogRecord, episode_id: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def get_raw_record(
        self, reference_id: str, episode_id: Optional[str] = None
    ) -> Optional[RawDialogRecord]: ...

    @abc.abstractmethod
    def list_raw_records(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]: ...

    def get_raw_records(self, reference_ids: Sequence[str], episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        """Batch exact-id lookup (used by capacity merges and the full-context baseline)."""
        found = []
        for ref in reference_ids:
            record = self.get_raw_record(ref, episode_id=episode_id)
            if record is not None:
                found.append(record)
        return found

    def delete_raw_record(self, reference_id: str, episode_id: Optional[str] = None) -> bool:
        """
        Destroy a raw record.

        Only the MemGPT-style destructive baseline calls this; the three-layer
        system never does (its raw store is permanent by contract).
        """
        raise NotImplementedError(f"{self.backend_name} cannot delete raw records")

    def count_raw_records(self, episode_id: Optional[str] = None) -> int:
        return len(self.list_raw_records(episode_id))

    # ------------------------------------------------------------------ #
    # Layer 1: active chain (ordered)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def add_active_summary(self, summary: ActiveSummary, episode_id: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def remove_active_summary(
        self, summary_id: str, episode_id: Optional[str] = None
    ) -> Optional[ActiveSummary]: ...

    @abc.abstractmethod
    def list_active_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]: ...

    def get_active_summary(
        self, summary_id: str, episode_id: Optional[str] = None
    ) -> Optional[ActiveSummary]:
        for summary in self.list_active_summaries(episode_id):
            if summary.summary_id == summary_id:
                return summary
        return None

    def count_active_summaries(self, episode_id: Optional[str] = None) -> int:
        return len(self.list_active_summaries(episode_id))

    def clear_active_summaries(self, episode_id: Optional[str] = None) -> None:
        for summary in list(self.list_active_summaries(episode_id)):
            self.remove_active_summary(summary.summary_id, episode_id)
        self.clear_indexes(episode_id)

    # ------------------------------------------------------------------ #
    # Layer 1 upper level: index entries (titles over groups of summaries)
    # ------------------------------------------------------------------ #
    # An index points at its members; it never replaces them.  Filing a summary
    # under an index therefore moves it out of the *rendered* chain while keeping
    # it fully readable by id -- that is the whole point of the hierarchy.
    @abc.abstractmethod
    def add_index_entry(self, entry: "IndexEntry", episode_id: Optional[str] = None) -> None: ...

    @abc.abstractmethod
    def list_index_entries(self, episode_id: Optional[str] = None) -> List["IndexEntry"]: ...

    @abc.abstractmethod
    def get_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> Optional["IndexEntry"]: ...

    def clear_indexes(self, episode_id: Optional[str] = None) -> None:
        for entry in list(self.list_index_entries(episode_id)):
            self.remove_index_entry(entry.index_id, episode_id)

    def remove_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> None:
        raise NotImplementedError(f"{self.backend_name} cannot remove index entries")

    def summaries_under_index(self, index_id: str, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        """Resolve an index to its member summaries (the catalogue behind a title)."""
        entry = self.get_index_entry(index_id, episode_id)
        if entry is None:
            return []
        members = {s.summary_id: s for s in self.list_active_summaries(episode_id)}
        return [members[sid] for sid in entry.members if sid in members]

    def chain_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        """
        The summaries rendered directly in the prompt: those not yet filed under an
        index.  ``list_active_summaries`` still returns *everything* (filed or not)
        so id lookups keep working.
        """
        return [s for s in self.list_active_summaries(episode_id) if not s.index_id]

    # ------------------------------------------------------------------ #
    # Layer 2: archived summaries (permanent, exact-id read)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def add_archived_summary(
        self, archived: ArchivedSummary, episode_id: Optional[str] = None
    ) -> None: ...

    @abc.abstractmethod
    def get_archived_summary(
        self, summary_id: str, episode_id: Optional[str] = None
    ) -> Optional[ArchivedSummary]: ...

    @abc.abstractmethod
    def list_archived_summaries(
        self, episode_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ArchivedSummary]: ...

    def mark_superseded(
        self, summary_id: str, superseded_by: str, episode_id: Optional[str] = None
    ) -> bool:
        """Record the forward pointer ``summary_id -> superseded_by``."""
        archived = self.get_archived_summary(summary_id, episode_id=episode_id)
        if archived is None:
            return False
        archived.superseded_by = superseded_by
        archived.is_overridden = True
        self.add_archived_summary(archived, episode_id=episode_id)
        return True

    def count_archived_summaries(self, episode_id: Optional[str] = None) -> int:
        return len(self.list_archived_summaries(episode_id))

    def clear_archive(self, episode_id: str) -> int:
        """
        Delete one episode's archived summaries, returning how many were removed.

        Raw records are **never** touched -- they are permanent by contract.  This
        exists because the archive namespace is ``<system>/<episode>`` and carries no
        run id: re-running an episode (``--force``, or a new run id over the same
        SQLite file) would otherwise inherit dead rows from the previous run, which
        inflates ``Avg_Archived`` and feeds stale summaries into the memory-point
        probes.  Batch evaluation calls this once, before ingestion starts.
        """
        raise NotImplementedError

    def find_archived_summary(self, summary_id: str) -> Optional[ArchivedSummary]:
        """Global exact-id search, used by the archival tool across episode scopes."""
        return self.get_archived_summary(summary_id, episode_id=None)

    # ------------------------------------------------------------------ #
    # Recent sliding window (verbatim dialogue)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def append_window_record(self, record: RawDialogRecord, episode_id: Optional[str] = None) -> List[RawDialogRecord]: ...

    @abc.abstractmethod
    def get_window(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]: ...

    @abc.abstractmethod
    def set_window(self, records: Sequence[RawDialogRecord], episode_id: Optional[str] = None) -> None: ...

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def reset(self, episode_id: Optional[str] = None, flush_hot_keys: bool = False) -> None:
        """
        Prepare a fresh episode.

        Drops the episode's hot state (active chain + sliding window) so the next
        dataset sample cannot inherit a single token of it.  Permanent cold data
        (archived summaries, raw records) is *kept*.
        """

    # ------------------------------------------------------------------ #
    # Bookkeeping shared by both backends
    # ------------------------------------------------------------------ #
    def checksum(self) -> Dict[str, int]:
        """Cheap integrity summary, handy in logs and checkpoints."""
        return {
            "active": self.count_active_summaries(self._bound_episode),
            "window": len(self.get_window(self._bound_episode)),
            "raw": self.count_raw_records(self._bound_episode),
            "archived": self.count_archived_summaries(self._bound_episode),
        }

    def purge_episode(self, episode_id: str) -> None:
        """
        Hard-delete *all* of one episode's data, archive and raw records included.

        Deliberately separate from :meth:`reset`: batch evaluation must never lose
        permanent data by accident, so only the interactive "new session" path
        calls this.
        """
        raise NotImplementedError(f"{self.backend_name} cannot purge an episode")

    def close(self) -> None:  # pragma: no cover - overridden where it matters
        pass


# --------------------------------------------------------------------------- #
# In-memory backend (fast debugging, zero dependencies)
# --------------------------------------------------------------------------- #
class InMemoryStore(BaseMemoryStore):
    """Pure-Python implementation: single episode debug, unit tests, CI."""

    backend_name = "memory"

    def __init__(self, recent_window_turns: int = 4):
        super().__init__(recent_window_turns)
        # raw records are global (like SQLite) but indexed per episode
        self._raw: "OrderedDict[str, RawDialogRecord]" = OrderedDict()
        self._raw_by_episode: Dict[str, List[str]] = {}
        # hot state per episode
        self._active: Dict[str, "OrderedDict[str, ActiveSummary]"] = {}
        self._indexes: Dict[str, "OrderedDict[str, IndexEntry]"] = {}
        self._window: Dict[str, List[RawDialogRecord]] = {}
        # archive is global
        self._archived: "OrderedDict[str, ArchivedSummary]" = OrderedDict()
        self.reset_calls = 0

    # -- raw ---------------------------------------------------------------- #
    def add_raw_record(self, record: RawDialogRecord, episode_id: Optional[str] = None) -> None:
        episode = record.episode_id or self._ep(episode_id)
        record.episode_id = episode
        self._raw[record.reference_id] = record
        self._raw_by_episode.setdefault(episode, []).append(record.reference_id)

    def get_raw_record(self, reference_id: str, episode_id: Optional[str] = None) -> Optional[RawDialogRecord]:
        record = self._raw.get(reference_id)
        if record is None:
            return None
        if episode_id is not None and record.episode_id != episode_id:
            return None
        return record

    def list_raw_records(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        if episode_id is None:
            return list(self._raw.values())
        return [self._raw[r] for r in self._raw_by_episode.get(episode_id, []) if r in self._raw]

    def delete_raw_record(self, reference_id: str, episode_id: Optional[str] = None) -> bool:
        record = self._raw.pop(reference_id, None)
        if record is None:
            return False
        refs = self._raw_by_episode.get(record.episode_id)
        if refs and reference_id in refs:
            refs.remove(reference_id)
        return True

    # -- active ------------------------------------------------------------- #
    def add_active_summary(self, summary: ActiveSummary, episode_id: Optional[str] = None) -> None:
        episode = summary.episode_id or self._ep(episode_id)
        summary.episode_id = episode
        self._active.setdefault(episode, OrderedDict())[summary.summary_id] = summary

    def remove_active_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ActiveSummary]:
        episode = self._ep(episode_id)
        return self._active.get(episode, OrderedDict()).pop(summary_id, None)

    def list_active_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        episode = self._ep(episode_id)
        return list(self._active.get(episode, OrderedDict()).values())

    # -- index entries ------------------------------------------------------ #
    def add_index_entry(self, entry: IndexEntry, episode_id: Optional[str] = None) -> None:
        episode = entry.episode_id or self._ep(episode_id)
        entry.episode_id = episode
        self._indexes.setdefault(episode, OrderedDict())[entry.index_id] = entry

    def list_index_entries(self, episode_id: Optional[str] = None) -> List[IndexEntry]:
        episode = self._ep(episode_id)
        return list(self._indexes.get(episode, OrderedDict()).values())

    def get_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> Optional[IndexEntry]:
        episode = self._ep(episode_id)
        return self._indexes.get(episode, OrderedDict()).get(index_id)

    def remove_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> None:
        self._indexes.get(self._ep(episode_id), OrderedDict()).pop(index_id, None)

    # -- archive ------------------------------------------------------------ #
    def add_archived_summary(self, archived: ArchivedSummary, episode_id: Optional[str] = None) -> None:
        episode = archived.episode_id or episode_id or self._bound_episode or ""
        archived.episode_id = episode
        self._archived[archived.summary_id] = archived

    def get_archived_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ArchivedSummary]:
        archived = self._archived.get(summary_id)
        if archived is None:
            return None
        if episode_id is not None and archived.episode_id != episode_id:
            return None
        return archived

    def list_archived_summaries(
        self, episode_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ArchivedSummary]:
        items = [
            a for a in self._archived.values()
            if episode_id is None or a.episode_id == episode_id
        ]
        if limit is not None:
            items = items[-limit:]
        return items

    def clear_archive(self, episode_id: str) -> int:
        episode = self._ep(episode_id)
        doomed = [sid for sid, item in self._archived.items() if item.episode_id == episode]
        for summary_id in doomed:
            self._archived.pop(summary_id, None)
        return len(doomed)

    # -- window ------------------------------------------------------------- #
    def append_window_record(self, record: RawDialogRecord, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        episode = record.episode_id or self._ep(episode_id)
        window = self._window.setdefault(episode, [])
        window.append(record)
        while len(window) > self.recent_window_turns:
            window.pop(0)
        return list(window)

    def get_window(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        return list(self._window.get(self._ep(episode_id), []))

    def set_window(self, records: Sequence[RawDialogRecord], episode_id: Optional[str] = None) -> None:
        episode = self._ep(episode_id)
        self._window[episode] = list(records)[-self.recent_window_turns :]

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self, episode_id: Optional[str] = None, flush_hot_keys: bool = False) -> None:
        self.reset_calls += 1
        episode = episode_id or self._bound_episode
        if episode is None:
            # Global wipe: used by --reset-db style maintenance and unit tests.
            self._active.clear()
            self._indexes.clear()
            self._window.clear()
            self._raw.clear()
            self._raw_by_episode.clear()
            self._archived.clear()
            return
        self._active.pop(episode, None)
        self._indexes.pop(episode, None)
        self._window.pop(episode, None)
        # Cold data is deliberately preserved (archived + raw records).

    def purge_episode(self, episode_id: str) -> None:
        self._active.pop(episode_id, None)
        self._indexes.pop(episode_id, None)
        self._window.pop(episode_id, None)
        for reference_id in list(self._raw_by_episode.pop(episode_id, [])):
            self._raw.pop(reference_id, None)
        for summary_id in [k for k, v in self._archived.items() if v.episode_id == episode_id]:
            self._archived.pop(summary_id, None)
