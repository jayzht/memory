"""
Redis (hot) + SQLite (cold) hybrid store.

Contract
--------
* **SQLite is the source of truth.**  Every read either hits Redis or falls back
  to SQLite, and every write goes to SQLite first.  When Redis dies mid-run the
  experiment degrades (slower, plus a warning) instead of losing data.
* **Redis only holds the bound episode's hot data** -- active chain + sliding
  window -- under ``episode:{episode_id}:`` keys.
* ``reset(episode_id)`` deletes that episode's Redis namespace but *keeps* the
  episode's archived summaries and raw records in SQLite, so the agent can still
  read its own overridden history (History_Fact_Acc depends on this).
* Resuming a crashed episode: ``rebuild_hot_state(episode_id)`` restores the
  active chain and sliding window from the SQLite mirrors written every turn.

All Redis calls are wrapped: a failure flips ``redis_degraded`` and the operation
falls through to SQLite.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..models import ActiveSummary, ArchivedSummary, IndexEntry, RawDialogRecord
from .base import BaseMemoryStore
from .redis_store import RedisHotStore, RedisUnavailable
from .sqlite_store import SQLiteColdStore

logger = logging.getLogger(__name__)


class RedisSQLiteHybridStore(BaseMemoryStore):
    """The experiment-grade backend."""

    backend_name = "redis_sqlite_hybrid"

    def __init__(
        self,
        redis_store: Optional[RedisHotStore] = None,
        sqlite_store: Optional[SQLiteColdStore] = None,
        *,
        sqlite_path: Optional[str] = None,
        recent_window_turns: int = 4,
        strict_redis: bool = False,
        redis_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(recent_window_turns)
        # Imported lazily so the module can be inspected without config side effects.
        from .. import config

        if sqlite_store is None:
            sqlite_store = SQLiteColdStore(sqlite_path or config.SQLITE_PATH)
        if redis_store is None:
            kwargs = dict(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD,
                key_prefix=config.REDIS_KEY_PREFIX,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                required=strict_redis,
            )
            kwargs.update(redis_kwargs or {})
            redis_store = RedisHotStore(**kwargs)
        self.cold = sqlite_store
        self.hot = redis_store
        self.reset_calls = 0
        # Diagnostics for the result CSV / logs.
        self.diagnostics: Dict[str, Any] = {
            "redis_available": self.hot.available,
            "redis_degraded_events": 0,
            "fallbacks": 0,
            "resyncs": 0,
        }

    # ------------------------------------------------------------------ #
    # episode binding & lifecycle
    # ------------------------------------------------------------------ #
    def bind_episode(
        self,
        episode_id: str,
        *,
        recent_window_turns: Optional[int] = None,
        reset: bool = False,
        flush_hot_keys: bool = False,
        resume: bool = False,
    ) -> None:
        if reset:
            self.reset(episode_id=episode_id, flush_hot_keys=flush_hot_keys)
        super().bind_episode(episode_id, recent_window_turns=recent_window_turns)
        if self.hot.available:
            self.hot.bind_episode(episode_id)
        if resume:
            self.rebuild_hot_state(episode_id)

    def reset(self, episode_id: Optional[str] = None, flush_hot_keys: bool = False) -> None:
        """
        Start a clean episode.

        Deletes the episode's Redis namespace (hot data) and clears the SQLite
        mirror tables for that episode.  ``archived_summaries`` and
        ``raw_records`` are intentionally **not** touched -- they are permanent.
        """
        episode = episode_id or self._bound_episode
        if episode is None:
            raise ValueError("reset() needs an episode_id (or bind one first)")
        self.reset_calls += 1
        if self.hot.available:
            try:
                deleted = self.hot.clear_episode(episode)
                logger.debug("reset(%s): deleted %d redis keys", episode, deleted)
            except RedisUnavailable as exc:
                self._degrade(f"reset: {exc}")
        self.cold.clear_active_summaries(episode)
        self.cold.clear_window(episode)
        logger.debug("reset(%s): hot state cleared, cold archive/raw data kept", episode)

    def purge_episode(self, episode_id: str) -> None:
        """Wipe one episode completely: Redis keys, mirrors, archive and raw store."""
        self.reset(episode_id=episode_id)
        self.cold.drop_episode(episode_id, keep_archive=False)

    def clear_archive(self, episode_id: str) -> int:
        """Drop this episode's archived summaries; raw records are permanent."""
        return self.cold.clear_archive(self._ep(episode_id))

    # The ledger lives in SQLite only: it is an audit record, not hot state, and it
    # must survive Redis being lost.
    def append_fact_ledger(self, rows) -> int:
        return self.cold.append_fact_ledger(rows)

    def list_fact_ledger(self, episode_id=None):
        return self.cold.list_fact_ledger(self._ep(episode_id))

    def update_fact_ledger_reason(self, summary_id, reason, superseded_by="", episode_id=None) -> int:
        return self.cold.update_fact_ledger_reason(
            summary_id, reason, superseded_by, episode_id=self._ep(episode_id)
        )

    def erase_fact_ledger(self, rows, episode_id=None) -> int:
        return self.cold.erase_fact_ledger(rows, self._ep(episode_id))

    def clear_fact_ledger(self, episode_id) -> int:
        return self.cold.clear_fact_ledger(self._ep(episode_id))

    def count_fact_ledger(self, episode_id=None) -> int:
        return self.cold.count_fact_ledger(self._ep(episode_id))

    def rebuild_hot_state(self, episode_id: Optional[str] = None) -> bool:
        """Restore active chain + window into Redis from the SQLite mirrors."""
        episode = self._ep(episode_id)
        if not self.hot.available:
            return False
        try:
            actives = self.cold.list_active_summaries(episode)
            window = self.cold.get_window(episode)
            indexes = self.cold.list_index_entries(episode)
            self.hot.save_active_all(actives, episode_id=episode)
            self.hot.set_window(window, episode_id=episode)
            self.hot.save_index_all(indexes, episode_id=episode)
            self.diagnostics["resyncs"] += 1
            logger.info(
                "resumed episode %s: %d active summaries, %d window records",
                episode, len(actives), len(window),
            )
            return True
        except RedisUnavailable as exc:
            self._degrade(f"rebuild_hot_state: {exc}")
            return False

    def _degrade(self, message: str) -> None:
        self.hot.degraded = True
        self.diagnostics["redis_available"] = False
        self.diagnostics["redis_degraded_events"] += 1
        logger.warning("Redis hot layer degraded (%s) -- using SQLite", message)

    def _fallback(self) -> None:
        self.diagnostics["fallbacks"] += 1

    # ------------------------------------------------------------------ #
    # Layer 3: raw records (SQLite only, permanent)
    # ------------------------------------------------------------------ #
    def add_raw_record(self, record: RawDialogRecord, episode_id: Optional[str] = None) -> None:
        episode = record.episode_id or self._ep(episode_id)
        record.episode_id = episode
        if record.turn_index < 0:
            record.turn_index = self._current_turn_index
        self.cold.add_raw_record(record)

    def get_raw_record(self, reference_id: str, episode_id: Optional[str] = None) -> Optional[RawDialogRecord]:
        record = self.cold.get_raw_record(reference_id)
        if record is None:
            return None
        if episode_id is not None and record.episode_id != episode_id:
            return None
        return record

    def list_raw_records(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        return self.cold.list_raw_records(episode_id)

    def count_raw_records(self, episode_id: Optional[str] = None) -> int:
        return self.cold.count_raw_records(episode_id)

    def delete_raw_record(self, reference_id: str, episode_id: Optional[str] = None) -> bool:
        existed = self.cold.get_raw_record(reference_id) is not None
        self.cold.delete_raw_record(reference_id)
        return existed

    def get_raw_records(
        self, reference_ids: Sequence[str], episode_id: Optional[str] = None
    ) -> List[RawDialogRecord]:
        return [
            record
            for record in (self.get_raw_record(ref, episode_id) for ref in reference_ids)
            if record is not None
        ]

    # ------------------------------------------------------------------ #
    # Layer 1: active chain (Redis hot, SQLite mirror)
    # ------------------------------------------------------------------ #
    def add_active_summary(self, summary: ActiveSummary, episode_id: Optional[str] = None) -> None:
        episode = summary.episode_id or self._ep(episode_id)
        summary.episode_id = episode
        self.cold.add_active_summary(summary)          # durability first
        if self.hot.available:
            try:
                self.hot.add_active_summary(summary, episode_id=episode)
            except RedisUnavailable as exc:
                self._degrade(f"add_active_summary: {exc}")

    def remove_active_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ActiveSummary]:
        episode = self._ep(episode_id)
        summary = self.get_active_summary(summary_id, episode_id=episode)
        self.cold.remove_active_summary(summary_id)
        if self.hot.available:
            try:
                self.hot.remove_active_summary(summary_id, episode_id=episode)
            except RedisUnavailable as exc:
                self._degrade(f"remove_active_summary: {exc}")
        return summary

    def get_active_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ActiveSummary]:
        episode = self._ep(episode_id)
        if self.hot.available:
            try:
                found = self.hot.get_active_summary(summary_id, episode_id=episode)
                if found is not None:
                    return found
            except RedisUnavailable as exc:
                self._degrade(f"get_active_summary: {exc}")
        # Hot miss -> cold mirror.  A cold hit is re-warmed into Redis.
        for summary in self.cold.list_active_summaries(episode):
            if summary.summary_id == summary_id:
                self._fallback()
                if self.hot.available:
                    try:
                        self.hot.add_active_summary(summary, episode_id=episode)
                    except RedisUnavailable:
                        pass
                return summary
        return None

    def list_active_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        episode = self._ep(episode_id)
        if self.hot.available:
            try:
                summaries = self.hot.list_active_summaries(episode_id=episode)
                if summaries:
                    return summaries
                # Empty hot chain: the cold mirror may still hold this episode's hot
                # state (Redis restarted, key evicted, or maxmemory pressure).
                cold_summaries = self.cold.list_active_summaries(episode)
                if cold_summaries:
                    self._fallback()
                    logger.debug(
                        "hot chain empty for %s: re-warming %d summaries from SQLite",
                        episode, len(cold_summaries),
                    )
                    self.hot.save_active_all(cold_summaries, episode_id=episode)
                    self.diagnostics["resyncs"] += 1
                    return cold_summaries
                return []
            except RedisUnavailable as exc:
                self._degrade(f"list_active_summaries: {exc}")
        self._fallback()
        return self.cold.list_active_summaries(episode)

    def count_active_summaries(self, episode_id: Optional[str] = None) -> int:
        # Count through the *same* path the reads use.  LLEN counts raw list entries
        # (including any legacy duplicates the list path dedupes) and returns 0 on a
        # hot miss where the list path falls back to SQLite, so the two disagreed.
        return len(self.list_active_summaries(episode_id))

    # ------------------------------------------------------------------ #
    # Layer 1 upper level: index entries (Redis hot, SQLite durable)
    # ------------------------------------------------------------------ #
    def add_index_entry(self, entry: IndexEntry, episode_id: Optional[str] = None) -> None:
        episode = entry.episode_id or self._ep(episode_id)
        entry.episode_id = episode
        self.cold.add_index_entry(entry)
        if self.hot.available:
            try:
                self.hot.add_index_entry(entry, episode_id=episode)
            except RedisUnavailable as exc:
                self._degrade(f"add_index_entry: {exc}")

    def list_index_entries(self, episode_id: Optional[str] = None) -> List[IndexEntry]:
        episode = self._ep(episode_id)
        if self.hot.available:
            try:
                entries = self.hot.list_index_entries(episode_id=episode)
                if entries:
                    return entries
                cold_entries = self.cold.list_index_entries(episode)
                if cold_entries:
                    self._fallback()
                    self.hot.save_index_all(cold_entries, episode_id=episode)
                    self.diagnostics["resyncs"] += 1
                    return cold_entries
                return []
            except RedisUnavailable as exc:
                self._degrade(f"list_index_entries: {exc}")
        self._fallback()
        return self.cold.list_index_entries(episode)

    def get_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> Optional[IndexEntry]:
        episode = self._ep(episode_id)
        if self.hot.available:
            try:
                found = self.hot.get_index_entry(index_id, episode_id=episode)
                if found is not None:
                    return found
            except RedisUnavailable as exc:
                self._degrade(f"get_index_entry: {exc}")
        return self.cold.get_index_entry(index_id, episode_id=episode)

    def remove_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> None:
        episode = self._ep(episode_id)
        self.cold.remove_index_entry(index_id, episode_id=episode)
        if self.hot.available:
            try:
                self.hot.remove_index_entry(index_id, episode_id=episode)
            except RedisUnavailable as exc:
                self._degrade(f"remove_index_entry: {exc}")

    # ------------------------------------------------------------------ #
    # Layer 2: archive (SQLite only, permanent)
    # ------------------------------------------------------------------ #
    def add_archived_summary(self, archived: ArchivedSummary, episode_id: Optional[str] = None) -> None:
        episode = archived.episode_id or episode_id or self._bound_episode or ""
        archived.episode_id = episode
        self.cold.add_archived_summary(archived)

    def get_archived_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ArchivedSummary]:
        return self.cold.get_archived_summary(summary_id, episode_id=episode_id)

    def list_archived_summaries(
        self, episode_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ArchivedSummary]:
        return self.cold.list_archived_summaries(episode_id=episode_id, limit=limit)

    def count_archived_summaries(self, episode_id: Optional[str] = None) -> int:
        return self.cold.count_archived_summaries(episode_id)

    def mark_superseded(self, summary_id: str, superseded_by: str, episode_id: Optional[str] = None) -> bool:
        archived = self.cold.get_archived_summary(summary_id)
        if archived is None:
            return False
        archived.superseded_by = superseded_by
        archived.is_overridden = True
        self.cold.add_archived_summary(archived)
        return True

    # ------------------------------------------------------------------ #
    # sliding window (Redis hot, SQLite mirror)
    # ------------------------------------------------------------------ #
    def append_window_record(self, record: RawDialogRecord, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        episode = record.episode_id or self._ep(episode_id)
        record.episode_id = episode
        # Cold first, like every other write in this class.  This used to update
        # Redis and mirror afterwards, which broke the "SQLite is the source of
        # truth, Redis is a cache" ordering: a crash in between left the mirror
        # stale, and rebuild_hot_state restores the window only from that mirror.
        window = self.cold.get_window(episode)
        window.append(record)
        window = window[-self.recent_window_turns :]
        self.cold.set_window(episode, window)
        if self.hot.available:
            try:
                self.hot.set_window(window, episode_id=episode)
            except RedisUnavailable as exc:
                self._degrade(f"append_window_record: {exc}")
        return window

    def get_window(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        episode = self._ep(episode_id)
        if self.hot.available:
            try:
                window = self.hot.get_window(episode_id=episode)
                if window:
                    return window
                cold_window = self.cold.get_window(episode)
                if cold_window:
                    self._fallback()
                    self.hot.set_window(cold_window, episode_id=episode)
                    self.diagnostics["resyncs"] += 1
                    return cold_window
                return []
            except RedisUnavailable as exc:
                self._degrade(f"get_window: {exc}")
        self._fallback()
        return self.cold.get_window(episode)

    def set_window(self, records: Sequence[RawDialogRecord], episode_id: Optional[str] = None) -> None:
        episode = self._ep(episode_id)
        trimmed = list(records)[-self.recent_window_turns :]
        self.cold.set_window(episode, trimmed)
        if self.hot.available:
            try:
                self.hot.set_window(trimmed, episode_id=episode)
            except RedisUnavailable as exc:
                self._degrade(f"set_window: {exc}")

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def checksum(self) -> Dict[str, int]:
        data = super().checksum()
        data["redis_keys"] = (
            len(self.hot.list_episode_keys(self._bound_episode)) if self.hot.available else 0
        )
        return data

    def close(self) -> None:
        self.cold.close()
