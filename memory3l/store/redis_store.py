"""
Hot layer: Redis.

Redis only ever holds *one running episode's* hot data -- the active summary
chain and the recent sliding window -- under a per-episode namespace derived
from ``REDIS_KEY_PREFIX`` (default ``episode:{episode_id}``).

Keys inside one namespace::

    episode:{episode_id}:active_ids   LIST  summary_id, creation order (newest last)
    episode:{episode_id}:active       HASH  summary_id -> JSON(ActiveSummary)
    episode:{episode_id}:window       LIST  JSON(RawDialogRecord), oldest -> newest
    episode:{episode_id}:meta         HASH  bookkeeping (turn counter, ...)

``reset(episode_id)`` deletes exactly that namespace (SCAN + UNLINK), which is
what guarantees dataset-sample isolation.

No permanent data is stored here: the hybrid store falls back to SQLite on any
Redis failure, so a batch run survives a Redis restart mid-experiment.
"""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, List, Optional, Sequence

from ..models import ActiveSummary, IndexEntry, RawDialogRecord

logger = logging.getLogger(__name__)

# Redis commands that mutate a single key; used by reset() as a fallback when
# UNLINK (Redis >= 4.0) is not available.
_WINDOW_KEY = "window"
_ACTIVE_IDS_KEY = "active_ids"
_ACTIVE_HASH_KEY = "active"
_META_KEY = "meta"
_INDEX_IDS_KEY = "index_ids"      # LIST : index_id order (the rendered "titles")
_INDEX_HASH_KEY = "index"         # HASH : index_id -> JSON(IndexEntry)


class RedisUnavailable(RuntimeError):
    """Raised when Redis cannot serve a request (connection/command failure)."""


class RedisHotStore:
    """
    Namespaced Redis access for one bound episode at a time.

    Parameters
    ----------
    client:
        Optional pre-built redis client (tests inject a fake; production passes
        ``None`` and the store builds one from config).
    key_prefix:
        Namespace template.  Must contain ``{episode_id}``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "episode:{episode_id}",
        socket_timeout: float = 5.0,
        client: Any = None,
        required: bool = False,
    ):
        if "{episode_id}" not in key_prefix:
            raise ValueError("REDIS_KEY_PREFIX must contain the {episode_id} placeholder")
        _validate_placeholders(key_prefix)
        self.key_prefix = key_prefix
        self.required = required
        self.degraded = False
        self._bound_episode: Optional[str] = None
        self._namespace_cache: Dict[str, str] = {}
        self._redis_module = None
        self._client = client
        if self._client is None:
            self._client = self._build_client(host, port, db, password, socket_timeout)

    # ------------------------------------------------------------------ #
    # connection
    # ------------------------------------------------------------------ #
    def _build_client(self, host, port, db, password, socket_timeout):
        try:
            import redis  # type: ignore
        except ImportError:
            self._redis_module = None
            message = "redis-py is not installed (pip install redis); Redis hot layer disabled"
            if self.required:
                raise RedisUnavailable(message)
            logger.warning(message)
            self.degraded = True
            return None
        self._redis_module = redis
        try:
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
                decode_responses=True,
            )
            client.ping()  # fail fast: a dead server must not poison the run
            return client
        except Exception as exc:  # noqa: BLE001
            message = f"cannot connect to Redis at {host}:{port}/{db}: {exc}"
            if self.required:
                raise RedisUnavailable(message) from exc
            logger.warning("%s -- falling back to SQLite for hot data", message)
            self.degraded = True
            return None

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def client(self):
        return self._client

    def bind_episode(self, episode_id: str) -> None:
        self._bound_episode = episode_id

    @property
    def bound_episode(self) -> Optional[str]:
        return self._bound_episode

    # ------------------------------------------------------------------ #
    # key helpers
    # ------------------------------------------------------------------ #
    def namespace(self, episode_id: Optional[str] = None) -> str:
        episode = episode_id or self._bound_episode
        if episode is None:
            raise ValueError("no episode bound for Redis namespace")
        cached = self._namespace_cache.get(episode)
        if cached is None:
            cached = self.key_prefix.format(episode_id=episode).rstrip(":")
            self._namespace_cache[episode] = cached
        return cached

    def key(self, suffix: str, episode_id: Optional[str] = None) -> str:
        return f"{self.namespace(episode_id)}:{suffix}"

    def namespace_glob(self, episode_id: Optional[str] = None) -> str:
        return f"{self.namespace(episode_id)}:*"

    # ------------------------------------------------------------------ #
    # low-level guarded calls
    # ------------------------------------------------------------------ #
    def _call(self, method: str, *args, **kwargs):
        if self._client is None:
            raise RedisUnavailable("Redis client is not available")
        try:
            return getattr(self._client, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RedisUnavailable(f"Redis {method} failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # active chain
    # ------------------------------------------------------------------ #
    def save_active_all(self, summaries: Sequence[ActiveSummary], episode_id: Optional[str] = None) -> None:
        """Replace the whole active chain (used on rebuild / resync)."""
        ep = episode_id or self._bound_episode
        ids_key = self.key(_ACTIVE_IDS_KEY, ep)
        hash_key = self.key(_ACTIVE_HASH_KEY, ep)
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(ids_key, hash_key)
        if summaries:
            pipe.rpush(ids_key, *[s.summary_id for s in summaries])
            pipe.hset(hash_key, mapping={s.summary_id: _dumps_summary(s) for s in summaries})
        pipe.execute()

    def add_active_summary(self, summary: ActiveSummary, episode_id: Optional[str] = None) -> None:
        """
        Idempotent upsert: one slot per ``summary_id``.

        The manager legitimately re-writes an existing summary (e.g. when the lazy
        sweep changes its ``index_id``), so an unconditional ``RPUSH`` made the id
        list accumulate duplicates.  Every read then multiplied them through
        ``HMGET``, which is how a 16-summary chain reported 610k rows.  LREM before
        RPUSH keeps the list a set-with-order.
        """
        ep = episode_id or self._bound_episode
        ids_key = self.key(_ACTIVE_IDS_KEY, ep)
        pipe = self._client.pipeline(transaction=True)
        pipe.lrem(ids_key, 0, summary.summary_id)
        pipe.rpush(ids_key, summary.summary_id)
        pipe.hset(self.key(_ACTIVE_HASH_KEY, ep), summary.summary_id, _dumps_summary(summary))
        pipe.execute()

    def get_active_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ActiveSummary]:
        raw = self._call("hget", self.key(_ACTIVE_HASH_KEY, episode_id), summary_id)
        if raw is None:
            return None
        return _load_summary(raw)

    def remove_active_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ActiveSummary]:
        summary = self.get_active_summary(summary_id, episode_id)
        ep = episode_id or self._bound_episode
        pipe = self._client.pipeline(transaction=True)
        pipe.lrem(self.key(_ACTIVE_IDS_KEY, ep), 0, summary_id)
        pipe.hdel(self.key(_ACTIVE_HASH_KEY, ep), summary_id)
        pipe.execute()
        return summary

    def list_active_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        ep = episode_id or self._bound_episode
        ids = self._call("lrange", self.key(_ACTIVE_IDS_KEY, ep), 0, -1) or []
        if not ids:
            return []
        # Deduplicate defensively: a stale list may still hold duplicates written by
        # an older version, and one duplicate would otherwise be paid for on every
        # read.  Order is preserved (newest last).
        seen: set = set()
        unique_ids: List[str] = []
        for summary_id in ids:
            if summary_id in seen:
                continue
            seen.add(summary_id)
            unique_ids.append(summary_id)
        if len(unique_ids) != len(ids):
            logger.warning(
                "redis active id list for %s contained %d duplicate entries; ignored",
                ep, len(ids) - len(unique_ids),
            )
        raws = self._call("hmget", self.key(_ACTIVE_HASH_KEY, ep), *unique_ids)
        out: List[ActiveSummary] = []
        for summary_id, raw in zip(unique_ids, raws):
            if raw is None:  # partially lost hot key: caller will resync from SQLite
                raise RedisUnavailable(f"active summary payload missing for {summary_id}")
            out.append(_load_summary(raw))
        return out

    # ------------------------------------------------------------------ #
    # index entries (hot: rendered in every prompt)
    # ------------------------------------------------------------------ #
    def save_index_all(self, entries, episode_id: Optional[str] = None) -> None:
        ep = episode_id or self._bound_episode
        ids_key = self.key(_INDEX_IDS_KEY, ep)
        hash_key = self.key(_INDEX_HASH_KEY, ep)
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(ids_key, hash_key)
        if entries:
            pipe.rpush(ids_key, *[e.index_id for e in entries])
            pipe.hset(hash_key, mapping={e.index_id: dumps_index(e) for e in entries})
        pipe.execute()

    def add_index_entry(self, entry: IndexEntry, episode_id: Optional[str] = None) -> None:
        ep = episode_id or self._bound_episode
        pipe = self._client.pipeline(transaction=True)
        pipe.rpush(self.key(_INDEX_IDS_KEY, ep), entry.index_id)
        pipe.hset(self.key(_INDEX_HASH_KEY, ep), entry.index_id, dumps_index(entry))
        pipe.execute()

    def list_index_entries(self, episode_id: Optional[str] = None) -> List[IndexEntry]:
        ep = episode_id or self._bound_episode
        ids = self._call("lrange", self.key(_INDEX_IDS_KEY, ep), 0, -1) or []
        if not ids:
            return []
        raws = self._call("hmget", self.key(_INDEX_HASH_KEY, ep), *ids)
        out: List[IndexEntry] = []
        for index_id, raw in zip(ids, raws):
            if raw is None:
                raise RedisUnavailable(f"index payload missing for {index_id}")
            out.append(loads_index(raw))
        return out

    def get_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> Optional[IndexEntry]:
        raw = self._call("hget", self.key(_INDEX_HASH_KEY, episode_id), index_id)
        return loads_index(raw) if raw is not None else None

    def remove_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> None:
        ep = episode_id or self._bound_episode
        pipe = self._client.pipeline(transaction=True)
        pipe.lrem(self.key(_INDEX_IDS_KEY, ep), 0, index_id)
        pipe.hdel(self.key(_INDEX_HASH_KEY, ep), index_id)
        pipe.execute()

    # ------------------------------------------------------------------ #
    # sliding window
    # ------------------------------------------------------------------ #
    def set_window(self, records: Sequence[RawDialogRecord], episode_id: Optional[str] = None) -> None:
        ep = episode_id or self._bound_episode
        key = self.key(_WINDOW_KEY, ep)
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(key)
        if records:
            pipe.rpush(key, *[dumps_record(r) for r in records])
        pipe.execute()

    def append_window(self, record: RawDialogRecord, keep: int, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        ep = episode_id or self._bound_episode
        key = self.key(_WINDOW_KEY, ep)
        pipe = self._client.pipeline(transaction=True)
        pipe.rpush(key, dumps_record(record))
        pipe.ltrim(key, -keep, -1)
        pipe.lrange(key, 0, -1)
        _, _, raws = pipe.execute()
        return [loads_record(r) for r in raws or []]

    def get_window(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        raw = self._call("lrange", self.key(_WINDOW_KEY, episode_id), 0, -1) or []
        return [loads_record(r) for r in raw]

    def set_turn_meta(self, turn_index: int, episode_id: Optional[str] = None) -> None:
        self._call("hset", self.key(_META_KEY, episode_id), "turn_index", int(turn_index))

    def get_turn_meta(self, episode_id: Optional[str] = None) -> int:
        raw = self._call("hget", self.key(_META_KEY, episode_id), "turn_index")
        try:
            return int(raw) if raw is not None else -1
        except (TypeError, ValueError):
            return -1

    # ------------------------------------------------------------------ #
    # reset / introspection
    # ------------------------------------------------------------------ #
    def list_episode_keys(self, episode_id: Optional[str] = None) -> List[str]:
        """SCAN (never KEYS) the episode namespace -- also used by unit tests."""
        pattern = self.namespace_glob(episode_id)
        found: List[str] = []
        cursor = 0
        while True:
            cursor, keys = self._call("scan", cursor=cursor, match=pattern, count=500)
            found.extend(keys or [])
            if cursor in (0, "0"):
                break
        return found

    def clear_episode(self, episode_id: Optional[str] = None) -> int:
        """Delete every key of one episode namespace; returns the number deleted."""
        keys = self.list_episode_keys(episode_id)
        if not keys:
            return 0
        try:
            self._call("unlink", *keys)
        except RedisUnavailable:
            self._call("delete", *keys)
        return len(keys)

    def scan_all_episode_namespaces(self) -> List[str]:
        """Glob pattern matching every episode namespace (for `--flush` tooling)."""
        fields = [f for _, f, _, _ in string.Formatter().parse(self.key_prefix) if f]
        sample = {f: "*" for f in fields}
        return [f"{self.key_prefix.format(**sample).rstrip(':')}:*"]

    def flush_all_episodes(self) -> int:
        """Maintenance helper: delete keys of *all* episodes (never used per-episode)."""
        deleted = 0
        for pattern in self.scan_all_episode_namespaces():
            cursor = 0
            while True:
                cursor, keys = self._call("scan", cursor=cursor, match=pattern, count=500)
                if keys:
                    self._call("delete", *keys)
                    deleted += len(keys)
                if cursor in (0, "0"):
                    break
        return deleted

    def ping(self) -> bool:
        try:
            return bool(self._call("ping"))
        except RedisUnavailable:
            return False


# --------------------------------------------------------------------------- #
# serialisation helpers
# --------------------------------------------------------------------------- #
def _dumps_summary(summary: ActiveSummary) -> str:
    import json

    return json.dumps(summary.to_dict(), ensure_ascii=False, default=str)


def _load_summary(raw: Any) -> ActiveSummary:
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return ActiveSummary(
        summary_id=data["summary_id"],
        text=data.get("text", ""),
        override_ids=list(data.get("override_ids") or []),
        timestamp=data.get("timestamp") or 0.0,
        raw_ref_id=data.get("raw_ref_id", ""),
        episode_id=data.get("episode_id", ""),
        seq=int(data.get("seq", -1)),
        origin=data.get("origin", "event"),
        raw_ref_ids=list(data.get("raw_ref_ids") or []),
        merged_from=list(data.get("merged_from") or []),
        # ``fact_keys`` and ``index_id`` are *not* recoverable from ``text`` (the
        # [FACTS: ...] line is stripped before storage) and they are load-bearing:
        # fact_keys drive the fact-safety rule and the index digest, index_id drives
        # the lazy sweep and the post-fold member re-pointing.  Dropping them here
        # made the hot path behave differently from the cold one.
        fact_keys=list(data.get("fact_keys") or []),
        index_id=data.get("index_id", "") or "",
    )


def dumps_index(entry: IndexEntry) -> str:
    import json

    return json.dumps(entry.to_dict(), ensure_ascii=False, default=str)


def loads_index(raw: Any) -> IndexEntry:
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    d = json.loads(raw)
    return IndexEntry(
        index_id=d["index_id"],
        title=d.get("title", ""),
        members=list(d.get("members") or []),
        span_start=d.get("span_start") or 0.0,
        span_end=d.get("span_end") or 0.0,
        turn_start=int(d.get("turn_start", -1)),
        turn_end=int(d.get("turn_end", -1)),
        fact_keys=list(d.get("fact_keys") or []),
        episode_id=d.get("episode_id", ""),
        seq=int(d.get("seq", -1)),
        timestamp=d.get("timestamp") or 0.0,
        member_summaries=list(d.get("member_summaries") or []),
        # ``previews`` is the *cheap* content of an index line.  Without it
        # ``IndexEntry.render`` falls back to ``member_summaries`` (full summary
        # lines), which makes the index layer cost what the chain it replaced cost.
        # ``child_index_ids`` marks a super index; losing it breaks fold bookkeeping.
        previews=list(d.get("previews") or []),
        child_index_ids=list(d.get("child_index_ids") or []),
    )


def dumps_record(record: RawDialogRecord) -> str:
    import json

    return json.dumps(record.to_dict(), ensure_ascii=False, default=str)


def loads_record(raw: Any) -> RawDialogRecord:
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return RawDialogRecord(
        reference_id=data["reference_id"],
        user_msg=data.get("user_msg", ""),
        agent_msg=data.get("agent_msg", ""),
        timestamp=data.get("timestamp") or 0.0,
        episode_id=data.get("episode_id", ""),
        turn_index=int(data.get("turn_index", -1)),
        meta=data.get("meta") or {},
    )


def _validate_placeholders(key_prefix: str) -> None:
    fields = [f for _, f, _, _ in string.Formatter().parse(key_prefix) if f]
    unknown = [f for f in fields if f != "episode_id"]
    if unknown:
        raise ValueError(f"unsupported placeholder(s) in REDIS_KEY_PREFIX: {unknown}")
