"""
Cold persistence layer: SQLite.

Owns **all** permanent experiment data:

``raw_records``          every original dialogue turn, forever
``archived_summaries``   every overridden / capacity-evicted summary, forever
``active_summaries``     mirror of the hot active chain (crash recovery only)
``sliding_window``       mirror of the recent window (crash recovery only)
``episode_progress``     batch checkpointing: which episodes are finished
``experiment_runs``      one row per (run, episode, system) result summary

The two mirror tables exist for one reason: Redis is explicitly *not* a
durability layer, so a crashed batch run must be able to rebuild the hot state of
the episode it was in the middle of.  They are written on every turn but are
small (a handful of rows per episode).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from ..models import ActiveSummary, ArchivedSummary, RawDialogRecord, pack_refs, unpack_refs

logger = logging.getLogger(__name__)

def _as_int(value, default: int) -> int:
    """Int conversion that keeps a legitimate ``0`` (``value or default`` would not)."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_records (
    reference_id TEXT PRIMARY KEY,
    episode_id   TEXT NOT NULL DEFAULT '',
    turn_index   INTEGER NOT NULL DEFAULT -1,
    user_msg     TEXT NOT NULL DEFAULT '',
    agent_msg    TEXT NOT NULL DEFAULT '',
    timestamp    REAL NOT NULL DEFAULT 0,
    meta         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_raw_episode ON raw_records(episode_id, turn_index);

CREATE TABLE IF NOT EXISTS archived_summaries (
    summary_id     TEXT PRIMARY KEY,
    episode_id     TEXT NOT NULL DEFAULT '',
    seq            INTEGER NOT NULL DEFAULT -1,
    text           TEXT NOT NULL DEFAULT '',
    override_ids   TEXT NOT NULL DEFAULT '',
    timestamp      REAL NOT NULL DEFAULT 0,
    raw_ref_id     TEXT NOT NULL DEFAULT '',
    raw_ref_ids    TEXT NOT NULL DEFAULT '',
    is_overridden  INTEGER NOT NULL DEFAULT 0,
    superseded_by  TEXT,
    archive_reason TEXT NOT NULL DEFAULT 'overridden',
    origin         TEXT NOT NULL DEFAULT 'event',
    merged_from    TEXT NOT NULL DEFAULT '',
    fact_keys      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_arch_episode ON archived_summaries(episode_id, seq);

CREATE TABLE IF NOT EXISTS active_summaries (
    summary_id  TEXT PRIMARY KEY,
    episode_id  TEXT NOT NULL DEFAULT '',
    seq         INTEGER NOT NULL DEFAULT -1,
    text        TEXT NOT NULL DEFAULT '',
    override_ids TEXT NOT NULL DEFAULT '',
    timestamp   REAL NOT NULL DEFAULT 0,
    raw_ref_id  TEXT NOT NULL DEFAULT '',
    raw_ref_ids TEXT NOT NULL DEFAULT '',
    origin      TEXT NOT NULL DEFAULT 'event',
    merged_from TEXT NOT NULL DEFAULT '',
    fact_keys   TEXT NOT NULL DEFAULT '',
    index_id    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_active_episode ON active_summaries(episode_id, seq);

CREATE TABLE IF NOT EXISTS index_entries (
    index_id      TEXT PRIMARY KEY,
    episode_id    TEXT NOT NULL DEFAULT '',
    seq           INTEGER NOT NULL DEFAULT -1,
    title         TEXT NOT NULL DEFAULT '',
    theme         TEXT NOT NULL DEFAULT '',
    members       TEXT NOT NULL DEFAULT '',
    member_lines  TEXT NOT NULL DEFAULT '',
    span_start    REAL NOT NULL DEFAULT 0,
    span_end      REAL NOT NULL DEFAULT 0,
    turn_start    INTEGER NOT NULL DEFAULT -1,
    turn_end      INTEGER NOT NULL DEFAULT -1,
    fact_keys     TEXT NOT NULL DEFAULT '',
    previews      TEXT NOT NULL DEFAULT '',
    child_index_ids TEXT NOT NULL DEFAULT '',
    timestamp     REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_index_episode ON index_entries(episode_id, seq);

CREATE TABLE IF NOT EXISTS fact_ledger (
    fact_id       TEXT PRIMARY KEY,
    episode_id    TEXT NOT NULL DEFAULT '',
    summary_id    TEXT NOT NULL DEFAULT '',
    slot          TEXT NOT NULL DEFAULT '',
    value         TEXT NOT NULL DEFAULT '',
    seq           INTEGER NOT NULL DEFAULT -1,
    observed_turn INTEGER NOT NULL DEFAULT -1,
    evidence      TEXT NOT NULL DEFAULT '',
    reason        TEXT NOT NULL DEFAULT '',
    superseded_by TEXT NOT NULL DEFAULT '',
    residual_mentions INTEGER NOT NULL DEFAULT 0,
    erased        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ledger_episode ON fact_ledger(episode_id, seq);

CREATE TABLE IF NOT EXISTS sliding_window (
    episode_id   TEXT NOT NULL,
    position     INTEGER NOT NULL,
    reference_id TEXT NOT NULL,
    user_msg     TEXT NOT NULL DEFAULT '',
    agent_msg    TEXT NOT NULL DEFAULT '',
    turn_index   INTEGER NOT NULL DEFAULT -1,
    timestamp    REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (episode_id, position)
);

CREATE TABLE IF NOT EXISTS episode_progress (
    run_id      TEXT NOT NULL DEFAULT 'default',
    episode_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'done',
    num_turns   INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0,
    detail      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, episode_id)
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id      TEXT NOT NULL,
    system_name TEXT NOT NULL,
    episode_id  TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT 0,
    payload     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, system_name, episode_id)
);
"""


class SQLiteColdStore:
    """Thin, thread-safe SQLite wrapper. No ORM, no magic."""

    backend_name = "sqlite_cold"

    def __init__(self, path: str = "./exp_memory.db", recent_window_turns: int = 4):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self.recent_window_turns = recent_window_turns
        self._bound_episode: Optional[str] = None
        self._current_turn_index: int = -1
        self.reset_calls = 0
        self._configure()
        self._create_schema()

    # ------------------------------------------------------------------ #
    # Episode binding and lifecycle
    #
    # These exist so this store can drive ``MemoryManager`` on its own, with no
    # Redis and no broker to install.  The audit surface never needed a hot
    # layer, and neither does a single-process run: the cold store is the source
    # of truth, so it can also be the only store.
    #
    # Note the deliberate asymmetry with ``BaseMemoryStore``: the query methods
    # here keep their ``episode_id=None`` meaning of "every episode", which is
    # what makes the store useful as a whole-file query surface.  The bound
    # episode is only needed by the turn bookkeeping below.
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
        """Monotonic turn ordinal inside the bound episode."""
        self._current_turn_index += 1
        return self._current_turn_index

    @property
    def current_turn_index(self) -> int:
        return self._current_turn_index

    def reset(self, episode_id: Optional[str] = None, flush_hot_keys: bool = False) -> None:
        """
        Start a clean episode: drop the working state, keep the permanent data.

        ``archived_summaries`` and ``raw_records`` are untouched, exactly as in the
        hybrid store -- an evaluation sample must not be able to inherit the
        previous one, but permanence is the whole point of the other two layers.
        ``flush_hot_keys`` is accepted and ignored: there is no hot layer here to
        flush, and inventing one would be worse than saying so.
        """
        episode = episode_id or self._bound_episode
        if episode is None:
            raise ValueError("reset() needs an episode_id (or bind one first)")
        self.reset_calls += 1
        self.clear_active_summaries(episode)
        self.clear_indexes(episode)
        self.clear_window(episode)
        logger.debug("reset(%s): working state cleared, archive/raw kept", episode)

    def purge_episode(self, episode_id: str) -> None:
        """Wipe one episode completely, archive and raw records included."""
        self.reset(episode_id=episode_id)
        self.drop_episode(episode_id, keep_archive=False)

    def checksum(self) -> Dict[str, int]:
        """Cheap integrity summary, handy in logs and checkpoints."""
        return {
            "active": len(self.list_active_summaries(self._bound_episode)),
            "window": len(self.get_window(self._bound_episode)),
            "raw": self.count_raw_records(self._bound_episode),
            "archived": self.count_archived_summaries(self._bound_episode),
        }

    # ------------------------------------------------------------------ #
    # connection plumbing
    # ------------------------------------------------------------------ #
    def _configure(self) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")       # concurrent readers
            cur.execute("PRAGMA synchronous=NORMAL")     # batch-run friendly
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
        finally:
            cur.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """
        Add columns introduced after a database was first created.

        Batch runs are expensive, so a schema change must never force a full
        re-run: existing files get the new columns in place.
        """
        additions = {
            "archived_summaries": {"fact_keys": "TEXT NOT NULL DEFAULT ''"},
            "active_summaries": {"fact_keys": "TEXT NOT NULL DEFAULT ''",
                                 "index_id": "TEXT NOT NULL DEFAULT ''"},
            # previews / child_index_ids were previously dropped on the SQLite
            # round-trip.  Losing ``previews`` made every index line render the
            # full member summaries instead of a ~10-token snippet; losing
            # ``child_index_ids`` broke super-index bookkeeping.
            "fact_ledger": {"residual_mentions": "INTEGER NOT NULL DEFAULT 0",
                            "erased": "INTEGER NOT NULL DEFAULT 0"},
            "index_entries": {"previews": "TEXT NOT NULL DEFAULT ''",
                              "child_index_ids": "TEXT NOT NULL DEFAULT ''",
                              "theme": "TEXT NOT NULL DEFAULT ''"},
        }
        for table, columns in additions.items():
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, ddl in columns.items():
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                    logger.info("schema migration: added %s.%s", table, name)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            rows = cur.fetchall()
            cur.close()
            return rows

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ #
    # raw records
    # ------------------------------------------------------------------ #
    def add_raw_record(
        self, record: RawDialogRecord, episode_id: Optional[str] = None
    ) -> None:
        record.episode_id = record.episode_id or self._ep(episode_id)
        self.execute(
            """INSERT INTO raw_records
                   (reference_id, episode_id, turn_index, user_msg, agent_msg, timestamp, meta)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(reference_id) DO UPDATE SET
                   episode_id=excluded.episode_id, turn_index=excluded.turn_index,
                   user_msg=excluded.user_msg, agent_msg=excluded.agent_msg,
                   timestamp=excluded.timestamp, meta=excluded.meta""",
            (
                record.reference_id,
                record.episode_id,
                int(record.turn_index),
                record.user_msg,
                record.agent_msg,
                float(record.timestamp),
                json.dumps(record.meta or {}, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _row_to_raw(row: sqlite3.Row) -> RawDialogRecord:
        try:
            meta = json.loads(row["meta"]) if row["meta"] else {}
        except (TypeError, ValueError):
            meta = {}
        return RawDialogRecord(
            reference_id=row["reference_id"],
            user_msg=row["user_msg"],
            agent_msg=row["agent_msg"],
            timestamp=row["timestamp"],
            episode_id=row["episode_id"],
            turn_index=row["turn_index"],
            meta=meta,
        )

    def get_raw_record(
        self, reference_id: str, episode_id: Optional[str] = None
    ) -> Optional[RawDialogRecord]:
        """
        Exact-id lookup.

        Accepts ``episode_id`` to match the :class:`BaseMemoryStore` contract: raw
        ids already embed the episode, so the parameter is a scope assertion, not a
        filter.  Without it this cold store could not be used as a standalone memory
        store (every caller passes ``episode_id=`` as a keyword), which is exactly
        the kind of hot/cold asymmetry that keeps producing surprises.
        """
        row = self.query_one("SELECT * FROM raw_records WHERE reference_id=?", (reference_id,))
        if row is None:
            return None
        record = self._row_to_raw(row)
        if episode_id is not None and record.episode_id and record.episode_id != episode_id:
            return None
        return record

    def list_raw_records(self, episode_id: Optional[str] = None) -> List[RawDialogRecord]:
        if episode_id is None:
            rows = self.query("SELECT * FROM raw_records ORDER BY episode_id, turn_index, timestamp")
        else:
            rows = self.query(
                "SELECT * FROM raw_records WHERE episode_id=? ORDER BY turn_index, timestamp",
                (episode_id,),
            )
        return [self._row_to_raw(r) for r in rows]

    def count_raw_records(self, episode_id: Optional[str] = None) -> int:
        if episode_id is None:
            row = self.query_one("SELECT COUNT(*) AS n FROM raw_records")
        else:
            row = self.query_one("SELECT COUNT(*) AS n FROM raw_records WHERE episode_id=?", (episode_id,))
        return int(row["n"]) if row else 0

    def delete_raw_record(
        self, reference_id: str, episode_id: Optional[str] = None
    ) -> None:
        """
        Hard-delete one raw record.

        Used exclusively by the MemGPT-style destructive baseline; the three-layer
        system's raw store is permanent by contract.
        """
        self.execute("DELETE FROM raw_records WHERE reference_id=?", (reference_id,))

    # ------------------------------------------------------------------ #
    # archived summaries
    # ------------------------------------------------------------------ #
    def add_archived_summary(
        self, archived: ArchivedSummary, episode_id: Optional[str] = None
    ) -> None:
        archived.episode_id = archived.episode_id or self._ep(episode_id)
        self.execute(
            """INSERT INTO archived_summaries
                   (summary_id, episode_id, seq, text, override_ids, timestamp, raw_ref_id,
                    raw_ref_ids, is_overridden, superseded_by, archive_reason, origin, merged_from,
                    fact_keys)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(summary_id) DO UPDATE SET
                   episode_id=excluded.episode_id, seq=excluded.seq, text=excluded.text,
                   override_ids=excluded.override_ids, timestamp=excluded.timestamp,
                   raw_ref_id=excluded.raw_ref_id, raw_ref_ids=excluded.raw_ref_ids,
                   is_overridden=excluded.is_overridden, superseded_by=excluded.superseded_by,
                   archive_reason=excluded.archive_reason, origin=excluded.origin,
                   merged_from=excluded.merged_from, fact_keys=excluded.fact_keys""",
            (
                archived.summary_id,
                archived.episode_id,
                int(archived.seq),
                archived.text,
                pack_refs(archived.override_ids),
                float(archived.timestamp),
                archived.raw_ref_id,
                pack_refs(archived.raw_ref_ids),
                1 if archived.is_overridden else 0,
                archived.superseded_by,
                archived.archive_reason,
                archived.origin,
                pack_refs(archived.merged_from),
                pack_refs(archived.fact_keys),
            ),
        )

    @staticmethod
    def _row_to_archived(row: sqlite3.Row) -> ArchivedSummary:
        raw_refs = unpack_refs(row["raw_ref_ids"]) or unpack_refs(row["raw_ref_id"])
        return ArchivedSummary(
            summary_id=row["summary_id"],
            text=row["text"],
            override_ids=unpack_refs(row["override_ids"]),
            timestamp=row["timestamp"],
            raw_ref_id=row["raw_ref_id"],
            is_overridden=bool(row["is_overridden"]),
            episode_id=row["episode_id"],
            seq=row["seq"],
            origin=row["origin"],
            superseded_by=row["superseded_by"],
            archive_reason=row["archive_reason"],
            raw_ref_ids=raw_refs,
            merged_from=unpack_refs(row["merged_from"]),
            fact_keys=unpack_refs(row["fact_keys"]) if "fact_keys" in row.keys() else [],
        )

    def get_archived_summary(self, summary_id: str, episode_id: Optional[str] = None) -> Optional[ArchivedSummary]:
        row = self.query_one("SELECT * FROM archived_summaries WHERE summary_id=?", (summary_id,))
        if row is None:
            return None
        archived = self._row_to_archived(row)
        if episode_id is not None and archived.episode_id != episode_id:
            return None
        return archived

    def list_archived_summaries(
        self, episode_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ArchivedSummary]:
        if episode_id is None:
            rows = self.query("SELECT * FROM archived_summaries ORDER BY timestamp, seq")
        else:
            rows = self.query(
                "SELECT * FROM archived_summaries WHERE episode_id=? ORDER BY timestamp, seq",
                (episode_id,),
            )
        items = [self._row_to_archived(r) for r in rows]
        if limit is not None:
            items = items[-limit:]
        return items

    def count_archived_summaries(self, episode_id: Optional[str] = None) -> int:
        if episode_id is None:
            row = self.query_one("SELECT COUNT(*) AS n FROM archived_summaries")
        else:
            row = self.query_one(
                "SELECT COUNT(*) AS n FROM archived_summaries WHERE episode_id=?", (episode_id,)
            )
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ #
    # active chain mirror (crash recovery only)
    # ------------------------------------------------------------------ #
    def add_active_summary(
        self, summary: ActiveSummary, episode_id: Optional[str] = None
    ) -> None:
        summary.episode_id = summary.episode_id or self._ep(episode_id)
        self.execute(
            """INSERT INTO active_summaries
                   (summary_id, episode_id, seq, text, override_ids, timestamp,
                    raw_ref_id, raw_ref_ids, origin, merged_from, fact_keys, index_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(summary_id) DO UPDATE SET
                   episode_id=excluded.episode_id, seq=excluded.seq, text=excluded.text,
                   override_ids=excluded.override_ids, timestamp=excluded.timestamp,
                   raw_ref_id=excluded.raw_ref_id, raw_ref_ids=excluded.raw_ref_ids,
                   origin=excluded.origin, merged_from=excluded.merged_from,
                   fact_keys=excluded.fact_keys, index_id=excluded.index_id""",
            (
                summary.summary_id,
                summary.episode_id,
                int(summary.seq),
                summary.text,
                pack_refs(summary.override_ids),
                float(summary.timestamp),
                summary.raw_ref_id,
                pack_refs(summary.raw_ref_ids),
                summary.origin,
                pack_refs(summary.merged_from),
                pack_refs(summary.fact_keys),
                summary.index_id,
            ),
        )

    def remove_active_summary(
        self, summary_id: str, episode_id: Optional[str] = None
    ) -> Optional[ActiveSummary]:
        """
        Delete an active summary and return what was deleted.

        Returning the object is not a convenience -- it is the contract, and the
        override path depends on it.  ``MemoryManager`` archives the summary it
        just removed, and it reads that summary from this return value:

            old = store.remove_active_summary(overridden_id, ...)
            if old is None: continue          # nothing to archive
            store.add_archived_summary(ArchivedSummary.from_active(old, ...))

        A DELETE that returned nothing therefore did not merely lose an object: it
        made every override skip its archive step, so the overridden text vanished
        with no forward pointer.  That is silent loss, and it is precisely what I1
        reports -- which is how this was found.  The hybrid store masked it by
        reading the summary itself before delegating here.

        Read and delete share one lock so no concurrent writer can archive a
        summary that has already been replaced.
        """
        params: tuple = (summary_id,)
        where = "summary_id=?"
        if episode_id is not None:
            # Summary ids are unique per episode, so scoping the delete prevents
            # removing another episode's summary that happens to share an id.
            where += " AND episode_id=?"
            params = (summary_id, episode_id)
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM active_summaries WHERE {where}", params
            ).fetchone()
            if row is None:
                return None
            summary = self._row_to_active(row)
            self._conn.execute(f"DELETE FROM active_summaries WHERE {where}", params)
            self._conn.commit()
        return summary

    # ------------------------------------------------------------------ #
    # read-side conveniences from the BaseMemoryStore contract
    # ------------------------------------------------------------------ #
    # This class is not a BaseMemoryStore (it has no episode binding), but tools that
    # only *read* -- an audit sidecar, a report generator -- need the same conveniences
    # a memory store offers.  They were missing, so a cold store could not stand in
    # for one; adding them here removes that asymmetry instead of working around it at
    # every call site.
    def get_active_summary(
        self, summary_id: str, episode_id: Optional[str] = None
    ) -> Optional[ActiveSummary]:
        for summary in self.list_active_summaries(episode_id):
            if summary.summary_id == summary_id:
                return summary
        return None

    def get_raw_records(
        self, reference_ids: Sequence[str], episode_id: Optional[str] = None
    ) -> List[RawDialogRecord]:
        out: List[RawDialogRecord] = []
        for reference_id in reference_ids:
            record = self.get_raw_record(reference_id, episode_id)
            if record is not None:
                out.append(record)
        return out

    def chain_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        """Summaries not filed under any index -- the same rule the prompt uses."""
        title_members = set()
        for entry in self.list_index_entries(episode_id):
            title_members.update(entry.members)
        return [
            summary for summary in self.list_active_summaries(episode_id)
            if summary.summary_id not in title_members
        ]

    @staticmethod
    def _row_to_active(row: sqlite3.Row) -> ActiveSummary:
        return ActiveSummary(
            summary_id=row["summary_id"],
            text=row["text"],
            override_ids=unpack_refs(row["override_ids"]),
            timestamp=row["timestamp"],
            raw_ref_id=row["raw_ref_id"],
            episode_id=row["episode_id"],
            seq=row["seq"],
            origin=row["origin"],
            raw_ref_ids=unpack_refs(row["raw_ref_ids"]) or unpack_refs(row["raw_ref_id"]),
            merged_from=unpack_refs(row["merged_from"]),
            fact_keys=unpack_refs(row["fact_keys"]) if "fact_keys" in row.keys() else [],
            index_id=(row["index_id"] if "index_id" in row.keys() else "") or "",
        )

    def list_active_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        if episode_id is None:
            rows = self.query("SELECT * FROM active_summaries ORDER BY timestamp, seq")
        else:
            rows = self.query(
                "SELECT * FROM active_summaries WHERE episode_id=? ORDER BY timestamp, seq",
                (episode_id,),
            )
        return [self._row_to_active(row) for row in rows]

    # ------------------------------------------------------------------ #
    # index entries (level 2 of layer 1)
    # ------------------------------------------------------------------ #
    def add_index_entry(self, entry, episode_id: Optional[str] = None) -> None:
        entry.episode_id = entry.episode_id or self._ep(episode_id)
        import json

        self.execute(
            """INSERT INTO index_entries
                   (index_id, episode_id, seq, title, theme, members, member_lines,
                    span_start, span_end, turn_start, turn_end, fact_keys,
                    previews, child_index_ids, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(index_id) DO UPDATE SET
                   episode_id=excluded.episode_id, seq=excluded.seq, title=excluded.title,
                   theme=excluded.theme,
                   members=excluded.members, member_lines=excluded.member_lines,
                   span_start=excluded.span_start, span_end=excluded.span_end,
                   turn_start=excluded.turn_start, turn_end=excluded.turn_end,
                   fact_keys=excluded.fact_keys, previews=excluded.previews,
                   child_index_ids=excluded.child_index_ids, timestamp=excluded.timestamp""",
            (
                entry.index_id, entry.episode_id, int(entry.seq), entry.title,
                getattr(entry, "theme", "") or "",
                pack_refs(entry.members), pack_refs(entry.member_summaries),
                float(entry.span_start), float(entry.span_end),
                int(entry.turn_start), int(entry.turn_end),
                pack_refs(entry.fact_keys),
                # JSON, not pack_refs: a preview is free text and may contain "|".
                json.dumps(list(entry.previews or []), ensure_ascii=False),
                json.dumps(list(entry.child_index_ids or []), ensure_ascii=False),
                float(entry.timestamp),
            ),
        )

    @staticmethod
    def _row_to_index(row) -> "IndexEntry":
        import json

        from ..models import IndexEntry

        keys = row.keys()

        def _json_list(column: str) -> List[str]:
            if column not in keys or not row[column]:
                return []
            try:
                return [str(item) for item in json.loads(row[column])]
            except (TypeError, ValueError):
                return []

        return IndexEntry(
            index_id=row["index_id"],
            title=row["title"],
            theme=row["theme"] if "theme" in keys else "",
            members=unpack_refs(row["members"]),
            span_start=row["span_start"],
            span_end=row["span_end"],
            turn_start=row["turn_start"],
            turn_end=row["turn_end"],
            fact_keys=unpack_refs(row["fact_keys"]) if "fact_keys" in keys else [],
            episode_id=row["episode_id"],
            seq=row["seq"],
            timestamp=row["timestamp"],
            member_summaries=unpack_refs(row["member_lines"]) if "member_lines" in keys else [],
            previews=_json_list("previews"),
            child_index_ids=_json_list("child_index_ids"),
        )

    def list_index_entries(self, episode_id: Optional[str] = None) -> List["IndexEntry"]:
        if episode_id is None:
            rows = self.query("SELECT * FROM index_entries ORDER BY timestamp, seq")
        else:
            rows = self.query(
                "SELECT * FROM index_entries WHERE episode_id=? ORDER BY timestamp, seq",
                (episode_id,),
            )
        return [self._row_to_index(r) for r in rows]

    def get_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> Optional["IndexEntry"]:
        row = self.query_one("SELECT * FROM index_entries WHERE index_id=?", (index_id,))
        if row is None:
            return None
        entry = self._row_to_index(row)
        if episode_id is not None and entry.episode_id != episode_id:
            return None
        return entry

    def summaries_under_index(
        self, index_id: str, episode_id: Optional[str] = None
    ) -> List[ActiveSummary]:
        """
        Resolve an index to its member summaries (the catalogue behind a title).

        Implemented here rather than inherited because this store does not extend
        ``BaseMemoryStore``.  Members are filtered against the *live* active
        summaries on purpose: an index that still lists an overridden member is
        exactly the stale-pointer failure this lookup must not paper over.
        """
        entry = self.get_index_entry(index_id, episode_id)
        if entry is None:
            return []
        members = {s.summary_id: s for s in self.list_active_summaries(entry.episode_id)}
        return [members[sid] for sid in entry.members if sid in members]

    def remove_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> None:
        self.execute("DELETE FROM index_entries WHERE index_id=?", (index_id,))
    def clear_indexes(self, episode_id: str) -> None:
        self.execute("DELETE FROM index_entries WHERE episode_id=?", (episode_id,))

    def clear_active_summaries(self, episode_id: str) -> None:
        self.execute("DELETE FROM active_summaries WHERE episode_id=?", (episode_id,))
        self.execute("DELETE FROM index_entries WHERE episode_id=?", (episode_id,))

    # ------------------------------------------------------------------ #
    # fact ledger (append-only audit record)
    # ------------------------------------------------------------------ #
    def append_fact_ledger(self, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        with self._lock:
            cursor = self._conn.executemany(
                """INSERT OR IGNORE INTO fact_ledger
                       (fact_id, episode_id, summary_id, slot, value, seq,
                        observed_turn, evidence, reason, superseded_by,
                        residual_mentions, erased)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        row.get("fact_id", ""), row.get("episode_id", ""),
                        row.get("summary_id", ""), row.get("slot", ""), row.get("value", ""),
                        # NOT `int(v or -1)`: that maps a legitimate 0 to -1, which
                        # silently rewrote every fact recorded on turn 0 (the first
                        # turn of every episode) as turn -1 on disk only.
                        _as_int(row.get("seq"), -1), _as_int(row.get("observed_turn"), -1),
                        row.get("evidence", ""), row.get("reason", ""),
                        row.get("superseded_by", ""),
                        _as_int(row.get("residual_mentions"), 0),
                        _as_int(row.get("erased"), 0),
                    )
                    for row in rows
                ],
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def list_fact_ledger(self, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if episode_id is None:
            rows = self.query("SELECT * FROM fact_ledger ORDER BY seq, fact_id")
        else:
            rows = self.query(
                "SELECT * FROM fact_ledger WHERE episode_id=? ORDER BY seq, fact_id",
                (episode_id,),
            )
        return [dict(row) for row in rows]

    def update_fact_ledger_reason(
        self, summary_id: str, reason: str, superseded_by: str = "",
        episode_id: Optional[str] = None,
    ) -> int:
        if episode_id is None:
            cursor = self.execute(
                "UPDATE fact_ledger SET reason=?, superseded_by=? "
                "WHERE summary_id=? AND reason=''",
                (reason, superseded_by, summary_id),
            )
        else:
            cursor = self.execute(
                "UPDATE fact_ledger SET reason=?, superseded_by=? "
                "WHERE summary_id=? AND episode_id=? AND reason=''",
                (reason, superseded_by, summary_id, episode_id),
            )
        return int(cursor.rowcount or 0)

    def erase_fact_ledger(self, rows: Sequence[Dict[str, Any]], episode_id: Optional[str] = None) -> int:
        """Tombstone rows: blank the value, keep the evidence pointer for I5."""
        if not rows:
            return 0
        with self._lock:
            cursor = self._conn.executemany(
                """UPDATE fact_ledger
                      SET erased=1, reason=?, value='', residual_mentions=?
                    WHERE fact_id=?""",
                [
                    (
                        row.get("reason", "erased"),
                        _as_int(row.get("residual_mentions"), 0),
                        row.get("fact_id", ""),
                    )
                    for row in rows
                ],
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def clear_fact_ledger(self, episode_id: str) -> int:
        cursor = self.execute("DELETE FROM fact_ledger WHERE episode_id=?", (episode_id,))
        return int(cursor.rowcount or 0)

    def count_fact_ledger(self, episode_id: Optional[str] = None) -> int:
        return len(self.list_fact_ledger(episode_id))

    def list_ledger_episodes(self) -> List[str]:
        """Episodes that have an audit ledger -- what a sidecar can serve."""
        rows = self.query("SELECT DISTINCT episode_id FROM fact_ledger ORDER BY episode_id")
        return [row["episode_id"] for row in rows if row["episode_id"]]

    def clear_archive(self, episode_id: str) -> int:
        """Drop this episode's archived summaries; raw records stay (permanent)."""
        cursor = self.execute("DELETE FROM archived_summaries WHERE episode_id=?", (episode_id,))
        return int(cursor.rowcount or 0)

    # ------------------------------------------------------------------ #
    # sliding window mirror (crash recovery only)
    # ------------------------------------------------------------------ #
    def append_window_record(
        self, record: RawDialogRecord, episode_id: Optional[str] = None
    ) -> List[RawDialogRecord]:
        """
        Append one turn to the window and trim it to ``recent_window_turns``.

        Reading and rewriting the whole window is not the cheapest possible
        implementation, but the window is a handful of rows and going through
        ``set_window`` keeps the trim rule in exactly one place -- a second,
        subtly different trim (``window[-0:]`` returns the *whole* list) is how
        sliding windows quietly stop sliding.
        """
        episode = record.episode_id or self._ep(episode_id)
        window = self.get_window(episode)
        window.append(record)
        limit = self.recent_window_turns
        if limit and len(window) > limit:
            window = window[-limit:]
        self.set_window(episode, window)
        return list(window)

    def set_window(self, episode_id: str, records: Sequence[RawDialogRecord]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sliding_window WHERE episode_id=?", (episode_id,))
            self._conn.executemany(
                """INSERT INTO sliding_window
                       (episode_id, position, reference_id, user_msg, agent_msg, turn_index, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (
                        episode_id,
                        pos,
                        rec.reference_id,
                        rec.user_msg,
                        rec.agent_msg,
                        int(rec.turn_index),
                        float(rec.timestamp),
                    )
                    for pos, rec in enumerate(records)
                ],
            )
            self._conn.commit()

    def get_window(self, episode_id: str) -> List[RawDialogRecord]:
        rows = self.query(
            "SELECT * FROM sliding_window WHERE episode_id=? ORDER BY position", (episode_id,)
        )
        return [
            RawDialogRecord(
                reference_id=r["reference_id"],
                user_msg=r["user_msg"],
                agent_msg=r["agent_msg"],
                timestamp=r["timestamp"],
                episode_id=episode_id,
                turn_index=r["turn_index"],
            )
            for r in rows
        ]

    def clear_window(self, episode_id: str) -> None:
        self.execute("DELETE FROM sliding_window WHERE episode_id=?", (episode_id,))

    # ------------------------------------------------------------------ #
    # batch checkpointing
    # ------------------------------------------------------------------ #
    def mark_episode_done(
        self, episode_id: str, run_id: str = "default", num_turns: int = 0, detail: str = ""
    ) -> None:
        self.execute(
            """INSERT INTO episode_progress (run_id, episode_id, status, num_turns, updated_at, detail)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(run_id, episode_id) DO UPDATE SET
                   status=excluded.status, num_turns=excluded.num_turns,
                   updated_at=excluded.updated_at, detail=excluded.detail""",
            (run_id, episode_id, "done", int(num_turns), time.time(), detail),
        )

    def mark_episode_failed(self, episode_id: str, run_id: str = "default", detail: str = "") -> None:
        self.execute(
            """INSERT INTO episode_progress (run_id, episode_id, status, num_turns, updated_at, detail)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(run_id, episode_id) DO UPDATE SET
                   status=excluded.status, updated_at=excluded.updated_at, detail=excluded.detail""",
            (run_id, episode_id, "failed", 0, time.time(), detail[:2000]),
        )

    def get_done_episodes(self, run_id: str = "default") -> set:
        rows = self.query(
            "SELECT episode_id FROM episode_progress WHERE run_id=? AND status='done'", (run_id,)
        )
        return {r["episode_id"] for r in rows}

    def get_progress_rows(self, run_id: str = "default") -> List[sqlite3.Row]:
        return self.query("SELECT * FROM episode_progress WHERE run_id=? ORDER BY updated_at", (run_id,))

    def delete_progress(self, run_id: str = "default") -> None:
        self.execute("DELETE FROM episode_progress WHERE run_id=?", (run_id,))

    # ------------------------------------------------------------------ #
    # raw per-episode result payloads
    # ------------------------------------------------------------------ #
    def save_run_payload(self, run_id: str, system_name: str, episode_id: str, payload: Dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO experiment_runs (run_id, system_name, episode_id, created_at, payload)
               VALUES (?,?,?,?,?)
               ON CONFLICT(run_id, system_name, episode_id) DO UPDATE SET
                   created_at=excluded.created_at, payload=excluded.payload""",
            (run_id, system_name, episode_id, time.time(), json.dumps(payload, ensure_ascii=False, default=str)),
        )

    def list_run_payloads(self, run_id: str = None) -> List[Dict[str, Any]]:
        if run_id is None:
            rows = self.query("SELECT * FROM experiment_runs ORDER BY created_at")
        else:
            rows = self.query("SELECT * FROM experiment_runs WHERE run_id=? ORDER BY created_at", (run_id,))
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError):
                payload = {}
            out.append(
                {
                    "run_id": row["run_id"],
                    "system_name": row["system_name"],
                    "episode_id": row["episode_id"],
                    "payload": payload,
                }
            )
        return out

    # ------------------------------------------------------------------ #
    # maintenance
    # ------------------------------------------------------------------ #
    def drop_episode(self, episode_id: str, keep_archive: bool = True) -> None:
        """Hard-delete an episode's data (used by tests/maintenance)."""
        self.execute("DELETE FROM raw_records WHERE episode_id=?", (episode_id,))
        self.execute("DELETE FROM active_summaries WHERE episode_id=?", (episode_id,))
        self.execute("DELETE FROM index_entries WHERE episode_id=?", (episode_id,))
        self.execute("DELETE FROM sliding_window WHERE episode_id=?", (episode_id,))
        if not keep_archive:
            self.execute("DELETE FROM archived_summaries WHERE episode_id=?", (episode_id,))

    def table_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for table in (
            "raw_records",
            "archived_summaries",
            "active_summaries",
            "index_entries",
            "sliding_window",
            "episode_progress",
            "experiment_runs",
        ):
            row = self.query_one(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(row["n"]) if row else 0
        return out
