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

    def __init__(self, path: str = "./exp_memory.db"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()

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
    def add_raw_record(self, record: RawDialogRecord) -> None:
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

    def get_raw_record(self, reference_id: str) -> Optional[RawDialogRecord]:
        row = self.query_one("SELECT * FROM raw_records WHERE reference_id=?", (reference_id,))
        return self._row_to_raw(row) if row else None

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

    def delete_raw_record(self, reference_id: str) -> None:
        """
        Hard-delete one raw record.

        Used exclusively by the MemGPT-style destructive baseline; the three-layer
        system's raw store is permanent by contract.
        """
        self.execute("DELETE FROM raw_records WHERE reference_id=?", (reference_id,))

    # ------------------------------------------------------------------ #
    # archived summaries
    # ------------------------------------------------------------------ #
    def add_archived_summary(self, archived: ArchivedSummary) -> None:
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
    def add_active_summary(self, summary: ActiveSummary) -> None:
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

    def remove_active_summary(self, summary_id: str) -> None:
        self.execute("DELETE FROM active_summaries WHERE summary_id=?", (summary_id,))

    def list_active_summaries(self, episode_id: Optional[str] = None) -> List[ActiveSummary]:
        if episode_id is None:
            rows = self.query("SELECT * FROM active_summaries ORDER BY timestamp, seq")
        else:
            rows = self.query(
                "SELECT * FROM active_summaries WHERE episode_id=? ORDER BY timestamp, seq",
                (episode_id,),
            )
        out: List[ActiveSummary] = []
        for row in rows:
            out.append(
                ActiveSummary(
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
            )
        return out

    # ------------------------------------------------------------------ #
    # index entries (level 2 of layer 1)
    # ------------------------------------------------------------------ #
    def add_index_entry(self, entry) -> None:
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

    def remove_index_entry(self, index_id: str, episode_id: Optional[str] = None) -> None:
        self.execute("DELETE FROM index_entries WHERE index_id=?", (index_id,))

    def clear_indexes(self, episode_id: str) -> None:
        self.execute("DELETE FROM index_entries WHERE episode_id=?", (episode_id,))

    def clear_active_summaries(self, episode_id: str) -> None:
        self.execute("DELETE FROM active_summaries WHERE episode_id=?", (episode_id,))
        self.execute("DELETE FROM index_entries WHERE episode_id=?", (episode_id,))

    def clear_archive(self, episode_id: str) -> int:
        """Drop this episode's archived summaries; raw records stay (permanent)."""
        cursor = self.execute("DELETE FROM archived_summaries WHERE episode_id=?", (episode_id,))
        return int(cursor.rowcount or 0)

    # ------------------------------------------------------------------ #
    # sliding window mirror (crash recovery only)
    # ------------------------------------------------------------------ #
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
