"""
Temporal-store adapter: the ledger projected into a system-versioned table.

Rationale
---------
The component should **not** own a storage engine.  "This fact changed at turn 16"
is exactly what a temporal table (SQL:2011 system-versioning, XTDB, Dolt, a
``valid_from``/``valid_to`` pair) already stores well.  What a customer needs from
this module is not an implementation but an *integration contract*:

    if your temporal store answers ``current(slot)`` and ``history(slot, upto)`` in
    the shape below, this component can use it instead of its own table.

``TemporalFactTable`` therefore implements that contract on a flat list of rows --
no delegation back to the ledger -- so :func:`TestTemporalEquivalence` comparing the
two is a real assertion about the projection rather than a tautology.  It also
carries the check that makes outsourcing safe: :meth:`anomalies` reports the ways a
versioned table goes wrong (overlapping intervals, two open rows for one slot, an
erased row that still holds a value), which is what you want to alert on when the
storage is somebody else's.

``ddl()`` emits the table plus the two queries the contract needs, so a customer can
stand it up in their own database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .audit import FactLedger

__all__ = ["TemporalRow", "TemporalFactTable"]


@dataclass
class TemporalRow:
    """One closed interval of one slot's value -- a system-versioned row."""

    episode_id: str
    slot: str
    value: str
    fact_id: str
    valid_from_turn: int
    valid_to_turn: Optional[int]     # None = still current
    superseded_by: str = ""
    reason: str = ""
    erased: bool = False

    @property
    def is_open(self) -> bool:
        return self.valid_to_turn is None


class TemporalFactTable:
    """A flat, system-versioned projection of :class:`FactLedger`."""

    def __init__(self, episode_id: str = ""):
        self.episode_id = episode_id
        self._rows: List[TemporalRow] = []

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_ledger(cls, ledger: FactLedger) -> "TemporalFactTable":
        """
        Project the ledger.  Each slot becomes a chain of half-open intervals
        ``[valid_from, valid_to)``; the newest row stays open.
        """
        table = cls(ledger.episode_id)
        by_slot: Dict[str, List[Any]] = {}
        for record in ledger.entries():
            by_slot.setdefault(record.slot, []).append(record)
        for slot, records in by_slot.items():
            records.sort(key=lambda r: (r.observed_turn, r.seq))
            for index, record in enumerate(records):
                following = records[index + 1] if index + 1 < len(records) else None
                table._rows.append(
                    TemporalRow(
                        episode_id=ledger.episode_id,
                        slot=slot,
                        value=record.value,
                        fact_id=record.fact_id,
                        valid_from_turn=record.observed_turn,
                        valid_to_turn=following.observed_turn if following else None,
                        superseded_by=following.fact_id if following else "",
                        reason=record.reason,
                        erased=record.erased,
                    )
                )
        table._rows.sort(key=lambda r: (r.slot, r.valid_from_turn))
        return table

    @classmethod
    def from_store(cls, store, episode_id: str) -> "TemporalFactTable":
        """Build from the persisted ledger -- what a sidecar does."""
        return cls.from_ledger(FactLedger(episode_id, store=store))

    # ------------------------------------------------------------------ #
    # the integration contract
    # ------------------------------------------------------------------ #
    def rows(self) -> List[TemporalRow]:
        return list(self._rows)

    def current(self, slot: str) -> Optional[TemporalRow]:
        """The open row for ``slot`` (``valid_to_turn IS NULL``)."""
        slot = (slot or "").strip().lower()
        open_rows = [r for r in self._rows if r.slot == slot and r.is_open]
        return open_rows[-1] if open_rows else None

    def history(self, slot: str, upto_turn: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Same shape as :meth:`FactLedger.history`, computed by filtering rows.

        "As of turn T" means: every interval that had started by T, re-closed against
        the next surviving interval, with the newest one left open.
        """
        slot = (slot or "").strip().lower()
        rows = [r for r in self._rows if r.slot == slot]
        rows.sort(key=lambda r: r.valid_from_turn)
        if upto_turn is not None:
            rows = [r for r in rows if r.valid_from_turn <= upto_turn]
        out: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            following = rows[index + 1] if index + 1 < len(rows) else None
            out.append({
                "slot": row.slot,
                "value": row.value,
                "fact_id": row.fact_id,
                "from_turn": row.valid_from_turn,
                "to_turn": following.valid_from_turn if following else None,
                "superseded_by": following.fact_id if following else "",
                "reason": row.reason,
                "erased": row.erased,
            })
        return out

    # ------------------------------------------------------------------ #
    # safety net for outsourced storage
    # ------------------------------------------------------------------ #
    def anomalies(self) -> List[str]:
        """
        Every way a versioned table can be wrong, reported rather than averaged away.

        If the temporal store is somebody else's, this is what you alert on: the
        ledger is the reference, and a projection that disagrees with it must be
        visible before it answers a compliance question.
        """
        problems: List[str] = []
        slots: Dict[str, List[TemporalRow]] = {}
        for row in self._rows:
            slots.setdefault(row.slot, []).append(row)
        for slot, rows in slots.items():
            rows.sort(key=lambda r: r.valid_from_turn)
            open_rows = [r for r in rows if r.is_open]
            if len(open_rows) > 1:
                problems.append(
                    f"{slot}: {len(open_rows)} open intervals (valid_to_turn IS NULL); "
                    "exactly one value can be current"
                )
            if not open_rows:
                problems.append(f"{slot}: no open interval -- the slot has no current value")
            for position, row in enumerate(rows):
                if row.erased and row.value:
                    problems.append(
                        f"{slot}@{row.valid_from_turn}: erased row still holds a value"
                    )
                if row.valid_to_turn is not None and row.valid_to_turn < row.valid_from_turn:
                    problems.append(
                        f"{slot}@{row.valid_from_turn}: valid_to {row.valid_to_turn} "
                        "precedes valid_from"
                    )
                if position + 1 < len(rows):
                    nxt = rows[position + 1]
                    if row.valid_to_turn is None:
                        problems.append(
                            f"{slot}@{row.valid_from_turn}: open interval followed by "
                            f"another at {nxt.valid_from_turn}"
                        )
                    elif row.valid_to_turn != nxt.valid_from_turn:
                        kind = "overlap" if row.valid_to_turn > nxt.valid_from_turn else "gap"
                        problems.append(
                            f"{slot}: {kind} between turn {row.valid_to_turn} and "
                            f"{nxt.valid_from_turn}"
                        )
        return problems

    # ------------------------------------------------------------------ #
    # hand-off to the customer's database
    # ------------------------------------------------------------------ #
    @staticmethod
    def ddl(table_name: str = "fact_history") -> str:
        """ANSI-flavoured DDL plus the two queries the contract requires."""
        return f"""-- System-versioned projection of the fact ledger.
-- This is what "outsource the storage" means concretely: the component keeps the
-- ledger, and this table answers current/history queries for it.
CREATE TABLE {table_name} (
    episode_id      TEXT    NOT NULL,
    slot            TEXT    NOT NULL,
    value           TEXT    NOT NULL DEFAULT '',   -- '' once compliance-erased
    fact_id         TEXT    NOT NULL,
    valid_from_turn INTEGER NOT NULL,
    valid_to_turn   INTEGER,                       -- NULL = still current
    superseded_by   TEXT    NOT NULL DEFAULT '',
    reason          TEXT    NOT NULL DEFAULT '',
    erased          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (fact_id)
);
CREATE INDEX idx_{table_name}_slot ON {table_name}(episode_id, slot, valid_from_turn);
-- at most one open row per slot; enforce this in your engine if it can
CREATE UNIQUE INDEX idx_{table_name}_open
    ON {table_name}(episode_id, slot) WHERE valid_to_turn IS NULL;

-- contract query 1: what is this slot now?
SELECT value, fact_id, valid_from_turn FROM {table_name}
 WHERE episode_id = :episode AND slot = :slot AND valid_to_turn IS NULL;

-- contract query 2: what did it hold as of turn :t?
SELECT value, fact_id, valid_from_turn FROM {table_name}
 WHERE episode_id = :episode AND slot = :slot AND valid_from_turn <= :t
 ORDER BY valid_from_turn;

-- On PostgreSQL 14+ the same thing can be expressed with a system-versioned table
-- (SQL:2011 FOR PORTION OF) so the engine maintains valid_to_turn itself.
"""
