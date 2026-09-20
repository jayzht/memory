"""
The audit surface as a reusable service: one SQLite file in, JSON answers out.

Why this is a module and not the sidecar's private class
--------------------------------------------------------
The audit answers are the product; HTTP is one way to ask for them and MCP is
another.  Keeping ``AuditService`` here means the sidecar (``audit_server.py``)
and the MCP server (``memory3l-mcp/``) answer from *the same code*, so an
integrator cannot get a different verdict from a different transport.  A bug
fixed once is fixed in both.

Everything is derived from the cold store, which is the source of truth.  Redis
is deliberately absent: an auditor usually asks *after* the writing process is
gone, so the answers must be about persisted state, not about a hot cache that
may no longer exist.

Read-only, with exactly one exception
-------------------------------------
No method writes summaries, indexes or raw dialogue.  ``record_facts`` is the
single accepted write: an append-only fact intake for a caller's own extractor.
It is idempotent (primary key ``<summary_id>#<slot>``) and it runs the audit
before answering, so a caller cannot append facts that break the invariants
without being told.
"""

from __future__ import annotations

from typing import Sequence

from .audit import FactLedger, audit_all, audit_episode, derive_current_values
from .store.sqlite_store import SQLiteColdStore
from .temporal import TemporalFactTable


def _int_or(value, default: int) -> int:
    """Keep a legitimate 0 (``int(value or default)`` would not)."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class AuditService:
    """Read-only queries over one SQLite file.  Shared by every request."""

    def __init__(self, sqlite_path: str, token: str = ""):
        self.sqlite_path = sqlite_path
        self.token = token
        self.store = SQLiteColdStore(sqlite_path)

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------------ #
    def episodes(self):
        lister = getattr(self.store, "list_ledger_episodes", None)
        return lister() if callable(lister) else []

    def audit(self, episode_id: str):
        return audit_episode(self.store, episode_id).to_dict()

    def current(self, episode_id: str):
        active = self.store.list_active_summaries(episode_id)
        values = derive_current_values(active)
        return {
            "episode_id": episode_id,
            "registry": "; ".join(f"{slot}={value}" for slot, value, _ in values),
            "values": [
                {"slot": slot, "value": value, "fact_id": f"{summary_id}#{slot}"}
                for slot, value, summary_id in values
            ],
        }

    def tombstones(self, episode_id: str):
        """
        What was erased, and what is provably gone.

        An auditor asks two questions, and they are opposites: "nothing was lost"
        (I1) and "what was meant to be deleted really is" (I5).  This answers the
        second one, listing tombstones plus the prose that still mentions them.
        """
        ledger = FactLedger(episode_id, store=self.store)
        rows = []
        for record in ledger.entries():
            if not record.erased:
                continue
            resolvable = [
                ref for ref in record.evidence
                if self.store.get_raw_record(ref, episode_id=episode_id) is not None
            ]
            rows.append({
                "fact_id": record.fact_id,
                "slot": record.slot,
                "observed_turn": record.observed_turn,
                "reason": record.reason,
                "value": record.value,               # empty once erased
                "evidence_pointers": list(record.evidence),
                "evidence_still_readable": resolvable,
                "residual_prose_mentions": record.residual_mentions,
            })
        return {"episode_id": episode_id, "tombstones": rows}

    def record_facts(self, episode_id: str, facts: Sequence[dict], max_facts: int = 500) -> dict:
        """
        Append pre-extracted facts to the audited ledger, idempotently.

        This lets an external extractor record facts against dialogue the component
        already stores, without adopting the manager.  Two properties make it safe to
        expose:

        * **idempotent** -- ``fact_id`` is ``<summary_id>#<slot>`` and the insert
          ignores an existing key, so a retry adds nothing;
        * **verified before answering** -- the audit runs first, so a fact that is not
          backed by a real summary comes back as an I1 violation, and a value that
          never becomes visible at the top level comes back as I2.

        That second point is a real constraint, not a formality: injecting a fact
        cannot *make* it visible, because the registry is derived from the summaries'
        ``fact_keys``.  An extractor that adds facts must add them to summaries whose
        text carries the value; the endpoint tells it immediately when it has not.
        """
        if len(facts) > max_facts:
            return {"error": f"too many facts in one request (max {max_facts})"}
        rows, rejected = [], []
        for fact in facts:
            if not isinstance(fact, dict):
                rejected.append({"fact": fact, "error": "not an object"})
                continue
            slot = str(fact.get("slot", "") or "").strip()
            value = str(fact.get("value", "") or "").strip()
            summary_id = str(fact.get("summary_id", "") or "").strip()
            if not (slot and value and summary_id):
                rejected.append({**fact, "error": "slot, value and summary_id are required"})
                continue
            evidence = fact.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            rows.append({
                "fact_id": f"{summary_id}#{slot}",
                "episode_id": episode_id,
                "summary_id": summary_id,
                "slot": slot.lower(),
                "value": value,
                "seq": _int_or(fact.get("seq"), -1),
                "observed_turn": _int_or(fact.get("observed_turn"), -1),
                "evidence": "|".join(str(ref) for ref in evidence if ref),
                "reason": "", "superseded_by": "",
                "residual_mentions": 0, "erased": 0,
            })
        added = self.store.append_fact_ledger(rows) if rows else 0
        report = self.audit(episode_id)
        return {
            "episode_id": episode_id,
            "accepted": len(rows),
            "added": added,
            "duplicates": len(rows) - added,
            "rejected": rejected,
            "audit_ok": report["ok"],
            "violations": report["violations"],
        }

    def history(self, episode_id: str, slot: str, upto_turn=None):
        """The slot's value over time -- the question a temporal store answers."""
        table = TemporalFactTable.from_store(self.store, episode_id)
        return {
            "episode_id": episode_id,
            "slot": slot.strip().lower(),
            "upto_turn": upto_turn,
            "history": table.history(slot, upto_turn=upto_turn),
        }

    def summary(self):
        """Aggregate audit over every episode that has a ledger."""
        return audit_all(self.store)

    def temporal(self, episode_id: str):
        """
        The versioned projection and its consistency check.

        Useful when storage is outsourced: the ledger is the reference, and this says
        whether the projection still agrees with it.
        """
        table = TemporalFactTable.from_store(self.store, episode_id)
        anomalies = table.anomalies()
        return {
            "episode_id": episode_id,
            "rows": len(table.rows()),
            "consistent": not anomalies,
            "anomalies": anomalies,
            "ddl": TemporalFactTable.ddl(),
        }

    def fact(self, fact_id: str):
        ledger = self._ledger_for(fact_id)
        if ledger is None:
            return None
        record = ledger.get(fact_id)
        if record is None:
            return None
        active = self.store.get_active_summary(record.summary_id, episode_id=record.episode_id)
        archived = self.store.get_archived_summary(record.summary_id, episode_id=None)
        state = "live" if active is not None else ("archived" if archived is not None else "MISSING")
        return {
            "fact_id": record.fact_id,
            "slot": record.slot,
            "value": record.value,
            "observed_turn": record.observed_turn,
            "state": state,
            "reason": record.reason,
            "superseded_by": record.superseded_by,
            "evidence": list(record.evidence),
        }

    def evidence(self, fact_id: str):
        ledger = self._ledger_for(fact_id)
        if ledger is None:
            return None
        record = ledger.get(fact_id)
        if record is None:
            return None
        messages = []
        for reference_id in record.evidence:
            raw = self.store.get_raw_record(reference_id, episode_id=record.episode_id)
            if raw is not None:
                messages.append(
                    {"turn": raw.turn_index, "user": raw.user_msg, "agent": raw.agent_msg}
                )
        return {
            "fact_id": fact_id,
            "slot": record.slot,
            "value": record.value,
            "raw_refs": list(record.evidence),
            "resolved": len(messages) == len(record.evidence),
            "messages": messages,
        }

    def _ledger_for(self, fact_id: str):
        """
        ``fact_id`` is ``<episode>/<kind><seq>@<hash>#<slot>``.

        The episode id itself may contain "/" (batch runs scope it as
        ``system/episode``), so it is the *summary id* with its last path segment
        removed -- no extra lookup index needed.
        """
        summary_id = fact_id.split("#", 1)[0]
        if "/" not in summary_id:
            return None
        ledger = FactLedger(summary_id.rsplit("/", 1)[0], store=self.store)
        return ledger if ledger.get(fact_id) is not None else None
