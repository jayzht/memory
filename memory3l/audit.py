"""
Auditable memory surface: an append-only fact ledger plus verifiable invariants.

Why this module exists
----------------------
Every memory system claims it "does not lose information".  Almost none can *prove*
it, because nothing keeps an independent record of what was ever extracted, so a
disappearance leaves no trace to compare against.  This module keeps that record:

    fact_ledger  =  append-only log of every (slot, value) ever extracted,
                    written at the moment the summary carrying it is created --
                    including summaries that are later overridden or capacity-merged
                    away (those are exactly where facts go missing).

``verify()`` then cross-checks the ledger against the live store and reports
violations instead of silently degrading.  Three invariants:

I1 conservation
    Every fact in the ledger still resolves to a summary that exists in the store
    (active *or* archived).  A fact that is in the ledger and in neither is a
    silent loss -- it was extracted, and now nothing can reach it.

I2 top-level current-value reachability
    The newest value of every known slot must be visible at the top level of the
    prompt: in the derived current-value registry, in a rendered summary, or in a
    live index title.  A value that exists only inside an index *member* is the
    dominant real-data failure mode -- the model is never told it.

I3 provenance resolvability
    Every ledger entry's evidence (``raw_ref_id``) resolves to a raw record, so
    "where did this come from" is answerable for 100% of facts, not most.

.. note::
   M1 keeps the ledger in memory for one episode.  Persisting it (so the audit
   survives a restart) is the obvious next step and needs a store adapter; the
   invariants and the report format are already store-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

__all__ = ["FactRecord", "FactLedger", "AuditReport", "verify_invariants"]


@dataclass
class FactRecord:
    """One observation of one slot's value, bound to the summary that carried it."""

    fact_id: str                     # "<summary_id>#<slot>"
    summary_id: str
    slot: str
    value: str
    seq: int                         # creation order inside the episode
    observed_turn: int = -1
    evidence: List[str] = field(default_factory=list)   # raw_ref_ids
    #: Why the carrying summary left the working set: "" (still live),
    #: "overridden" | "capacity".  Recorded so G2 ("every version is explainable")
    #: is answerable for archived facts too.
    reason: str = ""
    superseded_by: str = ""
    #: filled in by verify(): still present in the store (active or archived)
    resolved: bool = False

    def render(self) -> str:
        state = "live" if not self.reason else self.reason
        return f"{self.slot}={self.value} [{state}] via {self.summary_id}"


@dataclass
class AuditReport:
    """The machine-readable artifact a customer/auditor inspects."""

    episode_id: str
    invariants: Dict[str, Dict[str, Any]]
    counters: Dict[str, Any]
    violations: List[str] = field(default_factory=list)
    gold: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "ok": self.ok,
            "invariants": self.invariants,
            "counters": self.counters,
            "gold": self.gold,
            "violations": self.violations,
        }

    def render(self) -> str:
        lines = [f"=== audit: {self.episode_id} | {'OK' if self.ok else 'VIOLATIONS'} ==="]
        for name, result in self.invariants.items():
            mark = "ok " if result.get("ok") else "FAIL"
            detail = ", ".join(f"{k}={v}" for k, v in result.items() if k != "ok")
            lines.append(f"  [{mark}] {name:34} {detail}")
        if self.counters:
            lines.append(f"  counters: {self.counters}")
        if self.gold:
            lines.append(f"  gold    : {self.gold}")
        for violation in self.violations:
            lines.append(f"  !! {violation}")
        return "\n".join(lines)


def _split_fact_key(key: str) -> Optional[tuple]:
    """``"工位=3楼"`` -> ``("工位", "3楼")``; bare slots carry no value and are skipped."""
    if not key or "=" not in key:
        return None
    slot, _, value = key.partition("=")
    slot, value = slot.strip(), value.strip()
    if not slot or not value:
        return None
    return slot, value


class FactLedger:
    """
    Append-only log of every fact observation in one episode.

    Deliberately never removes anything: the whole point is to be the independent
    record that a disappearance can be detected *against*.
    """

    def __init__(self, episode_id: str = ""):
        self.episode_id = episode_id
        self._records: List[FactRecord] = []
        self._by_id: Dict[str, FactRecord] = {}
        self.duplicate_observations = 0

    # ------------------------------------------------------------------ #
    # writing (called by MemoryManager at the moment a summary is created)
    # ------------------------------------------------------------------ #
    def record_summary(
        self, summary_id: str, fact_keys: Sequence[str], *,
        seq: int = -1, turn_index: int = -1, evidence: Sequence[str] = (),
    ) -> List[FactRecord]:
        """Log every ``slot=value`` a summary carries.  Idempotent per fact_id."""
        created: List[FactRecord] = []
        for key in fact_keys or ():
            split = _split_fact_key(key)
            if split is None:
                continue
            slot, value = split
            fact_id = f"{summary_id}#{slot}"
            if fact_id in self._by_id:
                self.duplicate_observations += 1
                continue
            record = FactRecord(
                fact_id=fact_id, summary_id=summary_id, slot=slot, value=value,
                seq=seq, observed_turn=turn_index,
                evidence=[ref for ref in evidence if ref],
            )
            self._records.append(record)
            self._by_id[fact_id] = record
            created.append(record)
        return created

    def mark_left_working_set(self, summary_id: str, reason: str, superseded_by: str = "") -> int:
        """Record why a summary's facts left the active chain (G2)."""
        touched = 0
        for record in self._records:
            if record.summary_id == summary_id and not record.reason:
                record.reason = reason
                record.superseded_by = superseded_by
                touched += 1
        return touched

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #
    def entries(self) -> List[FactRecord]:
        return list(self._records)

    def get(self, fact_id: str) -> Optional[FactRecord]:
        return self._by_id.get(fact_id)

    def slots(self) -> Set[str]:
        return {record.slot for record in self._records}

    def current_by_slot(self) -> Dict[str, FactRecord]:
        """Newest observation per slot -- the ledger's view of "what is true now"."""
        current: Dict[str, FactRecord] = {}
        for record in self._records:               # appended in creation order
            current[record.slot] = record
        return current

    def stats(self) -> Dict[str, Any]:
        return {
            "ledger_facts": len(self._records),
            "ledger_slots": len(self.slots()),
            "ledger_archived": sum(1 for r in self._records if r.reason),
            "ledger_duplicate_observations": self.duplicate_observations,
        }


def verify_invariants(
    episode_id: str,
    ledger: FactLedger,
    *,
    active_summaries: Sequence[Any],
    archived_summaries: Sequence[Any],
    rendered_text: str,
    top_level_text: str,
    raw_lookup,
    reason_counts: Optional[Dict[str, int]] = None,
    gold_facts: Optional[Iterable[tuple]] = None,
) -> AuditReport:
    """
    Cross-check the ledger against the store and build the audit report.

    Everything is passed in (rather than reaching into the manager) so the checks
    work against any store backend -- including the ones a customer brings.
    """
    active_ids = {s.summary_id for s in active_summaries}
    archived_ids = {a.summary_id for a in archived_summaries}
    entries = ledger.entries()

    # --- I1 conservation -------------------------------------------------- #
    missing = [r for r in entries if r.summary_id not in active_ids and r.summary_id not in archived_ids]
    for record in entries:
        record.resolved = (
            record.summary_id in active_ids or record.summary_id in archived_ids
        )
    conservation = {
        "ok": not missing,
        "ledger": len(entries),
        "active": len(active_ids),
        "archived": len(archived_ids),
        "missing": len(missing),
    }

    # --- I3 provenance resolvability -------------------------------------- #
    unresolved: List[str] = []
    without_evidence = 0
    for record in entries:
        if not record.evidence:
            without_evidence += 1
            continue
        if any(raw_lookup(ref) is None for ref in record.evidence):
            unresolved.append(record.fact_id)
    provenance = {
        "ok": not unresolved and without_evidence == 0,
        "checked": len(entries),
        "unresolved_evidence": len(unresolved),
        "missing_evidence": without_evidence,
    }

    # --- I2 top-level current-value reachability -------------------------- #
    # A value may legitimately live in the rendered chain, in the derived registry,
    # or in a live index title.  If it is in none of them, the model is never told
    # it -- and that is the real-data failure mode this check exists for.
    haystack = f"{top_level_text}\n{rendered_text}"
    unreachable = [
        f"{slot}={record.value}"
        for slot, record in ledger.current_by_slot().items()
        if record.value not in haystack
    ]
    reachability = {
        "ok": not unreachable,
        "slots": len(ledger.current_by_slot()),
        "unreachable": len(unreachable),
    }

    invariants = {
        "I1_fact_conservation": conservation,
        "I2_top_level_current_value_reachable": reachability,
        "I3_provenance_resolvable": provenance,
    }
    violations: List[str] = []
    if missing:
        violations.append(
            f"I1: {len(missing)} fact(s) in the ledger resolve to no summary "
            f"(e.g. {missing[0].fact_id})"
        )
    if unreachable:
        violations.append(
            f"I2: {len(unreachable)} current value(s) are not visible at the top level "
            f"(e.g. {unreachable[0]})"
        )
    if unresolved or without_evidence:
        violations.append(
            f"I3: {len(unresolved)} unresolved evidence pointer(s), "
            f"{without_evidence} fact(s) with no evidence at all"
        )

    counters: Dict[str, Any] = dict(ledger.stats())
    if reason_counts:
        counters.update(reason_counts)

    gold_report: Dict[str, Any] = {}
    if gold_facts is not None:
        found = 0
        total = 0
        for slot, value in gold_facts:
            total += 1
            if any(
                r.slot == slot and r.value == value and r.resolved
                for r in entries
            ):
                found += 1
        gold_report = {
            "gold_facts": total,
            "captured_and_resolvable": found,
            "silent_loss_rate": round(1 - found / total, 4) if total else None,
        }

    return AuditReport(
        episode_id=episode_id,
        invariants=invariants,
        counters=counters,
        violations=violations,
        gold=gold_report,
    )
