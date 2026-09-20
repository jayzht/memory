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

from .gate import CHANGE, classify, extract_candidate_pairs

__all__ = ["FactRecord", "FactLedger", "AuditReport", "verify_invariants",
           "extraction_report", "normalise_fact"]


@dataclass
class FactRecord:
    """One observation of one slot's value, bound to the summary that carried it."""

    fact_id: str                     # "<summary_id>#<slot>"
    summary_id: str
    slot: str
    value: str
    seq: int                         # creation order inside the episode
    episode_id: str = ""
    observed_turn: int = -1
    evidence: List[str] = field(default_factory=list)   # raw_ref_ids
    #: Why the carrying summary left the working set: "" (still live),
    #: "overridden" | "capacity".  Recorded so G2 ("every version is explainable")
    #: is answerable for archived facts too.
    reason: str = ""
    superseded_by: str = ""
    #: filled in by verify(): still present in the store (active or archived)
    resolved: bool = False
    #: True once a compliance erasure destroyed this fact's content.  A real flag,
    #: not a magic ``reason`` string: "why it left the working set" (overridden /
    #: capacity) and "the content was destroyed" are orthogonal properties, and
    #: conflating them made an erasure with a legal basis ("gdpr-art17") invisible to
    #: the verification invariant.
    erased: bool = False
    #: How many places still contain this value *in prose* after a compliance
    #: erasure.  We blank the value and destroy the raw evidence, but an LLM-written
    #: summary is not safely rewritable, so the residue is reported rather than
    #: silently left behind.
    residual_mentions: int = 0

    def render(self) -> str:
        state = "live" if not self.reason else self.reason
        return f"{self.slot}={self.value} [{state}] via {self.summary_id}"

    # --- persistence: the store layer speaks plain dicts, so it never has to
    # --- import these dataclasses (keeps the dependency direction one-way)
    def to_row(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "episode_id": self.episode_id,
            "summary_id": self.summary_id,
            "slot": self.slot,
            "value": self.value,
            "seq": int(self.seq),
            "observed_turn": int(self.observed_turn),
            # packed with "|" like every other id list in the store schemas
            "evidence": "|".join(self.evidence),
            "reason": self.reason,
            "superseded_by": self.superseded_by,
            "residual_mentions": int(self.residual_mentions),
            "erased": 1 if self.erased else 0,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "FactRecord":
        evidence = row.get("evidence") or ""

        def _int(value, default: int) -> int:
            # NOT ``int(value or default)``: turn 0 is a legitimate value, and
            # collapsing it to -1 silently moved every first-turn fact to a
            # nonexistent turn on the way back out of the database.
            if value is None or value == "":
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return cls(
            fact_id=row["fact_id"],
            summary_id=row.get("summary_id", ""),
            episode_id=row.get("episode_id", ""),
            slot=row.get("slot", ""),
            value=row.get("value", ""),
            seq=_int(row.get("seq"), -1),
            observed_turn=_int(row.get("observed_turn"), -1),
            evidence=[ref for ref in str(evidence).split("|") if ref],
            reason=row.get("reason", "") or "",
            superseded_by=row.get("superseded_by", "") or "",
            residual_mentions=_int(row.get("residual_mentions"), 0),
            erased=bool(_int(row.get("erased"), 0)),
        )


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
            detail = ", ".join(
                f"{k}=<{len(v)} item(s)>" if isinstance(v, (list, dict, tuple)) else f"{k}={v}"
                for k, v in result.items() if k != "ok"
            )
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

    def __init__(self, episode_id: str = "", store=None):
        self.episode_id = episode_id
        #: Optional durable backing.  When given, every append/update is written
        #: through, and construction reloads whatever is already there -- an audit
        #: ledger that vanishes on restart is not an audit ledger.
        self.store = store
        self._records: List[FactRecord] = []
        self._by_id: Dict[str, FactRecord] = {}
        self.duplicate_observations = 0
        if store is not None:
            self.reload()

    # ------------------------------------------------------------------ #
    # durability
    # ------------------------------------------------------------------ #
    def reload(self) -> int:
        """Load this episode's ledger from the store (no-op without one)."""
        if self.store is None or not self.episode_id:
            return 0
        try:
            rows = self.store.list_fact_ledger(self.episode_id)
        except (AttributeError, NotImplementedError):
            return 0
        self._records = []
        self._by_id = {}
        for row in rows:
            record = FactRecord.from_row(row)
            self._records.append(record)
            self._by_id[record.fact_id] = record
        return len(self._records)

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
                fact_id=fact_id, summary_id=summary_id, episode_id=self.episode_id,
                slot=slot, value=value,
                seq=seq, observed_turn=turn_index,
                evidence=[ref for ref in evidence if ref],
            )
            self._records.append(record)
            self._by_id[fact_id] = record
            created.append(record)
        if created and self.store is not None:
            try:
                self.store.append_fact_ledger([r.to_row() for r in created])
            except (AttributeError, NotImplementedError):
                pass          # a store without durability still audits in-process
        return created

    def mark_left_working_set(self, summary_id: str, reason: str, superseded_by: str = "") -> int:
        """Record why a summary's facts left the active chain (G2)."""
        touched = 0
        for record in self._records:
            if record.summary_id == summary_id and not record.reason:
                record.reason = reason
                record.superseded_by = superseded_by
                touched += 1
        if touched and self.store is not None:
            try:
                self.store.update_fact_ledger_reason(
                    summary_id, reason, superseded_by, episode_id=self.episode_id
                )
            except (AttributeError, NotImplementedError):
                pass
        return touched

    def erase(
        self, fact_ids: Sequence[str], reason: str = "erased",
        residual_by_id: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tombstone facts: destroy the value, keep the shape and the evidence pointer.

        Keeping ``evidence`` is what makes the erasure *verifiable*: I5 can then check
        that the raw records it named are now unreachable, instead of trusting a flag.
        """
        rows: List[Dict[str, Any]] = []
        for fact_id in fact_ids:
            record = self._by_id.get(fact_id)
            if record is None or record.erased:
                continue
            record.erased = True
            record.reason = reason          # keeps the legal basis alongside the flag
            record.value = ""
            record.residual_mentions = int((residual_by_id or {}).get(fact_id, 0))
            rows.append({
                "fact_id": fact_id, "reason": reason,
                "residual_mentions": record.residual_mentions,
            })
        if rows and self.store is not None:
            try:
                self.store.erase_fact_ledger(rows, self.episode_id)
            except (AttributeError, NotImplementedError):
                pass
        return rows

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
            if record.erased:
                continue                           # the value was destroyed on purpose
            current[record.slot] = record
        return current

    def stats(self) -> Dict[str, Any]:
        return {
            "ledger_facts": len(self._records),
            "ledger_slots": len(self.slots()),
            "ledger_archived": sum(1 for r in self._records if r.reason),
            "ledger_duplicate_observations": self.duplicate_observations,
        }


def derive_current_values(summaries, max_slots: int = 12) -> List[tuple]:
    """
    ``[(slot, value, summary_id)]`` -- newest live summary per slot.

    Shared by the manager (which renders it into the prompt) and the audit (which
    checks that the prompt shows every current value), so the audit can only ever
    inspect exactly what the model is shown.
    """
    latest: Dict[str, tuple] = {}
    for summary in summaries:                      # oldest -> newest
        keys = getattr(summary, "fact_keys", None) or []
        for key in keys:
            if "=" not in key:
                continue
            slot, _, value = key.partition("=")
            slot, value = slot.strip().lower(), value.strip()
            if slot and value:
                latest[slot] = (slot, value, summary.summary_id, summary.seq)
    if not latest:
        return []
    ordered = sorted(latest.values(), key=lambda item: item[3], reverse=True)[: max(1, max_slots)]
    ordered.sort(key=lambda item: item[3])
    return [(slot, value, summary_id) for slot, value, summary_id, _ in ordered]


def render_current_values(summaries, max_slots: int = 12) -> str:
    items = derive_current_values(summaries, max_slots)
    if not items:
        return ""
    return "; ".join(f"{slot}={value}" for slot, value, _ in items)


def audit_episode(
    store, episode_id: str, gold_facts=None, max_slots: int = 12,
    extraction_min_recall: float = 0.0,
) -> AuditReport:
    """
    Audit a **persisted** episode without a live manager.

    This is what a sidecar service needs: the agent process may be long gone, and
    all that is left is the store.  The top-level view is reconstructed with the
    same rules the prompt uses (registry -> rendered chain -> live index titles), so
    a violation here means the *persisted* state would not have shown the value
    either.
    """
    from .models import LAZY_INDEX_ID

    active = store.list_active_summaries(episode_id)
    archived = store.list_archived_summaries(episode_id)
    entries = store.list_index_entries(episode_id)

    title_members = set()
    for entry in entries:
        title_members.update(entry.members)
    chain = [
        s for s in active
        if s.summary_id not in title_members and s.index_id != LAZY_INDEX_ID
    ]

    ledger = FactLedger(episode_id, store=store)
    def _raw_lookup(reference_id: str):
        return store.get_raw_record(reference_id, episode_id=episode_id)

    return verify_invariants(
        episode_id,
        ledger,
        active_summaries=active,
        archived_summaries=archived,
        rendered_text="\n".join(s.text for s in chain),
        top_level_text="\n".join(
            [render_current_values(active, max_slots),
             "\n".join(entry.title for entry in entries)]
        ),
        raw_lookup=_raw_lookup,
        gold_facts=gold_facts,
        raw_records=store.list_raw_records(episode_id),
        extraction_min_recall=extraction_min_recall,
    )


def normalise_fact(text: str) -> str:
    """Case/punctuation-insensitive form used to compare candidate to captured."""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _value_matches(candidate: str, captured: Set[str]) -> bool:
    """
    Does the memory hold this candidate value?

    Containment, not equality: the candidate detector reads verbatim text, so it
    keeps trailing particles ("下午2点**了**") and English adverbs ("alpha **now**")
    that the summariser's value cleaner strips.  Requiring equality made a perfectly
    correct extraction look like a miss and reported recall 0.0 on clean data.
    Short strings fall back to equality so "3" does not match "30".
    """
    normalised = normalise_fact(candidate)
    if not normalised:
        return False
    for have in captured:
        if not have:
            continue
        if normalised == have:
            return True
        if len(normalised) >= 2 and len(have) >= 2 and (
            normalised in have or have in normalised
        ):
            return True
    return False


def extraction_report(
    raw_records: Sequence[Any],
    ledger_entries: Sequence[FactRecord],
    max_gaps: int = 20,
    scope: str = "user",
) -> Dict[str, Any]:
    """
    How much of what was *said* actually made it into the memory?

    I1-I3 audit the storage path: once a fact is extracted, nothing can lose it.  None
    of them can see a fact that was **never extracted** -- the single real gap the M1
    run exposed (``silent_loss_rate = 0.2`` with every storage invariant passing).
    With no ground truth available at runtime, the independent signal is a high
    -precision lexical detector over the *original turn text*: whatever it finds and
    the memory did not record is a reported gap rather than an invisible one.

    Two denominators are returned on purpose:

    * ``strict_candidates`` -- turns carrying an explicit change verb or ``X=Y``.
      This is the defensible recall figure: the detector rarely fires by accident
      here, so a gap is very likely a real miss.
    * ``all_candidates`` -- every pattern hit, including weak copula matches.  On
      conversational text these over-generate (a question such as "…是**怎么**配合
      的" looks like a fact), so the count is reported for coverage only and is not
      used for the recall figure.
    """
    by_turn: Dict[int, List[FactRecord]] = {}
    for record in ledger_entries:
        by_turn.setdefault(record.observed_turn, []).append(record)

    strict_total = strict_captured = 0
    loose_total = loose_captured = 0
    gaps: List[Dict[str, Any]] = []
    for raw in raw_records:
        # ``scope="user"`` by default: the *user's* words are where a fact the memory
        # must keep is stated.  Including the assistant reply swamped the synthetic set
        # with false candidates ("把中间格式换成列式之后" inside a confident filler
        # answer), which made a correct extraction look like a miss.
        if scope == "both":
            text = f"{getattr(raw, 'user_msg', '')}\n{getattr(raw, 'agent_msg', '')}"
        else:
            text = getattr(raw, "user_msg", "") or ""
        candidates = extract_candidate_pairs(text)
        if not candidates:
            continue
        turn = getattr(raw, "turn_index", -1)
        have = {normalise_fact(r.value) for r in by_turn.get(turn, [])}
        strong = classify(text) == CHANGE
        for slot, value in candidates:
            hit = _value_matches(value, have)
            loose_total += 1
            loose_captured += 1 if hit else 0
            if strong:
                strict_total += 1
                strict_captured += 1 if hit else 0
            if not hit and strong and len(gaps) < max_gaps:
                gaps.append(
                    {"turn": turn, "slot": slot, "value": value,
                     "evidence": getattr(raw, "reference_id", "")}
                )

    return {
        "strict_candidates": strict_total,
        "strict_captured": strict_captured,
        "strict_recall": round(strict_captured / strict_total, 4) if strict_total else None,
        "all_candidates": loose_total,
        "all_captured": loose_captured,
        "all_recall": round(loose_captured / loose_total, 4) if loose_total else None,
        "gaps": gaps,
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
    raw_records: Optional[Sequence[Any]] = None,
    extraction_min_recall: float = 0.0,
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
    erased_entries = [r for r in entries if r.erased]
    for record in entries:
        if record.erased:
            # A compliance erasure deliberately destroys the evidence, so the
            # provenance invariant does not apply; I5 checks the opposite property.
            continue
        if not record.evidence:
            without_evidence += 1
            continue
        if any(raw_lookup(ref) is None for ref in record.evidence):
            unresolved.append(record.fact_id)
    provenance = {
        "ok": not unresolved and without_evidence == 0,
        # Counts only what was actually checked: erased facts are exempt (I5 covers
        # them), and reporting them as "checked" would overstate the coverage.
        "checked": len(entries) - len(erased_entries),
        "erased_exempt": len(erased_entries),
        "unresolved_evidence": len(unresolved),
        "missing_evidence": without_evidence,
    }

    # --- I5 deletion verifiability ---------------------------------------- #
    # The mirror image of I1: I1 proves nothing was lost, I5 proves what was
    # *supposed* to be deleted is actually gone.  An erasure that only flips a flag
    # leaves the value readable and must be reported as a violation.
    still_present_value = [r.fact_id for r in erased_entries if r.value]
    # One raw record can back several facts.  If a *surviving* fact still needs it,
    # the record must be kept -- so "the evidence still resolves" is only a violation
    # when nothing else depends on it.  Without this distinction the invariant would
    # force us to destroy data that is still in use.
    surviving_refs = {
        ref for r in entries if not r.erased for ref in r.evidence
    }
    still_resolvable: List[str] = []
    kept_shared = 0
    for record in erased_entries:
        for ref in record.evidence:
            if ref in surviving_refs:
                kept_shared += 1
                continue
            if raw_lookup(ref) is not None:
                still_resolvable.append(record.fact_id)
                break
    residual_total = sum(r.residual_mentions for r in erased_entries)
    deletion = {
        "ok": not still_present_value and not still_resolvable,
        "erased_facts": len(erased_entries),
        "value_still_present": len(still_present_value),
        "evidence_still_resolvable": len(still_resolvable),
        "evidence_kept_shared": kept_shared,
        "residual_prose_mentions": residual_total,
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

    # --- I4 extraction completeness --------------------------------------- #
    # Informational by default: a lexical detector over-generates, so a recall below
    # 1.0 is expected rather than a defect.  Setting ``extraction_min_recall`` turns
    # it into a violation for a deployment that knows its own floor.
    extraction: Optional[Dict[str, Any]] = None
    if raw_records is not None:
        extraction = extraction_report(raw_records, entries)
        recall = extraction["strict_recall"]
        extraction["ok"] = recall is None or recall >= extraction_min_recall
        extraction["min_recall"] = extraction_min_recall

    invariants = {
        "I1_fact_conservation": conservation,
        "I2_top_level_current_value_reachable": reachability,
        "I3_provenance_resolvable": provenance,
    }
    if extraction is not None:
        invariants["I4_extraction_completeness"] = extraction
    invariants["I5_deletion_verifiable"] = deletion
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
    if not deletion["ok"]:
        violations.append(
            f"I5: an erasure did not take effect -- {len(still_present_value)} fact(s) "
            f"still hold a value, {len(still_resolvable)} still resolve to raw evidence"
        )
    if extraction is not None and not extraction["ok"]:
        violations.append(
            f"I4: extraction recall {extraction['strict_recall']} is below the "
            f"configured floor {extraction_min_recall} "
            f"({len(extraction['gaps'])} reported gap(s))"
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
