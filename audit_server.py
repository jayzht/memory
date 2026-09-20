#!/usr/bin/env python3
"""
Audit sidecar: a read-only HTTP endpoint over the persisted fact ledger.

    python3 audit_server.py --sqlite-path exp_lm.db --port 8020 --token mem2024

Why a sidecar rather than a library call: an auditor wants **an endpoint**, not a
Python object, and usually wants it *after* the agent process is gone.  This
process opens the same SQLite file read-only, so the answers it gives are about the
persisted state, not about whatever happened to be in someone's memory.

Endpoints (all GET, all require ``?token=``):

    GET /health                                   liveness + which db
    GET /episodes                                 episodes that have a ledger
    GET /audit?episode=<id>                       the AuditReport (I1/I2/I3 + counters)
    GET /current?episode=<id>                     the top-level current values
    GET /fact?fact_id=<id>                        why a fact left the working set
    GET /fact/evidence?fact_id=<id>               the original dialogue it came from
    GET /tombstones?episode=<id>                  compliance erasures (what is gone)
    GET /history?episode=<id>&slot=<slot>         every value the slot held
    GET /temporal?episode=<id>                    the versioned projection + anomalies
    GET /audit/summary                            aggregate over every episode

    POST /facts?episode=<id>                      append pre-extracted facts (idempotent)

**Use the query-parameter form.** A ``fact_id`` is
``<episode>/<kind><seq>@<hash>#<slot>`` and contains ``#``, which in a URL starts
the fragment -- a client that puts it in the path silently loses the slot (and the
server then cannot find the fact).  The path forms ``/fact/<id>`` and
``/fact/<id>/evidence`` are kept for convenience and percent-decode the id, so
``/fact/three_layer%2Flong_0000%2Fs002%40abc%23工位楼层`` works too.

Design notes
------------
* **No Redis.** Everything here is derived from SQLite, which is the source of
  truth; the hot cache is an implementation detail of the writing process.
* **Read-only over memory content.**  The service never writes summaries or raw
  records; the single accepted write is ``POST /facts``, an append-only *fact intake*
  for a customer's own extractor.  It is idempotent (the primary key is
  ``<summary_id>#<slot>``) and it **runs the audit before answering**, so a caller
  cannot append facts that break the invariants without being told.
* **Token-gated and localhost by default**, because ``/evidence`` returns raw
  dialogue.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory3l.audit import (  # noqa: E402
    FactLedger, audit_all, audit_episode, derive_current_values,
)
from memory3l.temporal import TemporalFactTable  # noqa: E402
from memory3l.store.sqlite_store import SQLiteColdStore  # noqa: E402

logger = logging.getLogger("audit_server")

MAX_BODY_BYTES = 1 << 20        # 1 MiB: a fact batch, not a document upload


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

    def __init__(self, sqlite_path: str, token: str):
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
            "fact_id": record.fact_id,
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


def make_handler(service: AuditService, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AuditSidecar/1.0"

        def log_message(self, fmt, *args):        # quieter default logging
            logger.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, payload, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:                 # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if token and (query.get("token") or [""])[0] != token:
                self._send({"error": "forbidden: pass ?token=<token>"}, 403)
                return
            raw_path = parsed.path
            # Percent-decode: an id contains "/" and "#", so a client that has to
            # put it in the path must encode it.
            path = unquote(raw_path).rstrip("/") or "/"

            if path == "/health":
                self._send({"ok": True, "sqlite_path": service.sqlite_path})
                return
            if path == "/episodes":
                self._send({"episodes": service.episodes()})
                return
            if path == "/audit":
                episode_id = (query.get("episode") or [""])[0]
                if not episode_id:
                    self._send({"error": "missing ?episode=<id>"}, 400)
                    return
                self._send(service.audit(episode_id))
                return
            if path == "/current":
                episode_id = (query.get("episode") or [""])[0]
                if not episode_id:
                    self._send({"error": "missing ?episode=<id>"}, 400)
                    return
                self._send(service.current(episode_id))
                return
            # Preferred form: fact ids carry "#", which a URL treats as a fragment
            # start, so they belong in the query string rather than the path.
            if path in ("/fact", "/fact/evidence"):
                fact_id = (query.get("fact_id") or [""])[0]
                if not fact_id:
                    self._send({"error": "missing ?fact_id=<id>"}, 400)
                    return
                payload = (service.evidence if path.endswith("/evidence") else service.fact)(fact_id)
                self._send(payload if payload is not None else {"error": "unknown fact_id"},
                           200 if payload is not None else 404)
                return
            if path == "/tombstones":
                episode_id = (query.get("episode") or [""])[0]
                if not episode_id:
                    self._send({"error": "missing ?episode=<id>"}, 400)
                    return
                self._send(service.tombstones(episode_id))
                return
            if path == "/history":
                episode_id = (query.get("episode") or [""])[0]
                slot = (query.get("slot") or [""])[0]
                if not episode_id or not slot:
                    self._send({"error": "missing ?episode=<id>&slot=<slot>"}, 400)
                    return
                raw_upto = (query.get("upto_turn") or [""])[0]
                try:
                    upto = int(raw_upto) if raw_upto else None
                except ValueError:
                    self._send({"error": "upto_turn must be an integer"}, 400)
                    return
                self._send(service.history(episode_id, slot, upto))
                return
            if path == "/audit/summary":
                payload = service.summary()
                payload.pop("reports", None)      # per-episode detail lives at /audit
                self._send(payload)
                return
            if path == "/temporal":
                episode_id = (query.get("episode") or [""])[0]
                if not episode_id:
                    self._send({"error": "missing ?episode=<id>"}, 400)
                    return
                self._send(service.temporal(episode_id))
                return
            if path.startswith("/fact/"):
                rest = path[len("/fact/"):]
                want_evidence = rest.endswith("/evidence")
                fact_id = rest[: -len("/evidence")] if want_evidence else rest
                payload = (service.evidence if want_evidence else service.fact)(fact_id)
                self._send(payload if payload is not None else {"error": "unknown fact_id"},
                           200 if payload is not None else 404)
                return
            self._send({"error": "unknown endpoint", "path": path}, 404)

        def do_POST(self) -> None:                # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if token and (query.get("token") or [""])[0] != token:
                self._send({"error": "forbidden: pass ?token=<token>"}, 403)
                return
            path = unquote(parsed.path).rstrip("/") or "/"
            if path != "/facts":
                self._send(
                    {"error": "read-only service: the only accepted write is POST /facts"},
                    405,
                )
                return
            episode_id = (query.get("episode") or [""])[0]
            if not episode_id:
                self._send({"error": "missing ?episode=<id>"}, 400)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send({"error": f"body must be 1..{MAX_BODY_BYTES} bytes"}, 413)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                self._send({"error": f"invalid JSON body: {exc}"}, 400)
                return
            facts = payload.get("facts") if isinstance(payload, dict) else None
            if not isinstance(facts, list):
                self._send({"error": 'body must be {"facts": [...]}'}, 400)
                return
            self._send(service.record_facts(episode_id, facts))

    return Handler


def build_server(host: str, port: int, sqlite_path: str, token: str) -> ThreadingHTTPServer:
    service = AuditService(sqlite_path, token)
    httpd = ThreadingHTTPServer((host, port), make_handler(service, token))
    httpd.audit_service = service          # type: ignore[attr-defined]
    return httpd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sqlite-path", default="exp_memory.db",
                        help="SQLite file the agent wrote (read-only here)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--token", default="mem2024")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if not os.path.exists(args.sqlite_path):
        print(f"error: {args.sqlite_path} does not exist", file=sys.stderr)
        return 2
    httpd = build_server(args.host, args.port, args.sqlite_path, args.token)
    print(f"audit sidecar on http://{args.host}:{args.port}  (db={args.sqlite_path})")
    print(f"  GET /health | /episodes | /audit?episode=<id> | /current?episode=<id>")
    print(f"  GET /fact/<fact_id> | /fact/<fact_id>/evidence")
    print(f"  GET /tombstones?episode=<id> | /history?episode=<id>&slot=<s> | /temporal?episode=<id>")
    print(f"  all requests need ?token={args.token}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.audit_service.close()        # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
