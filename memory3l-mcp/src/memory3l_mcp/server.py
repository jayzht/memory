"""
memory3l as an MCP server: the auditable-memory checks on the model's tool list.

What this is for
----------------
A three-layer memory system claims it never loses a fact.  Almost no such system
can *prove* it, because nothing keeps an independent record to compare against.
``memory3l`` keeps one -- an append-only fact ledger written at the moment a
summary is created, including the summaries that are later overridden or
capacity-merged away -- and cross-checks the live store against it.

This server exposes those checks as tools, so an agent can ask "is this value
still reachable at the top level?", "why did this fact leave the working set?",
"what was erased, and is the prose that mentioned it gone too?" and get an
answer derived from persisted state.

Two design choices worth knowing
--------------------------------
* **Read-only by default.**  Nine of the ten tools only read.  The one write
  (``append_facts``, an intake for a caller's own extractor) is registered only
  when ``--allow-write`` or ``MEMORY3L_ALLOW_WRITE=1`` is set, so a default
  install cannot mutate anybody's ledger.
* **Answers come from SQLite, never a cache.**  An audit is usually interesting
  *after* the writing process is gone, so the hot layer is irrelevant here.  One
  file is the whole input, which is also why this can run anywhere.

Transport is stdio, which is what every MCP client can launch without a port.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .bootstrap import MemoryCoreNotFound, ensure_memory3l
from .config import DB_ENV, DEFAULT_DIR_ENV, resolve_db_path

logger = logging.getLogger("memory3l_mcp")

SERVER_NAME = "memory3l-mcp"
WRITE_ENV = "MEMORY3L_ALLOW_WRITE"

INSTRUCTIONS = (
    "Audited three-layer memory: an append-only fact ledger plus invariants I1-I5 "
    "checked against persisted state. Use `list_episodes` first to learn the valid "
    "episode ids; the other tools require one. `store_info` says which database is "
    "being served. A `fact_id` looks like `<episode>/<kind><seq>@<hash>#<slot>` and "
    "contains a '#'."
)

# Below this many characters a payload is re-indented for readability; above it,
# compact JSON is returned instead, because indentation roughly doubles the token
# cost of a large audit report and buys nothing the model cannot parse.
_PRETTY_LIMIT = 4000


def _json(payload: Any) -> str:
    """Serialise a payload, pretty-printing only while that is cheap."""
    compact = json.dumps(payload, ensure_ascii=False, default=str)
    if len(compact) <= _PRETTY_LIMIT:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return compact


def _error(message: str, **extra: Any) -> str:
    """A tool failure as data, so the caller sees the reason instead of a fault."""
    return _json({"error": message, **extra})


def _as_int(value: Any) -> int | None:
    """Coerce an optional turn number, keeping a legitimate 0."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_server(db_path: Path, allow_write: bool = False):
    """
    Construct the MCP server with its tools registered.

    The store is opened eagerly so that a misconfigured path is reported by
    ``store_info`` rather than surprising the first real query -- but a failure
    does *not* stop the server: the tools still register and each one returns the
    reason it cannot answer.  A server that refuses to start would leave the user
    with no tool list and no diagnosis.
    """
    ensure_memory3l()

    from mcp.server.mcpserver import MCPServer

    from memory3l.service import AuditService

    service: AuditService | None = None
    open_error: str | None = None
    try:
        service = AuditService(str(db_path))
    except Exception as error:  # noqa: BLE001 - reported through the tools
        open_error = f"{type(error).__name__}: {error}"
        logger.warning("cannot open %s: %s", db_path, open_error)

    def _service() -> tuple[AuditService | None, str | None]:
        return service, open_error

    def _guard(episode_id: str) -> str | None:
        """
        A shared precondition message, or None when the call may proceed.

        The existence check is not politeness -- it prevents a *false clean bill of
        health*.  An episode with no facts trivially satisfies every invariant, so
        auditing a misspelled or wrong id returns ``ok: true``, which is
        indistinguishable from a genuine pass.  For a tool whose whole purpose is
        "did anything get lost?", answering "no" about an episode that does not
        exist is the worst possible failure, so an unknown id is an explicit error
        that also lists the ids that do exist.
        """
        active, failure = _service()
        if active is None:
            return _error("ledger unavailable", path=str(db_path), detail=failure)
        episode = (episode_id or "").strip()
        if not episode:
            return _error("episode_id is required", hint="call list_episodes")
        try:
            known = active.episodes()
        except Exception as error:  # noqa: BLE001
            return _error("cannot list episodes", detail=f"{type(error).__name__}: {error}")
        if episode not in known and not active.store.list_active_summaries(episode):
            return _error(
                "unknown episode",
                episode_id=episode,
                available=known[:50],
                hint=(
                    "call list_episodes for the real ids. This is an error rather "
                    "than an empty result on purpose: an episode with no facts has "
                    "no violations, so reporting a pass here would be a false "
                    "all-clear rather than a clean audit."
                ),
            )
        return None

    mcp = MCPServer(
        name=SERVER_NAME,
        title="Auditable memory (memory3l)",
        instructions=INSTRUCTIONS,
    )

    # ---------------------------------------------------------------- info --
    @mcp.tool(
        title="Which ledger is being served",
        description=(
            "Report the database this server reads, whether it exists, its size, and how "
            "many episodes carry an audit ledger. Call this first when a query returns "
            "nothing: an empty or wrong path is the usual cause, not an absent feature."
        ),
    )
    def store_info() -> str:
        active, failure = _service()
        episodes = []
        if active is not None:
            try:
                episodes = active.episodes()
            except Exception as error:  # noqa: BLE001
                failure = f"{type(error).__name__}: {error}"
        return _json({
            "path": str(db_path),
            "exists": db_path.is_file(),
            "size_bytes": db_path.stat().st_size if db_path.is_file() else 0,
            "episodes": len(episodes),
            "readable": active is not None,
            "detail": failure,
            "writes_enabled": allow_write,
            "configured_by": (
                f"{DB_ENV}" if os.environ.get(DB_ENV)
                else f"{DEFAULT_DIR_ENV}" if os.environ.get(DEFAULT_DIR_ENV)
                else "default"
            ),
        })

    @mcp.tool(
        title="List audited episodes",
        description=(
            "List the episode ids that have an audit ledger. Every other tool needs one "
            "of these ids, so call this when you do not already know the episode."
        ),
    )
    def list_episodes() -> str:
        active, failure = _service()
        if active is None:
            return _error("ledger unavailable", path=str(db_path), detail=failure)
        try:
            return _json({"episodes": active.episodes()})
        except Exception as error:  # noqa: BLE001
            return _error("cannot list episodes", detail=f"{type(error).__name__}: {error}")

    # --------------------------------------------------------------- audit --
    @mcp.tool(
        title="Audit one episode against I1-I5",
        description=(
            "Run the invariant checks over one episode and return the violations. This is "
            "the tool to call to answer 'did this memory lose anything?'. I1 = a fact in "
            "the ledger no longer reaches any summary (silent loss); I2 = a slot's newest "
            "value is not visible at the top level, so the model is never told it; I3 = a "
            "fact's evidence no longer resolves to stored dialogue; I4 = extraction "
            "completeness against the raw turns; I5 = an erasure is verifiable and its "
            "evidence is gone. `ok: true` with no violations means none of the checked "
            "failure modes is present -- report that as the result, not as a guarantee "
            "about meanings."
        ),
    )
    def audit(episode_id: str) -> str:
        blocked = _guard(episode_id)
        if blocked:
            return blocked
        try:
            return _json(service.audit(episode_id))
        except Exception as error:  # noqa: BLE001
            return _error("audit failed", episode_id=episode_id,
                          detail=f"{type(error).__name__}: {error}")

    @mcp.tool(
        title="Audit every episode",
        description=(
            "Aggregate the invariant checks over every episode that has a ledger, "
            "including the `silent_loss_rate`. Use this for an overall health answer; use "
            "`audit` when you already know the episode and want the detail."
        ),
    )
    def audit_summary() -> str:
        active, failure = _service()
        if active is None:
            return _error("ledger unavailable", path=str(db_path), detail=failure)
        try:
            return _json(active.summary())
        except Exception as error:  # noqa: BLE001
            return _error("aggregate audit failed", detail=f"{type(error).__name__}: {error}")

    # ------------------------------------------------------------- current --
    @mcp.tool(
        title="Current value of every known slot",
        description=(
            "Return the derived current-value registry for an episode: the newest value of "
            "each attribute, with the summary it came from. This is what the model should "
            "be told at the top level. If a value the conversation established is missing "
            "here, that is an I2 failure -- confirm it with `audit`."
        ),
    )
    def current(episode_id: str) -> str:
        blocked = _guard(episode_id)
        if blocked:
            return blocked
        try:
            return _json(service.current(episode_id))
        except Exception as error:  # noqa: BLE001
            return _error("current lookup failed", episode_id=episode_id,
                          detail=f"{type(error).__name__}: {error}")

    @mcp.tool(
        title="Every value a slot ever held",
        description=(
            "Return one slot's value over time, oldest first, bounded by `upto_turn` when "
            "given. Use this for 'what was it before?' and for any question about change; "
            "`current` only answers 'what is it now?'. A slot is a lower-case attribute "
            "name such as `工位` or `office` -- call `current` first if you do not know the "
            "exact spelling, since this matches the slot literally."
        ),
    )
    def history(episode_id: str, slot: str, upto_turn: int | None = None) -> str:
        blocked = _guard(episode_id)
        if blocked:
            return blocked
        if not slot or not slot.strip():
            return _error("slot is required", hint="call current to list slot names")
        try:
            result = service.history(episode_id, slot, upto_turn=_as_int(upto_turn))
            if not result.get("history"):
                result["hint"] = (
                    "no entry for this slot; the name must match exactly -- "
                    "call current for the available names"
                )
            return _json(result)
        except Exception as error:  # noqa: BLE001
            return _error("history lookup failed", episode_id=episode_id, slot=slot,
                          detail=f"{type(error).__name__}: {error}")

    # --------------------------------------------------------------- facts --
    @mcp.tool(
        title="Why a fact left the working set",
        description=(
            "Given a `fact_id`, report the fact's full record and where it now lives: "
            "`live` (still in an active summary), `archived` (moved to the permanent "
            "archive, reachable by id) or `MISSING` (in the ledger but unreachable, which "
            "is an I1 silent loss). Also returns `reason` and `superseded_by`. Use this "
            "when asked why a piece of information is no longer in context."
        ),
    )
    def fact(fact_id: str) -> str:
        active, failure = _service()
        if active is None:
            return _error("ledger unavailable", path=str(db_path), detail=failure)
        if not fact_id or not fact_id.strip():
            return _error("fact_id is required",
                          hint="fact_id looks like <episode>/<kind><seq>@<hash>#<slot>")
        try:
            result = active.fact(fact_id.strip())
        except Exception as error:  # noqa: BLE001
            return _error("fact lookup failed", fact_id=fact_id,
                          detail=f"{type(error).__name__}: {error}")
        if result is None:
            return _error("no such fact in any ledger", fact_id=fact_id,
                          hint="check the episode with list_episodes, and keep the '#' in the id")
        return _json(result)

    @mcp.tool(
        title="Original dialogue behind a fact",
        description=(
            "Return the raw user/agent turns a fact was extracted from, resolved through "
            "its evidence pointers. Use this to justify a stored value with the actual "
            "conversation rather than the summary's paraphrase. `resolved: false` means at "
            "least one evidence pointer no longer resolves -- an I3 provenance failure; "
            "that matters and should be reported, not hidden."
        ),
    )
    def evidence(fact_id: str) -> str:
        active, failure = _service()
        if active is None:
            return _error("ledger unavailable", path=str(db_path), detail=failure)
        if not fact_id or not fact_id.strip():
            return _error("fact_id is required",
                          hint="fact_id looks like <episode>/<kind><seq>@<hash>#<slot>")
        try:
            result = active.evidence(fact_id.strip())
        except Exception as error:  # noqa: BLE001
            return _error("evidence lookup failed", fact_id=fact_id,
                          detail=f"{type(error).__name__}: {error}")
        if result is None:
            return _error("no such fact in any ledger", fact_id=fact_id)
        return _json(result)

    # ------------------------------------------------- compliance erasure --
    @mcp.tool(
        title="What was erased, and what remains",
        description=(
            "List the compliance tombstones for an episode: facts deliberately erased, "
            "with their evidence pointers, which of those are still readable, and how many "
            "residual prose mentions survive. This is the deletion-verification view (I5). "
            "A non-empty `residual_prose_mentions` or a readable evidence pointer means the "
            "content is not fully gone from the store -- report that plainly instead of "
            "calling the erasure complete."
        ),
    )
    def tombstones(episode_id: str) -> str:
        blocked = _guard(episode_id)
        if blocked:
            return blocked
        try:
            return _json(service.tombstones(episode_id))
        except Exception as error:  # noqa: BLE001
            return _error("tombstone lookup failed", episode_id=episode_id,
                          detail=f"{type(error).__name__}: {error}")

    @mcp.tool(
        title="Versioned projection and its anomalies",
        description=(
            "Return the episode's temporal (bitemporal-style) projection: row count, "
            "whether it is consistent with the ledger, any anomalies, and the DDL for "
            "recreating the table. Use this when memory state is projected into an "
            "external store and you need to know whether the two still agree."
        ),
    )
    def temporal(episode_id: str) -> str:
        blocked = _guard(episode_id)
        if blocked:
            return blocked
        try:
            return _json(service.temporal(episode_id))
        except Exception as error:  # noqa: BLE001
            return _error("temporal projection failed", episode_id=episode_id,
                          detail=f"{type(error).__name__}: {error}")

    # ------------------------------------------------------- optional write --
    if allow_write:
        @mcp.tool(
            title="Record externally extracted facts",
            description=(
                "Append pre-extracted facts to an episode's ledger, idempotently, for a "
                "caller that runs its own extractor. Each fact needs slot, value and "
                "summary_id; the key is `<summary_id>#<slot>`, so a retry adds nothing. The "
                "audit runs before the answer, so facts that are not backed by a real "
                "summary come back as I1 violations. Note that adding a fact cannot *make* "
                "it visible: the current-value registry is derived from the summaries, so "
                "the summary's own text must carry the value."
            ),
        )
        def append_facts(episode_id: str, facts: Sequence[dict]) -> str:
            blocked = _guard(episode_id)
            if blocked:
                return blocked
            if not facts:
                return _error("facts must be a non-empty list")
            try:
                return _json(service.record_facts(episode_id, list(facts)))
            except Exception as error:  # noqa: BLE001
                return _error("fact intake failed", episode_id=episode_id,
                              detail=f"{type(error).__name__}: {error}")

    return mcp


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, resolve storage, and serve MCP over stdio."""
    parser = argparse.ArgumentParser(
        prog=SERVER_NAME,
        description=(
            "Auditable three-layer memory over MCP. Serves one SQLite fact-ledger file "
            "read-only; the file is created on demand, so a fresh install answers with an "
            "empty episode list instead of failing."
        ),
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        help=(
            f"ledger database. Overrides ${DB_ENV}; default "
            f"${DEFAULT_DIR_ENV}/memory3l.db or ~/.memory3l/memory3l.db"
        ),
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help=f"also expose the append_facts tool (same as {WRITE_ENV}=1)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("MEMORY3L_LOG_LEVEL", "WARNING"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="stderr log level (stdout carries the protocol)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,           # stdout is the MCP transport, never logs
        format="%(levelname)s %(name)s: %(message)s",
    )

    allow_write = args.allow_write or os.environ.get(WRITE_ENV, "").lower() in {
        "1", "true", "yes", "on",
    }
    db_path = resolve_db_path(args.db)

    try:
        mcp = build_server(db_path, allow_write=allow_write)
    except MemoryCoreNotFound as error:
        print(error, file=sys.stderr)
        return 2

    logger.info("serving %s (writes %s)", db_path, "enabled" if allow_write else "disabled")
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
