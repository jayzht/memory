"""
Function calling by text parsing.

Three grammars are handled here, all deliberately strict so that failures are
*measurable* (they end up in ``Tool_Call_Success_Rate``) instead of silent:

1. **Summariser output**
   ``<summary text>`` + a final line ``[OVERRIDES: id1,id2]`` / ``[OVERRIDES: none]``.
   Tolerated deviations: the tag may be on the same line, may use ``；``/``:``
   instead of ``,``/``:``, may be missing (then ``missing_tag=True`` and the
   caller counts a format failure), and markdown fences are stripped.
   An id that is not in the active chain is *rejected* -- the chain is the only
   authority on valid ids, and a hallucinated id must not silently corrupt the
   override graph.
1b. **Self-written memory** (same grammar, delivered inside a ``<MEMORY_UPDATE>``
   block appended to the answering model's own final reply; see
   :func:`split_selfwrite_reply`).  Identical grammar on purpose: one parser and
   one apply path serve both sources, so the override graph never depends on
   *who* wrote a summary.
2. **Agent tool calls**
   ``get_archived_summary(summary_id="...")`` / ``get_raw_record(reference_id="...")``
   with or without quotes, single/double quotes, ``=`` or ``:``, python- or
   json-style kwargs, and an optional code fence.

No vectors, no embedding, no fuzzy matching anywhere: ids are compared
character-by-character.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .models import LAZY_INDEX_ID, ToolCall, ToolResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Summariser output parsing
# --------------------------------------------------------------------------- #
_OVERRIDES_RE = re.compile(
    r"\[\s*OVERRIDES\s*[:：]\s*([^\]\[]*?)\s*\]", re.IGNORECASE | re.DOTALL
)
# Optional machine-readable fact line:  [FACTS: slot=value; slot2=value2]
# It is stripped from the summary body and used as the authoritative fact-key list
# (falling back to lexical extraction when a model omits it).
_FACTS_RE = re.compile(
    r"\[\s*FACTS\s*[:：]\s*([^\]\[]*?)\s*\]", re.IGNORECASE | re.DOTALL
)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

NONE_TOKENS = {"none", "null", "无", "空", "-", "n/a", "na", ""}

#: Resolved by :class:`ToolExecutor` to the most recently archived (overridden)
#: summary of the bound episode.  It exists so a caller that knows *which*
#: attribute it wants but not *which* summary holds it can still drive the exact
#: id lookup path -- used by the offline heuristic backend.
#: ``__MOST_RECENT_OVERRIDDEN__`` optionally takes a ``|<slot>`` suffix so the
#: resolved summary is the newest archived one mentioning that attribute.
PLACEHOLDER_RECENT_OVERRIDDEN = "__MOST_RECENT_OVERRIDDEN__"


class SummaryParseError(ValueError):
    """Raised when a summariser response cannot be parsed at all."""


def _strip_tag(body: str, pattern: "re.Pattern") -> "tuple":
    """Remove the first match of ``pattern`` from ``body``; return (body, payload)."""
    match = pattern.search(body)
    if not match:
        return body, None
    return (body[: match.start()] + body[match.end() :]).strip(), match.group(1)


def parse_summary_response(
    text: str,
    valid_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Parse one summariser response.

    Returns
    -------
    dict with keys:
        ``summary_text``   cleaned summary body
        ``override_ids``   validated ids (subset of ``valid_ids`` when given)
        ``raw_override_ids`` ids as literally emitted by the model
        ``invalid_override_ids`` emitted ids that are not in the active chain
        ``missing_tag``    True when no OVERRIDES tag was found
        ``valid``          False when the body is empty
    """
    if text is None:
        raise SummaryParseError("empty model response")
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    if not cleaned:
        raise SummaryParseError("empty model response")

    missing_tag = _OVERRIDES_RE.search(cleaned) is None
    raw_ids: List[str] = []

    # Both tag lines are stripped from the body before anything else, so the
    # stored summary text contains only the summary itself.
    body, facts_payload = _strip_tag(cleaned, _FACTS_RE)
    body, overrides_payload = _strip_tag(body, _OVERRIDES_RE)
    fact_keys: List[str] = []
    if facts_payload:
        for chunk in re.split(r"[;；|]", facts_payload):
            if "=" in chunk:
                key, _, value = chunk.partition("=")
                key = key.strip().strip("，,。.；;:： ")
                value = value.strip().strip("，,。.；;:： ")
                if key and key.lower() not in NONE_TOKENS:
                    # Keep the value: the chain renderer shows this line to the
                    # summariser, which needs the current value to detect conflicts.
                    fact_keys.append(f"{key}={value}" if value else key)
    if overrides_payload is not None:
        raw_ids = [p.strip() for p in re.split(r"[,，;；\s]+", overrides_payload) if p.strip()]
        raw_ids = [i for i in raw_ids if i.lower() not in NONE_TOKENS]
    # Trailing labels the model sometimes echoes.
    body = re.sub(r"^\s*(?:摘要|summary)\s*[:：]\s*", "", body, flags=re.IGNORECASE).strip()
    body = body.strip("` \n\t")

    valid_set = set(valid_ids) if valid_ids is not None else None
    accepted: List[str] = []
    invalid: List[str] = []
    for summary_id in raw_ids:
        if valid_set is not None and summary_id not in valid_set:
            invalid.append(summary_id)
            continue
        if summary_id not in accepted:
            accepted.append(summary_id)

    if invalid:
        logger.debug("summariser emitted %d invalid override id(s): %s", len(invalid), invalid)

    return {
        "summary_text": body,
        "fact_keys": fact_keys,
        "override_ids": accepted,
        "raw_override_ids": raw_ids,
        "invalid_override_ids": invalid,
        "missing_tag": missing_tag,
        "valid": bool(body),
    }


# --------------------------------------------------------------------------- #
# 1b. Self-written memory block
# --------------------------------------------------------------------------- #
# A second way to obtain a turn summary: instead of paying for a *separate*
# summariser call, the answering model appends the same grammar to its own final
# reply inside a <MEMORY_UPDATE> block.  One call then both answers the user and
# records the turn.  Harvesting is opportunistic -- when the block is missing or
# malformed the caller falls back to the dedicated summariser, so a model that
# ignores the instruction costs one extra call but never loses an update.
_MEMORY_UPDATE_RE = re.compile(
    r"<\s*MEMORY_UPDATE\s*>(.*?)(?:<\s*/\s*MEMORY_UPDATE\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Tolerated variant: a fenced block labelled memory_update (some models cannot
# resist wrapping machine-readable output in a fence).
_MEMORY_UPDATE_FENCE_RE = re.compile(
    r"```[ \t]*(?:memory[_-]?update|memory)[ \t]*\r?\n?(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def split_selfwrite_reply(text: str) -> "tuple":
    """
    Split one agent reply into ``(visible_answer, memory_block_or_None)``.

    The visible answer is what the user (and the judge) sees; the block is the
    summariser grammar, parsed by :func:`parse_summary_response`.
    """
    if not text:
        return "", None
    match = _MEMORY_UPDATE_RE.search(text)
    if match is None:
        match = _MEMORY_UPDATE_FENCE_RE.search(text)
    if match is None:
        return text.strip(), None
    block = (match.group(1) or "").strip()
    visible = (text[: match.start()] + text[match.end() :]).strip()
    # A reply that is *only* the block still has to answer something.
    return (visible or text.strip()), (block or None)


# --------------------------------------------------------------------------- #
# 2. Tool call parsing
# --------------------------------------------------------------------------- #
TOOL_GET_ARCHIVED = "get_archived_summary"
TOOL_GET_RAW = "get_raw_record"
TOOL_LIST_ACTIVE = "list_active_summaries"          # optional debug convenience
#: Exact-attribute lookup over the archive: "give me the most recent archived fact
#: for attribute X".  This is the deterministic way to answer "what was it before":
#: it walks the archive in reverse chronological order and returns the newest entry
#: for that attribute, instead of the model having to guess among many old ids.
#: Still pure id/attribute matching -- no vectors, no similarity ranking.
TOOL_GET_PREDECESSOR = "get_predecessor_summary"
#: Expand a catalogue entry: "index_id -> the summaries it points at".  This is the
#: drill-down step of the hierarchy (title -> catalogue entries -> raw dialogue).
TOOL_EXPAND_INDEX = "expand_index"
KNOWN_TOOLS = (
    TOOL_GET_ARCHIVED,
    TOOL_GET_RAW,
    TOOL_LIST_ACTIVE,
    TOOL_GET_PREDECESSOR,
    TOOL_EXPAND_INDEX,
)

_CALL_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?P<args>[^)]*?)\s*\)", re.DOTALL
)
_KWARG_RE = re.compile(
    r"""["']?(?P<key>[A-Za-z_][A-Za-z0-9_]*)["']?\s*(?:=|:)\s*
        (?:["'](?P<qval>[^"']*)["']|(?P<val>[^,)\s]+))""",
    re.VERBOSE | re.DOTALL,
)
# Some chat templates render a JSON tool call instead of the textual grammar.
_JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*?\})\s*\}', re.DOTALL)


def is_tool_call(text: str) -> bool:
    """True when ``text`` looks like a tool call rather than a final answer."""
    if not text:
        return False
    return parse_tool_calls(text) != []


#: Lines the harness itself writes back into the conversation.  They must never be
#: re-parsed as a fresh tool call, otherwise a model that quotes a tool result
#: would loop forever.
_TOOL_ECHO_RE = re.compile(
    r"^\s*(?:\[TOOL RESULT[^\]]*\]|-{2,}\s*TOOL RESULT|\[SUPERSEDED BY[^\]]*\]|\[NOTE\]).*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_tool_echoes(text: str) -> str:
    """Remove harness-written tool result lines from a model reply."""
    if not text:
        return ""
    return _TOOL_ECHO_RE.sub("", text).strip()


# --------------------------------------------------------------------------- #
# DeepSeek DSML markup
# --------------------------------------------------------------------------- #
# Some DeepSeek checkpoints emit their internal markup instead of plain text::
#
#   <|DSML|invoke name="get_raw_record"><|DSML|parameter name="reference_id">
#   ep/raw030@c73002</|DSML|parameter></|DSML|invoke>
#
# (pipes may be ASCII ``|`` or full-width ``｜``).  The model *did* make a
# well-formed call, so scoring it as a format failure would understate the system.
_DSML_OPEN = r"<[|｜]{1,2}\s*DSML\s*[|｜]{1,2}\s*"
_DSML_CLOSE = r"</[|｜]{1,2}\s*DSML\s*[|｜]{1,2}\s*"
_DSML_INVOKE_RE = re.compile(
    _DSML_OPEN + r"invoke\s+name\s*=\s*[\"'](?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"'][^>]*>"
    r"(?P<body>.*?)" + _DSML_CLOSE + r"invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM_RE = re.compile(
    _DSML_OPEN + r"parameter\s+name\s*=\s*[\"'](?P<key>[A-Za-z_][A-Za-z0-9_]*)[\"'][^>]*>"
    r"(?P<value>.*?)" + _DSML_CLOSE + r"parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
# A bare DSML control token with no tool name attached is model chatter.
_DSML_NOISE_RE = re.compile(r"<[|｜]{1,2}\s*DSML\s*[|｜]{1,2}\s*/?\s*(?:calls?|invoke|parameter)?\s*>",
                            re.IGNORECASE)


def parse_tool_calls(text: str) -> List[ToolCall]:
    """
    Extract tool calls from a model reply (empty list when there are none).

    Handles four shapes, all of which real models emit in practice:

    * ``name(key="value")`` / ``name(key=value)`` -- the grammar the prompt asks for;
    * the JSON ``{"name": ..., "arguments": {...}}`` form;
    * DeepSeek DSML markup (``<|DSML|invoke name="...">...``);
    * the same call wrapped in a markdown code fence.

    Harness-written tool results (``[TOOL RESULT ...]``) are never treated as calls.
    """
    if not text:
        return []
    # Strip code fences and DSML chatter *tokens* only, never whole lines: a DSML
    # tool call is emitted on its own line, and a line-based echo filter would
    # delete it before the DSML pattern ever sees it.
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    cleaned = _DSML_NOISE_RE.sub(" ", cleaned)
    if not cleaned.strip():
        return []
    calls: List[ToolCall] = []

    # DeepSeek internal markup form.
    for match in _DSML_INVOKE_RE.finditer(cleaned):
        name = match.group("name").strip()
        if name not in KNOWN_TOOLS:
            continue
        args = {
            param.group("key"): param.group("value").strip().strip("\"'")
            for param in _DSML_PARAM_RE.finditer(match.group("body"))
        }
        calls.append(ToolCall(name=name, args=args, raw=match.group(0)[:200]))
    if calls:
        return calls

    # JSON form (more specific than the textual grammar).
    for match in _JSON_CALL_RE.finditer(cleaned):
        name = match.group("name").strip()
        if name not in KNOWN_TOOLS:
            continue
        try:
            import json

            args = json.loads(match.group("args"))
        except Exception:  # noqa: BLE001
            args = {}
        calls.append(
            ToolCall(name=name, args={str(k): str(v) for k, v in dict(args).items()}, raw=match.group(0))
        )
    if calls:
        return calls

    for match in _CALL_RE.finditer(cleaned):
        name = match.group("name").strip()
        if name not in KNOWN_TOOLS:
            continue
        args: Dict[str, str] = {}
        for kw in _KWARG_RE.finditer(match.group("args")):
            value = kw.group("qval")
            if value is None:
                value = kw.group("val") or ""
            args[kw.group("key")] = value.strip()
        calls.append(ToolCall(name=name, args=args, raw=match.group(0)))
    return calls


def normalise_tool_args(call: ToolCall) -> Dict[str, str]:
    """
    Map the many argument names models invent onto the two canonical ones.

    ``get_archived_summary(summary_id=...)`` also accepts ``id``/``sid``;
    ``get_raw_record(reference_id=...)`` also accepts ``ref``/``ref_id``/``id``.
    """
    out: Dict[str, str] = {}
    for key, value in call.args.items():
        lowered = re.sub(r"[^a-z]", "", key.lower())
        if lowered in ("summaryid", "sid", "summary", "id"):
            out.setdefault("summary_id", value)
        elif lowered in ("referenceid", "refid", "ref", "rawrefid", "reference", "rawid"):
            out.setdefault("reference_id", value)
        elif lowered in ("indexid", "index", "entryid", "catalogueid", "catalogid"):
            out.setdefault("index_id", value)
        elif lowered in ("factkey", "key", "attribute", "attr", "slot", "field", "fact"):
            out.setdefault("fact_key", value)
        elif lowered in ("referencevalue", "reference", "refvalue", "anchor", "anchorvalue",
                          "newvalue", "became", "currentvalue"):
            out.setdefault("reference_value", value)
        elif lowered in ("excludevalue", "exclude", "except", "oldvalue", "othervalue"):
            out.setdefault("exclude_value", value)
        elif lowered in ("value",):
            # "value" is ambiguous; treat it as the anchor (what the fact became),
            # which is the more useful meaning for a "before it became X" question.
            out.setdefault("reference_value", value)
        else:
            out.setdefault(key, value)
    return out


# --------------------------------------------------------------------------- #
# 3. Tool execution
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_FACT_LINE_RE = re.compile(r"([^=;|\n]+?)\s*=\s*([^;|\n]+)")


def _normalise_value(text: str) -> str:
    """Lowercase, strip whitespace/punctuation -- for value comparison only."""
    return re.sub(r"[\s\u3000。.,;；:：!！?？\"'“”‘’()（）\[\]【】]+", "", str(text or "").lower())


def _slot_value(text: str, slot: str) -> str:
    """
    Extract ``slot``'s value from a summary text of the form ``Facts: a=1; b=2``.

    Returns ``""`` when the slot is absent.  Purely lexical (exact-id retrieval is
    the system's contract; this helper only serves the offline heuristic).
    """
    if not text:
        return ""
    lowered = text.lower()
    for match in _FACT_LINE_RE.finditer(lowered):
        if slot and slot in match.group(1).strip():
            value = match.group(2).strip().strip("。.,;；!！?？\"'“”")
            if value:
                return value
    if slot and slot in lowered:
        tail = lowered.split(slot, 1)[1]
        # Natural-language summaries ("把工位楼层从3楼改为7楼") put the current value
        # after the change verb; the naive "everything after the slot" rule would
        # return the whole clause and make candidate ranking meaningless.
        current = re.findall(r"(?:改为|改成|改到|变成|变为|更新为|现在(?:是|为)|更改为)\s*([^，,。.;；\n]{1,40})", tail)
        if current:
            return current[-1].strip().strip("。.,;；!！?？\"'“”")
        after_from = re.findall(r"从\s*[^，,。.;；\n]{1,40}?\s*(?:改为|改成|变成)\s*([^，,。.;；\n]{1,40})", tail)
        if after_from:
            return after_from[-1].strip().strip("。.,;；!！?？\"'“”")
        tail = re.sub(r"^[\s=:：\-–—]+", "", tail).split(";")[0]
        return tail.strip().strip("。.,;；!！?？\"'“”")
    return ""


class ToolExecutor:
    """
    Executes the two archival tools against a store.

    Every call returns a :class:`ToolResult`; nothing raises, so the evaluation
    loop can keep going and simply record ``ok=False``.
    """

    def __init__(self, store, episode_id: Optional[str] = None):
        self.store = store
        self.episode_id = episode_id

    # -- individual tools --------------------------------------------------- #
    def resolve_placeholder(self, summary_id: str) -> str:
        """
        Expand the offline-heuristic placeholder into a real summary id.

        Encodings understood (a real id is returned unchanged, so honest tool
        calls are never affected)::

            __MOST_RECENT_OVERRIDDEN__
            __MOST_RECENT_OVERRIDDEN__|<slot>
            __MOST_RECENT_OVERRIDDEN__|<slot>|<exclude_value>

        Selection: newest archived (overridden first) summary whose text mentions
        ``<slot>`` and, when ``<exclude_value>`` is given, whose fact value for the
        slot differs from it.  Falls back progressively so the tool still returns
        something rather than failing outright.
        """
        if not summary_id.startswith(PLACEHOLDER_RECENT_OVERRIDDEN):
            return summary_id
        parts = summary_id.split("|")
        slot = parts[1].strip().lower() if len(parts) > 1 else ""
        exclude = parts[2].strip().lower() if len(parts) > 2 else ""

        archived = self.store.list_archived_summaries(self.episode_id)
        overridden = [a for a in archived if a.is_overridden]
        for pool in (overridden, archived):
            if not pool:
                continue
            candidates = pool
            if slot:
                matching = [a for a in candidates if slot in a.text.lower()]
                candidates = matching or candidates
            if exclude:
                differing = [
                    a for a in candidates if _slot_value(a.text, slot) not in ("", exclude)
                ]
                candidates = differing or candidates
            if candidates:
                return candidates[-1].summary_id
        return summary_id

    def get_archived_summary(self, summary_id: str) -> ToolResult:
        started = time.perf_counter()
        if not summary_id:
            return self._error(TOOL_GET_ARCHIVED, "missing summary_id argument", parsed=True)
        summary_id = self.resolve_placeholder(summary_id)
        # An index id passed here is a common model slip: expand it instead of failing.
        if summary_id.startswith(PLACEHOLDER_RECENT_OVERRIDDEN) is False and "/idx" in summary_id:
            entry = self.store.get_index_entry(summary_id, episode_id=self.episode_id)
            if entry is not None:
                return self.expand_index(summary_id)

        archived = self.store.get_archived_summary(summary_id, episode_id=None)
        if archived is None:
            # Not archived: maybe it is still in layer 1 -- rendered, filed under a
            # title, or moved to the lazy store.  All three resolve by exact id.
            lazy = self.store.get_active_summary(summary_id, episode_id=self.episode_id)
            if lazy is not None:
                where = (
                    "LAZY STORE (not rendered; id-only)"
                    if lazy.index_id == LAZY_INDEX_ID
                    else ("FILED UNDER INDEX (expand the title to see it in context)"
                          if lazy.index_id else "RENDERED IN THE ACTIVE CHAIN")
                )
                return ToolResult(
                    ok=True,
                    name=TOOL_GET_ARCHIVED,
                    resolved=True,
                    content=(
                        f"summary_id={lazy.summary_id} is in layer 1 -- {where}. "
                        f"text: {lazy.text} {lazy.overrides_tag} raw_ref: {lazy.raw_ref_id}"
                    ),
                    path=f"layer-1 exact id hit ({where})",
                    candidates=[{
                        "summary_id": lazy.summary_id, "short_id": lazy.summary_id.split("/")[-1],
                        "kind": "lazy" if lazy.index_id == LAZY_INDEX_ID else "active",
                        "value": "", "outcome": "chosen", "reason": where, "text": lazy.text,
                    }],
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            return ToolResult(
                ok=False,
                name=TOOL_GET_ARCHIVED,
                parsed=True,
                resolved=False,
                error="not found",
                content=f"no archived summary with id {summary_id!r} (ids are exact; check for typos)",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        content = archived.render()
        # Follow the supersede chain forward so the agent can answer both
        # "what was it" and "what is it now".
        chain_note: List[str] = []
        chain_ids: List[str] = [archived.summary_id]
        cursor = archived
        seen = {archived.summary_id}
        while cursor.superseded_by and cursor.superseded_by not in seen:
            successor = self.store.get_archived_summary(cursor.superseded_by, episode_id=None)
            if successor is None:
                break
            seen.add(successor.summary_id)
            chain_note.append(f"{successor.summary_id}: {successor.text}")
            chain_ids.append(successor.summary_id)
            cursor = successor
        if chain_note:
            content += (
                f"\n[SUPERSEDED BY (latest last)] " + " -> ".join(chain_note)
            )
        candidates = [
            {
                "summary_id": archived.summary_id,
                "short_id": archived.summary_id.split("/")[-1],
                "kind": "archived",
                "value": "",
                "outcome": "chosen",
                "reason": "exact id hit",
                "text": archived.text,
            }
        ]
        for step in chain_ids[1:]:
            successor = self.store.get_archived_summary(step, episode_id=None)
            candidates.append(
                {
                    "summary_id": step,
                    "short_id": step.split("/")[-1],
                    "kind": "supersede-chain",
                    "value": "",
                    "outcome": "traversed",
                    "reason": "successor in the replace chain",
                    "text": successor.text if successor else "",
                }
            )
        path = f"exact id hit -> {len(chain_ids)} entry supersede chain" if chain_note else "exact id hit"
        return ToolResult(
            ok=True,
            name=TOOL_GET_ARCHIVED,
            resolved=True,
            content=content,
            path=path,
            candidates=candidates,
            details={
                "requested_summary_id": summary_id,
                "supersede_chain": chain_ids,
                "archive_size": self.store.count_archived_summaries(self.episode_id),
            },
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def get_raw_record(self, reference_id: str) -> ToolResult:
        started = time.perf_counter()
        if not reference_id:
            return self._error(TOOL_GET_RAW, "missing reference_id argument", parsed=True)
        record = self.store.get_raw_record(reference_id, episode_id=self.episode_id)
        if record is None and self.episode_id:
            # Ids are globally unique; allow a cross-episode fallback for tooling.
            record = self.store.get_raw_record(reference_id, episode_id=None)
        if record is None:
            return ToolResult(
                ok=False,
                name=TOOL_GET_RAW,
                parsed=True,
                resolved=False,
                error="not found",
                content=f"no raw record with reference_id {reference_id!r}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        content = (
            f"reference_id={record.reference_id} turn={record.turn_index} "
            f"timestamp={record.timestamp:.0f}\n"
            f"user: {record.user_msg}\nagent: {record.agent_msg}"
        )
        return ToolResult(
            ok=True,
            name=TOOL_GET_RAW,
            resolved=True,
            content=content,
            path=f"raw layer exact id hit (turn {record.turn_index})",
            candidates=[
                {
                    "summary_id": record.reference_id,
                    "short_id": record.reference_id.split("/")[-1],
                    "kind": "raw",
                    "value": "",
                    "outcome": "chosen",
                    "reason": "exact reference_id hit",
                    "text": f"user: {record.user_msg}\nagent: {record.agent_msg}",
                }
            ],
            details={
                "reference_id": record.reference_id,
                "turn_index": record.turn_index,
                "raw_layer_size": self.store.count_raw_records(self.episode_id),
            },
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def get_predecessor_summary(
        self,
        fact_key: str,
        exclude_value: str = "",
        reference_value: str = "",
    ) -> ToolResult:
        """
        Return the earlier value of ``fact_key``.

        Two selection modes, in priority order:

        **Anchor mode** (``reference_value`` given).  The question names the value
        the fact became ("before it became X, what was it?"), so X is an exact
        anchor.  We find the summary that records ``X`` and return the value it
        *directly* superseded, following the override/supersede relation.  Without
        this anchor, an attribute with three updates has three valid old values and
        the tool can only guess the newest one -- which is why it used to score
        5/9 on multi-update attributes.

        **Fallback mode** (no anchor).  Newest archived entry for the attribute
        whose value differs from ``exclude_value``.

        Both modes are exact attribute/id matching: no vectors, no similarity.
        Every candidate considered is reported in ``ToolResult.candidates`` so the
        inspection UI can show why one entry won and the others lost.
        """
        started = time.perf_counter()
        slot = (fact_key or "").strip().lower()
        if not slot:
            return self._error(TOOL_GET_PREDECESSOR, "missing fact_key argument", parsed=True)
        exclude = (exclude_value or "").strip()
        anchor = (reference_value or "").strip()

        archived = self.store.list_archived_summaries(self.episode_id)
        active = self.store.list_active_summaries(self.episode_id)
        if not archived and not active:
            return ToolResult(
                ok=False, name=TOOL_GET_PREDECESSOR, parsed=True, resolved=False,
                error="empty memory",
                content=(
                    "no summaries exist yet, so there is no earlier value for "
                    f"{fact_key!r}"
                ),
                path="archive empty + active chain empty",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # -- 1. collect candidates (single summaries first, then merges) ------ #
        pools = [
            ("single", [a for a in archived if not a.merged_from]),
            ("merged", [a for a in archived if a.merged_from]),
            ("active", list(active)),
        ]
        candidates: List[Dict[str, Any]] = []
        by_id: Dict[str, Any] = {}
        for kind, pool in pools:
            for entry in pool:
                value = _slot_value(entry.text, slot)
                if not value:
                    continue
                by_id[entry.summary_id] = entry
                candidates.append(
                    {
                        "summary_id": entry.summary_id,
                        "short_id": entry.summary_id.split("/")[-1],
                        "kind": kind,
                        "value": value,
                        "timestamp": round(float(entry.timestamp), 3),
                        "seq": entry.seq,
                        "is_overridden": bool(getattr(entry, "is_overridden", False)),
                        "superseded_by": getattr(entry, "superseded_by", None),
                        "override_ids": list(entry.override_ids),
                        "raw_ref": ",".join(entry.raw_ref_ids) or entry.raw_ref_id,
                        "text": entry.text,
                        "outcome": "candidate",
                    }
                )
        # newest first for every decision below
        candidates.sort(key=lambda c: (c["timestamp"], c["seq"]), reverse=True)

        def mark(chosen_id: str, reason: str) -> None:
            for candidate in candidates:
                candidate["outcome"] = "chosen" if candidate["summary_id"] == chosen_id else "rejected"
                if candidate["summary_id"] == chosen_id:
                    candidate["reason"] = reason

        if not candidates:
            return ToolResult(
                ok=False, name=TOOL_GET_PREDECESSOR, parsed=True, resolved=False,
                error="no entry for attribute",
                content=f"no summary records fact_key {fact_key!r}",
                path=f"scanned {len(archived)} archived + {len(active)} active: no slot match",
                candidates=[],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        def build(entry, value: str, path: str, reason: str) -> ToolResult:
            lines = [
                f"fact_key={fact_key} predecessor_summary_id={entry.summary_id}",
                f"value: {value}",
                f"text: {entry.text}",
                f"overridden={getattr(entry, 'is_overridden', False)} "
                f"superseded_by={getattr(entry, 'superseded_by', None) or 'none'} "
                f"raw_ref: {','.join(entry.raw_ref_ids) or entry.raw_ref_id}",
            ]
            if getattr(entry, "merged_from", None):
                lines.append(
                    "NOTE: this entry is a capacity merge of several summaries; for "
                    f"finer detail call get_archived_summary on one of "
                    f"{','.join(entry.merged_from)}"
                )
            mark(entry.summary_id, reason)
            return ToolResult(
                ok=True, name=TOOL_GET_PREDECESSOR, resolved=True,
                content="\n".join(lines), path=path, candidates=candidates,
                details={
                    "fact_key": fact_key,
                    "reference_value": anchor,
                    "chosen_summary_id": entry.summary_id,
                    "chosen_value": value,
                },
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # -- 2. anchor mode: the value the fact became is an exact anchor ----- #
        if anchor:
            target = _normalise_value(anchor)
            # (a) an entry that *records* the anchor value
            recorder = next(
                (c for c in candidates if _normalise_value(c["value"]) == target), None
            )
            recorder_id = recorder["summary_id"] if recorder else None
            # (b) entries the recorder itself overrode -> the direct predecessors.
            # Direction matters: the recorder's ``override_ids`` is the set of old
            # summaries *it* replaced.  (An earlier version tested the reverse
            # relation and always returned the newest value.)
            if recorder_id:
                recorder_overrides = list(recorder.get("override_ids") or [])
                overridden = [
                    c for c in candidates if c["summary_id"] in recorder_overrides
                ]
                if overridden:
                    chosen = overridden[0]
                    return build(
                        by_id[chosen["summary_id"]],
                        chosen["value"],
                        f"anchor={anchor} -> recorder {recorder['short_id']} "
                        f"-> it overrode {chosen['short_id']}",
                        f"directly overridden by {recorder['short_id']}",
                    )
            # (c) no explicit relation: newest entry strictly older than the recorder
            older = [
                c for c in candidates
                if _normalise_value(c["value"]) != target
                and (recorder is None or (c["timestamp"], c["seq"]) < (recorder["timestamp"], recorder["seq"]))
            ]
            if older:
                chosen = older[0]
                return build(
                    by_id[chosen["summary_id"]],
                    chosen["value"],
                    f"anchor={anchor} -> no override relation -> newest earlier value "
                    f"{chosen['short_id']}",
                    "newest entry older than the anchor",
                )
            # (d) anchor not found anywhere -> fall through to fallback mode below

        # -- 3. fallback: newest entry whose value differs from exclude_value -- #
        usable = candidates
        dropped_by_exclude = 0
        if exclude:
            differing = [
                c for c in candidates
                if _normalise_value(c["value"]) != _normalise_value(exclude)
            ]
            dropped_by_exclude = len(candidates) - len(differing)
            if differing:
                usable = differing
        chosen = usable[0]
        path = "fallback (no anchor): newest archived value"
        if anchor:
            path = f"anchor={anchor} not found -> " + path
        if dropped_by_exclude and not usable:
            path += f" (exclude_value={exclude} matched everything)"
        return build(
            by_id[chosen["summary_id"]],
            chosen["value"],
            path,
            "newest value" + (f" differing from {exclude}" if exclude and dropped_by_exclude else ""),
        )

    def expand_index(self, index_id: str, limit: int = 0) -> ToolResult:
        """
        Return the summaries an index entry points at -- the "next level down".

        Each member line carries its own ``summary_id``, ``[OVERRIDES]`` tag and
        ``raw_ref``, so from here the model can either read the summary or jump
        straight to the raw dialogue.  Nothing is generated: this is stored content.
        """
        started = time.perf_counter()
        if not index_id:
            return self._error(TOOL_EXPAND_INDEX, "missing index_id argument", parsed=True)
        entry = self.store.get_index_entry(index_id, episode_id=self.episode_id)
        if entry is None:
            # A summary id was given by mistake: point at the right id shape.
            return ToolResult(
                ok=False, name=TOOL_EXPAND_INDEX, parsed=True, resolved=False,
                error="not found",
                content=(
                    f"no index entry {index_id!r}; index ids look like "
                    f"'<episode>/idxNNN@xxxxxx' and appear in <INDEX_LAYER>"
                ),
                path="index lookup failed",
                candidates=[],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        members = self.store.summaries_under_index(index_id, self.episode_id)
        missing = [m for m in entry.members if m not in {s.summary_id for s in members}]
        lines = [
            f"index_id={entry.index_id} title={entry.title}",
            f"covers {entry.time_label()} · {entry.size} entries · facts: {','.join(entry.fact_keys) or 'n/a'}",
            "members (newest last):",
        ]
        for summary in members:
            lines.append(summary.render())
        if missing:
            lines.append(f"[note] {len(missing)} member id(s) no longer resolvable: {','.join(missing)}")
        candidates = [
            {
                "summary_id": summary.summary_id,
                "short_id": summary.summary_id.split("/")[-1],
                "kind": "index-member",
                "value": "",
                "outcome": "listed",
                "reason": "pointed at by this index",
                "override_ids": list(summary.override_ids),
                "raw_ref": summary.raw_ref_id,
                "text": summary.text,
            }
            for summary in members
        ]
        return ToolResult(
            ok=True, name=TOOL_EXPAND_INDEX, resolved=True,
            content="\n".join(lines),
            path=f"index {index_id.split('/')[-1]} -> {len(members)} member summaries",
            candidates=candidates,
            details={"index_id": entry.index_id, "title": entry.title, "members": entry.members},
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def list_active_summaries(self) -> ToolResult:
        started = time.perf_counter()
        summaries = self.store.list_active_summaries(self.episode_id) if self.episode_id else self.store.list_active_summaries()
        content = "\n".join(s.render() for s in summaries) or "(empty)"
        return ToolResult(
            ok=True,
            name=TOOL_LIST_ACTIVE,
            resolved=True,
            content=content,
            path=f"active chain snapshot ({len(summaries)} entries)",
            candidates=[
                {
                    "summary_id": s.summary_id,
                    "short_id": s.summary_id.split("/")[-1],
                    "kind": "active",
                    "value": "",
                    "outcome": "listed",
                    "text": s.text,
                }
                for s in summaries
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- dispatch ----------------------------------------------------------- #
    def execute(self, call: ToolCall) -> ToolResult:
        args = normalise_tool_args(call)
        if call.name == TOOL_GET_ARCHIVED:
            return self.get_archived_summary(args.get("summary_id", ""))
        if call.name == TOOL_GET_RAW:
            return self.get_raw_record(args.get("reference_id", ""))
        if call.name == TOOL_GET_PREDECESSOR:
            return self.get_predecessor_summary(
                args.get("fact_key", ""),
                args.get("exclude_value", ""),
                args.get("reference_value", ""),
            )
        if call.name == TOOL_LIST_ACTIVE:
            return self.list_active_summaries()
        if call.name == TOOL_EXPAND_INDEX:
            return self.expand_index(args.get("index_id", ""))
        return self._error(call.name, f"unknown tool {call.name!r}", parsed=True)

    def execute_text(self, text: str) -> List[ToolResult]:
        """Parse and execute every tool call found in ``text``."""
        calls = parse_tool_calls(text)
        if not calls:
            return [
                ToolResult(
                    ok=False,
                    parsed=False,
                    error="no parsable tool call",
                    content=(
                        "your reply did not contain a parsable tool call; use exactly "
                        'get_archived_summary(summary_id="<id>") or get_raw_record(reference_id="<ref>")'
                    ),
                )
            ]
        return [self.execute(call) for call in calls]

    def _error(self, name: str, message: str, parsed: bool = True) -> ToolResult:
        return ToolResult(ok=False, name=name, error=message, content=message, parsed=parsed)
