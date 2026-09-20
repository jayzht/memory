---
name: memory3l
description: Give an agent memory across turns, and verify that memory did not silently lose anything. Use when you must remember what a user said earlier, when a stored value seems to have disappeared, when asked what a value was before it changed, or to audit whether a deletion really happened. Provides the remember/memory_context turn loop plus the audit tools (audit, current, history, fact, evidence, tombstones).
license: MIT
compatibility: Needs an MCP client with the `memory3l` server configured with write access for the memory half. Tools may appear under a client-specific namespace (e.g. mcp__memory__remember); the bare names below are what to look for.
metadata:
  component: memory3l
  layer: memory-and-audit
---

# Using and auditing this memory

This component does two things, and they are used at different moments.

1. **Memory** — `remember` and `memory_context` let you keep what was said across
   turns and get it back into your prompt.
2. **Audit** — `audit`, `current`, `history`, `fact`, `evidence`, `tombstones`
   check whether anything that was stored has been lost or become unreachable.

Storage is a three-layer design: short summaries carry the active facts, an archive
holds everything that was replaced, and the original dialogue is kept verbatim. A
sliding window of recent turns is shown word for word. A separate append-only
**fact ledger** records every `(slot, value)` ever extracted, which is what makes
the audit possible — a disappearance leaves a trace to compare against.

## The turn loop

If your server exposes `remember`, memory only grows if you call it. Two calls per
turn, in this order:

| When | Call | Why |
| --- | --- | --- |
| Before you answer | `memory_context(episode_id)` | Gives you the memory block to put in your prompt |
| After you answer | `remember(episode_id, user_message, agent_message)` | Records the turn and maintains the layers |

`memory_context` returns a text block in a fixed order — recent raw dialogue, the
current-value registry, index titles, the active summary chain — and nothing else:
no persona, no instructions. Insert it into whatever prompt you already use.

An **empty block is correct for a new conversation**. It is not an error, so do not
treat it as one.

**If you skip `memory_context`, you are answering without memory** even though
earlier turns were recorded. The tool has no way to inject itself.

One `episode_id` is one conversation. Reuse it to continue, including after the
server restarts: the chain and the turn counter are restored from the database, so
you will not begin numbering turns at zero again. Use a new id for an unrelated
conversation — that is the isolation boundary.

## What memory can and cannot answer

The current-value registry tells you each attribute's newest value, and it is
already inside the block `memory_context` returns. For anything else, ask:

| The question | Use |
| --- | --- |
| "What does it know about X now?" | `current` |
| "What was it before?" / "when did it change?" | `history` — `current` only shows the newest value |
| "Why did this fact leave context?" | `fact` |
| "What was actually said?" | `evidence` — quote the source, not the summary |
| "Was the deletion complete?" | `tombstones` |
| "Is anything being lost?" | `audit` (one episode) or `audit_summary` (all) |
| "Which database is this?" | `store_info` |

`history` needs an exact slot name; read the registry from `memory_context` or
`current` first rather than guessing a spelling.

**There is no semantic search.** Retrieval is by exact id only — no embeddings, no
similarity. The registry and the summary chain are what bring facts into the
prompt; you cannot "look for related memories".

## How to report audit results honestly

The invariants: **I1** a ledger fact no longer reaches any summary (real silent
loss); **I2** a slot's newest value exists only inside an index member, so it is
stored but never shown; **I3** a fact's evidence no longer resolves to dialogue;
**I4** the extractor itself missed facts present in the raw turns; **I5** an erasure
is verified rather than assumed.

**Never report a vacuous pass.** An episode with no facts satisfies every invariant
trivially, so an unknown episode id is an explicit error rather than an empty
result. If you get `unknown episode`, fix the id — do not describe it as "no
problems found".

**Do not upgrade "no violations" into a guarantee.** Passing means none of the five
checked failure modes was detected. The invariants check *conservation*, not
*correctness*: a fact that was extracted wrongly is conserved, reachable and has
provenance, so it passes all five. I4 is the only one that reads the raw turns, and
it catches facts **missed**, not facts **invented**. Say "no violations were found"
and stop there.

**Report failures with their evidence.** Every violation names concrete ids — give
the fact id and the reason (`overridden`, capacity merge, erasure), not a count.

**A non-empty `residual_prose_mentions` or a still-readable evidence pointer after
an erasure means the content is not fully gone.** Say so plainly. The tombstone
records the intent; I5 checks whether the intent was carried out.

**Empty results are information.** `tombstones: []` means nothing was erased, not
that erasure is unsupported.

## Practical notes

- A `fact_id` looks like `<episode>/<kind><seq>@<hash>#<slot>` and **contains a
  `#`**. Pass it whole as a parameter; do not splice it into a URL, where `#` starts
  a fragment and silently truncates the slot.
- Episode ids may themselves contain `/` (batch runs scope them as
  `system/episode`), so do not split on `/` to recover an episode.
- **A `summariser: "verbatim"` result from `store_info` means no model is
  configured.** Memory still works — turns are kept word for word — but without
  extraction the current-value registry stays empty and the audit tools have
  nothing to check. Mention that rather than reporting an empty registry as a bug.
- The server is read-only unless started with write access. If there is no
  `remember` tool, memory writing is disabled by configuration; that is deliberate.
- `store_info` reporting `readable: false` or an unexpected `path` explains a
  surprising empty result faster than any query.
