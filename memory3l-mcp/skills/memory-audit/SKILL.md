---
name: memory-audit
description: Verify what a three-layer memory system still knows and what it silently dropped. Use when asked whether stored facts were lost, why a value disappeared from context, what an earlier value was, whether a deletion was complete, or to audit memory provenance. Requires the memory3l MCP server (tools named list_episodes, audit, current, history, fact, evidence, tombstones, temporal, store_info).
license: MIT
compatibility: Needs an MCP client with the `memory3l` server configured. Tools may appear under a client-specific namespace (e.g. mcp__memory__audit); the bare names below are what to look for.
metadata:
  component: memory3l
  layer: audit
---

# Auditing a memory system

A memory system claims it never loses a fact. This skill is how you check that
claim instead of repeating it, and how to report the answer without overstating it.

The tools read an append-only **fact ledger** — an independent record of every
`(slot, value)` ever extracted, written at the moment each summary was created,
*including summaries later overridden or merged away*, which is exactly where
facts vanish. The ledger is the reference; the live store is what is checked
against it.

## Which tool answers which question

| The user asks | Use | Not |
| --- | --- | --- |
| "Is the memory losing anything?" | `audit` (one episode) or `audit_summary` (all) | guessing from a summary |
| "What does it currently know about X?" | `current` | reading a summary and paraphrasing |
| "What was it before?" / "when did it change?" | `history` | `current` — it only shows the newest value |
| "Why did this fact leave context?" | `fact` | assuming deletion |
| "What was actually said?" | `evidence` | quoting the summary as if it were the source |
| "Was the deletion complete?" | `tombstones` | trusting that a delete succeeded |
| "Do we have an episode to look at?" | `list_episodes` | inventing an id |
| "Which database is this?" | `store_info` | — |

Call `list_episodes` first if you do not already have an episode id. Every other
tool needs one, and a wrong id is an error rather than an empty answer — see
"Never report a vacuous pass" below.

`history` needs an exact slot name. Call `current` first to read the available
names rather than guessing a spelling.

## What the invariants mean

Report these by their substance, not as a wall of codes.

- **I1 conservation** — a fact in the ledger no longer resolves to any summary,
  active or archived. It was extracted and now nothing can reach it. This is the
  real silent loss.
- **I2 top-level reachability** — a slot's newest value exists only inside an
  index member, so the model is never told it. The value is technically stored
  and practically invisible.
- **I3 provenance** — a fact's evidence no longer resolves to stored dialogue.
  The value is there but cannot be justified from the source.
- **I4 extraction completeness** — the extractor itself missed facts present in
  the raw turns. This is the blind spot the other invariants cannot see: they all
  pass when nothing was extracted at all.
- **I5 deletion** — an erasure is verified, not assumed: the value is gone *and*
  its evidence is destroyed, with residual prose mentions counted.

## How to report results honestly

**Never report a vacuous pass.** An episode with no facts satisfies every
invariant trivially. The tools return an explicit `unknown episode` error for an
id that does not exist precisely so that you cannot mistake this for a clean
audit. If you get that error, fix the episode id — do not describe it as "no
problems found".

**Do not upgrade "no violations" into a guarantee.** Passing means none of the
five checked failure modes was detected. It does not mean nothing was lost: I4
bounds extraction only against the raw turns it can see, and an extractor that
dropped a fact from its own candidate list is invisible to all five. Say "no
violations were found" and stop there. If the user needs a stronger claim, say
what would be required to make it.

**Report failures with their evidence.** Every violation names concrete ids.
Give the user the fact id and the reason (`overridden`, capacity merge, erasure)
rather than a count alone.

**A non-empty `residual_prose_mentions` or a still-readable evidence pointer
after an erasure means the content is not fully gone.** Say so plainly. Do not
call an erasure complete because a tombstone exists — the tombstone records the
intent, and I5 checks whether the intent was carried out.

**Empty results are information.** `tombstones: []` means nothing was erased, not
that erasure is unsupported. `history` empty for a slot usually means the name
did not match; the tool adds a hint when it suspects this.

## Practical notes

- A `fact_id` looks like `<episode>/<kind><seq>@<hash>#<slot>` and **contains a
  `#`**. Keep it whole and pass it as a parameter; do not splice it into a URL,
  where `#` starts a fragment and silently truncates the slot.
- Episode ids may themselves contain `/` (batch runs scope them as
  `system/episode`), so do not split on `/` to recover an episode.
- The server is **read-only** unless it was started with write access. If there is
  no `append_facts` tool, fact intake is disabled by configuration — that is a
  deliberate default, not a fault.
- `store_info` reporting `readable: false` or an unexpected `path` explains a
  surprising empty result faster than any query. Check it before concluding that
  memory is empty.
