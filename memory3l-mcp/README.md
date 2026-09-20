# memory3l-mcp

<!-- mcp-name: io.github.jayzht/memory3l-mcp -->

Auditable memory over [MCP](https://modelcontextprotocol.io): an append-only fact
ledger plus verifiable invariants, served read-only from SQLite.

> The `mcp-name:` string above and `server.json`'s `name` must stay identical: the
> MCP Registry verifies PyPI package ownership by matching them. Changing one
> without the other fails verification.

## What this is for

A memory system claims it never loses a fact. Almost none can *prove* it, because
nothing keeps an independent record to compare against.

`memory3l` keeps one: an append-only ledger of every `(slot, value)` ever
extracted, written when each summary is created — **including the summaries later
overridden or merged away**, which is exactly where facts disappear. This server
exposes the cross-check as tools, so an agent can ask whether a value is still
reachable, why a fact left the working set, or whether a deletion actually
completed, and get an answer derived from persisted state rather than from the
summary's own paraphrase.

The checks are the product. The storage is ordinary SQLite.

## Install

```bash
uvx memory3l-mcp                  # no install step, runs from PyPI
```

The audit core (`memory3l`) is installed as an ordinary dependency. `MEMORY3L_ROOT`
is only needed when developing against a source checkout:

```bash
MEMORY3L_ROOT=/path/to/Memory uvx memory3l-mcp
```

The ledger defaults to `~/.memory3l/memory3l.db` and is **created on demand**, so a
fresh install starts and answers `episodes: []` instead of failing. Point it
somewhere else with `MEMORY3L_DB`, or `--db PATH`.

### Try it without your own pipeline

```bash
python3 scripts/seed_demo.py --db /tmp/demo.db   # generates a real ledger offline
MEMORY3L_DB=/tmp/demo.db uvx memory3l-mcp
```

No network, no API key, no Redis.

## Tools

Read-only. `append_facts` appears only with `--allow-write` or
`MEMORY3L_ALLOW_WRITE=1`.

| Tool | Answers |
| --- | --- |
| `store_info` | which database is served, and whether it is readable |
| `list_episodes` | which episode ids exist — call this first |
| `audit` | the invariant report for one episode |
| `audit_summary` | the same, aggregated over every episode |
| `current` | the newest value of every slot, as the model should be told it |
| `history` | every value a slot ever held, with supersession |
| `fact` | why one fact left the working set (live / archived / missing) |
| `evidence` | the original dialogue a fact was extracted from |
| `tombstones` | what was erased, what is provably gone, what prose remains |
| `temporal` | the versioned projection, its anomalies, and its DDL |

### The invariants

- **I1 conservation** — a ledger fact resolves to no summary, active or archived:
  it was extracted and now nothing can reach it. Real silent loss.
- **I2 top-level reachability** — a slot's newest value lives only inside an index
  member, so the model is never told it: stored, and practically invisible.
- **I3 provenance** — a fact's evidence no longer resolves to stored dialogue.
- **I4 extraction completeness** — the extractor itself missed facts present in the
  raw turns. The blind spot the other invariants cannot see: they all pass when
  nothing was extracted at all.
- **I5 deletion** — an erasure is verified rather than assumed: the value is gone
  *and* its evidence destroyed, with residual prose mentions counted.

An episode with no facts satisfies every invariant trivially, so an unknown episode
is an explicit error rather than an empty pass. A tool that answers "no violations"
about an episode that does not exist is worse than one that fails.

## Configure your client

Any MCP client that can launch a stdio server. For **DSH**, add one row to
`$DSH_HOME/profiles/<name>/cordis.patch.yml`:

```yaml
- insert:
    - id: mcp-memory
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: memory
        transport: stdio
        command: uvx
        args: ['memory3l-mcp']
        env:
          MEMORY3L_DB: /path/to/your/memory3l.db
```

Tools then appear as `mcp__memory__audit`, `mcp__memory__current`, and so on —
the same `mcp__<server>__<tool>` shape Claude Code and Codex use. DSH passes the
`env` entries through to the server (credential-shaped variables are scrubbed from
the ambient environment, which is why the ledger path is set explicitly here).

## The skill

`skills/memory-audit/SKILL.md` is an [Agent Skills](https://agentskills.io/specification)
bundle that teaches an agent *when* to reach for these tools and how to report the
answer without overstating it — chiefly that "no violations" is not a guarantee
that nothing was lost.

Install it into the cross-client directory so every compliant agent sees it:

```bash
memory3l-mcp-install-skill           # → ~/.agents/skills/memory-audit
```

Skills installed there are visible to any client that scans `.agents/skills/`,
and vice versa. No per-client adaptation. The command refuses to overwrite an
existing skill whose content differs — pass `--force` if you mean it, or
`--target DIR` to install somewhere else.

## Development

```bash
MEMORY3L_ROOT=$PWD PYTHONPATH=$PWD/src python -m unittest discover -s tests
python scripts/seed_demo.py --db /tmp/demo.db
```

Requires `mcp>=2`. Note that `mcp` 2.0 renamed `FastMCP` to `MCPServer` and made
`mcp.server.fastmcp` raise on import; this package targets 2.x.

## License

MIT
