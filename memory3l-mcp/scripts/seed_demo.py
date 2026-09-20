#!/usr/bin/env python3
"""
Build a small ledger so a fresh install has something to query.

Why this exists
---------------
An MCP server whose tools all answer "no episodes" is indistinguishable from a
broken one, and "install it, then write your own memory pipeline first" is not a
first-run experience.  This produces a real ledger from the offline synthetic
generator -- no network, no API key, no Redis -- so ``store_info`` and
``list_episodes`` have something to say and the audit tools can be exercised
before anyone adopts the library.

It is deliberately a *demo*: the data is generated, the values are meaningless,
and the point is the audit surface, not the facts.

    python3 memory3l-mcp/scripts/seed_demo.py --db /tmp/demo.db
    MEMORY3L_DB=/tmp/demo.db memory3l-mcp       # then query it
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from memory3l.dataset import build_long_context_episodes  # noqa: E402
from memory3l.llm import HeuristicLLM                     # noqa: E402
from memory3l.memory_manager import MemoryManager         # noqa: E402
from memory3l.store.sqlite_store import SQLiteColdStore   # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="SQLite file to create")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    args = parser.parse_args(argv)

    path = Path(args.db).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Re-seeding into an existing ledger would mix two demo runs into one
        # episode namespace and make the audit numbers uninterpretable.
        print(f"refusing to overwrite existing ledger: {path}", file=sys.stderr)
        return 1

    # The cold store is the source of truth and satisfies the manager's contract
    # on its own, so no Redis is involved: a demo that needed a broker would not
    # be a demo.
    store = SQLiteColdStore(str(path))
    episodes = build_long_context_episodes(
        num_episodes=args.episodes, turns_per_episode=args.turns,
        seed=args.seed, language=args.language,
    )
    for episode in episodes:
        manager = MemoryManager(
            store, HeuristicLLM(), episode_id=f"demo/{episode.episode_id}",
            active_chain_token_limit=400, reset_on_bind=True,
        )
        for user, reply in episode.dialogues:
            manager.add_dialog_turn(user, reply)
    store.close()

    print(f"seeded {len(episodes)} episode(s) -> {path}")
    print(f"query it with:  MEMORY3L_DB={path} memory3l-mcp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
