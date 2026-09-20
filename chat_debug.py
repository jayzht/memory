#!/usr/bin/env python3
"""
chat_debug.py -- minimal interactive debugger (NOT a demo UI).

Purpose: manually verify the memory mechanics on a handful of turns, printing the
sliding window, the active chain, and every archive mutation after each turn.
Works with the in-memory store by default, so no Redis is required:

    python chat_debug.py                       # in-memory + heuristic LLM
    python chat_debug.py --store redis_sqlite_hybrid --llm-backend ollama --model qwen2.5:7b
    python chat_debug.py --system naive_chain  # compare a baseline by hand

Commands inside the REPL:
    /state        dump memory layers
    /archive      list archived summaries
    /raw <ref>    fetch a raw record (shows what the get_raw_record tool returns)
    /arch <id>    fetch an archived summary (shows what get_archived_summary returns)
    /ask <q>      interrogate the agent without adding a turn
    /tools        list tools
    /reset        start a fresh episode
    /config       show the effective configuration
    /help, /quit
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

import config
from memory3l.agents import build_agent
from memory3l.llm import build_llm
from memory3l.store import InMemoryStore
from memory3l.store.base import BaseMemoryStore
from memory3l.store.hybrid_store import RedisSQLiteHybridStore
from memory3l.token_utils import TOKENIZER_NAME
from memory3l.tools import ToolExecutor

BANNER = """\
========================================================
 three-layer memory debugger  (type /help for commands)
 store={store}  llm={llm}/{model}  system={system}
 window={window} turns  chain_limit={limit} tokens  tokenizer={tokenizer}
========================================================"""


def build_store(args: argparse.Namespace) -> BaseMemoryStore:
    if args.store == "memory":
        return InMemoryStore(recent_window_turns=args.recent_window_turns)
    store = RedisSQLiteHybridStore(
        sqlite_path=args.sqlite_path or config.SQLITE_PATH,
        recent_window_turns=args.recent_window_turns,
        strict_redis=False,
    )
    if not store.hot.available:
        print(
            "[warn] Redis unavailable -- the hybrid store is running in degraded mode:\n"
            "       cold data still goes to SQLite, hot data is served from SQLite too.\n"
        )
    return store


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="interactive memory debugger")
    parser.add_argument("--store", default=config.STORE_BACKEND, choices=["memory", "redis_sqlite_hybrid"])
    parser.add_argument("--system", default="three_layer")
    parser.add_argument("--llm-backend", default=config.LLM_BACKEND, choices=["ollama", "openai", "heuristic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--episode-id", default="debug_ep")
    parser.add_argument("--recent-window-turns", type=int, default=config.RECENT_WINDOW_TURNS)
    parser.add_argument("--active-chain-token-limit", type=int, default=config.ACTIVE_CHAIN_TOKEN_LIMIT)
    parser.add_argument("--raw-context-token-limit", type=int, default=config.RAW_CONTEXT_TOKEN_LIMIT)
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-reset", action="store_true", help="continue the existing episode instead of resetting")
    parser.add_argument(
        "--self-write",
        action=argparse.BooleanOptionalAction,
        default=config.SELF_WRITE_MEMORY,
        help="let the answering call also write the turn summary (1 LLM call/turn instead of 2)",
    )
    args = parser.parse_args(argv)
    config.SELF_WRITE_MEMORY = args.self_write

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    llm = build_llm(args.llm_backend, args.model, args.temperature)
    store = build_store(args)
    agent = build_agent(
        args.system,
        store,
        llm,
        llm,
        recent_window_turns=args.recent_window_turns,
        active_chain_token_limit=args.active_chain_token_limit,
        raw_context_token_limit=args.raw_context_token_limit,
    )
    agent.debug = True
    agent.reset_episode(args.episode_id)
    executor = ToolExecutor(store, episode_id=args.episode_id)

    print(
        BANNER.format(
            store=store.backend_name,
            llm=llm.name,
            model=llm.model_name,
            system=agent.system_name,
            window=args.recent_window_turns,
            limit=args.active_chain_token_limit,
            tokenizer=TOKENIZER_NAME,
        )
    )

    while True:
        try:
            user_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.startswith("/"):
            if _handle_command(user_input, agent, store, executor, args):
                break
            continue

        # The "agent reply" is produced by the agent itself (tool loop included)
        # so what gets summarised is exactly what the real system would produce.
        # With SELF_WRITE_MEMORY the same call also records the turn (one LLM call
        # per turn); otherwise the agent answers and a second call summarises.
        if config.SELF_WRITE_MEMORY and hasattr(agent, "answer_and_remember"):
            run, stats = agent.answer_and_remember(user_input)
        else:
            run = agent.answer(user_input)
            stats = agent.add_turn(user_input, run.answer or "(empty answer)")
        agent_output = run.answer or "(empty answer)"
        print(f"agent> {agent_output}")
        if run.tool_call_log:
            for entry in run.tool_call_log:
                status = "ok" if entry.get("ok") else f"FAIL({entry.get('error') or 'unresolved'})"
                print(f"  [tool] {entry.get('name')}{entry.get('args') or ''} -> {status}")
        if getattr(run, "selfwrite_requested", False):
            print(
                "  [mem] self-write "
                + ("USED (1 call this turn)" if run.selfwrite_used
                   else "MISSING -> fell back to the summariser (2 calls)")
            )
        if hasattr(stats, "to_dict"):
            print(
                f"  [turn] new={stats.new_summary_id} overrides={stats.overridden_ids or '-'} "
                f"capacity={stats.capacity_merged_ids or '-'} active={stats.active_chain_size} "
                f"({stats.active_chain_text_tokens}tok bodies) window={stats.window_size} "
                f"archived={stats.archived_total}"
            )
            if stats.invalid_override_ids:
                print(f"  [turn] REJECTED invalid override ids: {stats.invalid_override_ids}")
        print(agent.debug_state() if hasattr(agent, "debug_state") else "")
    return 0


def _handle_command(
    line: str, agent, store: BaseMemoryStore, executor: ToolExecutor, args: argparse.Namespace
) -> bool:
    """Returns True when the REPL should exit."""
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return True
    if command == "/help":
        print(__doc__)
    elif command == "/state":
        print(agent.debug_state() if hasattr(agent, "debug_state") else store.checksum())
    elif command == "/archive":
        items = store.list_archived_summaries(args.episode_id)
        if not items:
            print("(archive empty)")
        for item in items:
            print("  " + item.render())
    elif command in ("/raw", "/arch"):
        if not argument:
            print(f"usage: {command} <id>")
            return False
        result = (
            executor.get_raw_record(argument)
            if command == "/raw"
            else executor.get_archived_summary(argument)
        )
        print(f"  ok={result.ok} resolved={result.resolved}")
        print("  " + result.render().replace("\n", "\n  "))
    elif command == "/ask":
        if not argument:
            print("usage: /ask <question>")
            return False
        run = agent.answer(argument)
        print(f"agent> {run.answer}")
        print(f"  tools={run.tool_calls_resolved}/{run.tool_calls_attempted} ctx={run.context_tokens}tok")
    elif command == "/tools":
        from memory3l.prompts import TOOL_DESCRIPTIONS

        print(TOOL_DESCRIPTIONS)
    elif command == "/reset":
        new_id = argument or args.episode_id
        agent.reset_episode(new_id)
        print(f"episode reset -> {new_id} (hot state cleared; SQLite archive kept)")
    elif command == "/config":
        for key in (
            "STORE_BACKEND", "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_KEY_PREFIX",
            "SQLITE_PATH", "RECENT_WINDOW_TURNS", "ACTIVE_CHAIN_TOKEN_LIMIT",
            "CAPACITY_MIN_MERGE", "CAPACITY_MAX_MERGE", "MAX_TOOL_ITERATIONS",
            "RAW_CONTEXT_TOKEN_LIMIT", "LLM_BACKEND", "MODEL_NAME", "TEMPERATURE",
        ):
            print(f"  {key} = {getattr(config, key)}")
    else:
        print(f"unknown command {command!r}; try /help")
    return False


if __name__ == "__main__":
    raise SystemExit(main())
