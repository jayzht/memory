#!/usr/bin/env python3
"""
Wire memory3l into your own agent: the four calls that matter.

    python3 examples/agent_with_memory.py                    # offline, no API key
    LLM_BACKEND=deepseek python3 examples/agent_with_memory.py

The whole integration is a loop:

    1. build the prompt FROM MEMORY      -> build_agent_messages(...)
    2. call your model                   -> llm.generate(messages)
    3. record the completed turn         -> manager.add_dialog_turn(user, answer)
    4. (when you care) audit it          -> audit_episode(store, episode_id)

Steps 1 and 3 are the contract. Everything else -- which store, which summariser,
which model -- is configuration, and the memory logic does not change.

Two things worth knowing before you adopt this
---------------------------------------------
**It costs a model call per turn.** Recording a turn makes the manager summarise it,
and by default that is a second LLM call (the summariser). If you want one call, the
manager supports "self-write": the agent emits a ``<MEMORY_UPDATE>`` block in its own
answer and the manager uses that instead of calling out again
(``add_dialog_turn_selfwritten``). This example uses the simple two-call path.

**You build the prompt, not the manager.** The manager exposes the pieces
(``get_window``, ``chain_summaries``, ``rendered_index_entries``,
``render_current_values``) and ``build_agent_messages`` assembles them in the fixed
order. That is deliberate: the prompt order is part of the design, so it lives in
one function rather than in every caller.

Storage: this uses SQLite alone. No Redis, no broker.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Copy this file into your own project as-is: it only reaches for the source tree
# when `memory3l` is not installed, which is what running it inside the checkout
# looks like. With `pip install memory3l` the insertion is skipped.
try:
    import memory3l  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_agent_prompt(manager, question: str, *, recent_window_turns: int = 4,
                       tools_available: bool = False) -> list[dict]:
    """
    The one call that turns memory state into chat messages.

    The order is fixed and load-bearing:
        system : instructions
        user   : [sliding window][active summary chain][question]
    """
    from memory3l.prompts import build_agent_messages

    return build_agent_messages(
        manager.get_window(),                 # layer 3, verbatim recent turns
        manager.chain_summaries(),            # layer 1, unfiled summaries
        question,
        recent_window_turns=recent_window_turns,
        indexes=manager.rendered_index_entries(),
        current_values=manager.render_current_values(),
        tools_available=tools_available,
    )


def make_agent(db: Path, episode_id: str, backend: str | None = None):
    """
    Assemble the three pieces an agent needs.

    ``summarizer`` may be the same model as your agent or a cheaper one -- it runs
    once per turn, so a small model is usually the right call.
    """
    from memory3l.llm import build_llm
    from memory3l.memory_manager import MemoryManager
    from memory3l.store.sqlite_store import SQLiteColdStore

    store = SQLiteColdStore(str(db))
    agent_llm = build_llm(backend)
    summarizer = build_llm(backend)
    manager = MemoryManager(
        store,
        summarizer,
        episode_id=episode_id,
        recent_window_turns=4,
        active_chain_token_limit=800,
        reset_on_bind=True,          # start this episode clean; archive/raw survive
    )
    return store, manager, agent_llm


def run_turn(manager, llm, question: str, *, show_prompt: bool = False) -> str:
    """One full turn: prompt from memory -> model -> record."""
    messages = build_agent_prompt(manager, question)
    if show_prompt:
        print("\n--- what the agent actually sees ---")
        for message in messages:
            print(f"[{message['role']}]")
            print(message["content"])
        print("--- end ---")
    answer = llm.generate(messages).text.strip()
    manager.add_dialog_turn(question, answer)      # <- memory is maintained here
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="agent_memory.db", help="SQLite ledger path")
    parser.add_argument("--episode", default="chat/demo", help="episode id (one conversation)")
    parser.add_argument("--backend", default=None,
                        help="deepseek | openai | ollama | heuristic (default: from config/env)")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the assembled prompt for the first turn")
    parser.add_argument("--purge", action="store_true",
                        help="delete this episode's archive and raw records first")
    args = parser.parse_args(argv)

    backend = args.backend or os.environ.get("LLM_BACKEND")
    if backend is None and not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        # A sensible default for a first look: fully offline, deterministic, no key.
        backend = "heuristic"
    store, manager, llm = make_agent(Path(args.db), args.episode, backend)
    if args.purge:
        manager.reset_episode(args.episode, purge=True)

    print(f"backend={llm.name}  model={llm.model_name}  episode={args.episode}  db={args.db}")
    if llm.name == "heuristic":
        print(
            "\nNOTE: the `heuristic` backend is a rule-based offline stand-in. Its\n"
            "      answers are not real, and its extractor reads explicit statements\n"
            "      only -- it will even read a *question* like \"我的X是什么？\" as the\n"
            "      fact X=什么. That artifact is the stub's, not the memory system's;\n"
            "      use --backend deepseek for real behaviour.\n"
        )

    # A conversation with one fact update, so the interesting behaviour shows: the
    # current value is replaced, the earlier value stays retrievable, and the audit
    # can prove nothing was lost.
    #
    # The phrasings are deliberate. The `heuristic` backend is a rule-based offline
    # stand-in that only recognises explicit statements of the form
    # "我的X是Y" / "我的X改成Y了" -- it is a way to see the plumbing without a key, not
    # a real extractor. Point --backend at deepseek (or any OpenAI-compatible model)
    # and ordinary conversational phrasing is summarised properly.
    conversation = [
        "你好，跟你说一下，我的工位楼层是3楼。",
        "对了，我的工位楼层改成7楼了。",
        "我的工位楼层是多少？",
        "在改成7楼之前，我的工位楼层是什么？",
    ]

    for index, user_input in enumerate(conversation):
        answer = run_turn(manager, llm, user_input,
                          show_prompt=args.show_prompt and index == 0)
        print(f"\nuser > {user_input}")
        print(f"agent> {answer[:200]}")

    # ---- what memory holds now ------------------------------------------- #
    print("\n" + "=" * 72)
    print("Current values (what every prompt carries at the top level):")
    print("  " + (manager.render_current_values() or "(none)"))

    now = manager.current("工位楼层")
    if now:
        print(f"\nNow: {now['slot']}={now['value']}  (since turn {now['since_turn']}, "
              f"evidence {now['evidence']})")

    print("\nEvery value that slot ever held (layer 2; exact-id lookup, oldest first):")
    for row in manager.history("工位楼层"):
        superseded = row.get("superseded_by") or "-"
        print(f"  turn {row.get('from_turn')}..{row.get('to_turn')}: {row.get('value')} "
              f"({row.get('reason') or 'live'}, by {superseded})")

    # ---- the audit -------------------------------------------------------- #
    print("\n" + "=" * 72)
    from memory3l.audit import audit_all

    report = audit_all(store)
    print(f"audit: {report.get('episodes', 0)} episode(s), "
          f"{report.get('clean_episodes', 0)} clean, "
          f"{report.get('episodes_with_violations', 0)} with violations")
    print(f"  totals: {report.get('totals')}")
    print(
        "\nWhat the audit does and does not claim: the invariants check that nothing\n"
        "extracted was lost (I1), that current values stay reachable (I2), that\n"
        "provenance resolves (I3) and that erasures are real (I5). They do NOT check\n"
        "that an extracted fact is *correct*. A bogus fact is conserved and reachable,\n"
        "so it passes -- which is why the question-extraction above shows up as a\n"
        "green audit rather than a violation. I4 is the one that looks at the raw\n"
        "turns, and it catches facts *missed*, not facts *invented*."
    )

    # To check one episode in detail, and to explain a single fact:
    #   from memory3l.audit import audit_episode
    #   audit_episode(store, args.episode).to_dict()
    #   manager.explain_fact("<fact_id>")
    store.close()
    print("\nThe same database is what `memory3l-mcp` serves, so an MCP-capable "
          "agent can audit it with the audit/current/history tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
