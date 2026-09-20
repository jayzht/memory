#!/usr/bin/env python3
"""
Verify the *installed* distributions work outside the checkout.

Why this exists, given a full unit-test suite
---------------------------------------------
The suite cannot answer this question, and it actively hides it:

* It imports repo-level modules (`evaluation`, `audit_server`) that are
  deliberately not packaged, so it only runs from a source checkout.
* Running from a checkout puts the checkout on `sys.path`, so `import memory3l`
  resolves to the source tree and the installed copy is never exercised.

That combination is exactly how a real defect shipped: `config.py` lived at the
repository root while four modules inside the package imported it, so the wheel
imported fine from the checkout and failed everywhere else. Every test passed.

So this script is run from a directory *outside* the checkout, with the built
wheels installed and nothing else on the path, and it asserts that what it
imported is the installed package before exercising a real audit round trip.

Run (from anywhere that is not the checkout):

    python3 scripts/check_installed_package.py --checkout /path/to/memory
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _resolve_installed(module_name: str, checkout: Path):
    """
    Import a module and refuse the *source* copy.

    The comparison is against the exact source path rather than "anything under
    the checkout", because a virtualenv legitimately lives inside the working
    copy (CI creates one at the workspace root), and rejecting that would reject
    the very install this script exists to check.
    """
    module = __import__(module_name, fromlist=["__file__"])
    origin = Path(module.__file__).resolve()
    source_copy = (checkout / module_name / "__init__.py").resolve()
    if origin != source_copy:
        return module, origin
    raise SystemExit(
        f"{module_name} resolved to the source tree at {origin}. The installed "
        "distribution was not tested -- install the built wheels and run this from "
        "a directory where the source package is not importable."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", required=True,
                        help="the source checkout these packages were built from")
    args = parser.parse_args(argv)

    checkout = Path(args.checkout)
    if Path.cwd().resolve() == checkout.resolve():
        raise SystemExit("run this from outside the checkout; the source tree would shadow the install")

    # ── the packaged modules, imported from the install ──────────────────────
    core, core_origin = _resolve_installed("memory3l", checkout)
    print(f"memory3l      {core.__version__:8} {core_origin}")

    # `config` is imported by memory_manager/llm/prompts/agents. Importing all of
    # them is the actual regression test for the packaging defect: an unpackage
    # `config` fails here and nowhere in the unit suite.
    from memory3l.memory_manager import MemoryManager
    from memory3l.store.sqlite_store import SQLiteColdStore
    from memory3l.dataset import build_long_context_episodes
    from memory3l.llm import HeuristicLLM
    from memory3l.audit import audit_episode
    import memory3l.agents.base_agent  # noqa: F401  (also imports config)
    print("core modules  imported: memory_manager, store, dataset, llm, audit, agents")

    server, server_origin = _resolve_installed("memory3l_mcp", checkout)
    print(f"memory3l-mcp  {server.__version__:8} {server_origin}")

    # The skill ships inside the wheel and the installer copies it from there; a
    # wheel built without it would break `memory3l-mcp-install-skill` for anyone
    # who only ever ran `uvx`.
    from memory3l_mcp.skills import BUNDLED_SKILL_DIR, install_skill
    skill = BUNDLED_SKILL_DIR / "memory-audit" / "SKILL.md"
    if not skill.is_file():
        raise SystemExit(f"the bundled skill is missing from the wheel: {skill}")
    print(f"bundled skill present: {skill.relative_to(BUNDLED_SKILL_DIR.parent)}")

    # ── a real round trip through the installed code ─────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "ledger.db"
        store = SQLiteColdStore(str(db))
        episode = build_long_context_episodes(num_episodes=1, turns_per_episode=24, seed=777)[0]
        manager = MemoryManager(store, HeuristicLLM(), episode_id=f"check/{episode.episode_id}",
                                active_chain_token_limit=400, reset_on_bind=True)
        for user, reply in episode.dialogues:
            manager.add_dialog_turn(user, reply)
        report = audit_episode(store, f"check/{episode.episode_id}").to_dict()
        store.close()

        ok = report["ok"]
        i1 = report["invariants"]["I1_fact_conservation"]
        print(f"audit round trip: ok={ok} ledger={i1['ledger']} archived={i1['archived']} missing={i1['missing']}")
        if i1["ledger"] == 0:
            raise SystemExit("the round trip extracted nothing -- the install is not functional")
        if not ok:
            raise SystemExit(f"the audit reports violations: {report['violations']}")

        destination = Path(tmp) / "skills"
        installed, written = install_skill(destination)
        if not (installed / "SKILL.md").is_file():
            raise SystemExit(f"install_skill did not produce a skill at {installed}")
        print(f"install_skill : wrote={written} -> {installed}")

    print("\nPASS: both installed distributions import and work outside the checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
