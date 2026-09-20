"""
Tests for the MCP surface: tool registration, answer shape, and the failure modes.

These run without network, Redis or an LLM.  A tiny ledger is built directly
through the store API rather than through ``MemoryManager``, so the tests stay
fast and pin the *tool* behaviour instead of re-testing the pipeline.

Requires the ``mcp`` package.  Run from the repository root::

    MEMORY3L_ROOT=$PWD python -m unittest discover -s memory3l-mcp/tests -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE.parents[1] / "src"))
os.environ.setdefault("MEMORY3L_ROOT", str(_ROOT))

try:
    import mcp  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    raise unittest.SkipTest("the `mcp` package is required for these tests")

from memory3l_mcp.config import DB_ENV, DEFAULT_DIR_ENV, resolve_db_path  # noqa: E402
from memory3l_mcp.server import build_server  # noqa: E402

from memory3l.models import ActiveSummary, RawDialogRecord  # noqa: E402
from memory3l.store.sqlite_store import SQLiteColdStore  # noqa: E402

EPISODE = "t/long_0000"


def _seed(path: Path) -> None:
    """One episode: an active summary, its ledger fact, and the raw turn behind it."""
    store = SQLiteColdStore(str(path))
    record = RawDialogRecord(
        reference_id=f"{EPISODE}/r1", episode_id=EPISODE, turn_index=1,
        user_msg="我搬到3楼了", agent_msg="好的", timestamp=1.0,
    )
    store.add_raw_record(record, episode_id=EPISODE)
    active = ActiveSummary(
        summary_id=f"{EPISODE}/s001", text="工位楼层=3楼", episode_id=EPISODE, seq=1,
        timestamp=1.0, raw_ref_id=f"{EPISODE}/r1", raw_ref_ids=[f"{EPISODE}/r1"],
        origin="event", fact_keys=["工位楼层=3楼"],
    )
    store.add_active_summary(active, episode_id=EPISODE)
    store.append_fact_ledger([{
        "fact_id": f"{EPISODE}/s001#工位楼层", "episode_id": EPISODE,
        "summary_id": f"{EPISODE}/s001", "slot": "工位楼层", "value": "3楼", "seq": 1,
        "observed_turn": 1, "evidence": f"{EPISODE}/r1", "reason": "",
        "superseded_by": "", "residual_mentions": 0, "erased": 0,
    }])
    store.close()


def _call(mcp, name: str, args: dict) -> dict:
    """Invoke a tool and parse its JSON text payload."""
    result = asyncio.run(mcp.call_tool(name, args))
    content = getattr(result, "content", result)
    text = getattr(content[0], "text", str(content[0])) if content else str(result)
    return json.loads(text)


class ToolRegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "ledger.db"
        _seed(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_read_tools_registered(self) -> None:
        mcp = build_server(self.db)
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        self.assertEqual(
            names,
            {"store_info", "list_episodes", "audit", "audit_summary", "current",
             "history", "fact", "evidence", "tombstones", "temporal"},
        )

    def test_write_tool_is_opt_in(self) -> None:
        """A default install must not be able to mutate anyone's ledger."""
        self.assertNotIn("append_facts",
                         {t.name for t in asyncio.run(build_server(self.db).list_tools())})
        self.assertIn("append_facts",
                      {t.name for t in asyncio.run(build_server(self.db, allow_write=True).list_tools())})


class AnswerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "ledger.db"
        _seed(self.db)
        self.mcp = build_server(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_store_info_reports_the_served_file(self) -> None:
        info = _call(self.mcp, "store_info", {})
        self.assertEqual(info["path"], str(self.db.resolve()))
        self.assertTrue(info["readable"])
        self.assertEqual(info["episodes"], 1)
        self.assertFalse(info["writes_enabled"])

    def test_list_episodes_and_current(self) -> None:
        self.assertEqual(_call(self.mcp, "list_episodes", {})["episodes"], [EPISODE])
        registry = _call(self.mcp, "current", {"episode_id": EPISODE})
        self.assertEqual(registry["registry"], "工位楼层=3楼")
        self.assertEqual(registry["values"][0]["fact_id"], f"{EPISODE}/s001#工位楼层")

    def test_audit_passes_on_a_consistent_ledger(self) -> None:
        report = _call(self.mcp, "audit", {"episode_id": EPISODE})
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["invariants"]["I1_fact_conservation"]["ok"])

    def test_history_reports_the_slot_over_time(self) -> None:
        history = _call(self.mcp, "history", {"episode_id": EPISODE, "slot": "工位楼层"})
        self.assertEqual(history["history"][0]["value"], "3楼")

    def test_fact_and_evidence_resolve(self) -> None:
        fact_id = f"{EPISODE}/s001#工位楼层"
        self.assertEqual(_call(self.mcp, "fact", {"fact_id": fact_id})["state"], "live")
        evidence = _call(self.mcp, "evidence", {"fact_id": fact_id})
        self.assertEqual(evidence["raw_refs"], [f"{EPISODE}/r1"])

    def test_tombstones_lists_only_erasures(self) -> None:
        self.assertEqual(_call(self.mcp, "tombstones", {"episode_id": EPISODE})["tombstones"], [])

    def test_temporal_projection_is_consistent(self) -> None:
        temporal = _call(self.mcp, "temporal", {"episode_id": EPISODE})
        self.assertTrue(temporal["consistent"])
        self.assertIn("CREATE TABLE", temporal["ddl"])


class FailureModeTest(unittest.TestCase):
    """
    The failure modes matter more than the happy path here.

    An audit tool that answers "no violations" for an episode that does not exist
    is worse than one that errors, because the caller cannot tell a real pass from
    a vacuous one.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "ledger.db"
        _seed(self.db)
        self.mcp = build_server(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unknown_episode_is_an_error_not_a_pass(self) -> None:
        audit = _call(self.mcp, "audit", {"episode_id": "typo"})
        self.assertEqual(audit["error"], "unknown episode")
        self.assertNotIn("ok", audit)
        self.assertEqual(audit["available"], [EPISODE])

    def test_blank_episode_is_rejected(self) -> None:
        self.assertEqual(_call(self.mcp, "current", {"episode_id": "   "})["error"],
                         "episode_id is required")

    def test_unknown_fact_id_is_reported(self) -> None:
        result = _call(self.mcp, "fact", {"fact_id": "no-hash-here"})
        self.assertIn("error", result)
        self.assertIn("hint", result)

    def test_empty_ledger_serves_an_empty_list_rather_than_failing(self) -> None:
        """A fresh install must start and answer, not crash."""
        empty = build_server(Path(self._tmp.name) / "fresh.db")
        self.assertEqual(_call(empty, "list_episodes", {})["episodes"], [])
        self.assertEqual(_call(empty, "store_info", {})["episodes"], 0)

    def test_store_info_reports_an_unopenable_path(self) -> None:
        blocked = Path(self._tmp.name) / "as_dir"
        blocked.mkdir()
        info = _call(build_server(blocked), "store_info", {})
        self.assertFalse(info["readable"])
        self.assertTrue(info["detail"])


class StorageLocationTest(unittest.TestCase):
    """The default has to work on a machine that has never run the component."""

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in (DB_ENV, DEFAULT_DIR_ENV)}
        for key in self._saved:
            os.environ.pop(key, None)
        self._home = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._home.cleanup()

    def test_explicit_path_wins_and_parents_are_created(self) -> None:
        target = Path(self._home.name) / "nested" / "ledger.db"
        self.assertEqual(resolve_db_path(str(target)), target.resolve())
        self.assertTrue(target.parent.is_dir())

    def test_env_var_selects_the_file(self) -> None:
        target = Path(self._home.name) / "from_env.db"
        os.environ[DB_ENV] = str(target)
        self.assertEqual(resolve_db_path(None), target.resolve())

    def test_home_env_selects_the_directory(self) -> None:
        os.environ[DEFAULT_DIR_ENV] = self._home.name
        self.assertEqual(resolve_db_path(None), (Path(self._home.name) / "memory3l.db").resolve())

    def test_default_is_per_user_and_created(self) -> None:
        # ``Path.home()`` follows $HOME, so pointing it at a temporary directory
        # tests the *default* rule without writing to the real home.  (On a
        # machine whose home is read-only, creating the default directory fails
        # loudly -- which is the honest behaviour, not something to paper over.)
        os.environ["HOME"] = self._home.name
        path = resolve_db_path(None)
        self.assertTrue(path.parent.is_dir())
        self.assertEqual(path.name, "memory3l.db")
        self.assertEqual(path.parent, Path(self._home.name).resolve() / ".memory3l")


class BootstrapTest(unittest.TestCase):
    def test_missing_core_names_the_fix(self) -> None:
        """The message must say what to set; a bare ImportError is not enough."""
        import builtins

        from memory3l_mcp import bootstrap

        saved = os.environ.pop("MEMORY3L_ROOT", None)
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "memory3l" or name.startswith("memory3l."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return real_import(name, *args, **kwargs)

        try:
            with mock.patch.object(bootstrap, "_candidate_roots", return_value=[]):
                with mock.patch.object(builtins, "__import__", side_effect=blocked):
                    with self.assertRaises(bootstrap.MemoryCoreNotFound) as ctx:
                        bootstrap.ensure_memory3l()
            self.assertIn("MEMORY3L_ROOT", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["MEMORY3L_ROOT"] = saved


if __name__ == "__main__":
    unittest.main()
