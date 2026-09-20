"""
Store-level tests: episode isolation, Redis namespacing, SQLite persistence and
batch checkpointing.

Redis is exercised through a small in-process fake that implements exactly the
commands ``RedisHotStore`` uses (list/hash/scan/unlink/ping/pipeline), so the
namespace and reset semantics are verified *without* a Redis server:

* keys are ``episode:{episode_id}:...`` and cannot collide between episodes;
* ``reset(episode_id)`` deletes exactly that episode's keys and nothing else;
* the SQLite cold layer keeps the archive and raw records across resets.
"""

from __future__ import annotations

import fnmatch
import os
import sys
import tempfile
import unittest

# Only when `memory3l` is not installed -- see the same guard in test_core.py: an
# unconditional insert would shadow the wheel that release CI installs, and the
# suite would test the checkout instead of the artifact.
try:
    import memory3l  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory3l.models import ActiveSummary, ArchivedSummary, IndexEntry, RawDialogRecord
from memory3l.store import InMemoryStore, SQLiteColdStore
from memory3l.store.hybrid_store import RedisSQLiteHybridStore
from memory3l.store.redis_store import RedisHotStore


class FakePipeline:
    def __init__(self, fake):
        self.fake = fake
        self.ops = []

    def delete(self, *keys):
        self.ops.append(("delete", keys))

    def rpush(self, key, *values):
        self.ops.append(("rpush", (key,) + values))

    def ltrim(self, key, start, stop):
        self.ops.append(("ltrim", (key, start, stop)))

    def lrange(self, key, start, stop):
        self.ops.append(("lrange", (key, start, stop)))

    def lrem(self, key, count, value):
        self.ops.append(("lrem", (key, count, value)))

    def hset(self, key, field=None, value=None, mapping=None):
        self.ops.append(("hset", (key, field, value, mapping)))

    def hdel(self, key, *fields):
        self.ops.append(("hdel", (key,) + fields))

    def execute(self):
        results = []
        for name, args in self.ops:
            results.append(getattr(self.fake, name)(*args))
        self.ops.clear()
        return results


class FakeRedis:
    """Minimal stand-in for redis-py (strings/lists/hashes/scan/unlink)."""

    def __init__(self):
        self.store = {}

    # -- plumbing --------------------------------------------------------- #
    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def ping(self):
        return True

    # -- strings / hashes -------------------------------------------------- #
    def hset(self, key, field=None, value=None, mapping=None):
        table = self.store.setdefault(key, {})
        if mapping:
            table.update(mapping)
        elif field is not None:
            table[field] = value
        return 1

    def hget(self, key, field):
        return self.store.get(key, {}).get(field)

    def hmget(self, key, *fields):
        table = self.store.get(key, {})
        return [table.get(field) for field in fields]

    def hdel(self, key, *fields):
        table = self.store.get(key, {})
        removed = 0
        for field in fields:
            if field in table:
                del table[field]
                removed += 1
        return removed

    # -- lists ------------------------------------------------------------- #
    def rpush(self, key, *values):
        self.store.setdefault(key, []).extend(values)
        return len(self.store[key])

    def lrange(self, key, start, stop):
        items = list(self.store.get(key, []))
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    def ltrim(self, key, start, stop):
        items = list(self.store.get(key, []))
        self.store[key] = items[start:] if stop == -1 else items[start : stop + 1]
        return True

    def llen(self, key):
        return len(self.store.get(key, []))

    def lrem(self, key, count, value):
        items = self.store.get(key, [])
        self.store[key] = [i for i in items if i != value]
        return len(items) - len(self.store[key])

    # -- keyspace ---------------------------------------------------------- #
    def scan(self, cursor=0, match=None, count=500):
        keys = [k for k in self.store if match is None or fnmatch.fnmatch(k, match)]
        return 0, keys

    def delete(self, *keys):
        removed = 0
        for key in keys:
            removed += 1 if self.store.pop(key, None) is not None else 0
        return removed

    def unlink(self, *keys):
        return self.delete(*keys)


def make_hybrid(**kwargs):
    fake = FakeRedis()
    hot = RedisHotStore(key_prefix="episode:{episode_id}", client=fake)
    cold = SQLiteColdStore(":memory:")
    store = RedisSQLiteHybridStore(redis_store=hot, sqlite_store=cold, **kwargs)
    return store, fake, cold


class TestRedisNamespacing(unittest.TestCase):
    def test_key_layout(self):
        store, fake, _ = make_hybrid()
        store.bind_episode("ep_A")
        store.add_active_summary(
            ActiveSummary(summary_id="ep_A/s001@aaaaaa", text="t", episode_id="ep_A", seq=1)
        )
        keys = sorted(fake.store.keys())
        self.assertTrue(all(key.startswith("episode:ep_A:") for key in keys), keys)
        self.assertIn("episode:ep_A:active_ids", keys)
        self.assertIn("episode:ep_A:active", keys)

    def test_episodes_never_collide(self):
        store, fake, _ = make_hybrid()
        store.bind_episode("ep_A", reset=True)
        store.add_active_summary(
            ActiveSummary(summary_id="ep_A/s001@aaaaaa", text="A", episode_id="ep_A", seq=1)
        )
        store.bind_episode("ep_B", reset=True)
        store.add_active_summary(
            ActiveSummary(summary_id="ep_B/s001@bbbbbb", text="B", episode_id="ep_B", seq=1)
        )
        self.assertEqual(len(store.list_active_summaries("ep_A")), 1)
        self.assertEqual(len(store.list_active_summaries("ep_B")), 1)
        self.assertEqual(store.list_active_summaries("ep_A")[0].text, "A")
        self.assertEqual(store.list_active_summaries("ep_B")[0].text, "B")

    def test_reset_deletes_only_that_episode_and_keeps_cold_data(self):
        store, fake, cold = make_hybrid()
        for episode in ("ep_A", "ep_B"):
            store.bind_episode(episode, reset=True)
            store.add_raw_record(
                RawDialogRecord(reference_id=f"{episode}/raw000@aaaaaa", user_msg="u",
                                agent_msg="a", episode_id=episode, turn_index=0),
                episode_id=episode,
            )
            store.add_active_summary(
                ActiveSummary(summary_id=f"{episode}/s001@aaaaaa", text="t",
                              episode_id=episode, seq=1, raw_ref_id=f"{episode}/raw000@aaaaaa")
            )
            store.add_archived_summary(
                ArchivedSummary(
                    summary_id=f"{episode}/a001@aaaaaa", text="old", episode_id=episode
                )
            )
        self.assertTrue(any(key.startswith("episode:ep_A:") for key in fake.store))
        store.reset(episode_id="ep_A")
        self.assertFalse(any(key.startswith("episode:ep_A:") for key in fake.store),
                         "reset must delete every key of the episode namespace")
        self.assertTrue(any(key.startswith("episode:ep_B:") for key in fake.store),
                        "reset must not touch another episode")
        self.assertEqual(cold.count_raw_records("ep_A"), 1, "cold raw data is permanent")
        self.assertEqual(cold.count_archived_summaries("ep_A"), 1, "archive is permanent")
        self.assertEqual(store.list_active_summaries("ep_A"), [])

    def test_redis_failure_degrades_to_sqlite(self):
        store, fake, _ = make_hybrid()
        store.bind_episode("ep_A", reset=True)
        store.add_active_summary(
            ActiveSummary(summary_id="ep_A/s001@aaaaaa", text="t", episode_id="ep_A", seq=1)
        )
        fake.store.clear()  # simulate losing the hot layer
        summaries = store.list_active_summaries("ep_A")
        self.assertEqual(len(summaries), 1, "cold mirror must serve the active chain")
        self.assertGreaterEqual(store.diagnostics["fallbacks"], 1)

    def test_resume_rebuilds_hot_state(self):
        store, fake, _ = make_hybrid()
        store.bind_episode("ep_A", reset=True)
        store.add_active_summary(
            ActiveSummary(summary_id="ep_A/s001@aaaaaa", text="t", episode_id="ep_A", seq=1)
        )
        fake.store.clear()
        store.rebuild_hot_state("ep_A")
        self.assertEqual(len(store.hot.list_active_summaries("ep_A")), 1)
        self.assertEqual(store.diagnostics["resyncs"], 1)


class TestHotStateRoundTrip(unittest.TestCase):
    """
    Every dataclass field must survive the Redis hot path.

    Regression: ``_load_summary`` dropped ``fact_keys``/``index_id`` and
    ``loads_index`` dropped ``previews``/``child_index_ids``.  Because every real
    experiment runs on the hybrid store, the hot path silently became poorer than
    the cold one: index titles lost their ``属性=值`` digest, index lines fell back
    to full member lines, and the lazy sweep re-wrote summaries on every turn.
    """

    def test_active_summary_fields_survive_the_hot_path(self):
        import dataclasses

        store, _fake, _cold = make_hybrid()
        store.bind_episode("ep_A", reset=True)
        summary = ActiveSummary(
            summary_id="ep_A/s007@abc123",
            text="工位换到 3 楼",
            override_ids=["ep_A/s002@ffffff"],
            timestamp=1234.5,
            raw_ref_id="ep_A/raw007@aaaaaa",
            episode_id="ep_A",
            seq=7,
            origin="event",
            raw_ref_ids=["ep_A/raw007@aaaaaa", "ep_A/raw008@bbbbbb"],
            merged_from=["ep_A/s001@cccccc"],
            fact_keys=["工位=3楼"],
            index_id="ep_A/idx003@dddddd",
        )
        store.add_active_summary(summary, episode_id="ep_A")

        hot = store.hot.list_active_summaries("ep_A")[0]
        cold = store.cold.list_active_summaries("ep_A")[0]
        for label, back in (("hot", hot), ("cold", cold)):
            for field in dataclasses.fields(ActiveSummary):
                self.assertEqual(
                    getattr(back, field.name), getattr(summary, field.name),
                    f"ActiveSummary.{field.name} lost on the {label} path",
                )

    def test_index_entry_fields_survive_the_hot_path(self):
        import dataclasses

        store, _fake, _cold = make_hybrid()
        store.bind_episode("ep_A", reset=True)
        entry = IndexEntry(
            index_id="ep_A/sidx002@abc123",
            title="工位=3楼[12楼→3楼] ｜ 搬办公室",
            members=["ep_A/s001@aaaaaa", "ep_A/s002@bbbbbb"],
            span_start=10.0,
            span_end=20.0,
            turn_start=3,
            turn_end=8,
            fact_keys=["工位=3楼"],
            episode_id="ep_A",
            seq=11,
            timestamp=99.0,
            member_summaries=["- line one", "- line two"],
            previews=["3楼", "12楼"],
            child_index_ids=["ep_A/idx001@cccccc"],
        )
        store.add_index_entry(entry, episode_id="ep_A")

        hot = store.hot.list_index_entries("ep_A")[0]
        cold = store.cold.list_index_entries("ep_A")[0]
        for label, back in (("hot", hot), ("cold", cold)):
            for field in dataclasses.fields(IndexEntry):
                self.assertEqual(
                    getattr(back, field.name), getattr(entry, field.name),
                    f"IndexEntry.{field.name} lost on the {label} path",
                )


class TestHotFailuresDegrade(unittest.TestCase):
    """
    A Redis failure mid-write must degrade, not abort the episode.

    Only ``_call`` used to be wrapped, so a connection error inside the
    pipeline-based writes escaped the hybrid store's ``except RedisUnavailable`` --
    and since SQLite had already committed the same write, the episode died for no
    reason.  That is the difference between "Redis is a cache" and "Redis is a
    dependency".
    """

    class _FailingPipeline:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def execute(self):
            raise ConnectionError("redis went away")

    class _FailingRedis(FakeRedis):
        def pipeline(self, transaction=True):
            return TestHotFailuresDegrade._FailingPipeline()

    def test_pipeline_failure_is_contained(self):
        fake = self._FailingRedis()
        hot = RedisHotStore(key_prefix="episode:{episode_id}", client=fake)
        store = RedisSQLiteHybridStore(
            redis_store=hot, sqlite_store=SQLiteColdStore(":memory:")
        )
        store.bind_episode("ep_A", reset=True)
        summary = ActiveSummary(
            summary_id="ep_A/s001@aaaaaa", text="t", episode_id="ep_A", seq=1
        )
        store.add_active_summary(summary, episode_id="ep_A")  # must not raise
        self.assertFalse(store.diagnostics["redis_available"])
        self.assertGreaterEqual(store.diagnostics["redis_degraded_events"], 1)
        # SQLite is the source of truth, so the write is not lost.
        self.assertEqual(len(store.cold.list_active_summaries("ep_A")), 1)


class TestInMemoryIdempotence(unittest.TestCase):
    def test_raw_add_is_idempotent(self):
        """
        Re-adding a deterministic raw id must not double-count.

        SQLite upserts on the primary key; the in-memory backend appended blindly,
        so ``count_raw_records`` (and full_context's token totals, which walk the
        same list) inflated on the default backend.
        """
        from memory3l.store import InMemoryStore

        store = InMemoryStore(recent_window_turns=2)
        store.bind_episode("ep_A", reset=True)
        record = RawDialogRecord(
            reference_id="ep_A/raw000@aaaaaa", user_msg="u", agent_msg="a",
            episode_id="ep_A", turn_index=0,
        )
        store.add_raw_record(record, episode_id="ep_A")
        store.add_raw_record(record, episode_id="ep_A")
        self.assertEqual(store.count_raw_records("ep_A"), 1)
        self.assertEqual(len(store.list_raw_records("ep_A")), 1)


class TestHybridWindowMirror(unittest.TestCase):
    def test_window_is_mirrored_before_the_hot_write(self):
        """
        The window write must reach SQLite unconditionally.

        It used to update Redis first and mirror afterwards, so a Redis failure (or a
        crash in between) left the crash-recovery mirror stale -- and
        ``rebuild_hot_state`` restores the window only from that mirror.
        """
        store, fake, cold = make_hybrid()
        store.bind_episode("ep_A", reset=True)
        record = RawDialogRecord(
            reference_id="ep_A/raw000@aaaaaa", user_msg="u", agent_msg="a",
            episode_id="ep_A", turn_index=0,
        )
        store.add_raw_record(record, episode_id="ep_A")
        store.append_window_record(record, episode_id="ep_A")
        self.assertEqual(len(cold.get_window("ep_A")), 1)
        fake.store.clear()  # lose the hot layer
        self.assertEqual(len(store.get_window("ep_A")), 1, "the mirror must serve the window")


class TestSQLitePersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")

    def test_tables_are_created_and_data_survives_reopen(self):
        cold = SQLiteColdStore(self.db_path)
        cold.add_raw_record(
            RawDialogRecord(reference_id="ep/raw000@aaaaaa", user_msg="hello", agent_msg="hi",
                            episode_id="ep", turn_index=0)
        )
        cold.mark_episode_done("ep", run_id="run1", num_turns=1)
        cold.close()

        reopened = SQLiteColdStore(self.db_path)
        record = reopened.get_raw_record("ep/raw000@aaaaaa")
        self.assertIsNotNone(record)
        self.assertEqual(record.user_msg, "hello")
        self.assertEqual(reopened.get_done_episodes("run1"), {"ep"})
        counts = reopened.table_counts()
        for table in ("raw_records", "archived_summaries", "active_summaries",
                      "sliding_window", "episode_progress"):
            self.assertIn(table, counts)
        reopened.close()

    def test_archived_roundtrip_preserves_fields(self):
        cold = SQLiteColdStore(":memory:")
        archived = ArchivedSummary(
            summary_id="ep/a001@aaaaaa",
            text="home city=Beijing",
            override_ids=["ep/s000@000000"],
            raw_ref_id="ep/raw000@aaaaaa",
            is_overridden=True,
            episode_id="ep",
            seq=3,
            superseded_by="ep/s005@555555",
            archive_reason="overridden",
            raw_ref_ids=["ep/raw000@aaaaaa", "ep/raw001@111111"],
        )
        cold.add_archived_summary(archived)
        loaded = cold.get_archived_summary("ep/a001@aaaaaa")
        self.assertEqual(loaded.text, archived.text)
        self.assertEqual(loaded.override_ids, archived.override_ids)
        self.assertEqual(loaded.raw_ref_ids, archived.raw_ref_ids)
        self.assertTrue(loaded.is_overridden)
        self.assertEqual(loaded.superseded_by, archived.superseded_by)
        self.assertEqual(loaded.seq, 3)


class TestColdStoreStandalone(unittest.TestCase):
    """
    The SQLite cold store has to be usable *as the only store*.

    It was not: it had no episode binding and no turn bookkeeping, so it could not
    drive ``MemoryManager`` at all, and the missing methods were papered over at
    every call site instead.  These tests pin the contract that made it usable --
    including the return value of ``remove_active_summary``, which is not a
    convenience but the thing the override path archives from.
    """

    def setUp(self):
        self.store = SQLiteColdStore(":memory:")

    def tearDown(self):
        self.store.close()

    @staticmethod
    def _summary(summary_id="ep/s001", episode_id="ep", text="x", seq=1):
        return ActiveSummary(
            summary_id=summary_id, text=text, episode_id=episode_id, seq=seq,
            timestamp=float(seq), raw_ref_id=f"{episode_id}/r{seq}",
            raw_ref_ids=[f"{episode_id}/r{seq}"], origin="event",
            fact_keys=["k=v"],
        )

    def test_remove_active_summary_returns_what_it_deleted(self):
        """
        Regression: a bare DELETE returned None, so MemoryManager's
        ``if old is None: continue`` skipped archiving on every override and the
        overridden text vanished with no forward pointer.  I1 reports exactly that.
        """
        self.store.add_active_summary(self._summary(), episode_id="ep")
        removed = self.store.remove_active_summary("ep/s001", episode_id="ep")
        self.assertIsNotNone(removed, "the removed summary must be returned, or overrides lose their archive step")
        self.assertEqual(removed.summary_id, "ep/s001")
        self.assertEqual(removed.fact_keys, ["k=v"])
        self.assertIsNone(self.store.get_active_summary("ep/s001", episode_id="ep"))

    def test_remove_of_an_unknown_id_is_none(self):
        self.assertIsNone(self.store.remove_active_summary("ep/nope", episode_id="ep"))

    def test_remove_with_a_mismatched_episode_is_a_no_op(self):
        """
        ``summary_id`` is the table's primary key, so two episodes cannot actually
        hold the same id through this API -- a second insert overwrites the first.
        The episode scoping is therefore defence in depth for raw SQL and future
        callers, and "nothing was deleted" is the property worth pinning.
        """
        self.store.add_active_summary(self._summary(summary_id="ep/s001", episode_id="A"), episode_id="A")
        self.assertIsNone(self.store.remove_active_summary("ep/s001", episode_id="B"))
        self.assertIsNotNone(self.store.get_active_summary("ep/s001", episode_id="A"))

    def test_override_round_trip_keeps_the_fact_reachable(self):
        """The consequence that matters: remove-then-archive must not lose the value."""
        self.store.add_active_summary(self._summary(), episode_id="ep")
        old = self.store.remove_active_summary("ep/s001", episode_id="ep")
        self.store.add_archived_summary(
            ArchivedSummary.from_active(old, is_overridden=True, superseded_by="ep/s002"),
            episode_id="ep",
        )
        self.assertEqual(self.store.count_archived_summaries("ep"), 1)
        self.assertEqual(self.store.get_archived_summary("ep/s001", episode_id="ep").superseded_by, "ep/s002")

    def test_bind_and_turn_bookkeeping(self):
        self.store.bind_episode("ep", reset=True)
        self.assertEqual(self.store.bound_episode, "ep")
        self.assertEqual(self.store.next_turn_index(), 0)
        self.assertEqual(self.store.next_turn_index(), 1)
        self.store.set_turn_index(7)
        self.assertEqual(self.store.current_turn_index, 7)

    def test_reset_clears_working_state_and_keeps_permanent_data(self):
        self.store.bind_episode("ep", reset=True)
        self.store.add_active_summary(self._summary(), episode_id="ep")
        self.store.add_archived_summary(
            ArchivedSummary.from_active(self._summary(), is_overridden=True), episode_id="ep")
        self.store.add_raw_record(
            RawDialogRecord(reference_id="ep/r1", episode_id="ep", turn_index=0,
                            user_msg="u", agent_msg="a", timestamp=1.0),
            episode_id="ep")
        self.store.reset(episode_id="ep")
        self.assertEqual(self.store.list_active_summaries("ep"), [])
        self.assertEqual(self.store.count_archived_summaries("ep"), 1, "archive is permanent")
        self.assertEqual(self.store.count_raw_records("ep"), 1, "raw dialogue is permanent")

    def test_append_window_record_trims_to_the_window(self):
        self.store.recent_window_turns = 2
        self.store.bind_episode("ep", reset=True)
        for turn in range(4):
            window = self.store.append_window_record(
                RawDialogRecord(reference_id=f"ep/r{turn}", episode_id="ep", turn_index=turn,
                                user_msg=f"u{turn}", agent_msg="a", timestamp=float(turn)),
                episode_id="ep")
        self.assertEqual([r.turn_index for r in window], [2, 3])
        self.assertEqual([r.turn_index for r in self.store.get_window("ep")], [2, 3])

    def test_summaries_under_index_resolves_live_members_only(self):
        self.store.add_active_summary(self._summary(summary_id="ep/s001"), episode_id="ep")
        self.store.add_active_summary(self._summary(summary_id="ep/s002", seq=2), episode_id="ep")
        self.store.add_index_entry(
            IndexEntry(index_id="ep/idx", episode_id="ep", seq=1, title="t",
                       members=["ep/s001", "ep/gone"]),
            episode_id="ep")
        members = [s.summary_id for s in self.store.summaries_under_index("ep/idx", episode_id="ep")]
        self.assertEqual(members, ["ep/s001"], "a stale pointer must not resolve")


class TestInMemoryStore(unittest.TestCase):
    def test_reset_isolates_episodes(self):
        store = InMemoryStore(recent_window_turns=1)
        store.bind_episode("A", reset=True)
        store.add_active_summary(ActiveSummary(summary_id="A/s1@aaaaaa", text="A", episode_id="A", seq=1))
        store.bind_episode("B", reset=True)
        store.add_active_summary(ActiveSummary(summary_id="B/s1@bbbbbb", text="B", episode_id="B", seq=1))
        self.assertEqual(len(store.list_active_summaries("A")), 1)
        store.reset(episode_id="A")
        self.assertEqual(store.list_active_summaries("A"), [])
        self.assertEqual(len(store.list_active_summaries("B")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
