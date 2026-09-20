"""
Unit tests for parsing, override/compression semantics, and episode isolation.

These tests need **no** network, no Redis server and no LLM: they use the
in-memory store, the SQLite store and (in ``test_store.py``) a fake Redis client.
Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory3l.dataset import build_synthetic_episodes
from memory3l.llm import HeuristicLLM, ScriptedLLM
from memory3l.memory_manager import MemoryManager
from memory3l.models import ActiveSummary, ArchivedSummary, RawDialogRecord
from memory3l.store import InMemoryStore
from memory3l.token_utils import estimate_tokens
from memory3l.tools import (
    ToolExecutor,
    normalise_tool_args,
    parse_summary_response,
    parse_tool_calls,
    strip_tool_echoes,
)


class TestSummaryParsing(unittest.TestCase):
    """The summariser grammar is the contract the whole archive graph rests on."""

    VALID = ["ep/s001@aaaaaa", "ep/s002@bbbbbb"]

    def test_plain_none(self):
        parsed = parse_summary_response("User moved to Shanghai.\n[OVERRIDES: none]", self.VALID)
        self.assertEqual(parsed["summary_text"], "User moved to Shanghai.")
        self.assertEqual(parsed["override_ids"], [])
        self.assertFalse(parsed["missing_tag"])

    def test_multiple_ids_comma_and_cjk_separators(self):
        for payload in ("ep/s001@aaaaaa,ep/s002@bbbbbb", "ep/s001@aaaaaa，ep/s002@bbbbbb",
                        "ep/s001@aaaaaa; ep/s002@bbbbbb", "ep/s001@aaaaaa ep/s002@bbbbbb"):
            parsed = parse_summary_response(f"Updated.\n[OVERRIDES: {payload}]", self.VALID)
            self.assertEqual(parsed["override_ids"], self.VALID, payload)

    def test_hallucinated_id_is_rejected(self):
        parsed = parse_summary_response("Updated.\n[OVERRIDES: ep/s999@ffffff,ep/s001@aaaaaa]", self.VALID)
        self.assertEqual(parsed["override_ids"], ["ep/s001@aaaaaa"])
        self.assertEqual(parsed["invalid_override_ids"], ["ep/s999@ffffff"])

    def test_missing_tag_is_flagged_not_fatal(self):
        parsed = parse_summary_response("Just a summary without any tag.", self.VALID)
        self.assertTrue(parsed["missing_tag"])
        self.assertEqual(parsed["summary_text"], "Just a summary without any tag.")

    def test_code_fence_and_label_are_stripped(self):
        parsed = parse_summary_response("```\n摘要: Moved to Chengdu.\n[OVERRIDES: none]\n```", self.VALID)
        self.assertEqual(parsed["summary_text"], "Moved to Chengdu.")
        self.assertFalse(parsed["missing_tag"])

    def test_none_variants(self):
        for token in ("none", "None", "无", "-"):
            parsed = parse_summary_response(f"x\n[OVERRIDES: {token}]", self.VALID)
            self.assertEqual(parsed["override_ids"], [], token)

    def test_tag_on_same_line_as_text(self):
        parsed = parse_summary_response("Moved to Chengdu. [OVERRIDES: none]", self.VALID)
        self.assertEqual(parsed["summary_text"], "Moved to Chengdu.")


class TestToolCallParsing(unittest.TestCase):
    def test_standard_forms(self):
        text = 'get_archived_summary(summary_id="ep/s001@aaaaaa")'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_archived_summary")
        self.assertEqual(normalise_tool_args(calls[0])["summary_id"], "ep/s001@aaaaaa")

    def test_unquoted_and_alias_arguments(self):
        for text, expected in (
            ("get_raw_record(reference_id=ep/raw001@aaaaaa)", "ep/raw001@aaaaaa"),
            ('get_raw_record(ref="ep/raw001@aaaaaa")', "ep/raw001@aaaaaa"),
            ("get_raw_record(reference_id: 'ep/raw001@aaaaaa')", "ep/raw001@aaaaaa"),
            ('get_archived_summary(id="ep/s001@aaaaaa")', "ep/s001@aaaaaa"),
        ):
            calls = parse_tool_calls(text)
            self.assertEqual(len(calls), 1, text)
            args = normalise_tool_args(calls[0])
            value = args.get("summary_id") or args.get("reference_id")
            self.assertEqual(value, expected, text)

    def test_json_form(self):
        text = '{"name": "get_archived_summary", "arguments": {"summary_id": "ep/s001@aaaaaa"}}'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args["summary_id"], "ep/s001@aaaaaa")

    def test_deepseek_dsml_markup_form(self):
        """Real DeepSeek checkpoints sometimes emit internal markup, not text."""
        for pipe in ("|", "\uff5c"):
            for pipes in (pipe, pipe * 2):
                text = (
                    f'<{pipes}DSML{pipes}invoke name="get_raw_record">'
                    f'<{pipes}DSML{pipes}parameter name="reference_id">ep/raw030@c73002'
                    f'</{pipes}DSML{pipes}parameter></{pipes}DSML{pipes}invoke>'
                )
                calls = parse_tool_calls(text)
                self.assertEqual(len(calls), 1, text)
                self.assertEqual(calls[0].name, "get_raw_record")
                args = normalise_tool_args(calls[0])
                self.assertEqual(args["reference_id"], "ep/raw030@c73002", text)

    def test_plain_answer_is_not_a_tool_call(self):
        self.assertEqual(parse_tool_calls("Your home city is Shanghai."), [])

    def test_tool_result_echo_is_not_reparsed(self):
        """A model quoting a tool result must not be re-read as a new call."""
        echoed = (
            "[TOOL RESULT: get_archived_summary] [OVERRIDDEN] ep/s001@aaaaaa [OVERRIDES: none] "
            "(raw_ref: ep/raw001@aaaaaa) Facts: home city=Beijing"
        )
        self.assertEqual(parse_tool_calls(echoed), [])
        self.assertEqual(strip_tool_echoes(echoed), "")
        # ...including when the echoed line itself contains call grammar.
        quoted = (
            '[TOOL RESULT: get_archived_summary ERROR] use '
            'get_archived_summary(summary_id="ep/s001@ab12cd")'
        )
        self.assertEqual(parse_tool_calls(quoted), [])

    def test_full_width_punctuation_still_parses(self):
        """
        A Chinese-oriented prompt gets full-width punctuation back.

        It used to produce *no* call at all: not counted as an attempt, and the
        tool-call text became the final answer.
        """
        for text in (
            'get_archived_summary(summary_id："ep/s001@ab12cd")',
            'get_archived_summary（summary_id="ep/s001@ab12cd"）',
            'get_raw_record（reference_id＝"ep/raw001@aaaaaa"）',
        ):
            calls = parse_tool_calls(text)
            self.assertEqual(len(calls), 1, text)
            args = normalise_tool_args(calls[0])
            self.assertTrue(args.get("summary_id") or args.get("reference_id"), text)

    def test_ambiguous_aliases_are_resolved_per_tool(self):
        """
        ``reference``/``id`` mean different things to different tools.

        A flat alias table sent the history *anchor* into ``reference_id``, which
        silently downgraded ``get_predecessor_summary`` to "newest old value".
        """
        anchor = normalise_tool_args(
            parse_tool_calls('get_predecessor_summary(fact_key="楼层", reference="12楼")')[0]
        )
        self.assertEqual(anchor["reference_value"], "12楼")
        self.assertNotIn("reference_id", anchor)

        raw = normalise_tool_args(parse_tool_calls('get_raw_record(id="ep/raw001@aaaaaa")')[0])
        self.assertEqual(raw["reference_id"], "ep/raw001@aaaaaa")

        archived = normalise_tool_args(
            parse_tool_calls('get_archived_summary(id="ep/s001@aaaaaa")')[0]
        )
        self.assertEqual(archived["summary_id"], "ep/s001@aaaaaa")

    def test_two_overrides_tags_leave_no_tag_in_the_body(self):
        parsed = parse_summary_response(
            "Summary body.\n[FACTS: a=1]\n[OVERRIDES: s001]\n[OVERRIDES: s002]"
        )
        self.assertEqual(parsed["summary_text"], "Summary body.")
        self.assertEqual(parsed["override_ids"], ["s001", "s002"])
        self.assertEqual(parsed["fact_keys"], ["a=1"])


class TestMemoryManagerOverrides(unittest.TestCase):
    """Event-driven override semantics (layer 1 -> layer 2)."""

    def setUp(self):
        self.summaries = []

        def scripted(messages):
            return self.summaries.pop(0)

        self.llm = ScriptedLLM(scripted)
        self.store = InMemoryStore(recent_window_turns=2)
        self.manager = MemoryManager(self.store, self.llm, episode_id="ep1",
                                     active_chain_token_limit=10_000)

    def test_override_moves_old_summary_to_archive(self):
        self.summaries = ["Lives in Beijing.\n[OVERRIDES: none]",
                          "Lives in Shanghai.\n[OVERRIDES: OVERRIDE_S001]"]
        first = self.manager.add_dialog_turn("I live in Beijing.", "ok")
        # patch the placeholder with the real id, as a real model would cite it
        self.summaries[0] = self.summaries[0].replace("OVERRIDE_S001", first.new_summary_id)
        second = self.manager.add_dialog_turn("I moved to Shanghai.", "ok")

        self.assertEqual(second.overridden_ids, [first.new_summary_id])
        self.assertTrue(second.event_triggered)
        self.assertEqual(second.capacity_merged_ids, [])

        active = self.manager.list_active_summaries()
        self.assertEqual([s.summary_id for s in active], [second.new_summary_id])
        archived = self.manager.list_archived_summaries()
        self.assertEqual(len(archived), 1)
        self.assertTrue(archived[0].is_overridden)
        self.assertEqual(archived[0].summary_id, first.new_summary_id)
        self.assertEqual(archived[0].superseded_by, second.new_summary_id)
        self.assertEqual(archived[0].archive_reason, "overridden")
        # the new summary carries the OVERRIDES tag
        self.assertEqual(active[0].override_ids, [first.new_summary_id])
        self.assertIn("[OVERRIDES:", active[0].render())

    def test_invalid_override_id_is_dropped(self):
        self.summaries = ["A.\n[OVERRIDES: none]", "B.\n[OVERRIDES: ep/nope@123456]"]
        self.manager.add_dialog_turn("a", "b")
        stats = self.manager.add_dialog_turn("c", "d")
        self.assertEqual(stats.overridden_ids, [])
        self.assertEqual(stats.invalid_override_ids, ["ep/nope@123456"])
        self.assertEqual(len(self.manager.list_active_summaries()), 2)
        self.assertEqual(self.manager.list_archived_summaries(), [])

    def test_capacity_compression_has_no_overrides_tag(self):
        self.store = InMemoryStore(recent_window_turns=2)
        manager = MemoryManager(
            self.store,
            ScriptedLLM(lambda messages: "some summary\n[FACTS: k=v]\n[OVERRIDES: none]"),
            episode_id="ep2",
            active_chain_token_limit=1,   # force compression every turn
            capacity_min_merge=2,
            capacity_max_merge=2,
            chain_strategy="merge",       # legacy flat strategy, pinned explicitly
        )
        for index in range(4):
            manager.add_dialog_turn(f"turn {index} " + "x" * 20, "ok")
        merged = [s for s in manager.list_active_summaries() if s.origin == "capacity_merge"]
        self.assertTrue(merged, "capacity compression never fired")
        for summary in merged:
            self.assertEqual(summary.override_ids, [])
            self.assertNotIn("[OVERRIDES: ", summary.text)
        for archived in manager.list_archived_summaries():
            if archived.archive_reason == "capacity":
                self.assertFalse(archived.is_overridden)
                self.assertIsNone(archived.superseded_by)

    def test_raw_record_and_window_layers(self):
        self.summaries = [f"s{i}\n[OVERRIDES: none]" for i in range(6)]
        for index in range(6):
            self.manager.add_dialog_turn(f"user {index}", f"agent {index}")
        records = self.store.list_raw_records("ep1")
        self.assertEqual(len(records), 6, "raw store must keep every turn")
        window = self.manager.get_window()
        self.assertEqual(len(window), 2, "window must respect recent_window_turns")
        self.assertEqual(window[-1].user_msg, "user 5")
        # evicted window entries survive in the raw layer, reachable by exact id
        evicted = records[0]
        self.assertIsNotNone(self.store.get_raw_record(evicted.reference_id, "ep1"))
        self.assertFalse(any(r.reference_id == evicted.reference_id for r in window))

    def test_episode_reset_isolates_and_keeps_archive(self):
        self.summaries = ["A.\n[OVERRIDES: none]"] * 3
        self.manager.add_dialog_turn("a", "b")
        archived_before = len(self.manager.list_archived_summaries())
        self.manager.reset_episode("other_episode")
        self.assertEqual(self.manager.list_active_summaries(), [])
        self.assertEqual(self.manager.get_window(), [])
        self.assertEqual(self.store.count_raw_records("ep1"), 1, "cold raw data must survive reset")
        self.assertEqual(len(self.manager.list_archived_summaries()), archived_before)


class TestArchiveTool(unittest.TestCase):
    """get_archived_summary must follow the supersede chain forward."""

    def setUp(self):
        self.store = InMemoryStore(recent_window_turns=4)
        self.store.bind_episode("ep1", reset=True)
        self.s1 = ActiveSummary(summary_id="ep/s001@aaaaaa", text="home city=Beijing",
                                episode_id="ep1", seq=1, raw_ref_id="ep/raw0@aaaaaa")
        self.s2 = ActiveSummary(summary_id="ep/s002@bbbbbb", text="home city=Shanghai",
                                episode_id="ep1", seq=2, raw_ref_id="ep/raw1@bbbbbb")
        self.store.add_archived_summary(
            ArchivedSummary.from_active(self.s1, is_overridden=True, superseded_by="ep/s002@bbbbbb")
        )
        self.store.add_archived_summary(
            ArchivedSummary.from_active(self.s2, is_overridden=True, superseded_by="ep/s003@cccccc")
        )
        self.s3 = ActiveSummary(summary_id="ep/s003@cccccc", text="home city=Chengdu",
                                episode_id="ep1", seq=3)
        self.store.add_archived_summary(
            ArchivedSummary.from_active(self.s3, is_overridden=False, archive_reason="capacity")
        )
        self.executor = ToolExecutor(self.store, episode_id="ep1")

    def test_supersede_chain_is_reported(self):
        result = self.executor.get_archived_summary("ep/s001@aaaaaa")
        self.assertTrue(result.ok and result.resolved)
        self.assertIn("SUPERSEDED BY", result.content)
        self.assertIn("Chengdu", result.content)

    def test_missing_id_is_a_clean_failure(self):
        result = self.executor.get_archived_summary("ep/nope@000000")
        self.assertFalse(result.ok)
        self.assertFalse(result.resolved)
        self.assertEqual(result.error, "not found")

    def test_raw_lookup(self):
        record = RawDialogRecord(reference_id="ep/raw0@aaaaaa", user_msg="hi", agent_msg="hello",
                                 episode_id="ep1", turn_index=0)
        self.store.add_raw_record(record, episode_id="ep1")
        result = self.executor.get_raw_record("ep/raw0@aaaaaa")
        self.assertTrue(result.ok)
        self.assertIn("hello", result.content)

    def test_execute_text_reports_unparsable_calls(self):
        results = self.executor.execute_text("I think I should call the tool now.")
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].parsed)


class TestIndexHierarchy(unittest.TestCase):
    """
    The catalogue design: when the chain grows, add a *title level* over groups of
    summaries.  Nothing is deleted or rewritten -- an index only points at members.
    """

    def _manager(self, token_limit=1, keep_recent=2, group_size=2):
        store = InMemoryStore(recent_window_turns=2)
        manager = MemoryManager(
            store,
            ScriptedLLM(lambda messages: "摘要正文\n[FACTS: 工位=3楼]\n[OVERRIDES: none]"),
            episode_id="ep_idx",
            active_chain_token_limit=token_limit,
            chain_strategy="index",
            index_keep_recent=keep_recent,
            index_group_size=group_size,
        )
        return manager, store

    def test_filing_keeps_members_readable_and_shrinks_the_chain(self):
        manager, store = self._manager()
        for index in range(8):
            manager.add_dialog_turn(f"turn {index}", "ok")

        entries = manager.list_index_entries()
        self.assertTrue(entries, "no index entry was built")

        entry = entries[0]
        self.assertGreaterEqual(len(entry.members), 2)
        # every member still exists in layer 1 and resolves by id
        for member_id in entry.members:
            self.assertIsNotNone(
                store.get_active_summary(member_id, episode_id="ep_idx"),
                "a filed summary must remain readable by id",
            )
        expanded = manager.expand_index(entry.index_id)
        self.assertEqual([s.summary_id for s in expanded], entry.members)
        # the rendered chain excludes filed members
        rendered_ids = {s.summary_id for s in manager.chain_summaries()}
        self.assertFalse(rendered_ids & set(entry.members))
        # and the rendered prompt is smaller than rendering every summary flat
        self.assertLess(manager.active_chain_tokens(), manager.all_rendered_tokens())

    def test_index_title_carries_no_override_tag(self):
        """The index line is a title; OVERRIDES belongs to the member summaries."""
        manager, _ = self._manager()
        for index in range(8):
            manager.add_dialog_turn(f"turn {index}", "ok")
        for entry in manager.list_index_entries():
            self.assertNotIn("[OVERRIDES", entry.title)
            # the bare title line must not fabricate a tag...
            self.assertNotIn("[OVERRIDES", entry.render(preview=0))
            # ...while previewed member lines legitimately show their own tags
            preview_lines = entry.render(preview=1).splitlines()[1:]
            for line in preview_lines:
                self.assertNotIn("[INDEX", line)

    def test_members_are_not_split_from_their_raw_refs(self):
        manager, store = self._manager()
        for index in range(6):
            manager.add_dialog_turn(f"turn {index}", "ok")
        for entry in manager.list_index_entries():
            for summary in manager.expand_index(entry.index_id):
                self.assertTrue(summary.raw_ref_id)
                self.assertIsNotNone(
                    store.get_raw_record(summary.raw_ref_id, "ep_idx"),
                    "the catalogue entry must still reach its raw dialogue",
                )

    def test_fact_digest_reports_the_newest_value_as_current(self):
        """
        The digest used to iterate newest-first, so ``seq[-1]`` -- labelled the
        current value -- was the group's OLDEST value and the history read backwards.
        """
        from memory3l.models import ActiveSummary

        members = [
            ActiveSummary(summary_id=f"e/s{i}@x", text=f"工位={i}楼",
                          fact_keys=[f"工位={i}楼"], seq=i, episode_id="e")
            for i in (1, 2, 3)
        ]
        digest = MemoryManager._fact_digest(members)
        self.assertEqual(digest, "工位=3楼[1楼→2楼→3楼]")
        self.assertTrue(digest.startswith("工位=3楼"), digest)

    def test_override_removes_the_member_from_its_index_entry(self):
        """
        An override must update the entry that pointed at the summary.

        Before the fix the entry kept the archived member: ``N entries`` overstated
        the group, ``expand_index`` returned fewer summaries than advertised, and the
        preview/title still showed the superseded value.
        """
        manager, store = self._manager()
        for index in range(10):
            manager.add_dialog_turn(f"turn {index}", "ok")
        entries = manager.list_index_entries()
        entry = entries[0]
        target = entry.members[0]
        short = target.split("/")[-1].split("@")[0]

        manager.summarizer = ScriptedLLM(
            lambda messages: f"工位改到99楼\n[FACTS: 工位=99楼]\n[OVERRIDES: {short}]"
        )
        manager.add_dialog_turn("工位换了吗", "换成99楼了")

        fresh = {e.index_id: e for e in manager.list_index_entries()}
        if entry.index_id in fresh:
            updated = fresh[entry.index_id]
            self.assertNotIn(target, updated.members)
            self.assertEqual(updated.size, len(manager.expand_index(updated.index_id)))
        self.assertTrue(
            sum(s.index_updates for s in manager.stats) >= 1,
            "the override should have rewritten an index entry",
        )

    def test_current_values_registry_tracks_the_newest_value(self):
        """
        The registry is the top-level guarantee: the current value must be visible
        without a tool call even when the summary that holds it is filed away.
        """
        manager, _ = self._manager()
        for index in range(10):
            manager.add_dialog_turn(f"turn {index}", "ok")

        values = {slot: value for slot, value, _sid in manager.current_values()}
        self.assertEqual(values.get("工位"), "3楼")
        rendered = manager.render_current_values()
        self.assertIn("工位=3楼", rendered)
        # ...and it costs almost nothing
        self.assertLess(len(rendered), 60)
        # even though nothing holding that fact is rendered directly
        self.assertTrue(manager.list_index_entries(), "expected the hierarchy to file summaries")

    def test_lazy_summaries_are_not_rendered(self):
        """A summary parked in the lazy store must not render (documented contract)."""
        from memory3l.models import LAZY_INDEX_ID

        manager, store = self._manager()
        for index in range(8):
            manager.add_dialog_turn(f"turn {index}", "ok")
        actives = manager.list_active_summaries()
        if not actives:
            self.skipTest("no summaries")
        actives[0].index_id = LAZY_INDEX_ID
        store.add_active_summary(actives[0], episode_id="ep_idx")
        rendered = {s.summary_id for s in manager.chain_summaries()}
        self.assertNotIn(actives[0].summary_id, rendered)


class TestHeuristicPipeline(unittest.TestCase):
    """End-to-end mechanics with the dependency-free backend."""

    def test_overrides_fire_on_synthetic_episode(self):
        llm = HeuristicLLM()
        store = InMemoryStore(recent_window_turns=4)
        manager = MemoryManager(store, llm, episode_id="ep")
        episode = build_synthetic_episodes(num_episodes=1, seed=11, turns_per_episode=12)[0]
        for user, agent in episode.dialogues:
            manager.add_dialog_turn(user, agent)
        metrics = manager.episode_metrics()
        self.assertGreater(metrics["overrides_events"], 0)
        self.assertGreater(metrics["archived_count"], 0)
        for archived in manager.list_archived_summaries():
            self.assertTrue(archived.is_overridden)
            self.assertIsNotNone(archived.superseded_by)


class TestTokenizer(unittest.TestCase):
    def test_monotonic_and_nonzero(self):
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 20)
        self.assertGreater(short, 0)
        self.assertGreater(long, short)

    def test_cjk_is_not_underestimated(self):
        chinese = estimate_tokens("我的工位在十二楼")
        self.assertGreaterEqual(chinese, 5)


class TestAgentToolLoop(unittest.TestCase):
    """The bounded tool loop must execute textual calls and count them."""

    def test_tool_call_then_answer(self):
        from memory3l.agents import build_agent
        from memory3l.models import ActiveSummary

        store = InMemoryStore(recent_window_turns=2)
        agent = build_agent("three_layer", store, ScriptedLLM(["unused-summariser"]), None)
        agent.summarizer = ScriptedLLM(["summary\n[OVERRIDES: none]"] * 4)
        agent.manager.summarizer = agent.summarizer
        agent.reset_episode("ep_tool")
        agent.add_turn("my home city is Beijing", "ok")
        sid = agent.manager.list_active_summaries()[0].summary_id
        store.add_archived_summary(
            ArchivedSummary.from_active(
                agent.manager.list_active_summaries()[0], is_overridden=True
            )
        )
        store.remove_active_summary(sid, episode_id="ep_tool")

        agent.llm = ScriptedLLM([
            f'get_archived_summary(summary_id="{sid}")',
            "It said home city=Beijing.",
        ])
        run = agent.answer("What does the archived entry say?")
        self.assertEqual(run.tool_calls_attempted, 1)
        self.assertEqual(run.tool_calls_resolved, 1)
        self.assertEqual(run.answer, "It said home city=Beijing.")

    def test_unparsable_tool_intent_is_counted(self):
        from memory3l.agents import build_agent

        store = InMemoryStore(recent_window_turns=2)
        agent = build_agent("three_layer", store, ScriptedLLM(["s\n[OVERRIDES: none]"]), None)
        agent.reset_episode("ep_bad")
        agent.llm = ScriptedLLM([
            "I will call get_archived_summary(summary_id=...) now",
            "Final answer.",
        ])
        run = agent.answer("q")
        self.assertEqual(run.tool_calls_attempted, 1)
        self.assertEqual(run.tool_calls_resolved, 0)

    def test_tool_call_is_never_accepted_as_the_answer(self):
        """
        At the iteration cap a looping model offers a call as its "answer".

        Accepting it put a call string into the prediction CSV, where the judge
        scored it wrong for a harness reason rather than a memory reason.
        """
        from memory3l.agents import build_agent

        store = InMemoryStore(recent_window_turns=2)
        agent = build_agent(
            "three_layer", store,
            ScriptedLLM(['get_raw_record(reference_id="ep_loop/raw000@aaaaaa")'] * 12), None,
        )
        agent.reset_episode("ep_loop")
        run = agent.answer("q")
        self.assertTrue(run.tool_call_as_answer)
        self.assertEqual(run.answer, "")
        self.assertNotIn("get_raw_record", run.answer)

    def test_tool_free_systems_are_not_told_to_call_tools(self):
        """
        Fairness: a system with no executor must not be given a tool manual.

        Advertising tools to ``memgpt_style``/``naive_chain`` made the model emit a
        call that could never execute; it fell through as the final answer and the
        baseline was scored wrong although the value was in its own context.
        """
        from memory3l.agents import build_agent

        store = InMemoryStore(recent_window_turns=2)
        for name in ("memgpt_style", "naive_chain"):
            agent = build_agent(name, store, ScriptedLLM(["answer"]), None)
            agent.reset_episode(f"ep_ns_{name}")
            system_content = agent.build_messages("q?")[0]["content"]
            self.assertFalse(agent.supports_tools(), name)
            self.assertNotIn("get_archived_summary", system_content, name)
            self.assertNotIn("INDEX_LAYER", system_content, name)
            self.assertNotIn("必须先调用工具", system_content, name)

        ours = build_agent("three_layer", store, ScriptedLLM(["answer"]), None)
        ours.reset_episode("ep_ns_ours")
        ours_system = ours.build_messages("q?")[0]["content"]
        self.assertTrue(ours.supports_tools())
        self.assertIn("get_archived_summary", ours_system)


class TestSelfWrittenMemory(unittest.TestCase):
    """
    One call answers *and* records the turn.

    The property that matters: the self-write path must produce exactly the same
    memory as the two-call path (override graph included), and a model that omits
    the block must degrade to the summariser, never to a lost update.
    """

    @staticmethod
    def _reply(visible: str, facts: str, overrides: str = "none") -> str:
        return (
            f"{visible}\n<MEMORY_UPDATE>\n{visible}\n"
            f"[FACTS: {facts}]\n[OVERRIDES: {overrides}]\n</MEMORY_UPDATE>"
        )

    def _agent(self, responses, episode="ep_self"):
        from memory3l.agents import build_agent

        llm = ScriptedLLM(responses)
        store = InMemoryStore(recent_window_turns=2)
        agent = build_agent("three_layer", store, llm, llm, recent_window_turns=2)
        agent.reset_episode(episode)
        return agent, store, llm

    def test_block_is_split_from_the_answer(self):
        agent, _store, _llm = self._agent([])
        run = agent.answer("ignored", selfwrite=False)
        self.assertEqual(run.memory_block, "")

    def test_one_call_per_turn_and_override_fires(self):
        agent, store, llm = self._agent([
            self._reply("你住在上海。", "居住城市=上海"),
            self._reply("已更新为北京。", "居住城市=北京", "s001"),
        ])
        run1, stats1 = agent.answer_and_remember("我住在上海")
        run2, stats2 = agent.answer_and_remember("我搬到北京了")

        # two turns -> exactly two LLM calls (not four)
        self.assertEqual(len(llm.seen_messages), 2)
        self.assertTrue(stats1.self_written)
        self.assertTrue(stats2.self_written)
        # the machine-readable block never leaks into the visible answer
        self.assertNotIn("MEMORY_UPDATE", run2.answer)
        self.assertEqual(run2.answer, "已更新为北京。")
        # the short id the model copied mapped back to the canonical id
        self.assertEqual(stats2.overridden_ids, [stats1.new_summary_id])
        active = store.list_active_summaries("ep_self")
        self.assertEqual([s.summary_id for s in active], [stats2.new_summary_id])
        archived = store.list_archived_summaries("ep_self")
        self.assertEqual([a.summary_id for a in archived], [stats1.new_summary_id])
        self.assertEqual(archived[0].superseded_by, stats2.new_summary_id)

    def test_missing_block_falls_back_without_double_writing(self):
        agent, store, llm = self._agent([
            "上海挺好的。",                                   # answer, no block
            "用户所在城市。\n[FACTS: 城市=上海]\n[OVERRIDES: none]",   # summariser
        ])
        _run, stats = agent.answer_and_remember("我在哪")
        self.assertFalse(stats.self_written)
        self.assertTrue(stats.self_write_fallback)
        self.assertEqual(len(llm.seen_messages), 2)          # answer + summariser
        self.assertTrue(stats.new_summary_id)
        # exactly one raw record: the fallback must not re-ingest the turn
        self.assertEqual(len(store.list_raw_records("ep_self")), 1)

    def test_unparsable_block_falls_back(self):
        agent, _store, llm = self._agent([
            "答案。\n<MEMORY_UPDATE>\n\n</MEMORY_UPDATE>",     # empty body
            "摘要。\n[OVERRIDES: none]",
        ])
        _run, stats = agent.answer_and_remember("q")
        self.assertTrue(stats.self_write_fallback)
        self.assertTrue(stats.new_summary_id)

    def test_split_helper_variants(self):
        from memory3l.tools import split_selfwrite_reply

        visible, block = split_selfwrite_reply("答。\n<MEMORY_UPDATE>\n摘要\n[OVERRIDES: none]\n</MEMORY_UPDATE>")
        self.assertEqual(visible, "答。")
        self.assertIn("摘要", block)
        # fenced variant
        visible, block = split_selfwrite_reply("答。\n```memory_update\n摘要\n[OVERRIDES: none]\n```")
        self.assertEqual(visible, "答。")
        self.assertIn("摘要", block)
        # unclosed tag (models forget the closing tag)
        _visible, block = split_selfwrite_reply("答。\n<MEMORY_UPDATE>\n摘要\n[OVERRIDES: none]")
        self.assertIn("摘要", block)
        # no block at all
        self.assertEqual(split_selfwrite_reply("只是回答")[1], None)

    def test_block_never_leaks_into_the_answer(self):
        """Every delimiter variant must strip the machine payload from the answer."""
        from memory3l.tools import split_selfwrite_reply

        tag = "[OVERRIDES: none]"
        cases = {
            "only block": (
                f"<MEMORY_UPDATE>\nfloor 3\n[FACTS: floor=3]\n{tag}\n</MEMORY_UPDATE>",
                "floor 3",
            ),
            "two blocks": (
                f"A\n<MEMORY_UPDATE>x1\n{tag}</MEMORY_UPDATE>\nmid\n"
                f"<MEMORY_UPDATE>x2\n{tag}</MEMORY_UPDATE>",
                "A",
            ),
            "closed fence": (f"answer\n```memory_update\nq\n{tag}\n```", "answer"),
            "unclosed fence": (f"answer\n```memory_update\nq\n{tag}", "answer"),
            "unclosed tag": (f"answer\n<MEMORY_UPDATE>\nq\n{tag}", "answer"),
            "lowercase tag": (f"answer\n<memory_update>q\n{tag}</memory_update>", "answer"),
        }
        for name, (text, expected_start) in cases.items():
            visible, block = split_selfwrite_reply(text)
            self.assertTrue(block, f"{name}: block not extracted")
            self.assertNotIn("MEMORY_UPDATE", visible.upper(), f"{name}: tag leaked")
            self.assertNotIn("[OVERRIDES", visible.upper(), f"{name}: tag leaked")
            self.assertTrue(
                visible.startswith(expected_start),
                f"{name}: answer body lost (got {visible!r})",
            )

    def test_prose_mention_is_not_a_block(self):
        """
        A sentence that merely *names* the marker must not truncate the answer.

        The unclosed-tag form is only trusted when the remainder carries the
        summary grammar, otherwise "the tag <MEMORY_UPDATE> means ..." would be
        harvested as memory and the real answer deleted.
        """
        from memory3l.tools import split_selfwrite_reply

        text = "The tag <MEMORY_UPDATE> is used to record memory; the answer is 3F."
        visible, block = split_selfwrite_reply(text)
        self.assertIsNone(block)
        self.assertEqual(visible, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
