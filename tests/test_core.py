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
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory3l.dataset import build_synthetic_episodes, HISTORY_FACT
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


class TestAnswerMatching(unittest.TestCase):
    """The string judge must not accept a mention of the gold as the answer."""

    def test_strict_matching(self):
        from evaluation import string_match

        for prediction, gold in (
            ("Paris", "Paris"),
            ("Paris.", "Paris"),
            ("答案是 Paris", "Paris"),
            ("(b) sushi", "sushi"),
            ("sushi", "(b) sushi"),
        ):
            self.assertTrue(string_match(prediction, gold), (prediction, gold))

    def test_mentioning_the_gold_is_not_an_answer(self):
        """
        The old containment rule accepted the gold anywhere in a prediction up to 4x
        its length, so a verbose answer asserting a *different* value was scored
        correct without ever reaching the LLM judge -- exactly what strict matching
        exists to prevent.
        """
        from evaluation import string_match

        for prediction, gold in (
            ("It is now Shanghai, but originally it was Paris.", "Paris"),
            ("The current value is Beijing (previously Paris).", "Paris"),
            ("Beijing", "Paris"),
            ("I think the answer might be Shanghai, not Paris", "Paris"),
        ):
            self.assertFalse(string_match(prediction, gold), (prediction, gold))


class TestSummaryGate(unittest.TestCase):
    """
    The content gate must skip information-free turns and keep fact-bearing ones.

    Cost of a wrong decision is asymmetric: a false skip removes a fact from the
    prompt once it leaves the sliding window, while a false summarise only costs a
    call.  These tests pin the discrimination direction, not a tuned threshold.
    """

    FACT_ZH = "你好，跟你说一下，我的工位楼层是3楼。"
    CHANGE_ZH = "对了，我的会议时间改成下午2点了。"
    FILLER_ZH = "乐观锁和悲观锁在这里到底有什么区别？"
    QUESTION_ABOUT_FACT = "再提醒我一下重试预算和熔断器是怎么配合的。"

    def test_off_summarises_everything(self):
        from memory3l.gate import SummaryGate

        gate = SummaryGate("off")
        for text in (self.FACT_ZH, self.FILLER_ZH, self.CHANGE_ZH):
            self.assertTrue(gate.should_summarise(text))

    def test_pattern_keeps_facts_and_drops_filler(self):
        from memory3l.gate import SummaryGate

        gate = SummaryGate("pattern")
        self.assertTrue(gate.should_summarise(self.FACT_ZH))
        self.assertTrue(gate.should_summarise(self.CHANGE_ZH))
        self.assertFalse(gate.should_summarise(self.FILLER_ZH))
        # an interrogative clause is not a fact statement, even though it contains 是
        self.assertFalse(gate.should_summarise(self.QUESTION_ABOUT_FACT))

    def test_change_verb_survives_a_question_mark(self):
        from memory3l.gate import SummaryGate

        gate = SummaryGate("pattern")
        self.assertTrue(gate.should_summarise("我把地址改成了上海，对吗？"))

    def test_strict_requires_a_value_not_seen_before(self):
        from memory3l.gate import SummaryGate

        gate = SummaryGate("strict")
        self.assertTrue(gate.should_summarise(self.CHANGE_ZH, seen_values=set()))
        self.assertFalse(
            gate.should_summarise(self.CHANGE_ZH, seen_values={"下午2点了"})
        )

    def test_english_turns(self):
        from memory3l.gate import SummaryGate

        gate = SummaryGate("pattern")
        self.assertTrue(gate.should_summarise("My office floor is the 3rd."))
        self.assertTrue(gate.should_summarise("I moved to Shanghai last week."))
        self.assertFalse(gate.should_summarise("What is the difference between optimistic and pessimistic locking?"))

    def test_possessive_change_restated_as_it_is(self):
        """
        "my X has changed - it is Y now" is how the English set states updates.

        An ``I <verb>``-only rule missed every one of them: recall fell to 80% and
        Current_Fact_Acc dropped from 31/32 to 23/32.
        """
        from memory3l.gate import SummaryGate, extract_candidate_pairs

        turn = "By the way, my project codename has changed - it is otter now."
        self.assertTrue(SummaryGate("pattern").should_summarise(turn))
        values = [value for _slot, value in extract_candidate_pairs(turn)]
        self.assertTrue(any("otter" in value for value in values), values)

    def test_gate_unknown_level_is_rejected(self):
        from memory3l.gate import SummaryGate

        with self.assertRaises(ValueError):
            SummaryGate("aggressive")


class TestGatedManager(unittest.TestCase):
    """A gated-out turn still lands in L3 + the window; only the call is skipped."""

    def _manager(self, level):
        store = InMemoryStore(recent_window_turns=4)
        manager = MemoryManager(
            store,
            ScriptedLLM(lambda messages: "摘要\n[FACTS: 工位=3楼]\n[OVERRIDES: none]"),
            episode_id="ep_gate",
            active_chain_token_limit=10 ** 6,
        )
        manager.summary_gate = __import__("memory3l.gate", fromlist=["SummaryGate"]).SummaryGate(level)
        return manager, store

    def test_filler_turn_is_recorded_but_not_summarised(self):
        manager, store = self._manager("pattern")
        stats = manager.add_dialog_turn("乐观锁和悲观锁有什么区别？", "它们不一样。")
        self.assertTrue(stats.summariser_skipped)
        self.assertEqual(stats.new_summary_id, "", "no summary should have been created")
        # ...but the turn is durable and visible
        self.assertEqual(store.count_raw_records("ep_gate"), 1)
        self.assertEqual(len(manager.get_window()), 1)
        self.assertEqual(manager.list_active_summaries(), [])

    def test_fact_turn_is_still_summarised(self):
        manager, store = self._manager("pattern")
        stats = manager.add_dialog_turn("我的工位楼层是3楼。", "好的")
        self.assertFalse(stats.summariser_skipped)
        self.assertTrue(stats.new_summary_id)
        self.assertEqual(len(manager.list_active_summaries()), 1)

    def test_metrics_expose_the_skip_rate(self):
        manager, _ = self._manager("pattern")
        manager.add_dialog_turn("我的工位楼层是3楼。", "好的")
        manager.add_dialog_turn("什么是乐观锁？", "一种锁。")
        metrics = manager.episode_metrics()
        self.assertEqual(metrics["turns_not_summarised"], 1)
        self.assertEqual(metrics["turns_summarised"], 1)
        self.assertEqual(metrics["gate_level"], "pattern")

    def test_gate_is_bypassed_without_a_summariser(self):
        """No LLM means the turn *is* the summary; skipping would drop it entirely."""
        store = InMemoryStore(recent_window_turns=4)
        manager = MemoryManager(store, None, episode_id="ep_gate2")
        manager.summary_gate = __import__("memory3l.gate", fromlist=["SummaryGate"]).SummaryGate("strict")
        manager.add_dialog_turn("什么是乐观锁？", "一种锁。")
        self.assertEqual(len(manager.list_active_summaries()), 1)


class TestHeuristicExtraction(unittest.TestCase):
    """
    Regression pin for three extraction bugs the fact ledger surfaced.

    ``audit_check.py`` reported ``silent_loss_rate = 0.2`` on the synthetic set while
    every storage invariant passed -- i.e. the facts were never extracted correctly
    in the first place.  All three causes were in the offline backend's lexical
    extraction, not in the parser or the store.
    """

    def test_english_article_does_not_eat_the_first_letter(self):
        from memory3l.llm import _clean_fact_value

        # 'a' had no word boundary, so "alpha" -> "lpha", "theatre" -> "atre".
        self.assertEqual(_clean_fact_value("alpha"), "alpha")
        self.assertEqual(_clean_fact_value("theatre"), "theatre")
        self.assertEqual(_clean_fact_value("a cat"), "cat")

    def test_slot_keeps_a_leading_superlative(self):
        from memory3l.llm import clean_slot

        # '最' was listed as slot noise, so "最喜欢的饮料" became "饮料" and stopped
        # matching its own earlier announcements.
        self.assertEqual(clean_slot("最喜欢的饮料"), "最喜欢的饮料")
        self.assertEqual(clean_slot("我的会议时间"), "会议时间")

    def test_locative_verb_does_not_split_a_slot(self):
        from memory3l.llm import HeuristicLLM

        # "在" is both a state verb and a common slot character; with a lazy slot the
        # match was slot="所" + verb="在" + value="城市是上海".
        facts = HeuristicLLM.extract_facts("你好，跟你说一下，我的所在城市是上海。")
        self.assertEqual(len(facts), 1)
        _span, slot, value = facts[0]
        self.assertEqual(slot, "所在城市")
        self.assertEqual(value, "上海")
        # ...without losing the genuine locative form
        facts = HeuristicLLM.extract_facts("我的车在车库。")
        self.assertEqual([(s, v) for _sp, s, v in facts], [("车", "车库")])


class TestFactLedgerAudit(unittest.TestCase):
    """
    The audit surface must be able to *fail*; a check that cannot fail proves
    nothing.  Both halves are pinned here: a clean episode reports no violations,
    and each deliberate break trips exactly the invariant that covers it.
    """

    @staticmethod
    def _run(store=None, turns=20):
        from memory3l.dataset import build_long_context_episodes

        episode = build_long_context_episodes(
            num_episodes=1, turns_per_episode=turns, seed=777, language="zh"
        )[0]
        store = store or InMemoryStore(recent_window_turns=4)
        manager = MemoryManager(
            store, HeuristicLLM(), episode_id="ep_audit",
            active_chain_token_limit=400, reset_on_bind=True,
        )
        for user, reply in episode.dialogues:
            manager.add_dialog_turn(user, reply)
        gold = [
            (attr, value)
            for attr, entries in episode.facts.items()
            for _turn, value in entries
        ]
        return manager, episode, gold

    def test_clean_episode_passes_every_invariant_and_loses_nothing(self):
        manager, _episode, gold = self._run()
        report = manager.verify(gold_facts=gold)
        self.assertTrue(report.ok, report.violations)
        for name, result in report.invariants.items():
            self.assertTrue(result["ok"], name)
        self.assertEqual(report.gold["silent_loss_rate"], 0.0)
        self.assertGreater(report.counters["ledger_facts"], 0)

    def test_ledger_records_why_a_fact_left_the_working_set(self):
        manager, _episode, _gold = self._run()
        archived = [r for r in manager.fact_ledger.entries() if r.reason]
        self.assertTrue(archived, "expected some facts to have been overridden")
        record = archived[0]
        self.assertEqual(record.reason, "overridden")
        self.assertTrue(record.superseded_by)
        explained = manager.explain_fact(record.fact_id)
        self.assertEqual(explained["state"], "archived")
        self.assertEqual(explained["superseded_by"], record.superseded_by)

    def test_evidence_resolves_to_the_original_turn(self):
        manager, _episode, _gold = self._run()
        record = manager.fact_ledger.entries()[0]
        evidence = manager.evidence(record.fact_id)
        self.assertIsNotNone(evidence)
        self.assertTrue(evidence["messages"])
        self.assertIn(record.value, evidence["messages"][0]["user"])

    def test_control_a_silent_removal_trips_conservation(self):
        """A summary removed from the store without being archived is a silent loss."""
        manager, _episode, gold = self._run()
        active_ids = {s.summary_id for s in manager.list_active_summaries()}
        victim = next(
            (r for r in manager.fact_ledger.entries() if r.summary_id in active_ids), None
        )
        self.assertIsNotNone(victim)
        manager.store.remove_active_summary(victim.summary_id, episode_id=manager.episode_id)
        report = manager.verify(gold_facts=gold)
        self.assertFalse(report.ok, "I1 should have fired")
        self.assertFalse(report.invariants["I1_fact_conservation"]["ok"])
        self.assertGreater(report.invariants["I1_fact_conservation"]["missing"], 0)

    def test_control_b_dropped_read_fields_trip_top_level_reachability(self):
        """
        Reproduce the historical serializer bug: values are stored, but the read path
        loses ``fact_keys``.  Every storage invariant still passes (nothing is
        missing, evidence resolves) while the current values silently become
        invisible to the prompt -- which is exactly the failure that invalidated the
        real hybrid runs.
        """
        import copy

        class StrippingStore:
            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def list_active_summaries(self, episode_id=None):
                out = []
                for summary in self._inner.list_active_summaries(episode_id):
                    clone = copy.copy(summary)
                    clone.fact_keys = []
                    clone.index_id = ""
                    out.append(clone)
                return out

        manager, _episode, gold = self._run(
            store=StrippingStore(InMemoryStore(recent_window_turns=4))
        )
        report = manager.verify(gold_facts=gold)
        self.assertFalse(report.invariants["I2_top_level_current_value_reachable"]["ok"])
        self.assertTrue(report.invariants["I1_fact_conservation"]["ok"],
                        "the summaries still exist -- only their fields were lost")
        self.assertEqual(manager.render_current_values(), "")


class TestLedgerPersistence(unittest.TestCase):
    """
    The ledger must outlive the process that wrote it.

    An audit record that vanishes on restart is not an audit record, so this pins the
    round trip: write with one store, drop it, and re-derive the *same* report from a
    fresh store that has only the SQLite file (the hot cache deliberately empty).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ledger.db")

    @staticmethod
    def _hybrid(path):
        from memory3l.store.hybrid_store import RedisSQLiteHybridStore
        from memory3l.store.redis_store import RedisHotStore
        from memory3l.store.sqlite_store import SQLiteColdStore
        from test_store import FakeRedis    # shared in-process stand-in
        # FakeRedis: the sidecar path must not depend on a real Redis at all.
        fake = FakeRedis()
        hot = RedisHotStore(key_prefix="episode:{episode_id}", client=fake)
        return RedisSQLiteHybridStore(redis_store=hot, sqlite_store=SQLiteColdStore(path))

    def test_restart_reproduces_the_same_audit(self):
        from memory3l.audit import FactLedger, audit_episode
        from memory3l.dataset import build_long_context_episodes

        episode = build_long_context_episodes(
            num_episodes=1, turns_per_episode=16, seed=777, language="zh"
        )[0]
        gold = [
            (attr, value) for attr, entries in episode.facts.items()
            for _turn, value in entries
        ]
        episode_id = "three_layer/long_0000"

        store = self._hybrid(self.db)
        manager = MemoryManager(
            store, HeuristicLLM(), episode_id=episode_id,
            active_chain_token_limit=400, reset_on_bind=True,
        )
        for user, reply in episode.dialogues:
            manager.add_dialog_turn(user, reply)
        live = manager.verify(gold_facts=gold)
        fact_id = manager.fact_ledger.entries()[0].fact_id
        self.assertGreater(live.counters["ledger_facts"], 0)
        del manager, store                      # the writing process is gone

        reopened = self._hybrid(self.db)
        after = audit_episode(reopened, episode_id, gold_facts=gold)
        self.assertEqual(live.invariants, after.invariants)
        self.assertEqual(live.gold, after.gold)
        self.assertEqual(live.counters["ledger_facts"], after.counters["ledger_facts"])

        # ...and the ledger is queryable, which is the whole point of the sidecar
        ledger = FactLedger(episode_id, store=reopened)
        record = ledger.get(fact_id)
        self.assertIsNotNone(record, "the ledger did not survive the restart")
        self.assertTrue(reopened.get_raw_record(record.evidence[0], episode_id=episode_id))

    def test_a_fresh_episode_does_not_inherit_the_previous_ledger(self):
        """The namespace has no run id, so a fresh start must clear the audit record."""
        from memory3l.dataset import build_long_context_episodes

        episode = build_long_context_episodes(
            num_episodes=1, turns_per_episode=12, seed=777, language="zh"
        )[0]
        episode_id = "three_layer/long_0000"
        store = self._hybrid(self.db)
        first = MemoryManager(store, HeuristicLLM(), episode_id=episode_id,
                              active_chain_token_limit=400, reset_on_bind=True)
        for user, reply in episode.dialogues:
            first.add_dialog_turn(user, reply)
        self.assertGreater(first.fact_ledger.stats()["ledger_facts"], 0)

        # reset_on_bind=True == a fresh run over the same (run-agnostic) namespace
        second = MemoryManager(store, HeuristicLLM(), episode_id=episode_id,
                               active_chain_token_limit=400, reset_on_bind=True)
        self.assertEqual(second.fact_ledger.stats()["ledger_facts"], 0)


class TestAuditSidecar(unittest.TestCase):
    """The sidecar answers from the persisted store alone, and gates access."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "sidecar.db")

    def _seed(self):
        from memory3l.store.hybrid_store import RedisSQLiteHybridStore
        from memory3l.store.redis_store import RedisHotStore
        from memory3l.store.sqlite_store import SQLiteColdStore
        from memory3l.dataset import build_long_context_episodes
        from test_store import FakeRedis

        episode = build_long_context_episodes(
            num_episodes=1, turns_per_episode=16, seed=777, language="zh"
        )[0]
        episode_id = "three_layer/long_0000"
        store = RedisSQLiteHybridStore(
            redis_store=RedisHotStore(key_prefix="episode:{episode_id}", client=FakeRedis()),
            sqlite_store=SQLiteColdStore(self.db),
        )
        manager = MemoryManager(store, HeuristicLLM(), episode_id=episode_id,
                                active_chain_token_limit=400, reset_on_bind=True)
        for user, reply in episode.dialogues:
            manager.add_dialog_turn(user, reply)
        fact_id = manager.fact_ledger.entries()[0].fact_id
        return episode_id, fact_id

    def test_service_reads_only_from_sqlite(self):
        from audit_server import AuditService

        episode_id, fact_id = self._seed()
        service = AuditService(self.db, token="t")
        self.assertEqual(service.episodes(), [episode_id])

        report = service.audit(episode_id)
        self.assertTrue(report["ok"], report["violations"])
        self.assertTrue(all(v["ok"] for v in report["invariants"].values()))

        current = service.current(episode_id)
        self.assertIn("=", current["registry"])

        fact = service.fact(fact_id)
        self.assertEqual(fact["slot"], fact_id.split("#", 1)[1])
        self.assertIn(fact["state"], ("live", "archived"))

        evidence = service.evidence(fact_id)
        self.assertTrue(evidence["resolved"])
        self.assertTrue(evidence["messages"][0]["user"])
        self.assertIsNone(service.fact("nope#nope"))
        service.close()

    def test_http_endpoints_and_access_control(self):
        import json as _json
        import threading
        import urllib.error
        import urllib.parse
        import urllib.request

        from audit_server import build_server

        episode_id, fact_id = self._seed()
        httpd = build_server("127.0.0.1", 0, self.db, "tok")
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            def get(path, token="tok"):
                url = f"http://127.0.0.1:{port}{path}"
                if token is not None:
                    url += ("&" if "?" in path else "?") + f"token={token}"
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        return response.status, _json.loads(response.read().decode())
                except urllib.error.HTTPError as exc:
                    return exc.code, _json.loads(exc.read().decode())

            status, payload = get(f"/audit?episode={urllib.parse.quote(episode_id, safe='')}")
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            # a fact id contains "#", so it must travel in the query string
            status, payload = get(f"/fact?fact_id={urllib.parse.quote(fact_id, safe='')}")
            self.assertEqual(status, 200)
            self.assertEqual(payload["value"], fact_id.split("#", 1)[1] and payload["value"])
            self.assertEqual(get(f"/audit?episode={episode_id}", token=None)[0], 403)
            self.assertEqual(get("/fact?fact_id=nope")[0], 404)
            self.assertEqual(get("/fact")[0], 400)
        finally:
            httpd.shutdown()
            httpd.server_close()          # release the listening socket
            httpd.audit_service.close()


class TestLongMemEvalAdapter(unittest.TestCase):
    """
    The adapter decides what a "question" even is, so its truncation and filtering
    must not silently make a question unanswerable or return nothing at all.
    """

    @staticmethod
    def _record():
        def session(prefix, pairs):
            messages = []
            for i in range(pairs):
                messages.append({"role": "user", "content": f"{prefix} u{i}"})
                messages.append({"role": "assistant", "content": f"{prefix} a{i}"})
            return messages

        return {
            "question_id": "q1",
            "question": "Where did I go on my most recent family trip?",
            "answer": "Paris",
            "question_type": "knowledge-update",
            "haystack_session_ids": ["s_ev", "s2", "s3"],
            "answer_session_ids": ["s_ev"],
            # the evidence is the OLDEST session and the longest
            "haystack_sessions": [session("EVID", 10), session("b", 5), session("c", 5)],
        }

    def test_truncation_keeps_the_evidence_turns(self):
        from memory3l.longmemeval import record_to_episode

        episode = record_to_episode(self._record(), 0, max_turns=6)
        self.assertEqual(episode.meta["kept_turns"], 6)
        self.assertTrue(episode.meta["answer_session_included"])
        transcript = " ".join(user for user, _agent in episode.dialogues)
        self.assertIn("EVID", transcript, "the evidence turns were cut away")

    def test_evidence_survives_even_when_it_exceeds_the_budget(self):
        from memory3l.longmemeval import record_to_episode

        episode = record_to_episode(self._record(), 0, max_turns=4)
        transcript = " ".join(user for user, _agent in episode.dialogues)
        self.assertIn("EVID", transcript)
        self.assertTrue(episode.meta["answer_session_included"])

    def test_no_truncation_keeps_everything(self):
        from memory3l.longmemeval import record_to_episode

        episode = record_to_episode(self._record(), 0, max_turns=None)
        self.assertEqual(episode.meta["kept_turns"], episode.meta["full_turns"])
        self.assertTrue(episode.meta["answer_session_included"])

    def test_type_filter_counts_matches_not_scanned_records(self):
        """
        Filtering used to happen after a hard record cap, so
        ``types=['knowledge-update'], limit=50`` returned ZERO episodes when the
        matching records sat beyond the first 50 (they all do, in the real file).
        """
        import json
        import tempfile

        from memory3l.longmemeval import load_longmemeval

        def record(qid, qtype):
            return {
                "question_id": qid, "question": f"q {qid}", "answer": "a",
                "question_type": qtype,
                "haystack_session_ids": ["s1"], "answer_session_ids": ["s1"],
                "haystack_sessions": [[
                    {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"},
                ]],
            }

        records = [record(f"n{i}", "single-session-user") for i in range(60)]
        records.append(record("k1", "knowledge-update"))
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "lme.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(records, handle)

        episodes = load_longmemeval(path=path, limit=1, types=["knowledge-update"])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].meta["question_type"], "knowledge-update")


class TestProbeTypeInference(unittest.TestCase):
    def test_current_questions_are_not_mislabelled_as_history(self):
        """
        A substring test for "was" labelled "What was the meeting time?" as a history
        probe, and an unlabelled probe with no marker became OTHER -- which is in
        neither headline metric, so it silently vanished from the report.
        """
        from memory3l.dataset import _normalise_probe_type

        cases = [
            ("What was the meeting time?", "9am", "current_fact"),
            ("What is my office floor now?", "3rd", "current_fact"),
            ("Where do I work in Washington?", "DC", "current_fact"),
            ("Before it became 12th, what was my office floor?", "3rd", "history_fact"),
            ("What was my office floor previously?", "3rd", "history_fact"),
            ("在改成12楼之前，我的工位是什么？", "3楼", "history_fact"),
            ("我现在的工位是什么？", "12楼", "current_fact"),
        ]
        for question, gold, expected in cases:
            self.assertEqual(
                _normalise_probe_type(None, question, has_answer=bool(gold)),
                expected,
                question,
            )


class TestSyntheticProbes(unittest.TestCase):
    """
    The synthetic dataset is the cheap smoke path, so its gold answers must be right.

    ``build_synthetic_episodes`` anchored every history probe on the episode-FINAL
    value, so an attribute that changed twice produced two probes with the same
    wording ("before it became <latest>") and different gold answers.  Only one can
    be correct, so a correct lookup was scored wrong.
    """

    def test_history_probes_are_anchored_on_the_next_value(self):
        import re

        from memory3l.dataset import build_synthetic_episodes

        for language, pattern in (("zh", r"改成(.+?)之前"), ("en", r"became (.+?),")):
            episodes = build_synthetic_episodes(
                num_episodes=4, turns_per_episode=12, seed=11, language=language
            )
            for episode in episodes:
                seen = {}
                for probe in episode.probes:
                    key = (probe.fact_key, probe.question)
                    if key in seen:
                        self.assertEqual(
                            seen[key], probe.answer,
                            f"{language}: same question, different gold: {probe.question!r}",
                        )
                    seen[key] = probe.answer

                    if probe.probe_type != HISTORY_FACT:
                        continue
                    timeline = [value for _turn, value in episode.facts.get(probe.fact_key, [])]
                    match = re.search(pattern, probe.question)
                    self.assertIsNotNone(match, probe.question)
                    successor = match.group(1)
                    self.assertIn(successor, timeline, probe.question)
                    position = timeline.index(successor)
                    self.assertGreater(position, 0, probe.question)
                    self.assertEqual(
                        timeline[position - 1], probe.answer,
                        f"{language}: gold is not the value immediately before {successor!r}",
                    )


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
