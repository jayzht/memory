#!/usr/bin/env python3
"""
Offline sweep of the summariser content gate: tokens x accuracy x gate quality.

Runs entirely locally (no network, no API cost) on the synthetic long set, where
the fact-bearing turns are known, so the gate can be scored for precision/recall
*and* its effect on Current/History accuracy can be measured on the same run.

    python3 gate_sweep.py --episodes 3 --turns 40
    python3 gate_sweep.py --episodes 10 --turns 40 --levels off,loose,pattern,strict

Why a sweep instead of a single number: the cost of a wrong gate decision is
asymmetric.  A false *skip* removes a fact from the prompt once it leaves the
sliding window (the raw record survives in L3, but only a guessed id reaches it);
a false *summarise* just costs one call.  The useful output is therefore the
frontier -- how far the skip rate can rise before accuracy moves -- not a
hand-tuned threshold.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from memory3l.agents import build_agent  # noqa: E402
from memory3l.dataset import LONG_ATTRS_ZH, LONG_ATTRS_EN, build_long_context_episodes  # noqa: E402
from memory3l.llm import HeuristicLLM  # noqa: E402
from memory3l.store.base import InMemoryStore  # noqa: E402
from memory3l.token_utils import count_message_tokens  # noqa: E402
from memory3l.tools import ToolExecutor  # noqa: E402


class CountingLLM(HeuristicLLM):
    """Offline backend that also records the prompt size of every call, by role."""

    def __init__(self):
        super().__init__()
        self.log = []

    @staticmethod
    def _role(messages):
        system = (messages[0].get("content") or "") if messages else ""
        if "记忆摘要器" in system:
            return "summariser"
        if "记忆索引器" in system:
            return "index_title"
        if "记忆合并器" in system:
            return "merge"
        if "三层记忆系统" in system:
            return "agent"
        return "other"

    def _generate(self, messages, temperature, max_tokens, json_mode):
        response = super()._generate(messages, temperature, max_tokens, json_mode)
        self.log.append((self._role(messages), count_message_tokens(messages)))
        return response


def gold_fact_turns(episodes, attrs):
    """Which turns carry a fact, according to the generator's own attribute names.

    This is ground truth for *scoring the gate*; it is deliberately not available
    to the gate itself.
    """
    pattern = re.compile("|".join(re.escape(a) for a, *_ in attrs))
    return [[bool(pattern.search(user)) for user, _agent in ep.dialogues] for ep in episodes]


def run_level(level, episodes, gold, attrs, turns_per_episode):
    config.SUMMARY_GATE = level
    llm = CountingLLM()
    store = InMemoryStore(recent_window_turns=4)
    agent = build_agent("three_layer", store, llm, llm)
    manager = agent.manager
    from evaluation import string_match

    tokens = collections.Counter()
    correct = collections.Counter()
    total = collections.Counter()
    gate = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for episode, gold_flags in zip(episodes, gold):
        manager.reset_episode(f"sweep/{episode.episode_id}", purge=True)
        agent.manager = manager
        agent.episode_id = manager.episode_id
        agent.turn_index = -1
        agent.executor = ToolExecutor(store, episode_id=manager.episode_id)

        for index, (user, reply) in enumerate(episode.dialogues):
            stats = manager.add_dialog_turn(user, reply)
            kept = not stats.summariser_skipped
            is_fact = gold_flags[index]
            gate["tp" if (is_fact and kept) else
                 "fn" if is_fact else
                 "fp" if kept else "tn"] += 1

        for probe in episode.probes:
            run = agent.answer(probe.question)
            total[probe.probe_type] += 1
            if string_match(run.answer, probe.answer):
                correct[probe.probe_type] += 1

    for role, prompt_tokens in llm.log:
        tokens[role] += prompt_tokens
    return tokens, correct, total, gate


def run_longmemeval(levels, records, path=None):
    """
    Score the gate on real conversations, using ``answer_session_ids`` as truth.

    There is no gold label for "this turn is filler" in real data, so the honest
    proxy is: *would the gate skip a turn from the session that holds the answer?*
    That is the failure mode that matters -- a skipped evidence turn removes the
    fact from the prompt.
    """
    from memory3l.longmemeval import DEFAULT_PATH, _iter_raw, sessions_to_dialogues
    from memory3l.gate import SummaryGate, extract_candidate_pairs, is_interrogative
    from memory3l import gate as G

    path = path or DEFAULT_PATH
    rows = collections.defaultdict(collections.Counter)
    seen_records = 0
    for record in _iter_raw(path, records):
        sessions = record.get("haystack_sessions") or []
        ids = record.get("haystack_session_ids") or []
        answers = set(record.get("answer_session_ids") or [])
        if not sessions or not answers:
            continue
        seen_records += 1
        turns = []
        for session_id, session in zip(ids, sessions):
            for user, reply in sessions_to_dialogues([session]):
                turns.append((f"{user}\n{reply}", session_id in answers))
        for level in levels:
            gate_state = SummaryGate(level)
            slots, values = set(), set()
            for text, is_evidence in turns:
                keep = gate_state.should_summarise(text, known_slots=slots, seen_values=values)
                for slot, value in extract_candidate_pairs(text):
                    slots.add(slot)
                    values.add(value)
                side = "ev" if is_evidence else "other"
                rows[level][(side, "kept" if keep else "skipped")] += 1
                if level == levels[0]:
                    rows["_signals"][(side, "pairs")] += 1 if extract_candidate_pairs(text) else 0
                    rows["_signals"][(side, "change")] += 1 if G._EN_CHANGE.search(text) or G._ZH_CHANGE_ANY.search(text) else 0
                    rows["_signals"][(side, "question")] += 1 if is_interrogative(text) else 0
                    rows["_signals"][(side, "n")] += 1

    print(f"数据集: LongMemEval（真实对话）| {seen_records} 条记录 | "
          f"{sum(rows['_signals'][(s, 'n')] for s in ('ev', 'other'))} 轮\n")
    print(f"{'gate':>8} | {'证据轮保留':>12} | {'证据轮跳过':>11} | {'非证据轮跳过':>13} | {'总跳过率':>9}")
    print("-" * 68)
    for level in levels:
        r = rows[level]
        ev_k, ev_s = r[("ev", "kept")], r[("ev", "skipped")]
        ot_k, ot_s = r[("other", "kept")], r[("other", "skipped")]
        ev_n, ot_n = ev_k + ev_s, ot_k + ot_s
        print(f"{level:>8} | {ev_k:>5}/{ev_n:<6} | {ev_s/max(ev_n,1):>10.1%} | "
              f"{ot_s/max(ot_n,1):>12.1%} | {100*(ev_s+ot_s)/max(ev_n+ot_n,1):>8.1f}%")
    print("\n信号存在率（判定门控是否有判别力）:")
    sig = rows["_signals"]
    for side, label in (("ev", "证据轮"), ("other", "非证据轮")):
        n = max(sig[(side, "n")], 1)
        print(f"  {label}: 候选对 {100*sig[(side,'pairs')]/n:5.1f}% | "
              f"变更动词 {100*sig[(side,'change')]/n:5.1f}% | "
              f"疑问句 {100*sig[(side,'question')]/n:5.1f}%  (n={n})")
    print("\n两列信号率越接近，说明门控越没有判别力 —— 词法门控不能用于真实数据。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "longmemeval"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--levels", default="off,loose,pattern,strict")
    parser.add_argument("--lme-records", type=int, default=40,
                        help="records to stream for --dataset longmemeval")
    parser.add_argument("--lme-path", default=None)
    args = parser.parse_args()

    levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]
    if args.dataset == "longmemeval":
        run_longmemeval(levels, args.lme_records, args.lme_path)
        return 0

    attrs = LONG_ATTRS_ZH if args.language == "zh" else LONG_ATTRS_EN
    episodes = build_long_context_episodes(
        num_episodes=args.episodes, turns_per_episode=args.turns,
        seed=args.seed, language=args.language,
    )
    gold = gold_fact_turns(episodes, attrs)
    n_turns = sum(len(e.dialogues) for e in episodes)
    n_facts = sum(sum(flags) for flags in gold)
    print(f"数据集: {len(episodes)} episodes x {args.turns} 轮 = {n_turns} 轮 "
          f"| 事实轮 {n_facts} | 填充轮 {n_turns - n_facts}  (seed={args.seed}, {args.language})")
    print(f"后端: heuristic（离线，仅用于验证机制与成本；准确率结论需真实模型）\n")

    header = (f"{'gate':>8} | {'跳过率':>6} | {'事实召回':>8} | {'假阳':>6} | "
              f"{'摘要器tok':>10} | {'索引tok':>8} | {'回答tok':>9} | {'合计':>10} | {'省':>5} | "
              f"{'Cur':>7} | {'His':>7}")
    print(header)
    print("-" * len(header))
    baseline = None
    for level in [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]:
        tokens, correct, total, gate = run_level(level, episodes, gold, attrs, args.turns)
        grand = sum(tokens.values())
        if baseline is None:
            baseline = grand
        tp, fp, fn, tn = gate["tp"], gate["fp"], gate["fn"], gate["tn"]
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        skip = (fn + tn) / max(n_turns, 1)
        cur_n, his_n = total["current_fact"], total["history_fact"]
        print(f"{level:>8} | {skip:>5.0%} | {tp:>3}/{tp + fn:<4} | {fp:>3}/{fp + tn:<3}| "
              f"{tokens['summariser']:>10,} | {tokens['index_title']:>8,} | {tokens['agent']:>9,} | "
              f"{grand:>10,} | {100 * (1 - grand / baseline):>4.0f}% | "
              f"{correct['current_fact']:>3}/{cur_n:<3}| {correct['history_fact']:>3}/{his_n:<3}|")

    print("\n说明: 『省』= 相对第一个档位的全部 prompt tokens 节省。")
    print("      『事实召回』是门控放过的事实轮比例；『假阳』是被误留的填充轮比例。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
