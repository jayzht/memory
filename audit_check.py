#!/usr/bin/env python3
"""
Audit the fact ledger, and prove the audit can fail.

Runs offline (no network, no API cost) on the synthetic long set, where every fact
change is annotated, so the audit report can carry a real ``silent_loss_rate``.

    python3 audit_check.py --episodes 3 --turns 40

Two things are reported:

1. **The audit itself** -- the three invariants plus, against gold, the fraction of
   known facts that are neither in the ledger nor resolvable.
2. **Two controls that must FAIL.** A check that cannot fail proves nothing, so the
   script deliberately breaks the system in two ways and asserts the right
   invariant fires.  It exits non-zero if a control does *not* fire:

   * *Control A -- silent loss*: remove a summary from the store without archiving
     it (what a store bug or an over-eager reset looks like).  I1 must fire.
   * *Control B -- the historical serializer bug*: reproduce the read path that
     dropped ``fact_keys``/``index_id`` (this is the defect that made every real
     hybrid run's index digests empty).  I2 must fire, because the derived
     current-value registry can no longer see any value.
   * *Control C -- a dead extractor*: make the summariser extract nothing at all.
     I1-I3 must keep passing (there is nothing to lose) and only I4 may notice --
     this is the blind spot I4 exists for, and the shape of the M1 finding where
     silent_loss_rate was 0.2 with every storage invariant green.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory3l.dataset import build_long_context_episodes  # noqa: E402
from memory3l.llm import HeuristicLLM  # noqa: E402
from memory3l.memory_manager import MemoryManager  # noqa: E402
from memory3l.store.base import InMemoryStore  # noqa: E402


def gold_pairs(episode):
    """``[(slot, value), ...]`` for every fact the generator announced."""
    seen = set()
    pairs = []
    for attr, entries in episode.facts.items():
        for _turn, value in entries:
            if (attr, value) not in seen:
                seen.add((attr, value))
                pairs.append((attr, value))
    return pairs


def run_episode(episode, store=None, limit=400):
    store = store or InMemoryStore(recent_window_turns=4)
    manager = MemoryManager(
        store, HeuristicLLM(), episode_id=f"audit/{episode.episode_id}",
        active_chain_token_limit=limit, reset_on_bind=True,
    )
    for user, reply in episode.dialogues:
        manager.add_dialog_turn(user, reply)
    return manager


class StrippingStore:
    """
    A store whose *reads* drop ``fact_keys`` and ``index_id``.

    This is a faithful reproduction of the historical defect: the values were
    written correctly (SQLite had them) but the hot read path reconstructed the
    object without those two fields, so every consumer downstream -- the index
    digest, the current-value registry, the fact-safety rule -- saw nothing.
    ``copy.copy`` gives the same serialisation boundary the real bug had.
    """

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    args = parser.parse_args()

    episodes = build_long_context_episodes(
        num_episodes=args.episodes, turns_per_episode=args.turns,
        seed=args.seed, language=args.language,
    )
    print(f"数据集: {len(episodes)} episodes × {args.turns} 轮 "
          f"(seed={args.seed}, {args.language}) — 离线，零成本\n")

    # ---------------------------------------------------------------- #
    # 1) the audit itself
    # ---------------------------------------------------------------- #
    print("=" * 78)
    print("1) 审计：事实台账 + 三条不变式")
    print("=" * 78)
    total_gold = total_found = 0
    violations = 0
    for episode in episodes:
        manager = run_episode(episode)
        gold = gold_pairs(episode)
        report = manager.verify(gold_facts=gold)
        print(report.render())
        print()
        stats = report.gold
        if stats:
            total_gold += stats["gold_facts"]
            total_found += stats["captured_and_resolvable"]
        violations += len(report.violations)

    rate = 1 - total_found / total_gold if total_gold else None
    print(f"汇总: gold 事实 {total_gold} | 已捕获且可解析 {total_found} | "
          f"silent_loss_rate = {rate:.4f}" if rate is not None else "汇总: 无 gold")
    print(f"      违规数 = {violations}")

    # ---------------------------------------------------------------- #
    # 2) controls: the audit must be able to fail
    # ---------------------------------------------------------------- #
    print()
    print("=" * 78)
    print("2) 对照：故意破坏，验证检查真的会失败")
    print("=" * 78)
    failures = []

    # Control A -- silent loss (I1)
    episode = episodes[0]
    manager = run_episode(episode)
    active_ids = {s.summary_id for s in manager.list_active_summaries()}
    victim = next(
        (r for r in manager.fact_ledger.entries() if r.summary_id in active_ids), None
    )
    if victim is None:
        print("  [skip] 对照 A：没有可用的活跃事实")
    else:
        manager.store.remove_active_summary(victim.summary_id, episode_id=manager.episode_id)
        report = manager.verify(gold_facts=gold_pairs(episode))
        fired = not report.invariants["I1_fact_conservation"]["ok"]
        print(f"  [{'PASS' if fired else 'FAIL'}] 对照 A（静默删除一条摘要，不归档）")
        print(f"         期望 I1 失败 -> {'I1 失败' if fired else 'I1 仍然通过！'}")
        print(f"         {report.violations[0] if report.violations else '(无违规)'}")
        if not fired:
            failures.append("control A did not trip I1")

    # Control B -- the historical serializer bug (I2)
    stripped = StrippingStore(InMemoryStore(recent_window_turns=4))
    manager_b = run_episode(episodes[0], store=stripped)
    report_b = manager_b.verify(gold_facts=gold_pairs(episodes[0]))
    fired_b = not report_b.invariants["I2_top_level_current_value_reachable"]["ok"]
    print(f"  [{'PASS' if fired_b else 'FAIL'}] 对照 B（读路径丢 fact_keys/index_id ＝ 历史序列化 bug）")
    print(f"         期望 I2 失败 -> {'I2 失败' if fired_b else 'I2 仍然通过！'}")
    print(f"         registry = {manager_b.render_current_values()!r}")
    print(f"         {report_b.violations[0] if report_b.violations else '(无违规)'}")
    if not fired_b:
        failures.append("control B did not trip I2")

    # Control C -- a dead extractor (I4 only)
    from unittest import mock as _mock

    from memory3l.dataset import build_long_context_episodes as _build
    from memory3l.llm import HeuristicLLM as _Heur

    episode_c = _build(num_episodes=1, turns_per_episode=24, seed=777, language=args.language)[0]
    with _mock.patch.object(_Heur, "extract_facts", side_effect=lambda text: []):
        manager_c = run_episode(episode_c)
    report_c = manager_c.verify(gold_facts=gold_pairs(episode_c))
    extraction = report_c.invariants.get("I4_extraction_completeness", {})
    storage_ok = all(
        report_c.invariants[name]["ok"]
        for name in ("I1_fact_conservation", "I2_top_level_current_value_reachable",
                     "I3_provenance_resolvable")
    )
    fired_c = storage_ok and extraction.get("strict_captured") == 0 and bool(extraction.get("gaps"))
    print(f"  [{'PASS' if fired_c else 'FAIL'}] 对照 C（抽取器完全失效）")
    print(f"         期望：I1-I3 仍全部通过，只有 I4 报警 -> "
          f"{'符合预期' if fired_c else '不符合预期！'}")
    print(f"         I1-I3 全部 ok = {storage_ok} | I4 strict_captured = "
          f"{extraction.get('strict_captured')} | gaps = {len(extraction.get('gaps') or [])}")
    if not fired_c:
        failures.append("control C did not show the storage invariants' blind spot")

    # ---------------------------------------------------------------- #
    print()
    if failures:
        for item in failures:
            print(f"!! 对照未按预期失败: {item}")
        print("→ 审计面不可信，必须先修检查本身。")
        return 1
    print("结论: 审计面通过，且三个对照都按预期失败 —— 这些检查是有判别力的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
