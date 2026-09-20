#!/usr/bin/env python3
"""
analyze_results.py -- turn evaluation artifacts into report-ready tables and figures.

Reads the files written by ``evaluation.py`` and produces:

* a printed comparison table (the four headline metrics per system);
* ``<out>/summary_table.md``  -- markdown table to paste into a technical report;
* ``<out>/summary_table.csv`` -- the same numbers as data;
* optional PNG figures (matplotlib if available):
  - accuracy comparison (Current vs History)
  - memory-cost comparison (active-chain tokens)
  - compression mechanism activity (overrides / capacity merges)

Usage
-----
    python analyze_results.py --results results/main
    python analyze_results.py --predictions results/main/predictions.csv \
        --probes results/main/probe_predictions.csv --out results/main/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional

HEADLINE = [
    ("Current_Fact_Acc", "Current Fact Acc"),
    ("History_Fact_Acc", "History Fact Acc"),
    ("Avg_Active_Chain_Tokens", "Active Chain Tokens"),
    ("Tool_Call_Success_Rate", "Tool Call Success"),
]


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_from_predictions(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Recompute the headline metrics straight from the per-episode prediction rows.

    Deliberately independent of ``evaluation.aggregate`` so the report table can be
    cross-checked against what the run printed (micro-averaging over probes).
    """
    order = ["three_layer", "full_context", "memgpt_style", "naive_chain"]
    grouped: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(row.get("system", "?"), []).append(row)
    if set(grouped) <= set(order):
        grouped = OrderedDict((name, grouped[name]) for name in order if name in grouped)

    out: List[Dict[str, Any]] = []
    for system, items in grouped.items():
        cur_n = sum(int(_f(i.get("current_total")) or 0) for i in items)
        cur_ok = sum(int(_f(i.get("current_correct")) or 0) for i in items)
        his_n = sum(int(_f(i.get("history_total")) or 0) for i in items)
        his_ok = sum(int(_f(i.get("history_correct")) or 0) for i in items)
        tool_n = sum(int(_f(i.get("tool_attempted")) or 0) for i in items)
        tool_ok = sum(int(_f(i.get("tool_resolved")) or 0) for i in items)
        chain_tokens = [
            _f(i.get("active_chain_text_tokens_mean")) for i in items
        ]
        chain_tokens = [t for t in chain_tokens if t is not None]
        context_tokens = [_f(i.get("active_chain_tokens_mean")) for i in items]
        context_tokens = [t for t in context_tokens if t is not None]
        out.append(
            {
                "system": system,
                "episodes": len(items),
                "Current_Fact_Acc": (cur_ok / cur_n) if cur_n else None,
                "Current_Fact_n": cur_n,
                "History_Fact_Acc": (his_ok / his_n) if his_n else None,
                "History_Fact_n": his_n,
                "Avg_Active_Chain_Tokens": statistics.fmean(chain_tokens) if chain_tokens else None,
                "Avg_Context_Tokens": statistics.fmean(context_tokens) if context_tokens else None,
                "Tool_Call_Success_Rate": (tool_ok / tool_n) if tool_n else None,
                "chain_is_discrete": any(
                    _f(i.get("active_chain_size_final")) not in (None, 0.0) for i in items
                ),
                "Tool_Calls": tool_n,
                "overrides_events": sum(int(_f(i.get("overrides_events")) or 0) for i in items),
                "capacity_compressions": sum(
                    int(_f(i.get("capacity_compressions")) or 0) for i in items
                ),
                "Avg_Archived": statistics.fmean(
                    [_f(i.get("archived_final")) or 0.0 for i in items]
                ) if items else 0.0,
                "wall_time_s": sum(_f(i.get("wall_time_s")) or 0.0 for i in items),
            }
        )
    return out


def error_breakdown(probes: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """Count wins/losses by probe type and judge method (why did a system fail?)."""
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in probes:
        system = row.get("system", "?")
        bucket = f"{row.get('probe_type','?')}|{row.get('judge_method','?')}"
        out[system][bucket + "|correct" if row.get("correct") == "1" else bucket + "|wrong"] += 1
    return {k: dict(v) for k, v in out.items()}


def read_rows_from_sqlite(db_path: str, run_id: str) -> List[Dict[str, str]]:
    """
    Rebuild per-episode prediction rows from the SQLite checkpoint table.

    Each leg writes its episode rows to ``experiment_runs``, so this survives a
    partial re-run that overwrote the CSV.  This is also the reason the checkpoint
    doubles as an experiment record rather than only a resume marker.
    """
    import sqlite3

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows: List[Dict[str, str]] = []
    for row in connection.execute(
        "SELECT payload FROM experiment_runs WHERE run_id=? ORDER BY system_name, episode_id",
        (run_id,),
    ):
        payload = json.loads(row["payload"])
        rows.append({k: ("" if v is None else str(v)) for k, v in payload.items()})
    connection.close()
    return rows


def per_episode_table(probes: List[Dict[str, str]]) -> str:
    """Episode-level accuracy, so a single bad episode cannot hide in the mean."""
    grouped: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in probes:
        key = (row.get("system", "?"), row.get("episode_id", "?"))
        ptype = row.get("probe_type", "?")
        grouped[key][f"{ptype}_n"] += 1
        grouped[key][f"{ptype}_ok"] += 1 if row.get("correct") == "1" else 0
    lines = [
        "| System | Episode | Current | History | Memory-point |",
        "|---|---|---|---|---|",
    ]
    for (system, episode), counts in grouped.items():
        def ratio(kind: str) -> str:
            n = counts.get(f"{kind}_n", 0)
            return "n/a" if not n else f"{counts.get(f'{kind}_ok', 0)}/{n}"

        lines.append(
            f"| {system} | {episode} | {ratio('current_fact')} | {ratio('history_fact')} | "
            f"{ratio('memory_point')} |"
        )
    return "\n".join(lines)


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| System | " + " | ".join(label for _, label in HEADLINE)
        + " | Context Tokens | Overrides | Cap. merges | Archived |"
    )
    sep = "|" + "---|" * (len(HEADLINE) + 5)
    lines = [header, sep]
    for row in rows:
        cells: List[str] = []
        for key, _ in HEADLINE:
            value = row.get(key)
            if value is None:
                cells.append("n/a")
            elif key == "Avg_Active_Chain_Tokens" and row.get("chain_is_discrete") is False:
                cells.append("n/a")
            elif value <= 1.0:
                cells.append(f"{value:.3f}")
            else:
                cells.append(f"{value:.1f}")
        context = row.get("Avg_Context_Tokens")
        cells.append("n/a" if context is None else f"{context:.1f}")
        cells.append(str(row.get("overrides_events", 0)))
        cells.append(str(row.get("capacity_compressions", 0)))
        archived = row.get("Avg_Archived")
        cells.append("n/a" if archived is None else f"{archived:.1f}")
        lines.append(f"| {row['system']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_figures(rows: List[Dict[str, Any]], out_dir: str) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[info] matplotlib unavailable ({exc}); skipping figures")
        return []

    systems = [r["system"] for r in rows]
    written: List[str] = []

    # 1. accuracy
    cur = [(r["Current_Fact_Acc"] or 0.0) for r in rows]
    his = [(r["History_Fact_Acc"] or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(systems)), 4))
    index = range(len(systems))
    ax.bar([i - 0.2 for i in index], cur, width=0.4, label="Current fact")
    ax.bar([i + 0.2 for i in index], his, width=0.4, label="History fact")
    ax.set_xticks(list(index))
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Current vs. overridden-history accuracy")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2. memory cost
    tokens = [(r["Avg_Active_Chain_Tokens"] or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(systems)), 4))
    ax.bar(systems, tokens, color="#4C72B0")
    ax.set_ylabel("tokens")
    ax.set_title("Mean memory tokens per episode")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    path = os.path.join(out_dir, "memory_tokens.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 3. mechanism activity
    overrides = [r.get("overrides_events", 0) for r in rows]
    capacity = [r.get("capacity_compressions", 0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(systems)), 4))
    ax.bar([i - 0.2 for i in index], overrides, width=0.4, label="event overrides")
    ax.bar([i + 0.2 for i in index], capacity, width=0.4, label="capacity compressions")
    ax.set_xticks(list(index))
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylabel("count")
    ax.set_title("Compression mechanism activity")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "mechanisms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse evaluation results")
    parser.add_argument("--results", default="results/main", help="directory produced by evaluation.py")
    parser.add_argument("--predictions", default=None, help="override predictions.csv path")
    parser.add_argument("--probes", default=None, help="override probe_predictions.csv path")
    parser.add_argument("--out", default=None, help="output dir (default: <results>/analysis)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--run-id", default=None,
                        help="read per-episode rows from SQLite (experiment_runs) instead of predictions.csv; "
                             "required when a leg was re-run and the CSV was overwritten")
    parser.add_argument("--sqlite", default=None, help="SQLite file used by the run")
    args = parser.parse_args(argv)

    if args.run_id:
        rows = read_rows_from_sqlite(args.sqlite or "exp_memory.db", args.run_id)
        if not rows:
            print(f"[error] no rows for run_id {args.run_id!r} in {args.sqlite or 'exp_memory.db'}")
            return 2
        summary = aggregate_from_predictions(rows)
        table = markdown_table(summary)
        print(table)
        out_dir = args.out or os.path.join(args.results, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary_table.md"), "w", encoding="utf-8") as handle:
            handle.write("# Memory-system comparison\n\n" + table + "\n")
        with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        if not args.no_figures:
            for path in make_figures(summary, out_dir):
                print(f"figure -> {path}")
        print(f"\nwritten -> {out_dir}/summary_table.md, summary_table.csv")
        return 0

    predictions = args.predictions or os.path.join(args.results, "predictions.csv")
    probes_path = args.probes or os.path.join(args.results, "probe_predictions.csv")
    out_dir = args.out or os.path.join(args.results, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    rows = aggregate_from_predictions(read_csv(predictions))
    table = markdown_table(rows)
    print(table)

    if os.path.exists(probes_path):
        detailed = per_episode_table(read_csv(probes_path))
        with open(os.path.join(out_dir, "per_episode.md"), "w", encoding="utf-8") as handle:
            handle.write("# Per-episode accuracy\n\n" + detailed + "\n")

    with open(os.path.join(out_dir, "summary_table.md"), "w", encoding="utf-8") as handle:
        handle.write("# Memory-system comparison\n\n" + table + "\n")
    with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if os.path.exists(probes_path):
        probes = read_csv(probes_path)
        breakdown = error_breakdown(probes)
        with open(os.path.join(out_dir, "error_breakdown.json"), "w", encoding="utf-8") as handle:
            json.dump(breakdown, handle, ensure_ascii=False, indent=2)
        print("\nFailure breakdown (probe_type|judge_method|outcome):")
        for system, buckets in breakdown.items():
            wrong = {k.split("|correct")[0]: v for k, v in buckets.items() if k.endswith("|wrong")}
            if wrong:
                print(f"  {system}: " + ", ".join(f"{k}={v}" for k, v in sorted(wrong.items())))

    if not args.no_figures:
        figures = make_figures(rows, out_dir)
        for path in figures:
            print(f"figure -> {path}")

    print(f"\nwritten -> {out_dir}/summary_table.md, summary_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
