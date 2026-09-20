#!/usr/bin/env python3
"""
web_ui.py -- interactive memory inspector.

Ask a question and watch exactly which memory path the agent takes: which layer
each lookup touched, which candidates the retrieval tool considered, which one
won, what it rejected, and whether the value the question wanted was reachable at
all.

Why a bespoke server instead of a notebook: the interesting object is the
*retrieval trace*, which only exists while the agent answers, and it must be
examined with the live four-layer memory state next to it.

Run
---
    python web_ui.py                      # http://127.0.0.1:8000
    python web_ui.py --port 8123 --llm-backend heuristic      # offline, no API cost
    python web_ui.py --dataset data/sample_episodes.json      # start from a dataset

Stdlib HTTP only (no Flask).  ``memory3l`` does the work; this file is transport
plus one JSON contract:

    GET  /                     -> the single-page UI
    GET  /api/state            -> episode list + current four-layer memory state
    POST /api/load             -> ingest a dataset episode (resets memory first)
    POST /api/ask              -> run the agent, return answer + full trace
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from memory3l.agents import build_agent  # noqa: E402
from memory3l.dataset import Episode, build_long_context_episodes, build_synthetic_episodes, load_dataset  # noqa: E402
from memory3l.llm import build_llm  # noqa: E402
from memory3l.store import InMemoryStore, SQLiteColdStore  # noqa: E402
from memory3l.store.hybrid_store import RedisSQLiteHybridStore  # noqa: E402
from memory3l.tools import ToolExecutor, _slot_value  # noqa: E402

logger = logging.getLogger("web_ui")

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui.html")


# --------------------------------------------------------------------------- #
# Session: one episode loaded in memory, one agent bound to it
# --------------------------------------------------------------------------- #
class InspectorSession:
    """Holds the dataset, the store and the agent currently being inspected."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.episodes: List[Episode] = []
        self.episode_index = 0
        self.loaded_index = -1          # which episode is currently in memory
        self.loading_log: List[str] = []
        self.agent = None
        self.store = None
        self.summarizer = None
        self.llm = None
        self.built = False
        self.blank_mode = bool(getattr(args, "blank", False))
        self._loaded_limit: Optional[int] = None
        self.load_dataset_source(args)

    # -- setup -------------------------------------------------------------- #
    def load_dataset_source(self, args: argparse.Namespace) -> None:
        if args.blank:
            self.episodes = []
            self.source = "blank（空白会话：从零开始，逐轮自己加对话）"
            logger.info("blank session: memory starts empty")
            return
        if args.dataset:
            self.episodes = load_dataset(path=args.dataset)
            self.source = args.dataset
        elif args.dataset_name:
            self.episodes = load_dataset(dataset_name=args.dataset_name)
            self.source = args.dataset_name
        else:
            self.episodes = build_long_context_episodes(
                num_episodes=args.num_episodes,
                turns_per_episode=args.synthetic_turns,
                seed=args.seed,
                language=args.synthetic_language,
            )
            self.source = (
                f"synthetic_long(turns={args.synthetic_turns}, lang={args.synthetic_language}, seed={args.seed})"
            )
        logger.info("loaded %d episodes from %s", len(self.episodes), self.source)

    def _build_agent(self) -> None:
        backend = self.args.llm_backend or config.LLM_BACKEND
        model = self.args.model or config.MODEL_NAME
        self.llm = build_llm(backend, model, self.args.temperature)
        self.summarizer = self.llm
        if self.args.store == "memory":
            self.store = InMemoryStore(recent_window_turns=self.args.recent_window_turns)
        else:
            self.store = RedisSQLiteHybridStore(
                sqlite_path=self.args.sqlite_path or config.SQLITE_PATH,
                recent_window_turns=self.args.recent_window_turns,
            )
        self.agent = build_agent(
            self.args.system,
            self.store,
            self.llm,
            self.summarizer,
            recent_window_turns=self.args.recent_window_turns,
            active_chain_token_limit=self.args.chain_token_limit,
            raw_context_token_limit=self.args.raw_token_limit,
            max_tool_iterations=self.args.max_tool_iterations,
        )
        self.agent.debug = True          # collect the full retrieval trace
        self.built = True
        logger.info("agent ready: system=%s llm=%s/%s store=%s", self.agent.system_name, backend, model, self.store.backend_name)

    # -- episode loading ---------------------------------------------------- #
    def load_episode(self, index: int, limit: Optional[int] = None, reset: bool = True) -> Dict[str, Any]:
        with self.lock:
            if not self.built:
                self._build_agent()
            if not self.episodes:
                raise RuntimeError(
                    "空白模式没有数据集 episode：请直接逐轮加对话，"
                    "或在启动时加 --dataset/去掉 --blank 来加载数据集。"
                )
            index = max(0, min(index, len(self.episodes) - 1))
            # Re-ingesting an already-loaded episode would duplicate summaries and
            # re-bill every summariser call; only reset when the limit changed.
            if (
                self.loaded_index == index
                and self._loaded_limit == limit
                and reset
                and self.agent is not None
            ):
                state = self.state()
                state["load_log"] = ["该 episode 已经灌入，跳过重复灌入（如需重来请先 /reset 或换一个 episode）"]
                state["reused"] = True
                return state
            episode = self.episodes[index]
            scoped = f"{self.agent.system_name}/{episode.episode_id}"
            self.agent.reset_episode(scoped, resume=False, purge=True)
            self.loading_log = []

            dialogues = episode.dialogues[:limit] if limit else episode.dialogues
            stats_by_turn: List[Dict[str, Any]] = []
            for turn, (user, agent_reply) in enumerate(dialogues):
                try:
                    stats = self.agent.add_turn(user, agent_reply)
                except Exception as exc:  # noqa: BLE001 - surface, do not kill the session
                    logger.exception("turn %d failed", turn)
                    self.loading_log.append(f"turn {turn}: ERROR {exc}")
                    continue
                payload = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats or {})
                stats_by_turn.append(payload)
                if payload.get("event_triggered"):
                    self.loading_log.append(
                        f"turn {turn}: 事件覆盖 -> 新摘要覆盖了 {len(payload.get('overridden_ids') or [])} 条旧摘要"
                    )
                if payload.get("capacity_triggered"):
                    self.loading_log.append(
                        f"turn {turn}: 容量压缩 -> 合并了 {len(payload.get('capacity_merged_ids') or [])} 条摘要"
                    )
                if payload.get("invalid_override_ids"):
                    self.loading_log.append(
                        f"turn {turn}: 丢弃幻觉 override id {payload['invalid_override_ids']}"
                    )
            self.episode_index = index
            self.loaded_index = index
            self._loaded_limit = limit
            state = self.state()
            state["load_log"] = self.loading_log
            state["turn_stats"] = stats_by_turn
            return state

    # -- state snapshot ----------------------------------------------------- #
    def state(self) -> Dict[str, Any]:
        with self.lock:
            episode = self.episodes[self.episode_index] if self.episodes else None
            payload: Dict[str, Any] = {
                "source": self.source,
                "num_episodes": len(self.episodes),
                "episode_index": self.episode_index,
                "episodes": [
                    {
                        "index": i,
                        "episode_id": e.episode_id,
                        "turns": e.num_turns,
                        "probes": len(e.probes),
                    }
                    for i, e in enumerate(self.episodes)
                ],
                "loaded": self.loaded_index == self.episode_index,
                "blank_mode": self.blank_mode,
                "built": self.built,
                "config": {
                    "llm_backend": self.args.llm_backend or config.LLM_BACKEND,
                    "model": (self.llm.model_name if self.llm else (self.args.model or config.MODEL_NAME)),
                    "system": self.args.system,
                    "store": self.args.store,
                    "recent_window_turns": self.args.recent_window_turns,
                    "active_chain_token_limit": self.args.chain_token_limit,
                    "max_tool_iterations": self.args.max_tool_iterations,
                },
                "episode": None,
                "layers": {"indexes": [], "active": [], "window": [], "archive": [], "raw": []},
                "probes": [],
            }
            if episode is not None:
                payload["episode"] = {
                    "episode_id": episode.episode_id,
                    "turns": episode.num_turns,
                    "facts": {k: list(v) for k, v in episode.facts.items()},
                    "probes": [
                        {
                            "question": p.question,
                            "answer": p.answer,
                            "probe_type": p.probe_type,
                            "fact_key": p.fact_key,
                        }
                        for p in episode.probes
                    ],
                }
            if not self.built or self.agent is None or self.loaded_index != self.episode_index:
                return payload

            store = self.store
            scoped = self.agent.episode_id
            payload["scoped_episode_id"] = scoped
            payload["layers"] = {
                "indexes": [
                    {
                        "index_id": e.index_id,
                        "short_id": e.index_id.split("/")[-1],
                        "title": e.title,
                        "size": len(e.members),
                        "time_label": e.time_label(),
                        "fact_keys": e.fact_keys,
                        "members": [
                            {"summary_id": m, "short_id": m.split("/")[-1]} for m in e.members
                        ],
                        "preview": e.member_summaries[:3],
                    }
                    for e in store.list_index_entries(scoped)
                ],
                "active": [
                    {
                        "summary_id": s.summary_id,
                        "short_id": s.summary_id.split("/")[-1],
                        "text": s.text,
                        "override_ids": s.override_ids,
                        "short_override_ids": [o.split("/")[-1] for o in s.override_ids],
                        "origin": s.origin,
                        "merged_from": s.merged_from,
                        "fact_keys": s.fact_keys,
                        "raw_ref": s.raw_ref_id,
                        "seq": s.seq,
                        "index_id": s.index_id,
                        "filed": bool(s.index_id),
                    }
                    for s in store.list_active_summaries(scoped)
                ],
                "window": [
                    {
                        "reference_id": r.reference_id,
                        "short_id": r.reference_id.split("/")[-1],
                        "turn_index": r.turn_index,
                        "user_msg": r.user_msg,
                        "agent_msg": r.agent_msg,
                    }
                    for r in store.get_window(scoped)
                ],
                "archive": [
                    {
                        "summary_id": a.summary_id,
                        "short_id": a.summary_id.split("/")[-1],
                        "text": a.text,
                        "override_ids": a.override_ids,
                        "short_override_ids": [o.split("/")[-1] for o in a.override_ids],
                        "is_overridden": a.is_overridden,
                        "superseded_by": a.superseded_by,
                        "superseded_by_short": (a.superseded_by or "").split("/")[-1],
                        "archive_reason": a.archive_reason,
                        "merged_from": a.merged_from,
                        "fact_keys": a.fact_keys,
                        "raw_ref": a.raw_ref_id,
                        "seq": a.seq,
                    }
                    for a in store.list_archived_summaries(scoped)
                ],
                "raw": [
                    {
                        "reference_id": r.reference_id,
                        "short_id": r.reference_id.split("/")[-1],
                        "turn_index": r.turn_index,
                        "user_msg": r.user_msg,
                        "agent_msg": r.agent_msg,
                    }
                    for r in store.list_raw_records(scoped)
                ],
            }
            payload["tokenizer_note"] = "active chain tokens = rendered chain (ids+tags+text)"
            return payload

    # -- asking ------------------------------------------------------------- #
    def ask(self, question: str, gold: str = "") -> Dict[str, Any]:
        with self.lock:
            if not self.built or self.agent is None:
                raise RuntimeError("load an episode first")
            if self.loaded_index != self.episode_index:
                raise RuntimeError("the selected episode is not loaded yet")

            run = self.agent.answer(question)
            layers_after = self.state()["layers"]

            # Ground-truth probe: where does the expected value actually live?
            grounding = None
            if gold:
                grounding = self.ground_truth_trace(gold)
                grounding["prediction"] = run.answer
                grounding["string_match"] = _string_match(run.answer, gold)

            return {
                "question": question,
                "answer": run.answer,
                "error": run.error,
                "steps": run.steps,
                "metrics": {
                    "llm_calls": run.llm_calls,
                    "tool_calls_attempted": run.tool_calls_attempted,
                    "tool_calls_resolved": run.tool_calls_resolved,
                    "context_tokens": run.context_tokens,
                    "prompt_tokens": run.prompt_tokens,
                    "completion_tokens": run.completion_tokens,
                    "latency_ms": round(run.latency_ms, 1),
                },
                "grounding": grounding,
                "layers": layers_after,
            }

    def reset_session(self, episode_id: str = "session") -> Dict[str, Any]:
        """Start a brand-new session: no window, no active chain, no archive."""
        with self.lock:
            if not self.built:
                self._build_agent()
            scoped = f"{self.agent.system_name}/{episode_id}"
            # purge=True: a *new session* must start with zero memory, archive and
            # raw store included.  Batch evaluation keeps the default (hot only).
            self.agent.reset_episode(scoped, resume=False, purge=True)
            self.loading_log = []
            self.blank_mode = True
            self.episode_index = 0
            self.loaded_index = 0
            self._loaded_limit = None
            state = self.state()
            state["load_log"] = ["新会话：记忆已彻底清空（滑动窗口 / 活跃链 / 归档 / 原文库 均为空）"]
            return state

    def add_turn_and_reply(self, user_input: str) -> Dict[str, Any]:
        """
        One conversational turn in a blank session.

        1. the agent answers ``user_input`` with whatever memory exists *now*
           (this is what the user sees, and it may be "I do not know" early on);
        2. the completed turn is ingested by the memory system exactly as in an
           episode, so the summariser / override / compression path runs live.

        With ``SELF_WRITE_MEMORY`` step 1 and step 2 share a single LLM call: the
        reply carries a trailing ``<MEMORY_UPDATE>`` block which is stripped from
        what the user sees and applied as this turn's summary.  When the block is
        absent the manager falls back to the ordinary summariser call.
        """
        with self.lock:
            if not self.built or self.agent is None:
                raise RuntimeError("not ready")
            if self.loaded_index != self.episode_index:
                raise RuntimeError("episode not loaded")
            if config.SELF_WRITE_MEMORY and hasattr(self.agent, "answer_and_remember"):
                run, stats = self.agent.answer_and_remember(user_input)
            else:
                run = self.agent.answer(user_input)
                stats = self.agent.add_turn(user_input, run.answer)
            payload = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats or {})
            payload["selfwrite_requested"] = bool(getattr(run, "selfwrite_requested", False))
            payload["selfwrite_used"] = bool(getattr(run, "selfwrite_used", False))
            state = self.state()
            state["reply"] = run.answer
            state["turn_stats"] = payload
            state["steps"] = run.steps
            state["load_log"] = self.loading_log[-6:]
            return state

    def ground_truth_trace(self, gold: str) -> Dict[str, Any]:
        """
        Locate the expected value in all four layers.

        This is the honesty check the whole inspector exists for: if the value is
        present in the raw layer but the retrieval path never touched that record,
        the failure is a *retrieval* failure, not a storage failure.
        """
        store = self.store
        scoped = self.agent.episode_id
        needle = (gold or "").strip()
        hits = {"active": [], "archive": [], "raw": [], "window": []}

        def contains(text: str) -> bool:
            return bool(needle) and needle in (text or "")

        for s in store.list_active_summaries(scoped):
            if contains(s.text):
                hits["active"].append({"id": s.summary_id.split("/")[-1], "text": s.text})
        for a in store.list_archived_summaries(scoped):
            if contains(a.text):
                hits["archive"].append(
                    {
                        "id": a.summary_id.split("/")[-1],
                        "text": a.text,
                        "is_overridden": a.is_overridden,
                        "superseded_by": (a.superseded_by or "").split("/")[-1],
                        "raw_ref": a.raw_ref_id.split("/")[-1],
                    }
                )
        for r in store.list_raw_records(scoped):
            if contains(r.user_msg) or contains(r.agent_msg):
                hits["raw"].append(
                    {
                        "id": r.reference_id.split("/")[-1],
                        "turn_index": r.turn_index,
                        "user_msg": r.user_msg,
                        "agent_msg": r.agent_msg,
                    }
                )
        for w in store.get_window(scoped):
            if contains(w.user_msg) or contains(w.agent_msg):
                hits["window"].append({"id": w.reference_id.split("/")[-1], "turn_index": w.turn_index})

        total = sum(len(v) for v in hits.values())
        return {
            "gold": gold,
            "hits": hits,
            "total_hits": total,
            "verdict": (
                "值在整个记忆中都不存在（存储问题）"
                if total == 0
                else "值存在于记忆中——如果答错了，就是检索路径问题，不是信息丢失"
            ),
        }


def _normalise(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    value = re.sub(r"[\s\u3000]+", "", value)
    return re.sub(r"[。，,.;；:：!！?？\"'“”‘’()（）\[\]【】]", "", value)


def _string_match(prediction: str, gold: str) -> bool:
    pred, target = _normalise(prediction), _normalise(gold)
    if not target or not pred:
        return False
    return pred == target or target in pred or pred in target


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    session: InspectorSession = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):  # quieter default logging
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------------ #
    def _authorised(self, query: Dict[str, List[str]]) -> bool:
        token = getattr(self.session, "token", None)
        if not token:
            return True
        supplied = (query.get("token") or [""])[0] or self.headers.get("X-Memory-Token", "")
        return supplied == token

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200):
        self._send(status, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routes ------------------------------------------------------------- #
    def do_GET(self):  # noqa: N802
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(self.path)
            if not self._authorised(parse_qs(parsed.query)):
                self._send(403, b"forbidden: add ?token=<token>", "text/plain; charset=utf-8")
                return
            if parsed.path in ("/", "/index.html"):
                with open(UI_PATH, "rb") as handle:
                    self._send(200, handle.read(), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/api/state"):
                self._json(self.session.state())
                return
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            logger.exception("GET %s failed", self.path)
            self._json({"error": str(exc), "trace": traceback.format_exc()[-1500:]}, 500)

    def do_POST(self):  # noqa: N802
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(self.path)
            if not self._authorised(parse_qs(parsed.query)) and not self._authorised({}):
                self._send(403, b"forbidden", "text/plain; charset=utf-8")
                return
            if parsed.path == "/api/load":
                body = self._read_json()
                state = self.session.load_episode(
                    int(body.get("index", 0)),
                    limit=int(body["limit"]) if body.get("limit") else None,
                    reset=bool(body.get("reset", True)),
                )
                self._json(state)
                return
            if parsed.path == "/api/reset":
                body = self._read_json()
                self._json(self.session.reset_session(body.get("episode_id") or "session"))
                return
            if parsed.path == "/api/turn":
                body = self._read_json()
                text = (body.get("user") or "").strip()
                if not text:
                    self._json({"error": "empty turn", "hint": "请输入一句对话内容。"}, 400)
                    return
                if not self.session.built or self.session.loaded_index != self.session.episode_index:
                    self._json(
                        {"error": "no session",
                         "hint": "先点「新会话（清空记忆）」再逐轮对话。"}, 409)
                    return
                self._json(self.session.add_turn_and_reply(text))
                return
            if parsed.path == "/api/ask":
                body = self._read_json()
                question = (body.get("question") or "").strip()
                if not question:
                    self._json({"error": "empty question", "hint": "问题不能为空。"}, 400)
                    return
                if not self.session.built or self.session.loaded_index != self.session.episode_index:
                    self._json(
                        {
                            "error": "episode not loaded",
                            "hint": (
                                "记忆是空的：请先在左栏点「灌入 episode」，"
                                "等它跑完（每轮一次摘要调用）再提问。"
                            ),
                        },
                        409,
                    )
                    return
                self._json(self.session.ask(question, gold=(body.get("gold") or "").strip()))
                return
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s failed", self.path)
            self._json(
                {"error": str(exc), "hint": "服务端异常，详见 .webui.log",
                 "trace": traceback.format_exc()[-1500:]},
                500,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="interactive memory-path inspector")
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address; 0.0.0.0 makes it reachable from your host over the tunnel")
    parser.add_argument("--port", type=int, default=8010,
                        help="port (8000 is often taken by other UIs; default 8010)")
    parser.add_argument("--llm-backend", default=None, choices=["deepseek", "ollama", "openai", "heuristic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--system", default="three_layer")
    parser.add_argument("--store", default="memory", choices=["memory", "redis_sqlite_hybrid"])
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--dataset", default=None, help="dataset JSON; omit for synthetic long episodes")
    parser.add_argument("--dataset-name", default=None, help="HuggingFace dataset name (MEME)")
    parser.add_argument("--blank", action="store_true",
                        help="start with empty memory and add turns interactively "
                             "(no pre-loaded episode)")
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--synthetic-turns", type=int, default=40)
    parser.add_argument("--synthetic-language", default="zh", choices=["zh", "en"])
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--recent-window-turns", type=int, default=config.RECENT_WINDOW_TURNS)
    parser.add_argument("--chain-token-limit", type=int, default=config.ACTIVE_CHAIN_TOKEN_LIMIT)
    parser.add_argument("--raw-token-limit", type=int, default=config.RAW_CONTEXT_TOKEN_LIMIT)
    parser.add_argument("--max-tool-iterations", type=int, default=config.MAX_TOOL_ITERATIONS)
    parser.add_argument("--token", default=None,
                        help="optional access token; required as ?token=... when set. "
                             "Recommended when binding 0.0.0.0 on a shared machine. "
                             "Pass --token '' to disable.")
    parser.add_argument("--detach", action="store_true",
                        help="fork into the background, write .webui.pid, and return immediately")
    parser.add_argument("--log-level", default="INFO")
    return parser


def detach_process(args: argparse.Namespace) -> int:
    """
    Re-exec in the background with a fresh session so the server outlives the
    shell that started it (a managed job dies with the session; this does not).
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, ".webui.log")
    pid_path = os.path.join(here, ".webui.pid")
    argv = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:] if a != "--detach"]
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            argv,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            cwd=here,
            start_new_session=True,   # survives the parent shell / session
        )
    # Record the pid that is actually addressable from later shells.
    real_pid = _find_server_pid()
    with open(pid_path, "w", encoding="utf-8") as handle:
        handle.write(str(real_pid or process.pid))
    print(
        f"started in background: pid={real_pid or process.pid} (spawned {process.pid})  "
        f"log={log_path}  pidfile={pid_path}"
    )
    return 0


def _find_server_pid() -> Optional[int]:
    """Locate a running ``web_ui.py`` process via its command line."""
    import subprocess

    try:
        out = subprocess.run(
            ["pgrep", "-f", "web_ui.py --"], capture_output=True, text=True, timeout=5
        ).stdout.split()
    except Exception:  # noqa: BLE001
        return None
    own = os.getpid()
    for token in out:
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid != own:
            return pid
    return None


def main(argv: Optional[List[str]] = None) -> int:
    if "--detach" in (argv if argv is not None else sys.argv[1:]):
        pre = build_parser().parse_args(argv)
        return detach_process(pre)
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not os.path.exists(UI_PATH):
        print(f"[fatal] UI file missing: {UI_PATH}")
        return 2

    session = InspectorSession(args)
    session.token = args.token
    Handler.session = session
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    backend = args.llm_backend or config.LLM_BACKEND
    token_hint = f"?token={args.token}" if args.token else ""
    print(
        f"\n  memory inspector  ->  http://{args.host}:{args.port}/{token_hint}\n"
        f"  llm={backend}/{args.model or config.MODEL_NAME}  system={args.system}  store={args.store}\n"
        f"  episodes={len(Handler.session.episodes)} from {Handler.session.source}\n"
        f"  Ctrl-C to stop\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
