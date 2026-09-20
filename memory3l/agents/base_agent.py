"""
Shared agent machinery: the tool-call loop and the agent factory.

The loop is deliberately identical for every system so that measured differences
come from the *memory mechanism*, not from the harness:

1. build messages from the system's context
2. call the LLM
3. if the reply is a textual tool call -> execute it, append
   ``[TOOL RESULTS]``, repeat (bounded by ``MAX_TOOL_ITERATIONS``)
4. otherwise treat the reply as the final answer

Tool-call accounting (``Tool_Call_Success_Rate``) is collected here:
``attempted``, ``parsed_ok``, ``resolved_ok`` and an error list.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config

from ..llm import BaseLLM, LLMError
from ..memory_manager import MemoryManager
from ..models import ToolResult
from ..prompts import build_tool_followup_messages
from ..store.base import BaseMemoryStore
from ..token_utils import count_message_tokens
from ..tools import (
    KNOWN_TOOLS,
    ToolExecutor,
    is_tool_call,
    parse_tool_calls,
    split_selfwrite_reply,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    """Everything one question produced (one row of the prediction CSV)."""

    answer: str = ""
    tool_calls_attempted: int = 0
    tool_calls_parsed: int = 0
    tool_calls_resolved: int = 0
    tool_calls_failed: int = 0
    tool_call_log: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    llm_calls: int = 0
    context_tokens: int = 0
    error: str = ""
    #: The ``<MEMORY_UPDATE>`` payload the model appended to its own final reply
    #: (empty when the model did not supply one).  Only used by the self-write path.
    memory_block: str = ""
    selfwrite_requested: bool = False
    selfwrite_used: bool = False
    #: True when a tool call had to be discarded instead of being accepted as the
    #: answer (iteration cap reached, or a system with no executor emitted one).
    tool_call_as_answer: bool = False
    #: Full per-step trace (model reply + every tool's retrieval path and
    #: candidates).  Collected only when ``agent.debug`` is on.
    steps: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def tool_success(self) -> Optional[float]:
        if self.tool_calls_attempted == 0:
            return None
        return self.tool_calls_resolved / float(self.tool_calls_attempted)


class BaseAgent(abc.ABC):
    """Interface every compared system implements."""

    #: canonical name written into result files
    system_name = "base"

    def __init__(
        self,
        store: BaseMemoryStore,
        llm: BaseLLM,
        summarizer: Optional[BaseLLM] = None,
        *,
        recent_window_turns: int = None,
        active_chain_token_limit: int = None,
        max_context_tokens: int = None,
        raw_context_token_limit: int = None,
        max_tool_iterations: int = None,
    ):
        self.store = store
        self.llm = llm
        self.summarizer = summarizer or llm
        self.recent_window_turns = (
            config.RECENT_WINDOW_TURNS if recent_window_turns is None else recent_window_turns
        )
        self.active_chain_token_limit = (
            config.ACTIVE_CHAIN_TOKEN_LIMIT
            if active_chain_token_limit is None
            else active_chain_token_limit
        )
        self.raw_context_token_limit = (
            config.RAW_CONTEXT_TOKEN_LIMIT
            if raw_context_token_limit is None
            else raw_context_token_limit
        )
        self.max_context_tokens = max_context_tokens or config.MAX_TOKENS
        self.max_tool_iterations = (
            config.MAX_TOOL_ITERATIONS if max_tool_iterations is None else max_tool_iterations
        )
        self.episode_id: Optional[str] = None
        self.turn_index = -1
        self.manager: Optional[MemoryManager] = None
        self.executor: Optional[ToolExecutor] = None
        self.debug = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def reset_episode(self, episode_id: str, *, resume: bool = False, purge: bool = False) -> None:
        ...


    @abc.abstractmethod
    def add_turn(self, user_input: str, agent_output: str) -> Any:
        ...

    def add_turns(self, turns) -> Any:
        """Ingest several turns.  Systems without batch support fall back to 1-by-1."""
        return [self.add_turn(u, a) for u, a in turns]

    def set_ingest_concurrency(self, concurrency: int) -> None:
        """Optional hint; systems that cannot ingest concurrently ignore it."""
        return None

    def finalize_episode(self) -> Dict[str, Any]:
        """Hook for systems that flush statistics at the end of an episode."""
        return {}

    # ------------------------------------------------------------------ #
    # answering
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def build_messages(self, question: str) -> List[Dict[str, str]]:
        ...

    def supports_tools(self) -> bool:
        return self.executor is not None

    def context_messages(self, question: str, selfwrite: bool = False) -> List[Dict[str, str]]:
        """
        Messages for one answering step.

        ``selfwrite`` asks the agent to append a ``<MEMORY_UPDATE>`` block to its
        own final reply.  Systems that cannot do that simply ignore the flag and
        the caller falls back to the separate summariser call.
        """
        return self.build_messages(question)

    def answer(self, question: str, selfwrite: bool = False) -> AgentRunResult:
        """Run the bounded tool-call loop and return the final answer."""
        started = time.perf_counter()
        result = AgentRunResult(selfwrite_requested=selfwrite)
        messages = self.context_messages(question, selfwrite=selfwrite)
        result.context_tokens = count_message_tokens(messages)
        final_text = ""
        for iteration in range(self.max_tool_iterations + 1):
            allow_tools = self.supports_tools() and iteration < self.max_tool_iterations
            try:
                response = self.llm.generate(messages, json_mode=False)
            except LLMError as exc:
                result.error = f"LLM error: {exc}"
                logger.warning("LLM error while answering (%s): %s", self.episode_id, exc)
                break
            result.llm_calls += 1
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            text = (response.text or "").strip()
            if self.debug:
                logger.info("[%s] agent reply (iter %d): %s", self.system_name, iteration, text[:400])

            if allow_tools and is_tool_call(text):
                # Lenient on purpose: some models emit a tool call *and* prose in one
                # reply.  Requiring the reply to contain nothing else made those calls
                # silently invisible (no tool ran, and the prose was treated as the
                # final answer).  Any parsable call is executed; the leftover prose is
                # kept in the transcript so the model can still use it.
                tool_results = self._run_tools(text, result)
                self._record_step(result, iteration, "tool_call", text, tool_results)
                messages = build_tool_followup_messages(
                    messages,
                    text,
                    "\n".join(r.render() for r in tool_results),
                    budget_note=self._remaining_budget_note(iteration),
                )
                continue

            if allow_tools and _looks_like_tool_intent(text):
                # The model wanted a tool but did not use the grammar: tell it so
                # once, and count it as a failed attempt.
                result.tool_calls_attempted += 1
                result.tool_calls_failed += 1
                result.tool_call_log.append(
                    {"name": "<unparsable>", "ok": False, "parsed": False,
                     "error": "unparsable tool intent", "raw": text[:200]}
                )
                self._record_step(result, iteration, "unparsable_tool_intent", text, [])
                messages = build_tool_followup_messages(
                    messages,
                    text,
                    "[TOOL RESULT: unparsable] your reply looked like a tool call but did not match the "
                    'grammar; use exactly get_archived_summary(summary_id="<id>") or '
                    'get_raw_record(reference_id="<ref>")',
                )
                continue

            final_text = text
            self._record_step(result, iteration, "final_answer", text, [])
            break
        else:  # pragma: no cover - loop exhausted
            final_text = ""

        # A tool call is never an answer.  Two ways one can end up here: the loop
        # hit its iteration cap while the model kept calling tools, or a system
        # without an executor emitted one anyway.  Left alone it becomes the
        # prediction, and the judge scores a call string as a wrong answer.
        if final_text and (is_tool_call(final_text) or _looks_like_tool_intent(final_text)):
            logger.debug(
                "discarding a tool call offered as the final answer: %s", final_text[:160]
            )
            result.tool_call_as_answer = True
            final_text = ""
        if not final_text:
            final_text = self._force_answer(question, messages, result)
            self._record_step(result, -1, "forced_answer", final_text, [])
            # Even the tool-free re-ask can return a call string (a model that is
            # stuck in the tool grammar).  Never hand that to the judge as the
            # prediction; an empty answer is scored as wrong, which is honest.
            if final_text and (is_tool_call(final_text) or _looks_like_tool_intent(final_text)):
                logger.debug("forced answer was still a tool call; discarding it")
                result.tool_call_as_answer = True
                final_text = ""
        if selfwrite and final_text:
            # The memory block is machine-readable payload, not part of the answer:
            # strip it before the answer reaches the user, the judge or the CSV.
            visible, block = split_selfwrite_reply(final_text)
            result.memory_block = block or ""
            result.selfwrite_used = bool(block)
            final_text = visible
        result.answer = final_text
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def _record_step(
        self,
        result: AgentRunResult,
        iteration: int,
        kind: str,
        model_text: str,
        tool_results: List[ToolResult],
    ) -> None:
        """
        Store one agent step with its full retrieval trace.

        Only populated when ``self.debug`` is set (the inspection UI turns it on),
        so batch evaluation does not pay for trace serialisation.
        """
        if not self.debug:
            return
        result.steps.append(
            {
                "iteration": iteration,
                "kind": kind,
                "model_text": model_text,
                "tools": [
                    {
                        "name": tool.name,
                        "ok": bool(tool.ok and tool.resolved),
                        "error": tool.error,
                        "path": tool.path,
                        "content": tool.content,
                        "candidates": tool.candidates,
                        "details": tool.details,
                        "latency_ms": round(tool.latency_ms, 1),
                    }
                    for tool in tool_results
                ],
            }
        )

    def _run_tools(self, text: str, result: AgentRunResult) -> List[ToolResult]:
        calls = parse_tool_calls(text)
        results: List[ToolResult] = []
        for call in calls:
            result.tool_calls_attempted += 1
            try:
                tool_result = self.executor.execute(call)
            except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the episode
                tool_result = ToolResult(
                    ok=False, name=call.name, error=str(exc), content=f"tool crashed: {exc}"
                )
            if tool_result.parsed:
                result.tool_calls_parsed += 1
            if tool_result.ok and tool_result.resolved:
                result.tool_calls_resolved += 1
            else:
                result.tool_calls_failed += 1
            result.tool_call_log.append(
                {
                    "name": call.name,
                    "args": call.args,
                    "ok": bool(tool_result.ok and tool_result.resolved),
                    "parsed": bool(tool_result.parsed),
                    "error": tool_result.error,
                    "raw": call.raw[:200],
                }
            )
            results.append(tool_result)
        return results

    def _remaining_budget_note(self, iteration: int) -> str:
        remaining = self.max_tool_iterations - iteration - 1
        if remaining <= 0:
            return "You have no tool calls left: answer the question NOW using what you have."
        return f"You may make at most {remaining} more tool call(s)."

    def _force_answer(self, question: str, messages: List[Dict[str, str]], result: AgentRunResult) -> str:
        """Budget exhausted or empty reply: ask once more, tools forbidden."""
        forced = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Do not call any tool. Answer the question now with your best knowledge "
                    f"from the memory shown above. Question: {question}"
                ),
            }
        ]
        if len(forced) > 1 and forced[0].get("role") == "system" and len(forced) > 8:
            # Keep the system prompt and the most recent turns only.
            forced = [forced[0]] + forced[-6:]
        try:
            response = self.llm.generate(forced, json_mode=False)
        except LLMError as exc:
            result.error = result.error or f"forced answer failed: {exc}"
            return ""
        result.llm_calls += 1
        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens
        return (response.text or "").strip()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
#: Every dispatchable tool, so a malformed call to *any* of them is at least
#: recognised as an attempt instead of silently becoming the final answer.
_TOOL_INTENT_HINTS = tuple(KNOWN_TOOLS)


def _looks_like_tool_intent(text: str) -> bool:
    lowered = text.lower()
    # Full-width parentheses are a realistic output for a Chinese-oriented prompt
    # and must not make an attempted call invisible.
    if "(" not in text and "（" not in text:
        return False
    return any(hint in lowered for hint in _TOOL_INTENT_HINTS)


def build_agent(
    system_name: str,
    store: BaseMemoryStore,
    llm: BaseLLM,
    summarizer: Optional[BaseLLM] = None,
    **kwargs,
) -> BaseAgent:
    """
    Factory shared by evaluation.py and chat_debug.py.

    Only the tuning options a class actually declares are forwarded.  This matters
    for correctness, not tidiness: a baseline that silently swallowed
    ``raw_context_token_limit`` through ``**kwargs`` would never compress, and the
    comparison would quietly measure nothing.
    """
    import inspect

    from . import AGENT_REGISTRY, CANONICAL_NAMES

    key = (system_name or "three_layer").strip().lower().replace("-", "_")
    if key not in AGENT_REGISTRY:
        raise ValueError(
            f"unknown system {system_name!r}; choose from {sorted(set(AGENT_REGISTRY))}"
        )
    cls = AGENT_REGISTRY[key]
    # Collect declared parameters across the MRO: subclasses often forward to
    # BaseAgent.__init__ through **kwargs, so a signature on the subclass alone
    # would under-report what is actually supported.
    supported: set = set()
    for klass in cls.__mro__:
        if "__init__" in vars(klass):
            supported.update(inspect.signature(klass.__init__).parameters)
    accepted = {name: value for name, value in kwargs.items() if name in supported and name != "self"}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        logger.debug(
            "[%s] ignoring unsupported options: %s", CANONICAL_NAMES.get(cls, key), dropped
        )
    agent = cls(store, llm, summarizer, **accepted)
    agent.system_name = CANONICAL_NAMES.get(cls, key)
    return agent
