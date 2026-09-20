"""
Unified LLM backends.

Everything the experiment talks to is a :class:`BaseLLM` with one method::

    resp = llm.generate(messages, temperature=..., json_mode=True)
    resp.text -> str

Backends shipped here:

* ``OllamaLLM``        -- local Ollama server (``/api/chat``).
* ``OpenAILLM``        -- any OpenAI-compatible endpoint (vLLM, LM Studio, ...).
* ``HeuristicLLM``     -- **no network, no model**: deterministic rule-based
  summariser / answerer used to smoke-test the whole pipeline (and to run the
  test-suite in CI).  It writes text that deliberately obeys the *exact* output
  grammar the real prompt demands, so the parsers are exercised for real.
* ``ScriptedLLM``      -- for unit tests: a queue of canned responses with call
  counting.

All backends record ``latency_ms`` and prompt/completion token estimates so the
evaluation CSV can report cost without extra plumbing.
"""

from __future__ import annotations

import abc
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import config
from .token_utils import estimate_tokens

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Response object
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    text: str
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLMError(RuntimeError):
    """Raised when a backend cannot produce an answer (network, auth, ...)."""


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #
class BaseLLM(abc.ABC):
    """Uniform interface for every backend."""

    name: str = "base"

    def __init__(self, model_name: str = "", temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0.0

    # -- public API --------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            resp = self._generate(
                list(messages),
                self.temperature if temperature is None else temperature,
                self.max_tokens if max_tokens is None else max_tokens,
                json_mode,
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend failures
            raise LLMError(f"{self.name} backend failed: {exc}") from exc
        resp.latency_ms = (time.perf_counter() - started) * 1000.0
        if not resp.prompt_tokens:
            resp.prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if not resp.completion_tokens:
            resp.completion_tokens = estimate_tokens(resp.text)
        if not resp.model:
            resp.model = self.model_name
        self.calls += 1
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
        self.total_latency_ms += resp.latency_ms
        return resp

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model_name,
            "calls": self.calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "latency_ms_total": round(self.total_latency_ms, 1),
        }

    # -- to implement ------------------------------------------------------- #
    @abc.abstractmethod
    def _generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        ...


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        host: str = None,
        timeout: float = None,
    ):
        super().__init__(
            model_name or config.MODEL_NAME,
            config.TEMPERATURE if temperature is None else temperature,
            config.MAX_TOKENS if max_tokens is None else max_tokens,
        )
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout or config.REQUEST_TIMEOUT

    def _generate(self, messages, temperature, max_tokens, json_mode) -> LLMResponse:
        import requests  # local import: keeps module import cheap

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        text = (data.get("message") or {}).get("content", "")
        return LLMResponse(
            text=text,
            model=data.get("model", self.model_name),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            raw=None,
        )


# --------------------------------------------------------------------------- #
# DeepSeek (first-class backend: OpenAI-compatible, no SDK required)
# --------------------------------------------------------------------------- #
class DeepSeekLLM(BaseLLM):
    """
    DeepSeek chat API (``https://api.deepseek.com/chat/completions``).

    Implemented over ``requests`` rather than the ``openai`` SDK so a batch run
    needs no extra dependency, and instrumented for evaluation:

    * retries transient failures (429 / 5xx / connection errors) with exponential
      backoff -- a rate limit must not cost an episode;
    * reads ``reasoning_content``: some DeepSeek models place the answer there, so
      an empty ``content`` would otherwise be scored as a wrong answer;
    * records token usage so the results CSV can report real cost.
    """

    name = "deepseek"

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        base_url: str = None,
        api_key: str = None,
        timeout: float = None,
        max_retries: int = None,
    ):
        super().__init__(
            model_name or config.DEEPSEEK_MODEL,
            config.TEMPERATURE if temperature is None else temperature,
            config.MAX_TOKENS if max_tokens is None else max_tokens,
        )
        self.base_url = (base_url or config.DEEPSEEK_BASE_URL).rstrip("/")
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
        self.retries = 0

    def _generate(self, messages, temperature, max_tokens, json_mode) -> LLMResponse:
        import requests

        if not self.api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not set; put it in .env or export it before running"
            )
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - network layer
                last_error = f"connection error: {exc}"
            else:
                if resp.status_code < 400:
                    data = resp.json()
                    message = (data.get("choices") or [{}])[0].get("message") or {}
                    text = message.get("content") or ""
                    if not text.strip():
                        # Reasoning models may place the answer in this field.
                        text = message.get("reasoning_content") or ""
                    usage = data.get("usage") or {}
                    return LLMResponse(
                        text=text,
                        model=data.get("model", self.model_name),
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                    )
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    raise LLMError(f"DeepSeek request failed ({last_error})")
            if attempt < self.max_retries:
                self.retries += 1
                delay = config.LLM_RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "DeepSeek call failed (%s); retry %d/%d in %.1fs",
                    last_error, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
        raise LLMError(
            f"DeepSeek request failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["retries"] = self.retries
        return data


# --------------------------------------------------------------------------- #
# OpenAI-compatible (self-hosted: vLLM / LM Studio / Ollama's OpenAI shim)
# --------------------------------------------------------------------------- #
class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        base_url: str = None,
        api_key: str = None,
        timeout: float = None,
    ):
        super().__init__(
            model_name or config.MODEL_NAME,
            config.TEMPERATURE if temperature is None else temperature,
            config.MAX_TOKENS if max_tokens is None else max_tokens,
        )
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or config.OPENAI_API_KEY
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self._client = None
            return None
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
        )
        return self._client

    def _generate(self, messages, temperature, max_tokens, json_mode) -> LLMResponse:
        client = self._get_client()
        if client is not None:
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                completion = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"OpenAI SDK call failed: {exc}") from exc
            choice = completion.choices[0]
            usage = getattr(completion, "usage", None)
            return LLMResponse(
                text=choice.message.content or "",
                model=getattr(completion, "model", self.model_name),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )

        # Fallback: raw HTTP so the project works with just `requests`.
        import requests

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"cannot reach {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"OpenAI HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model_name),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Offline deterministic backend (no model, no network)
# --------------------------------------------------------------------------- #
_FACT_PATTERNS = [
    # (en) "my X is Y" / "my X changed to Y" / "my X has changed - it is Y"
    re.compile(
        r"\bmy\s+([a-zA-Z][a-zA-Z0-9 _-]{1,30}?)\s+(?:is|are|was|were|became|"
        r"has changed to|have changed to|had changed to|changed to|"
        r"has changed|have changed|had changed|changed|updated to|is now|are now|now)\s*"
        r"(?:[-–—]\s*)?(?:it\s+is|it's|it\s+was)?\s*([^.;,\n]{1,60})",
        re.IGNORECASE,
    ),
    # (zh) "我的X是Y" / "我的X在Y" / "我的X改成了Y"
    re.compile(
        r"我的([\u4e00-\u9fffA-Za-z0-9 _-]{1,20}?)(?:是|为|在|变成了|改成了|改成|改到|调到了|调到|现在是|已经变成)"
        r"([^。；，\n]{1,40})"
    ),
    # (en) "I like / I prefer X"
    re.compile(r"\bI\s+(?:like|love|prefer|enjoy)\s+([^.;,\n]{1,40})", re.IGNORECASE),
    # (zh) "我喜欢X"
    re.compile(r"我喜欢([^。；，\n]{1,30})"),
]

_OPTION_RE = re.compile(r"\(([a-dA-D])\)\s*([^()\n]{1,60})")

# Question tails stripped before using a question as a fact slot.
_QUESTION_TAILS = [
    r"是什么[？?]?", r"是多少[？?]?", r"在几楼[？?]?", r"在哪儿[？?]?", r"在哪里[？?]?",
    r"几点[？?]?", r"现在[？?]?", r"呢[？?]?", r"吗[？?]?",
    r"\bnow\b", r"\bcurrently\b", r"\bthese days\b", r"\bright now\b", r"[?？]",
]

# Markers that a question asks about a superseded (historical) value.
_HISTORY_MARKERS_EN = ("before", "previously", "used to", "originally", "earlier", "prior to", "was my")
_HISTORY_MARKERS_ZH = ("原来", "以前", "之前", "原本", "早先", "当初")

# Leading function words stripped from extracted fact values.
_VALUE_LEADING = re.compile(
    r"^(?:在|是|为|到|了|is|are|was|were|the|a|an)\s*", re.IGNORECASE
)
_VALUE_TRAILING = re.compile(r"(?:了|呢|吗|的|now|please|ok|okay)[。.!！?？]*$", re.IGNORECASE)

#: Noise that gets glued onto a slot when a turn re-states a fact with an update
#: suffix ("my office floor has changed - it is 12th").  Stripped so that the
#: slot of an update matches the slot of the original announcement.
_SLOT_NOISE = re.compile(
    r"\b(?:has|have|had|is|are|was|were|been)\s+(?:changed|updated|switched|moved)"
    r"(?:\s+(?:to|into))?|(?:\bchanged\b|\bupdated\b|\bswitched\b|\bmoved\b)"
    r"|(?:\bit\b|\bnow\b|\banymore\b|\bagain\b|\bactually\b|\bjust\b|\bsorry\b)"
    r"|(?:改成|改到|调到|改为|变成|更新为|现在|已经|又)",
    re.IGNORECASE,
)


def _clean_fact_value(raw: str) -> str:
    value = str(raw or "").strip().strip("。.,，;；!！?？\"'“”")
    previous = None
    while previous != value:
        previous = value
        value = _VALUE_LEADING.sub("", value)
        value = _VALUE_TRAILING.sub("", value)
        value = value.strip().strip("。.,，;；!！?？\"'“”")
    return value.strip()


def clean_slot(raw: str) -> str:
    """Normalise a fact slot so updates match their original announcement."""
    slot = str(raw or "").strip().lower()
    previous = None
    while previous != slot:
        previous = slot
        slot = _SLOT_NOISE.sub(" ", slot)
        slot = re.sub(r"[\s\-–—_]+", " ", slot).strip()
        slot = re.sub(r"^(?:my|the|your|我|我的|最)\s*", "", slot).strip()
        slot = re.sub(r"^(?:喜欢|爱|讨厌)(?:的)?\s*", "", slot).strip()
        slot = re.sub(
            r"(?:是什么|是多少|是几|在哪儿|在哪里|在哪|在什么|叫啥|叫什么|多少钱|呢|吗)"
            r"(?:[一二三四五六七八九十百千0-9]+[层楼])?\s*$",
            "",
            slot,
        ).strip()
        slot = re.sub(r"在[一二三四五六七八九十百千0-9]+[层楼]\s*$", "", slot).strip()
        slot = re.sub(r"[\s:：,，。.]+$", "", slot).strip()
    return slot


def clean_text(text: str) -> str:
    """Trim quotes/punctuation/whitespace from an id-ish string."""
    return re.sub(r"\s+", "", str(text or "").strip().strip("\"'`，,。.；;：:"))


def _parse_fact_summary_line(line: str) -> List[tuple]:
    """
    Parse heuristic summary text of the form ``Facts: slot=value; slot=value``.

    Needed because a rendering line (``- <id> [OVERRIDES: none] (raw_ref: ..) ...``)
    does not itself match the natural-language fact patterns.
    """
    if "=" not in line:
        return []
    body = line
    marker = body.find("Facts:")
    if marker >= 0:
        body = body[marker + len("Facts:") :]
    out: List[tuple] = []
    for chunk in re.split(r"[;|]", body):
        if "=" not in chunk:
            continue
        slot, _, value = chunk.partition("=")
        slot, value = slot.strip().lower(), _clean_fact_value(value)
        if slot and value:
            out.append((chunk, slot, value))
    return out


def _strip_question_tail(text: str) -> str:
    value = str(text or "").strip()
    for pattern in _QUESTION_TAILS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:what|which|who|where|when|how many|how much|whats|what's)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:is|are|was|were|do|does|did|我的)\s*", "", value, flags=re.IGNORECASE)
    return value.strip()


def is_history_question(text: str) -> bool:
    """True when a question asks about a superseded value ('before X, what was Y')."""
    lowered = str(text or "").lower()
    if any(marker in lowered for marker in _HISTORY_MARKERS_EN):
        return True
    return any(marker in lowered for marker in _HISTORY_MARKERS_ZH)


#: History-question shapes that also name the *new* value, e.g.
#: "Before it became pizza, what was my favourite food?" -> ("favourite food", "pizza")
#: History-question shapes as ``(regex, slot_group_or_None)``.  ``slot_group`` is
#: the capture group holding the attribute; the *other* group (when present) holds
#: the value the question says the fact became.  ``None`` = single-group pattern
#: (only the attribute is named).
_HISTORY_SLOT_PATTERNS = [
    # English: "Before it became X, what was my Y?"
    (
        r"^\s*before\s+(?:it\s+)?(?:became|was|changed\s+to)\s+(.+?)\s*,\s*(?:what|which)\s+"
        r"(?:was|were|is|are)\s+(?:my|the|your)\s+(.+?)\s*[?？.]?\s*$",
        2,
    ),
    # English: "What was my Y before it became X?"
    (
        r"^\s*(?:what|which)\s+(?:was|were)\s+(?:my|the|your)\s+(.+?)\s+before\s+"
        r"(?:it\s+)?(?:became|was|changed\s+to)\s+(.+?)\s*[?？.]?\s*$",
        1,
    ),
    # English: "What was my Y originally/previously/earlier?"
    (
        r"^\s*(?:what|which)\s+(?:was|were)\s+(?:my|the|your)\s+(.+?)\s+"
        r"(?:originally|previously|earlier)\s*[?？.]?\s*$",
        1,
    ),
    # Chinese: "在改成X之前，我的Y是什么？"  (value first, slot second).
    # Note the anchor must be "X", never "在X" -- the leading 在 belongs to the
    # question frame, and including it made the anchor match nothing.
    (
        r"^\s*在?(?:改成|变成|改为|调整到|调到)\s*(.+?)\s*(?:之前|以前)[，,]?\s*我(?:的)?(.+?)"
        r"(?:是什么|是多少|是几|在哪儿|在哪里|在几楼|在几层|几点|呢|吗)?\s*[?？。]?\s*$",
        2,
    ),
    # Chinese: "我原来的Y是什么？"  (slot only)
    (
        r"^\s*我(?:原来|以前|之前|原本)的(.+?)"
        r"(?:是什么|是多少|是几|在哪儿|在哪里|在几楼|几点|呢|吗)\s*[?？。]?\s*$",
        None,
    ),
]


def parse_history_question(question: str) -> "tuple":
    """
    Return ``(slot, new_value)`` for a history question, either possibly empty.

    ``"Before it became pizza, what was my favourite food?"`` -> ``("favourite food", "pizza")``
    Used by the offline backend so it can ask the archive for the summary that
    holds *that* attribute with a *different* value.  Returns ``("", "")`` when the
    question does not follow a known history shape.
    """
    text = str(question or "").strip()
    for pattern, slot_group in _HISTORY_SLOT_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if match.re.groups == 1:
            return clean_slot(match.group(1)), ""
        slot_raw = match.group(slot_group)
        value_raw = match.group(3 - slot_group)
        return clean_slot(slot_raw), _clean_fact_value(value_raw)
    return "", ""
_STOPWORDS = set(
    "the a an of to and or in on at is are was were be been i my me you your it its "
    "for with that this these those as by from not no do does did have has had will "
    "would can could should now then than there here what which who whom whose when "
    "where why how if but so about into over after before also very just".split()
)


class HeuristicLLM(BaseLLM):
    """
    Deterministic, dependency-free stand-in for a real model.

    It implements just enough shallow semantics to drive the *whole* pipeline:

    * as a **summariser** it extracts ``slot -> value`` facts, and marks an
      override when the new turn re-states an active fact with a new value
      (this is exactly the grammar the real prompt asks the model to produce);
    * as an **agent** it looks for the answer inside the provided context using
      lexical overlap, and emits a real textual tool call when the needed id is
      only referenced in an ``[OVERRIDES: ...]`` tag.

    Runs produced with this backend are plumbing checks, **not** experimental
    results -- the CSV records ``llm_backend=heuristic`` so this is auditable.
    """

    name = "heuristic"

    def __init__(self, model_name: str = "heuristic-rule-based", **kwargs):
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 512)
        super().__init__(model_name, **kwargs)
        self._last_messages: List[Dict[str, str]] = []

    # -- fact helpers ------------------------------------------------------- #
    @staticmethod
    def extract_facts(text: str) -> List[tuple]:
        """Return ``(raw_span, slot, value)`` triples found in ``text``."""
        raw_facts: List[tuple] = []
        for pattern in _FACT_PATTERNS:
            for match in pattern.finditer(text):
                if pattern.groups == 1:  # "I like X" / "我喜欢X"
                    slot, value = "likes", match.group(1)
                else:
                    slot, value = match.group(1), match.group(2)
                raw_facts.append((match.group(0), slot, value))

        facts: List[tuple] = []
        for span, slot, value in raw_facts:
            slot = clean_slot(slot)
            value = _clean_fact_value(value)
            if value:
                facts.append((span, slot, value))
        return facts

    @staticmethod
    def question_slot(question: str) -> str:
        """
        Extract the fact slot a question asks about (if any).

        Interrogative forms ("What is my X now?", "我的X是什么？") are handled first,
        because the declarative patterns would otherwise swallow the question tail
        into the slot ('my meeting time now' != 'meeting time').
        """
        text = str(question or "").strip()
        patterns = [
            # history forms must come first: "before X, what was my Y?"
            r"^\s*before\s+(?:it\s+)?(?:became|was|changed\s+to)\s+.+?,\s*(?:what|which)\s+(?:was|were|is|are)\s+(?:my|the|your)\s+(.+?)\s*[?？.]?\s*$",
            r"^\s*(?:what|which)\s+(?:was|were)\s+(?:my|the|your)\s+(.+?)\s+before\s+(?:it\s+)?(?:became|was|changed\s+to)?\s*.+?\s*[?？.]?\s*$",
            r"^\s*(?:what|which)\s+(?:was|were)\s+(?:my|the|your)\s+(.+?)\s+(?:originally|previously|earlier|before)\s*[?？.]?\s*$",
            # present-tense forms
            r"^\s*(?:what|which)\s+(?:is|are|was|were)\s+(?:my|the|your)\s+(.+?)\s*(?:now|currently|right now|these days)?\s*[?？.]?\s*$",
            r"^\s*(?:what(?:'s| is)|whats)\s+(?:my|the|your)\s+(.+?)\s*(?:now|currently|right now|these days)?\s*[?？.]?\s*$",
            r"^\s*(?:how (?:many|much))\s+(.+?)\s+(?:do|does|did)\s+i\s+(?:have|own)\s*(?:now)?\s*[?？.]?\s*$",
            # Chinese
            r"^\s*(?:在)?(?:改成|变成|改为).+?(?:之前|以前|原来)[，,]?\s*我的(.+?)(?:是什么|是多少|是几|在哪儿|在哪里|在几楼|几点|呢|吗)?\s*[?？。]?\s*$",
            r"^\s*我(?:原来|以前|之前|原本)的(.+?)(?:是什么|是多少|是几|在哪儿|在哪里|在几楼|几点|呢|吗)?\s*[?？。]?\s*$",
            r"^\s*我(?:现在)?的(.+?)(?:是什么|是多少|是几|在哪儿|在哪里|在几楼|几点|呢|吗)?\s*[?？。]?\s*$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                slot = clean_slot(match.group(1))
                slot = re.sub(r"\s*(?:now|currently|right now|these days|现在|目前)\s*$", "", slot, flags=re.IGNORECASE)
                if slot:
                    return slot.strip()
        facts = HeuristicLLM.extract_facts(text)
        if facts:
            return facts[0][1]
        return clean_slot(_strip_question_tail(text))

    def _generate(self, messages, temperature, max_tokens, json_mode) -> LLMResponse:
        self._last_messages = list(messages)
        system = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
        user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if "SUMMARISER" in system.upper() or "摘要器" in system:
            return LLMResponse(text=self._summarise(messages))
        if "MERGE_SUMMARIES" in system.upper() or ("合并" in system and "摘要" in system):
            return LLMResponse(text=self._merge(messages))
        if "JUDGE" in system.upper() or "判分" in system:
            return LLMResponse(text=self._judge(messages))
        return LLMResponse(text=self._answer(user))

    # -- summariser --------------------------------------------------------- #
    def _summarise(self, messages) -> str:
        user_block = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        turn = _section(user_block, "NEW_TURN") or user_block
        chain = _section(user_block, "ACTIVE_CHAIN") or ""
        facts = self.extract_facts(turn)
        if facts:
            summary_text = "Facts: " + "; ".join(
                f"{slot}={value}" for _, slot, value in facts
            )
        else:
            compact = re.sub(r"\s+", " ", turn).strip()
            summary_text = (compact[:200] + "...") if len(compact) > 200 else compact

        # Override detection: same slot, different value already in the chain.
        overrides: List[str] = []
        if facts and chain:
            chain_slots: Dict[str, List[tuple]] = {}
            for line in chain.splitlines():
                m = re.match(r"\s*-\s*(\S+)\s+\[OVERRIDES:", line)
                if not m:
                    continue
                sid = m.group(1)
                line_facts = self.extract_facts(line) or _parse_fact_summary_line(line)
                for _, slot, value in line_facts:
                    chain_slots.setdefault(slot, []).append((sid, value))
            for _, slot, value in facts:
                for sid, old_value in chain_slots.get(slot, []):
                    if _normalise(old_value) != _normalise(value):
                        overrides.append(sid)
        overrides = list(dict.fromkeys(overrides))
        tag = ",".join(overrides) if overrides else "none"
        fact_line = "; ".join(f"{slot}={value}" for _, slot, value in facts) if facts else "none"
        # Mirror the real prompt grammar so the offline path exercises the same parser.
        return f"{summary_text}\n[FACTS: {fact_line}]\n[OVERRIDES: {tag}]"

    def _merge(self, messages) -> str:
        user_block = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        block = _section(user_block, "SUMMARIES") or user_block
        bullets: List[str] = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                line = re.sub(r"^-\s*\S+\s*(\[OVERRIDES:[^\]]*\])?\s*", "", line)
                line = re.sub(r"\(raw_ref:[^)]*\)", "", line)
                if line:
                    bullets.append(line.strip())
        merged = "Merged: " + " | ".join(bullets)[:600]
        keys = sorted({_parse_fact_summary_line(b)[0][1] for b in bullets if _parse_fact_summary_line(b)})
        fact_line = "; ".join(f"{k}=merged" for k in keys) if keys else "none"
        return f"{merged}\n[FACTS: {fact_line}]\n[OVERRIDES: none]"

    # -- judge -------------------------------------------------------------- #
    def _judge(self, messages) -> str:
        user_block = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        pred = _section(user_block, "PREDICTION") or ""
        gold = _section(user_block, "GOLD") or ""
        correct = _answer_matches(pred, gold)
        return json.dumps({"correct": bool(correct), "reason": "heuristic string match"})

    # -- answering agent ---------------------------------------------------- #
    def _answer(self, user_block: str) -> str:
        system = " ".join(
            m.get("content", "") for m in (self._last_messages or []) if m.get("role") == "system"
        )
        # The harness appends "Do not call any tool" once the tool budget is spent.
        tools_forbidden = "do not call any tool" in system.lower()
        question = _section(user_block, "QUESTION") or ""
        context = _section(user_block, "CONTEXT")
        tool_results = _section(user_block, "TOOL_RESULTS")
        if tool_results:
            # After a tool call the tool output is the authoritative context -- the
            # question text must not be mixed in, or the model would just echo it.
            context = tool_results
        elif context is None:
            context = user_block
        already_used_tools = "<TOOL_RESULTS>" in user_block
        options = _OPTION_RE.findall(question)

        # 1. A question that names a memory point -> exact-id tool lookup.  This
        #    is what exercises get_archived_summary / get_raw_record end to end.
        id_match = re.search(r"(\S*?/(?:s|a|m)\d{3}@[0-9a-f]{6})", question)
        if id_match and not already_used_tools and not tools_forbidden:
            return f'get_archived_summary(summary_id="{clean_text(id_match.group(1))}")'
        ref_match = re.search(r"(\S*?/raw\d+@[0-9a-f]{6})", question)
        if ref_match and not already_used_tools and not tools_forbidden:
            return f'get_raw_record(reference_id="{clean_text(ref_match.group(1))}")'

        # 2. A question about an overridden value: the current chain cannot answer
        #    it, so ask the archive (the three-layer system supports this tool).
        #    The placeholder is resolved by ToolExecutor to the newest archived
        #    summary matching the slot and differing from the value named in the
        #    question ("before it became X, what was my Y?").
        if is_history_question(question) and not already_used_tools and not tools_forbidden:
            # The value the fact *became* is the exact anchor ("before it became
            # X"): pass it as reference_value so the tool can pick the value X
            # directly superseded instead of guessing the newest old value.
            slot, anchor = parse_history_question(question)
            slot = slot or self.question_slot(question)
            return (
                f'get_predecessor_summary(fact_key="{slot}", '
                f'reference_value="{anchor}")'
            )

        # 3. The question refers to a historical value but tools are unavailable
        #    (a baseline without an archive): answer from whatever context exists.
        # 4. Otherwise answer from the provided context, preferring a fact slot
        #    match over free-text overlap.
        slot = self.question_slot(question)
        best = ""
        best_score = 0.0
        for line in context.splitlines():
            line_facts = self.extract_facts(line) or _parse_fact_summary_line(line)
            for _, fact_slot, value in line_facts:
                if slot and fact_slot and _slots_match(slot, fact_slot):
                    best, best_score = value, 1.0
                    break
                score = _overlap(question, value)
                if score > best_score:
                    best, best_score = value, score
            if best_score >= 1.0:
                break

        # Only fall back to an unparsed line when it overlaps the question almost
        # completely; otherwise a line echo would masquerade as an answer.
        if best_score < 0.6:
            for line in context.splitlines():
                score = _overlap(question, line)
                if score > best_score and score >= 0.6:
                    best, best_score = line.strip(), score

        if options and best:
            normalised_best = _normalise(best)
            for letter, option_text in options:
                normalised_option = _normalise(option_text)
                if normalised_option and (
                    normalised_option == normalised_best
                    or normalised_option in normalised_best
                    or normalised_best in normalised_option
                ):
                    return f"({letter}) {option_text.strip()}"
        if not best or best_score < 0.3:
            return "I do not have that information in my memory."
        return best


# --------------------------------------------------------------------------- #
# Scripted backend (unit tests)
# --------------------------------------------------------------------------- #
class ScriptedLLM(BaseLLM):
    """Returns canned responses in order; supports a callable for dynamic use."""

    name = "scripted"

    def __init__(self, responses: "Sequence[str] | Callable[[List[Dict[str, str]]], str]"):
        super().__init__("scripted", 0.0, 256)
        self._responses = list(responses) if not callable(responses) else responses
        self.seen_messages: List[List[Dict[str, str]]] = []

    def _generate(self, messages, temperature, max_tokens, json_mode) -> LLMResponse:
        self.seen_messages.append(messages)
        if callable(self._responses):
            return LLMResponse(text=self._responses(messages))
        if not self._responses:
            raise LLMError("ScriptedLLM exhausted")
        return LLMResponse(text=self._responses.pop(0))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_llm(
    backend: str = None,
    model_name: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> BaseLLM:
    """Instantiate a backend from config values (CLI overrides win)."""
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "deepseek":
        return DeepSeekLLM(model_name, temperature, max_tokens)
    if backend == "ollama":
        return OllamaLLM(model_name, temperature, max_tokens)
    if backend in ("openai", "openai_compatible", "vllm", "compatible"):
        return OpenAILLM(model_name, temperature, max_tokens)
    if backend in ("heuristic", "offline", "rule"):
        return HeuristicLLM(model_name or "heuristic-rule-based")
    raise ValueError(f"unknown LLM backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Text helpers shared with the parsers
# --------------------------------------------------------------------------- #
def _section(block: str, name: str) -> Optional[str]:
    """Extract ``<name> ... </name>`` section from a prompt block."""
    start = block.find(f"<{name}>")
    if start < 0:
        return None
    end = block.find(f"</{name}>", start)
    if end < 0:
        return block[start + len(name) + 2 :]
    return block[start + len(name) + 2 : end]


def _normalise(text: str) -> str:
    """Lowercase and drop punctuation/whitespace; used for fact-value comparison."""
    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    value = re.sub(r"[\s\u3000]+", "", value)
    return re.sub(r"[。，,.;；:：!！?？\"'“”‘’()（）\[\]【】]", "", value)


def _slots_match(question_slot: str, fact_slot: str) -> bool:
    """
    True when a question slot and a stored fact slot denote the same attribute.

    Deterministic rules: equal after normalisation, or one is a substring of the
    other with at least four shared characters (``meeting time`` vs
    ``my meeting time``, ``team size`` vs ``size``).  No fuzzy/vector matching.
    """
    a, b = _normalise(question_slot), _normalise(fact_slot)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and a in b:
        return True
    if len(b) >= 4 and b in a:
        return True
    return False


def _tokens(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(text or "").lower())
        if w not in _STOPWORDS
    }


def _overlap(question: str, candidate: str) -> float:
    q, c = _tokens(question), _tokens(candidate)
    if not q or not c:
        return 0.0
    return len(q & c) / float(len(q))


def _answer_matches(prediction: str, gold: str) -> bool:
    """
    Scoring used by the heuristic judge.

    Kept in sync with ``evaluation.string_match``; it is duplicated here so the
    backend has no dependency on the evaluation module (which would be a cycle).
    """
    p, g = _normalise(prediction), _normalise(gold)
    if not g or not p:
        return False
    if p == g or g in p:
        return True
    if len(p) >= 2 and p in g:
        return True
    g_body = re.sub(r"^\(?[a-d]\)?", "", g)
    p_body = re.sub(r"^\(?[a-d]\)?", "", p)
    if g_body and p_body and (g_body == p_body or g_body in p_body or p_body in g_body):
        return True
    return False
