"""
Token estimation.

The active-chain token monitor must be cheap (called on every turn of every
episode) and must not require network calls.  Strategy:

1. ``tiktoken`` (cl100k_base) when installed -- exact for OpenAI-family models;
2. a deterministic character-class estimator otherwise, which is accurate to
   roughly +-10% for mixed Chinese/English text and is *monotonic*, which is all
   the capacity-compression trigger needs.

The estimator is intentionally documented so reported *Avg_Active_Chain_Tokens*
can be interpreted unambiguously.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

try:  # optional dependency
    import tiktoken  # type: ignore

    _ENCODER = tiktoken.get_encoding("cl100k_base")
    TOKENIZER_NAME = "tiktoken:cl100k_base"
except Exception:  # pragma: no cover - exercised when tiktoken is absent
    _ENCODER = None
    TOKENIZER_NAME = "heuristic:v1"

# Heuristic weights (see module docstring).
_W_CJK = 1.0        # CJK ideographs / kana / hangul
_W_LATIN = 0.30     # latin/digit characters inside words
_W_WORD = 0.75      # per whitespace-delimited word surcharge
_W_PUNCT = 0.5      # punctuation and symbols
_W_OTHER = 0.5


def _is_cjk(code: int) -> bool:
    return (
        0x3040 <= code <= 0x30FF        # kana
        or 0x3400 <= code <= 0x4DBF     # CJK ext A
        or 0x4E00 <= code <= 0x9FFF     # CJK unified
        or 0xAC00 <= code <= 0xD7AF     # hangul
        or 0xF900 <= code <= 0xFAFF     # CJK compatibility
        or 0x20000 <= code <= 0x2FA1F   # CJK ext B-F
    )


def _heuristic_estimate(text: str) -> int:
    if not text:
        return 0
    cjk = latin = punct = other = 0
    for ch in text:
        code = ord(ch)
        if _is_cjk(code):
            cjk += 1
        elif ch.isalnum():
            latin += 1
        elif ch.isspace():
            continue
        elif ch.isascii():
            punct += 1
        else:
            other += 1
    words = len(text.split())
    total = (
        cjk * _W_CJK
        + latin * _W_LATIN
        + words * _W_WORD
        + punct * _W_PUNCT
        + other * _W_OTHER
    )
    return max(1, int(round(total)))


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception:  # pragma: no cover
            pass
    return _heuristic_estimate(text)


def estimate_tokens_many(texts: Iterable[str]) -> int:
    return sum(estimate_tokens(t) for t in texts)


def count_message_tokens(messages: Sequence[dict]) -> int:
    """
    Token count of a chat-style message list, including a small per-message
    overhead (role markers / separators), which matters when comparing prompt
    sizes across systems.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message framing overhead
        total += estimate_tokens(str(msg.get("role", "")))
        total += estimate_tokens(str(msg.get("content", "")))
    return total + 2


def truncate_to_tokens(text: str, limit: int, suffix: str = " ...") -> str:
    """Hard-truncate ``text`` so that it fits into ``limit`` tokens."""
    if limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    # Binary search on characters keeps this O(log n) encoder calls.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


# --------------------------------------------------------------------------- #
# Fact keys
# --------------------------------------------------------------------------- #
#: ``slot=value`` pairs, as produced by the summariser grammar
#: ("Facts: 工位楼层=12楼; 会议时间=上午11点") and by the offline backend.
_FACT_PAIR_RE = re.compile(r"([^=;|\n]{1,40}?)\s*=\s*([^;|\n]{1,80})")


def extract_fact_keys(text: str) -> "set":
    """
    Slot names (lower-cased, trimmed) mentioned in a summary.

    Capacity compression uses this to guarantee that the *latest* summary for each
    fact survives on the active chain.  Without it, merging several older
    summaries into one lossy "miscellaneous" entry can remove the only copy of a
    fact from the chain entirely -- which is exactly what happened in the first
    long-context run (current-fact accuracy collapsed to 0.48).

    Purely lexical: no embeddings, no similarity ranking.
    """
    if not text:
        return set()
    keys = set()
    for match in _FACT_PAIR_RE.finditer(text):
        slot = match.group(1).strip().strip("，,。.；;:：-–— ")
        slot = re.sub(r"^facts?\s*[:：]\s*", "", slot, flags=re.IGNORECASE)
        if slot and len(slot) <= 30:
            keys.add(slot.lower())
    return keys
