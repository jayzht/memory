"""
Cheap, LLM-free gate: does this turn carry anything worth remembering?

Why this exists (measured on the long synthetic set, 120 turns x 3 episodes):

    summarise every turn      452,455 prompt tok   Current 23/24  History 21/21
    summarise fact turns only 240,583 prompt tok   Current 23/24  History 21/21

i.e. **46.8% of all prompt tokens** were spent summarising turns that carried no
user-fact update, and skipping them changed nothing -- because a filler turn has
no *information* to lose, whereas merging information-bearing turns into coarser
summaries (batch summarisation) drops Current_Fact_Acc hard (17/24 at batch=4).

Two signals, both deterministic and free:

* a **fact-statement pattern** ("我的X是Y", "X改成Y", "my X is Y", "X=Y", ...),
* a **known slot** mentioned in the turn, where the slot lexicon is grown from the
  agent's own summaries -- so no external attribute list is required.

The default level is ``off`` (summarise everything, the pre-existing behaviour).

.. warning::
   **Do not enable this on real conversational data.**  Swept against LongMemEval
   (60 records, 15,156 turns, evidence sessions as ground truth) the lexical
   signals have *no discriminative power at all*:

   ==========================  ==========  ==============
   signal                      evidence    non-evidence
   ==========================  ==========  ==============
   yields a (slot, value) pair      7.3%           6.8%
   explicit change verb             2.1%           2.3%
   ``X=Y``                          0.0%           2.5%
   turn ends in a question         96.2%          91.8%
   finally kept by the gate         2.1%           5.0%
   ==========================  ==========  ==============

   Real facts are stated inside long first-person narratives that end in a question
   ("I graduated with a degree in Business Administration, ... Do you have any
   advice?"), so the gate keeps 5.0% of filler but only 2.1% of the turns that
   actually hold the answer -- it is mildly *anti*-correlated with what matters.
   Skipping turns on real data needs a semantic judgement, i.e. a cheap classifier
   model, not a regex.

   On template-like/synthetic data it is exact (100% fact recall, 0 false
   positives), which is what makes it useful for cost-mechanism experiments -- see
   ``GATE_EXPERIMENT.md`` for both tables.

.. warning::
   The cost of a wrong decision is **asymmetric**. A false "skip" removes a fact
   from the prompt once it leaves the sliding window (the raw record is still in
   L3, but only reachable by a guessed id); a false "summarise" merely costs one
   call.  Levels are therefore ordered from safest to most aggressive, and the
   sweep in ``gate_sweep.py`` exists to measure where accuracy starts to break
   before any level is made the default.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set, Tuple

#: Gate levels, ordered from "summarise the most" to "summarise the least".
#: ``off`` keeps the original behaviour; the others are strictly nested.
GATE_LEVELS = ("off", "loose", "pattern", "strict")

_PUNCT = "，,。.；;:：!！?？\"'“”‘’()（）[]【】 \t\u3000"

# -- slot/value extraction -------------------------------------------------- #
# Chinese change verbs: the slot is whatever precedes them, the value follows.
_ZH_CHANGE = (
    r"改成|改为|换成|换为|变为|变成|调整为|更新为|现在是|已经是|成为了"
)
_ZH_PATTERNS = (
    # 我的<slot>是<value> / 我<slot>是<value>
    re.compile(r"(?:我的|我)(?P<slot>[^，,。.；;:：!！?？\n]{1,20}?)(?:是|变成|成了)(?P<value>[^，,。.；;:：!！?？\n]{1,30})"),
    # <slot>改成<value>了
    re.compile(r"(?P<slot>[^，,。.；;:：!！?？\n]{1,20}?)(?:" + _ZH_CHANGE + r")(?P<value>[^，,。.；;:：!！?？\n]{1,30})"),
)
_EN_PATTERNS = (
    re.compile(r"\bmy\s+(?P<slot>[A-Za-z][A-Za-z0-9 _\-]{0,28}?)\s+(?:is|was|are|were)\s+(?P<value>[^.,;!?\n]{1,40})", re.I),
    re.compile(r"\bI\s+(?:have\s+)?(?:changed|switched|moved|updated|set|bought|started|stopped|finished|joined|left)\b[^.,;!?\n]{0,40}?\bto\s+(?P<value>[^.,;!?\n]{1,40})", re.I),
    re.compile(r"\b(?:my|the)\s+(?P<slot>[A-Za-z][A-Za-z0-9 _\-]{0,28}?)\s+(?:changed|switched|moved|became)\b[^.,;!?\n]{0,20}?\b(?:to|into)\s+(?P<value>[^.,;!?\n]{1,40})", re.I),
)
_KV_PATTERN = re.compile(r"(?P<slot>[^\s=,;，；]{1,24})\s*=\s*(?P<value>[^,;，；\n]{1,40})")

#: First-person *or possessive* change verbs.  The possessive form matters: the
#: English synthetic set states updates as "my project codename has changed - it is
#: otter now", which an "I <verb>"-only rule missed (recall dropped to 80% and
#: Current_Fact_Acc with it).
_EN_CHANGE = re.compile(
    r"\b(?:I|we)\s+(?:have\s+|just\s+|recently\s+)?"
    r"(?:moved|changed|switched|updated|set|bought|started|stopped|finished|joined|left|booked|cancelled)\b"
    r"|\b(?:my|our)\s+[A-Za-z][A-Za-z0-9 _\-]{0,28}?\s+(?:has|have|had)\s+"
    r"(?:changed|updated|switched|moved|become|became)\b"
    r"|\b(?:my|our)\s+[A-Za-z][A-Za-z0-9 _\-]{0,28}?\s+(?:changed|updated|switched|moved|became)\b"
    r"|\b(?:now|currently)\s+it'?s\b",
    re.I,
)
#: "… changed, it is <value> now" / "… , now <value>" -> recover the stated value.
_EN_RESTATED_VALUE = re.compile(
    r"(?:it'?s|it is|now)\s+(?P<value>[A-Za-z0-9][^.,;!?\n]{0,40})", re.I
)
_ZH_CHANGE_ANY = re.compile(r"(?:" + _ZH_CHANGE + r")")

#: Interrogative markers.  A *question about* a fact is not a fact *update*, and
#: measuring showed the copula pattern alone fired on questions such as
#: "再提醒我一下重试预算和熔断器是**怎么**配合的。" (27% false positives).
#: Explicit change verbs are exempt: "我把地址改成了 X，对吗？" still carries an update.
_INTERROGATIVE = re.compile(
    r"\?|？|\b(?:how|what|which|why|when|where|who|whose|whom|can you|could you|"
    r"do i|did i|does|is it|are there)\b|怎么|什么|哪|多少|吗|呢|如何|为什么|哪些|是否",
    re.IGNORECASE,
)

#: Classification of a turn's lexical evidence.
CHANGE = "change"   # explicit change verb + value  (strong)
STATE = "state"     # copula "X is Y" only          (weak)
NONE = "none"


def is_interrogative(text: str) -> bool:
    return bool(_INTERROGATIVE.search(text or ""))


def _clean(text: str, limit: int = 40) -> str:
    return (text or "").strip().strip(_PUNCT)[:limit]


def extract_candidate_pairs(text: str) -> List[Tuple[str, str]]:
    """
    ``[(slot, value), ...]`` guessed from a turn; empty when nothing looks factual.

    Deliberately lexical: the gate must cost nothing, because its whole purpose is
    to avoid a paid summariser call.
    """
    if not text:
        return []
    pairs: List[Tuple[str, str]] = []
    for pattern in _ZH_PATTERNS + _EN_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            value = _clean(groups.get("value") or "")
            slot = _clean(groups.get("slot") or "", limit=24)
            if value:
                pairs.append((slot.lower(), value.lower()))
    for match in _KV_PATTERN.finditer(text):
        pairs.append((_clean(match.group("slot"), 24).lower(),
                      _clean(match.group("value")).lower()))
    # "… changed - it is otter now": the value follows a restatement, not the verb.
    if _EN_CHANGE.search(text):
        for match in _EN_RESTATED_VALUE.finditer(text):
            pairs.append(("", _clean(match.group("value")).lower()))
    # De-duplicate, preserve order.
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def classify(text: str) -> str:
    """
    Strong / weak / no lexical evidence that this turn writes a fact.

    A copula match is only *weak* evidence, so an interrogative turn is not treated
    as an update; an explicit change verb is strong enough to survive the question
    mark.
    """
    if not text:
        return NONE
    if _ZH_CHANGE_ANY.search(text) or _EN_CHANGE.search(text):
        return CHANGE
    if _KV_PATTERN.search(text):
        return CHANGE
    if extract_candidate_pairs(text):
        return STATE
    return NONE


def matches_fact_pattern(text: str) -> bool:
    """True when the turn states a fact/change, by pattern alone."""
    evidence = classify(text)
    if evidence == CHANGE:
        return True
    if evidence == STATE:
        return not is_interrogative(text)
    return False


def mentioned_known_slots(text: str, known_slots: Iterable[str]) -> bool:
    """True when the turn mentions a slot the memory already tracks."""
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(slot and slot in lowered for slot in known_slots)


class SummaryGate:
    """
    Decide whether a turn is worth a summariser call.

    ``off``      always summarise (pre-existing behaviour, the default)
    ``loose``    summarise if it states a fact *or* mentions a known slot
    ``pattern``  summarise only if it states a fact
    ``strict``   summarise only if it states a fact with a *new* value
    """

    def __init__(self, level: str = "off"):
        level = (level or "off").strip().lower()
        if level not in GATE_LEVELS:
            raise ValueError(f"unknown SUMMARY_GATE level {level!r}; use {GATE_LEVELS}")
        self.level = level
        self.skipped = 0
        self.summarised = 0

    def should_summarise(
        self,
        turn_text: str,
        known_slots: Sequence[str] = (),
        seen_values: Iterable[str] = (),
    ) -> bool:
        if self.level == "off":
            self.summarised += 1
            return True

        evidence = classify(turn_text)
        interrogative = is_interrogative(turn_text)
        known = {v for v in (seen_values or ())}

        if self.level == "loose":
            keep = matches_fact_pattern(turn_text) or (
                not interrogative and mentioned_known_slots(turn_text, known_slots)
            )
        elif self.level == "pattern":
            keep = matches_fact_pattern(turn_text)
        else:  # strict: a stated value we have not seen before
            pairs = extract_candidate_pairs(turn_text)
            novel = any(value not in known for _slot, value in pairs)
            if evidence == CHANGE:
                # A change verb with nothing parseable still reads as an update.
                keep = novel or not pairs
            else:
                keep = novel and not interrogative

        if keep:
            self.summarised += 1
        else:
            self.skipped += 1
        return keep

    def stats(self) -> dict:
        total = self.skipped + self.summarised
        return {
            "gate_level": self.level,
            "gate_skipped": self.skipped,
            "gate_summarised": self.summarised,
            "gate_skip_rate": round(self.skipped / total, 4) if total else 0.0,
        }
