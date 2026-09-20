"""
Global configuration for the three-layer memory agent project.

Every value can be overridden by an environment variable, so a batch experiment
can be launched without touching the source::

    STORE_BACKEND=redis_sqlite_hybrid LLM_BACKEND=openai MODEL_NAME=gpt-4o-mini \
        python evaluation.py --dataset data/synth.json

Design note
-----------
The project targets *reproducible batch evaluation*, not a demo.  Therefore all
tunables live here (or in CLI flags that write back into this module) and are
written to the result CSV as metadata, so a result file can always be traced
back to the configuration that produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------- #
# .env loading (mini-implementation, no external dependency)
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: Optional[str] = None) -> None:
    """Load a very small subset of .env syntax (KEY=VALUE, '#' comments)."""
    path = path or os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip("'\"")
                os.environ.setdefault(key, value)
    except OSError:
        # A broken .env must never break a batch run.
        pass


_load_dotenv()

# Optional project-local dependency directory.
#
# Some evaluation hosts have a read-only virtualenv, which makes `pip install`
# impossible even though extra packages are allowed.  Installing here with
#
#     pip install --target ./.pylibs redis
#
# and re-running needs no environment fiddling: this hook makes the directory
# importable.  It is purely additive -- nothing breaks when .pylibs is absent.
_PYLIB_DIR = os.path.join(_PROJECT_ROOT, ".pylibs")
if os.path.isdir(_PYLIB_DIR):
    import sys as _sys

    if _PYLIB_DIR not in _sys.path:
        _sys.path.insert(0, _PYLIB_DIR)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# --------------------------------------------------------------------------- #
# Storage backend
# --------------------------------------------------------------------------- #
# "memory"               -> InMemoryStore, zero dependencies, fast single-episode debug
# "redis_sqlite_hybrid"  -> Redis hot cache (per-episode namespace) + SQLite cold persistence
STORE_BACKEND: str = _env_str("STORE_BACKEND", "memory")

# --- Redis (hot layer only, never the source of truth) --------------------- #
REDIS_HOST: str = _env_str("REDIS_HOST", "127.0.0.1")
REDIS_PORT: int = _env_int("REDIS_PORT", 6379)
REDIS_DB: int = _env_int("REDIS_DB", 0)
REDIS_PASSWORD: Optional[str] = os.environ.get("REDIS_PASSWORD") or None
REDIS_SOCKET_TIMEOUT: float = _env_float("REDIS_SOCKET_TIMEOUT", 5.0)
# Key namespace pattern.  {episode_id} is required so that episodes can never
# collide with each other and reset() can delete exactly one episode's keys.
REDIS_KEY_PREFIX: str = _env_str("REDIS_KEY_PREFIX", "episode:{episode_id}")

# --- SQLite (cold, permanent) --------------------------------------------- #
SQLITE_PATH: str = _env_str("SQLITE_PATH", os.path.join(_PROJECT_ROOT, "exp_memory.db"))

# --------------------------------------------------------------------------- #
# Memory hyper-parameters
# --------------------------------------------------------------------------- #
# Number of most recent complete turns kept verbatim and prepended to the prompt.
RECENT_WINDOW_TURNS: int = _env_int("RECENT_WINDOW_TURNS", 4)
# Capacity-driven compression threshold for the active summary chain.
ACTIVE_CHAIN_TOKEN_LIMIT: int = _env_int("ACTIVE_CHAIN_TOKEN_LIMIT", 800)
# When compressing by capacity, merge at least / at most this many summaries.
CAPACITY_MIN_MERGE: int = _env_int("CAPACITY_MIN_MERGE", 2)
CAPACITY_MAX_MERGE: int = _env_int("CAPACITY_MAX_MERGE", 4)
# --- hierarchy (the "catalogue" design) ---------------------------------- #
# How many of the newest summaries always stay directly visible in the prompt,
# i.e. outside any index entry.
INDEX_KEEP_RECENT: int = _env_int("INDEX_KEEP_RECENT", 2)
# When the chain is over budget, file the oldest summaries into index groups of
# this size (2-6).  Filing never deletes: an index only points at its members.
INDEX_GROUP_SIZE: int = _env_int("INDEX_GROUP_SIZE", 3)
# How many *fact snippets* each index previews inline (0 = bare titles).
INDEX_PREVIEW: int = _env_int("INDEX_PREVIEW", 1)
# Only the newest N index entries carry a preview; older ones render as bare
# titles.  Keeps the index layer from growing linearly with the episode.
INDEX_PREVIEW_RECENT: int = _env_int("INDEX_PREVIEW_RECENT", 1)
# Recursive level: when there are more than this many index entries, older groups
# are folded into a "super index" (a title over titles), and so on upward.
INDEX_MAX_ENTRIES: int = _env_int("INDEX_MAX_ENTRIES", 8)
SUPER_INDEX_GROUP: int = _env_int("SUPER_INDEX_GROUP", 4)
# Hard floor: always keep at least this many unfiled summaries directly visible,
# whatever the budget says.
INDEX_MIN_VISIBLE: int = _env_int("INDEX_MIN_VISIBLE", 1)
# --- lazy expansion (the step that actually removes cost) ----------------- #
# "index"  : file groups under titles (cheap titles, but every summary still renders)
# "lazy"   : same, plus: once titles are built, summaries that are NOT referenced by
#            any title are kept by id only and no longer rendered.  The model then
#            navigates title -> summary-by-id / raw_ref on demand.
LAZY_MODE: bool = _env_bool("LAZY_MODE", True)
# How many of the newest summaries always stay rendered verbatim in the prompt.
INDEX_RENDER_RECENT: int = _env_int("INDEX_RENDER_RECENT", 2)
# Hard cap on index entries built per turn (stability guard: without it a
# mis-calibrated loop once produced 800 entries and 11k-token prompts).
INDEX_MAX_PER_TURN: int = _env_int("INDEX_MAX_PER_TURN", 2)
# --- current-value registry ------------------------------------------------ #
# A small always-rendered block ``属性=最新值; ...`` derived from the live
# summaries' fact_keys (no extra LLM call).  It exists because the *current* value
# of an attribute used to be visible only if some index title happened to still
# carry it -- and index titles truncate, go stale on override, and are recomputed
# per group.  With the registry, "what is X now?" is answerable from the top level
# of the prompt no matter how the summaries are filed.
CURRENT_VALUES_ENABLED: bool = _env_bool("CURRENT_VALUES_ENABLED", True)
# Hard cap on registry entries.  Recency wins when an episode has many slots.
CURRENT_VALUES_MAX_SLOTS: int = _env_int("CURRENT_VALUES_MAX_SLOTS", 12)
# Characters allowed for an index title *as rendered*.  The digest is built from
# up to 6 attributes plus a theme, so a 60-char cap silently discarded most of it
# (the built string was allowed 220).
INDEX_TITLE_CHARS: int = _env_int("INDEX_TITLE_CHARS", 120)
# How many summariser calls to run concurrently while ingesting a dialogue.
# 0/1 = strictly sequential.  Concurrency changes only the wall clock: override
# resolution stays sequential, so the resulting memory is identical.
# NOTE: windowed concurrency currently produces a different override graph than
# sequential ingestion (batched turns share one chain snapshot, so a turn cannot
# see the summary of the turn before it).  Sequential is the default so that
# measured results are correct; concurrency stays available for experimentation.
INGEST_CONCURRENCY: int = _env_int("INGEST_CONCURRENCY", 1)
# "index" = build a level above the summaries (keeps every summary readable);
# "merge" = the older flat behaviour (LLM rewrites several summaries into one).
CHAIN_STRATEGY: str = _env_str("CHAIN_STRATEGY", "index").lower()

# A capacity merge must reduce the rendered chain by at least this fraction,
# otherwise it is rejected and the originals stay (a "merge" that grows the chain
# is not compression; real models do occasionally produce one).
CAPACITY_MIN_GAIN: float = _env_float("CAPACITY_MIN_GAIN", 0.15)
# Hard ceiling on a single generated summary.  Real models occasionally "summarise"
# by reproducing the whole chain they were shown, which snowballs turn over turn
# (observed: a 1.8M-token summariser request).  A summary is an index entry, not a
# transcript, so it is truncated at this many tokens and the truncation is counted.
MAX_SUMMARY_OUTPUT_TOKENS: int = _env_int("MAX_SUMMARY_OUTPUT_TOKENS", 220)
# Run the O(n) "manager view == store view" integrity check every turn (debug aid;
# turns a silent inconsistency into a logged error with a stack trace).
STRICT_INTEGRITY: bool = _env_bool("STRICT_INTEGRITY", False)
# A summary whose *body* exceeds this is a sign the model copied its input.
SUMMARY_SUSPECT_CHARS: int = _env_int("SUMMARY_SUSPECT_CHARS", 4000)
# Hard ceiling on the chain content shown to the *summariser*.  Independent of the
# per-summary cap: even if one summary slips through, the next prompt stays sendable.
SUMMARIZER_MAX_INPUT_TOKENS: int = _env_int("SUMMARIZER_MAX_INPUT_TOKENS", 3000)

# Upper bound on agent <-> tool iterations per question.
MAX_TOOL_ITERATIONS: int = _env_int("MAX_TOOL_ITERATIONS", 4)
# Token budget for *raw* dialogue context.  Used by the full-context baseline
# (older turns are dropped past this point) and as the trigger for the
# MemGPT-style rolling compression.  Independent of ACTIVE_CHAIN_TOKEN_LIMIT,
# which budgets the *summary* chain only.
RAW_CONTEXT_TOKEN_LIMIT: int = _env_int("RAW_CONTEXT_TOKEN_LIMIT", 4000)

# --------------------------------------------------------------------------- #
# LLM backend
# --------------------------------------------------------------------------- #
# "deepseek" | "ollama" | "openai" | "heuristic" (offline, no network) | "scripted" (tests)
#
# Default resolution: an explicit LLM_BACKEND wins; otherwise, when a DeepSeek /
# OpenAI key is present in the environment it is used, and only as a last resort
# do we fall back to a local Ollama.  That way a machine with `.env` containing
# DEEPSEEK_API_KEY runs the real experiment with no extra flags.
def _default_backend() -> str:
    if os.environ.get("LLM_BACKEND"):
        return os.environ["LLM_BACKEND"]
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"] not in ("", "EMPTY"):
        return "openai"
    return "ollama"


LLM_BACKEND: str = _default_backend()

# --- DeepSeek (OpenAI-compatible) ----------------------------------------- #
DEEPSEEK_API_KEY: str = _env_str("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
DEEPSEEK_BASE_URL: str = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# deepseek-flash = fast/cheap (recommended for batch runs);
# deepseek-v4-pro = stronger reasoning, slower and pricier.
DEEPSEEK_MODEL: str = _env_str("DEEPSEEK_MODEL", "deepseek-flash")

if LLM_BACKEND == "deepseek":
    MODEL_NAME: str = _env_str("MODEL_NAME", DEEPSEEK_MODEL)
elif LLM_BACKEND == "openai":
    MODEL_NAME = _env_str("MODEL_NAME", "gpt-4o-mini")
else:
    MODEL_NAME = _env_str("MODEL_NAME", "qwen2.5:7b")

TEMPERATURE: float = _env_float("TEMPERATURE", 0.0)
MAX_TOKENS: int = _env_int("MAX_TOKENS", 1024)
REQUEST_TIMEOUT: float = _env_float("REQUEST_TIMEOUT", 120.0)
# Retries for transient API failures (429 / 5xx / connection resets).  A batch
# run must survive a rate limit instead of losing an episode.
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", 3)
LLM_RETRY_BACKOFF: float = _env_float("LLM_RETRY_BACKOFF", 2.0)

OLLAMA_HOST: str = _env_str("OLLAMA_HOST", "http://127.0.0.1:11434")
OPENAI_BASE_URL: str = _env_str("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY: str = _env_str("OPENAI_API_KEY", "EMPTY")

# Summary generation may use a different model from the answering agent
# (summaries are cheaper/simpler); empty string means "reuse the agent model".
SUMMARIZER_MODEL_NAME: str = _env_str("SUMMARIZER_MODEL_NAME", "")
SUMMARIZER_BACKEND: str = _env_str("SUMMARIZER_BACKEND", "")
# Judge model for LLM-as-judge scoring; empty = reuse the agent model.
JUDGE_MODEL_NAME: str = _env_str("JUDGE_MODEL_NAME", "")

# Retry the summariser once, with an explicit format reminder, when its reply
# violates the mandatory [OVERRIDES: ...] grammar.  Format failures are counted
# either way (`summarizer_failures`), so the effect stays measurable.
SUMMARIZER_FORMAT_RETRY: bool = _env_bool("SUMMARIZER_FORMAT_RETRY", True)
# Drop override ids that are not present in the active chain (True) instead of
# accepting them.  Keep True for trustworthy experiments.
STRICT_OVERRIDE_IDS: bool = _env_bool("STRICT_OVERRIDE_IDS", True)

# Let the *answering* call write the turn's summary itself (a trailing
# <MEMORY_UPDATE> block), so a turn costs one LLM call instead of two.  Only the
# interactive path (chat_debug / inspector) uses it: batch datasets already
# contain the assistant side of every turn, so ingestion there has no answer call
# to piggyback on.  See README section 5.5.
SELF_WRITE_MEMORY: bool = _env_bool("SELF_WRITE_MEMORY", True)

# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
DATASET_PATH: str = _env_str("DATASET_PATH", os.path.join(_PROJECT_ROOT, "data", "sample_episodes.json"))
OUTPUT_CSV_PATH: str = _env_str("OUTPUT_CSV_PATH", os.path.join(_PROJECT_ROOT, "results", "predictions.csv"))
OUTPUT_METRICS_PATH: str = _env_str(
    "OUTPUT_METRICS_PATH", os.path.join(_PROJECT_ROOT, "results", "metrics.csv")
)
OUTPUT_LOG_PATH: str = _env_str(
    "OUTPUT_LOG_PATH", os.path.join(_PROJECT_ROOT, "results", "episode_logs.jsonl")
)
LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO")
# Exactly one of the four systems below is selected by --system / SYSTEM.
SYSTEM_NAME: str = _env_str("SYSTEM", "three_layer")
# Judge for Current_Fact_Acc / History_Fact_Acc: "auto" | "llm" | "string"
JUDGE_BACKEND: str = _env_str("JUDGE_BACKEND", "auto")
NUM_EPISODES: int = _env_int("NUM_EPISODES", 0)  # 0 = all
SEED: int = _env_int("SEED", 20240501)


# --------------------------------------------------------------------------- #
# Config snapshot helper
# --------------------------------------------------------------------------- #
@dataclass
class ConfigSnapshot:
    """Immutable-ish dump of the config actually used by one run."""

    store_backend: str = STORE_BACKEND
    redis_host: str = REDIS_HOST
    redis_port: int = REDIS_PORT
    redis_db: int = REDIS_DB
    redis_key_prefix: str = REDIS_KEY_PREFIX
    sqlite_path: str = SQLITE_PATH
    recent_window_turns: int = RECENT_WINDOW_TURNS
    active_chain_token_limit: int = ACTIVE_CHAIN_TOKEN_LIMIT
    chain_strategy: str = CHAIN_STRATEGY
    index_keep_recent: int = INDEX_KEEP_RECENT
    index_group_size: int = INDEX_GROUP_SIZE
    index_preview_recent: int = INDEX_PREVIEW_RECENT
    index_max_entries: int = INDEX_MAX_ENTRIES
    lazy_mode: bool = LAZY_MODE
    index_render_recent: int = INDEX_RENDER_RECENT
    index_max_per_turn: int = INDEX_MAX_PER_TURN
    index_min_visible: int = INDEX_MIN_VISIBLE
    capacity_min_merge: int = CAPACITY_MIN_MERGE
    capacity_max_merge: int = CAPACITY_MAX_MERGE
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    raw_context_token_limit: int = RAW_CONTEXT_TOKEN_LIMIT
    llm_backend: str = LLM_BACKEND
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    summarizer_model_name: str = SUMMARIZER_MODEL_NAME
    judge_model_name: str = JUDGE_MODEL_NAME
    system_name: str = SYSTEM_NAME
    judge_backend: str = JUDGE_BACKEND
    strict_override_ids: bool = STRICT_OVERRIDE_IDS
    seed: int = SEED
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)
