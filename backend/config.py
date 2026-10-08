"""Process configuration, read once from the environment (and the repo-root .env).

Workspace settings saved in the UI override the retrieval defaults per question;
everything else here is fixed for the life of the process. `.env.example` documents
every variable.
"""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    return default if not value else value in {"1", "true", "yes", "on"}


# --- OpenRouter: writer, checker, rewrites, and Jev all use one key -----------
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
# DeepSeek V4.1 Flash writes answers; V4 Flash runs the cheaper grounding check.
OPENROUTER_MODEL = env_str("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
CHECKER_MODEL = env_str("CHECKER_MODEL", "deepseek/deepseek-v4-flash")
# Sent as OpenRouter's `reasoning.effort`; models that do not reason ignore it. Empty disables it.
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "low").strip().lower()
# Budgets include hidden reasoning tokens: reasoning models spend part of max_tokens
# thinking, and a budget sized only for the visible reply comes back empty.
CONDENSE_MAX_TOKENS = env_int("CONDENSE_MAX_TOKENS", 800)
WRITER_MAX_TOKENS = env_int("WRITER_MAX_TOKENS", 4000)
CHECKER_MAX_TOKENS = env_int("CHECKER_MAX_TOKENS", 2500)
CONDENSE_HISTORY_TURNS = env_int("CONDENSE_HISTORY_TURNS", 8)
WRITER_ATTEMPTS = env_int("WRITER_ATTEMPTS", 2)

# --- Jev reranker, reached through OpenRouter ----------------------------------
JEV_ENDPOINT = env_str("JEV_ENDPOINT", "https://openrouter.ai/api/v1/systemone")
JEV_MODEL = env_str("JEV_MODEL", "typesafe/jev-1.13")
JEV_RELEVANCE_THRESHOLD = env_float("JEV_RELEVANCE_THRESHOLD", 0.20)
JEV_MAX_CONCURRENCY = env_int("JEV_MAX_CONCURRENCY", 4)
# The library defaults (180 s, 8 retries) can stall a chat for minutes; fail fast and fall back.
JEV_TIMEOUT_SECONDS = env_float("JEV_TIMEOUT_SECONDS", 30)
JEV_MAX_RETRIES = env_int("JEV_MAX_RETRIES", 2)
JEV_CACHE_TTL_SECONDS = env_float("JEV_CACHE_TTL_SECONDS", 3600)
# On both eval corpora the evidence sits in the fused top 15 for every answerable question,
# and scoring 15 instead of 30-40 halves rerank time. 0 = score every fused candidate.
JEV_CANDIDATE_LIMIT = env_int("JEV_CANDIDATE_LIMIT", 15)
RETRIEVAL_RERANK_MODE = env_str("RETRIEVAL_RERANK_MODE", "jev_filter")
CROSS_ENCODER_MODEL = env_str("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Retrieval defaults ----------------------------------------------------------
TOP_K_CHILDREN = env_int("NEEDLE_TOP_K", 30)
VECTOR_SIMILARITY_THRESHOLD = env_float("NEEDLE_SIMILARITY_THRESHOLD", 0.30)
RRF_K = env_int("NEEDLE_RRF_K", 60)
MAX_PARENTS = 5
RETRY_THRESHOLD = env_float("RETRY_THRESHOLD", 0.35)
RETRY_TOP_K = env_int("RETRY_TOP_K", 40)
RELATED_LIMIT = env_int("RELATED_LIMIT", 3)
CONFIDENCE_HIGH = env_float("CONFIDENCE_HIGH", 0.60)
CONFIDENCE_MEDIUM = env_float("CONFIDENCE_MEDIUM", 0.35)
MIN_QUOTE_CHARS = env_int("MIN_QUOTE_CHARS", 12)
MIN_SUPPORTED_CHARS = env_int("MIN_SUPPORTED_CHARS", 40)
CIRCUIT_FAILURES = env_int("CIRCUIT_FAILURES", 3)
CIRCUIT_RESET_SECONDS = env_float("CIRCUIT_RESET_SECONDS", 60)

# --- Ingestion -------------------------------------------------------------------
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
CHUNKING_STRATEGIES = ("Parent-child", "Fixed window", "Index card summary")
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
OCR_ENABLED = env_bool("OCR_ENABLED", False)

# --- Index refresh gate ----------------------------------------------------------
WORKSPACE_GOLDEN = env_str(
    "NEEDLE_WORKSPACE_GOLDEN", os.path.join(os.path.dirname(__file__), "eval", "datasets", "workspace.jsonl")
)
EVAL_MIN_GOLDEN = env_int("EVAL_MIN_GOLDEN", 30)
EVAL_RECALL_DROP = env_float("EVAL_RECALL_DROP", 0.10)
EVAL_K = env_int("EVAL_K", 5)
# Jev does not change between index versions, so the gate compares versions on the free
# fused ranking unless asked otherwise.
REFRESH_GATE_USE_JEV = env_bool("REFRESH_GATE_USE_JEV", False)

# --- Cost estimates (reporting only) ----------------------------------------------
JEV_COST_PER_CANDIDATE = env_float("COST_JEV_PER_CANDIDATE", 0.00002)
WRITER_INPUT_PER_M = env_float("COST_WRITER_INPUT_PER_M", 0.15)
WRITER_OUTPUT_PER_M = env_float("COST_WRITER_OUTPUT_PER_M", 0.60)
CHECKER_INPUT_PER_M = env_float("COST_CHECKER_INPUT_PER_M", 0.028)
CHECKER_OUTPUT_PER_M = env_float("COST_CHECKER_OUTPUT_PER_M", 0.056)
