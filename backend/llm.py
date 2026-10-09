"""OpenRouter chat client: guarded calls with retries and a circuit breaker, query rewrites, and spend tracking."""

import logging
import os
import threading
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import httpx

from config import (
    OPENROUTER_KEY_URL,
    CHECKER_INPUT_PER_M,
    CHECKER_MODEL,
    CHECKER_OUTPUT_PER_M,
    CIRCUIT_FAILURES,
    CIRCUIT_RESET_SECONDS,
    CONDENSE_HISTORY_TURNS,
    CONDENSE_MAX_TOKENS,
    OPENROUTER_CHAT_URL,
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
    WRITER_ATTEMPTS,
    WRITER_INPUT_PER_M,
    WRITER_OUTPUT_PER_M,
)
from pipeline_logic import (
    CircuitBreaker,
    InfraError,
    NeedleError,
    classify_http_infra_error,
)

log = logging.getLogger("needle")
writer_breaker = CircuitBreaker(CIRCUIT_FAILURES, CIRCUIT_RESET_SECONDS)


_cost_ledger: Dict[str, float] = {
    "jev": 0.0,
    "writer": 0.0,
    "checker": 0.0,
    "rewriter": 0.0,
    "golden_build": 0.0,
    "stt": 0.0,
    "other": 0.0,
}


_cost_ledger_lock = threading.Lock()


def record_cost(purpose: str, amount: float) -> None:
    with _cost_ledger_lock:
        bucket = purpose if purpose in _cost_ledger else "other"
        _cost_ledger[bucket] = round(float(_cost_ledger.get(bucket, 0.0)) + float(amount), 6)


def cost_ledger_snapshot() -> Dict[str, float]:
    with _cost_ledger_lock:
        total = sum(_cost_ledger.values())
        return {**{key: round(value, 6) for key, value in _cost_ledger.items()}, "total": round(total, 6)}


def openrouter_remaining_budget() -> Optional[float]:
    """Best-effort remaining USD on the OpenRouter key; None if unknown or unlimited."""
    try:
        response = httpx.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {openrouter_key()}"}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        remaining = (payload.get("data") or payload).get("limit_remaining")
        return float(remaining) if remaining is not None else None
    except (httpx.HTTPError, NeedleError, ValueError, AttributeError):
        return None


def openrouter_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise NeedleError("Set OPENROUTER_API_KEY in .env. The answer and checker models run on OpenRouter.")
    return api_key


def _one_line_query(text: str, fallback: str) -> str:
    return " ".join((text or "").split())[:500] or fallback


def condense_query(history: List[Dict[str, Any]], question: str, *, return_usage: bool = False):
    """Rewrite a follow-up into a standalone search query. Caller skips this when history is empty."""
    turns = history[-CONDENSE_HISTORY_TURNS:]
    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {(turn.get('content') or '').strip()}"
        for turn in turns
        if (turn.get("content") or "").strip()
    )
    prompt = (
        "Rewrite the latest question as one standalone search query. "
        "Keep the names and limits from the conversation. "
        "Return only the query.\n\n"
        f"{transcript}\n\nLatest question:\n{question}"
    )
    rewritten, usage = complete(
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=CONDENSE_MAX_TOKENS,
        cost_purpose="rewriter",
        return_usage=True,
    )
    text = _one_line_query(rewritten, question)
    return (text, usage) if return_usage else text


def complete(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    model: Optional[str] = None,
    return_usage: bool = False,
    cost_purpose: Optional[str] = None,
):
    openrouter_key()  # a missing key is a configuration error: say so instead of retrying
    if not writer_breaker.closed():
        raise NeedleError("The answer model is temporarily unavailable.")
    delay = 0.4
    last_error: Optional[Exception] = None
    for attempt in range(max(1, WRITER_ATTEMPTS)):
        try:
            result = _openrouter_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                return_usage=return_usage,
                cost_purpose=cost_purpose,
            )
            writer_breaker.success()
            return result
        except InfraError:
            writer_breaker.failure()
            raise
        except TokenBudgetExhausted as exc:
            # Not an outage: give the model room to finish thinking and try again.
            last_error = exc
            max_tokens *= 2
            continue
        except NeedleError as exc:
            last_error = exc
            writer_breaker.failure()
            if attempt + 1 < WRITER_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
    raise NeedleError("The answer model on OpenRouter did not return a draft.") from last_error


def rewrite_search_query(question: str, *, return_usage: bool = False):
    rewritten, usage = complete(
        [{
            "role": "user",
            "content": (
                "Rewrite this search query with different words and the same meaning. "
                f"Return only the query.\n\n{question}"
            ),
        }],
        temperature=0,
        max_tokens=CONDENSE_MAX_TOKENS,
        cost_purpose="rewriter",
        return_usage=True,
    )
    text = _one_line_query(rewritten, question)
    return (text, usage) if return_usage else text


def _usage_cost(usage: Dict[str, Any], *, model: str) -> float:
    prompt = float(usage.get("prompt_tokens") or 0)
    completion = float(usage.get("completion_tokens") or 0)
    if model == CHECKER_MODEL:
        return (prompt * CHECKER_INPUT_PER_M + completion * CHECKER_OUTPUT_PER_M) / 1_000_000
    return (prompt * WRITER_INPUT_PER_M + completion * WRITER_OUTPUT_PER_M) / 1_000_000


class TokenBudgetExhausted(NeedleError):
    """The model stopped at max_tokens before writing any visible text (usually all reasoning)."""


def _openrouter_chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    model: Optional[str] = None,
    return_usage: bool = False,
    cost_purpose: Optional[str] = None,
):
    chosen = model or OPENROUTER_MODEL
    try:
        response = httpx.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {openrouter_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": chosen,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **({"reasoning": {"effort": OPENROUTER_REASONING_EFFORT}} if OPENROUTER_REASONING_EFFORT else {}),
            },
            timeout=120,
        )
    except httpx.TimeoutException as exc:
        log.exception("OpenRouter chat timed out")
        raise InfraError("OpenRouter timed out.", status_code=None, kind="timeout") from exc
    except httpx.HTTPError as exc:
        log.exception("OpenRouter chat request failed")
        raise InfraError("OpenRouter could not be reached.", status_code=None, kind="network") from exc
    kind = classify_http_infra_error(response.status_code)
    if kind:
        log.error("OpenRouter infra failure HTTP %s (%s)", response.status_code, kind)
        raise InfraError(
            f"OpenRouter infra error HTTP {response.status_code}.",
            status_code=response.status_code,
            kind=kind,
        )
    if response.status_code != 200:
        log.error("OpenRouter chat failed with HTTP %s", response.status_code)
        raise NeedleError("The answer model on OpenRouter did not return a draft.")
    payload = response.json()
    choices = payload.get("choices") or []
    text = ""
    finish_reason = None
    if choices:
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "").strip()
        finish_reason = choices[0].get("finish_reason")
    usage = dict(payload.get("usage") or {})
    usage["model"] = chosen
    usage["cost_usd"] = round(_usage_cost(usage, model=chosen), 6)
    purpose = cost_purpose or ("checker" if chosen == CHECKER_MODEL else "writer")
    record_cost(purpose, float(usage["cost_usd"] or 0))
    if not text and finish_reason == "length":
        log.warning("%s used all %s tokens without a visible reply", chosen, max_tokens)
        raise TokenBudgetExhausted(f"{chosen} ran out of tokens before answering.")
    if return_usage:
        return text, usage
    return text
