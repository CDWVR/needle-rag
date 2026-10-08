"""Optional LLM judge for answer correctness (`--judge`).

Key facts give a cheap, deterministic correctness signal. The judge adds a graded
check that tolerates paraphrase, using the reference facts and evidence spans.
It runs only on released answers to answerable questions.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from config import CHECKER_MAX_TOKENS, CHECKER_MODEL
from llm import complete
from pipeline_logic import NeedleError, extract_json_object

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "").strip() or CHECKER_MODEL
_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge(row: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    reference = row.get("reference_answer") or "; ".join(
        item.get("answer_span") or "" for item in row.get("expected") or []
    )
    facts = ", ".join(fact.replace("|", " or ") for fact in row.get("key_facts") or []) or "none listed"
    prompt = (
        "You grade answers from a document assistant. Compare the ANSWER with the REFERENCE evidence. "
        "Return JSON only: {\"verdict\": \"correct\" | \"partial\" | \"incorrect\", \"reason\": \"<15 words\"}. "
        "correct = answers the question and agrees with the reference; partial = right but incomplete; "
        "incorrect = wrong, contradicts the reference, or does not answer. Ignore citation markers like [1].\n\n"
        f"QUESTION: {row.get('question')}\nREFERENCE: {reference}\nKEY FACTS: {facts}\nANSWER: {record.get('answer')}"
    )
    try:
        raw, usage = complete(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=CHECKER_MAX_TOKENS,  # reasoning models need room before the verdict
            model=JUDGE_MODEL,
            return_usage=True,
            cost_purpose="other",
        )
    except NeedleError as exc:
        return {"score": None, "verdict": "error", "reason": str(exc), "cost_usd": 0.0}
    parsed = extract_json_object(raw) or {}
    verdict = str(parsed.get("verdict") or "").strip().lower()
    return {
        "score": _SCORES.get(verdict),
        "verdict": verdict or "unparseable",
        "reason": str(parsed.get("reason") or "")[:200],
        "model": JUDGE_MODEL,
        "cost_usd": float(usage.get("cost_usd") or 0),
    }
