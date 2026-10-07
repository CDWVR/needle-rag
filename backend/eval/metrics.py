"""Metric functions for the eval. Pure Python: no engine, no network, safe to import in CI."""

from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


# --- retrieval ---------------------------------------------------------------

def hit_at_k(rank: Optional[int], k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(relevant_ranks: Sequence[int], n_relevant: int, k: int) -> float:
    """Binary-relevance nDCG@k. `relevant_ranks` are the 1-based ranks of relevant results."""
    if n_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks if rank <= k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


# --- answers -----------------------------------------------------------------

_NUMBER_COMMAS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize_answer(text: str) -> str:
    cleaned = (text or "").lower()
    cleaned = _NUMBER_COMMAS.sub("", cleaned)
    for left, right in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"), (" ", " ")):
        cleaned = cleaned.replace(left, right)
    return re.sub(r"\s+", " ", cleaned).strip()


def fact_present(answer: str, fact: str) -> bool:
    """A key fact is present if any `|`-separated alternative appears in the answer."""
    haystack = normalize_answer(answer)
    for option in fact.split("|"):
        needle = normalize_answer(option)
        if not needle:
            continue
        # Short alphanumeric facts ("6", "no", "F") must match as whole tokens.
        if len(needle) <= 3 and re.fullmatch(r"[\w.]+", needle):
            if re.search(rf"(?<![\w.]){re.escape(needle)}(?![\w]|\.\d)", haystack):
                return True
        elif needle in haystack:
            return True
    return False


def key_fact_recall(answer: str, facts: Sequence[str]) -> Optional[float]:
    if not facts:
        return None
    return sum(1 for fact in facts if fact_present(answer, fact)) / len(facts)


def leaked_phrases(answer: str, phrases: Sequence[str]) -> List[str]:
    haystack = normalize_answer(answer)
    return [phrase for phrase in phrases or [] if normalize_answer(phrase) in haystack]


def citation_numbers(answer: str) -> List[int]:
    return [int(number) for number in re.findall(r"\[(\d+)\]", answer or "")]


# --- classification ----------------------------------------------------------

def binary_scores(predictions: Sequence[bool], labels: Sequence[bool]) -> Dict[str, Any]:
    """Precision / recall / F1 where True is the positive class (here: abstain on unanswerable)."""
    tp = sum(1 for p, y in zip(predictions, labels) if p and y)
    fp = sum(1 for p, y in zip(predictions, labels) if p and not y)
    fn = sum(1 for p, y in zip(predictions, labels) if not p and y)
    tn = sum(1 for p, y in zip(predictions, labels) if not p and not y)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# --- aggregation -------------------------------------------------------------

def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = [float(value) for value in values if value is not None]
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    position = (len(ordered) - 1) * (p / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 1)


def bootstrap_ci(values: Sequence[float], *, resamples: int = 2000, alpha: float = 0.05, seed: int = 13) -> Optional[Dict[str, float]]:
    """Percentile bootstrap CI for the mean. Deterministic for a given seed."""
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    rng = random.Random(seed)
    n = len(numbers)
    means = sorted(sum(numbers[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "low": round(means[int((alpha / 2) * (resamples - 1))], 4),
        "high": round(means[int((1 - alpha / 2) * (resamples - 1))], 4),
    }


def paired_delta(current: Dict[str, float], baseline: Dict[str, float], *, resamples: int = 2000, seed: int = 13) -> Optional[Dict[str, Any]]:
    """Mean difference (current - baseline) over question ids present in both, with a bootstrap CI.

    Pairing by question removes most between-question variance, so small real changes show up
    and noise on a handful of questions does not.
    """
    shared = sorted(set(current) & set(baseline))
    if not shared:
        return None
    diffs = [float(current[key]) - float(baseline[key]) for key in shared]
    interval = bootstrap_ci(diffs, resamples=resamples, seed=seed)
    return {
        "delta": round(sum(diffs) / len(diffs), 4),
        "ci": interval,
        "n": len(shared),
        "significant": bool(interval and (interval["high"] < 0 or interval["low"] > 0)),
    }


def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None
