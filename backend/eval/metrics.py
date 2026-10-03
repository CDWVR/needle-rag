"""Retrieval and harness metrics that do not require a model."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    expected = [item for item in expected_ids if item]
    if not expected:
        return 1.0
    found = set(list(retrieved_ids)[: max(0, k)])
    return len(found.intersection(expected)) / len(set(expected))


def mean_reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    expected = set(item for item in expected_ids if item)
    for index, item in enumerate(retrieved_ids, start=1):
        if item in expected:
            return 1.0 / index
    return 0.0


def hit_from_rank(rank: Optional[int], k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def mrr_from_rank(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize_latencies(rows: Iterable[Dict[str, float]], stages: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for stage in stages:
        values = [float(row.get(stage) or 0) for row in rows if stage in row]
        if not values:
            summary[stage] = {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "n": 0}
            continue
        summary[stage] = {
            "p50_ms": round(percentile(values, 50), 2),
            "p95_ms": round(percentile(values, 95), 2),
            "mean_ms": round(sum(values) / len(values), 2),
            "n": len(values),
        }
    return summary


def abstention_scores(predictions: Sequence[bool], labels: Sequence[bool]) -> Dict[str, float]:
    """predictions/labels True means abstain / unanswerable."""
    tp = fp = tn = fn = 0
    for predicted, label in zip(predictions, labels):
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif (not predicted) and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def sweep_thresholds(scores: Sequence[float], thresholds: Sequence[float]) -> List[dict]:
    ordered = sorted(float(score) for score in scores)
    rows = []
    for threshold in thresholds:
        kept = [score for score in ordered if score >= threshold]
        rows.append({"threshold": threshold, "kept": len(kept), "best": max(kept) if kept else 0.0})
    return rows


def mean_spread(values: Sequence[float]) -> Dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "spread": 0.0}
    mean = sum(numbers) / len(numbers)
    return {
        "mean": round(mean, 4),
        "min": round(min(numbers), 4),
        "max": round(max(numbers), 4),
        "spread": round(max(numbers) - min(numbers), 4),
    }
