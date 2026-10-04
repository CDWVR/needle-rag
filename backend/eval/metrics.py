"""Retrieval and harness metrics that do not require a model."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


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


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> Dict[str, float]:
    """Percentile bootstrap 95% CI for the mean of `values`."""
    import random

    numbers = [float(value) for value in values]
    if not numbers:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    rng = random.Random(seed)
    means = []
    n = len(numbers)
    for _ in range(max(1, n_resamples)):
        sample = [numbers[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_i = int((alpha / 2) * (len(means) - 1))
    high_i = int((1 - alpha / 2) * (len(means) - 1))
    mean = sum(numbers) / n
    return {
        "mean": round(mean, 4),
        "low": round(means[low_i], 4),
        "high": round(means[high_i], 4),
        "n": n,
    }


def bootstrap_diff_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> Dict[str, Any]:
    """Bootstrap CI for mean(left) - mean(right). Flags when 0 is inside the interval."""
    import random

    a = [float(value) for value in left]
    b = [float(value) for value in right]
    if not a or not b or len(a) != len(b):
        # Pairwise when same length; otherwise compare independent means on min length.
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    if not a:
        return {"diff": 0.0, "low": 0.0, "high": 0.0, "within_noise": True, "n": 0}
    rng = random.Random(seed)
    diffs = []
    n = len(a)
    for _ in range(max(1, n_resamples)):
        idx = [rng.randrange(n) for _ in range(n)]
        left_mean = sum(a[i] for i in idx) / n
        right_mean = sum(b[i] for i in idx) / n
        diffs.append(left_mean - right_mean)
    diffs.sort()
    low_i = int((alpha / 2) * (len(diffs) - 1))
    high_i = int((1 - alpha / 2) * (len(diffs) - 1))
    diff = (sum(a) / n) - (sum(b) / n)
    low, high = diffs[low_i], diffs[high_i]
    return {
        "diff": round(diff, 4),
        "low": round(low, 4),
        "high": round(high, 4),
        "within_noise": low <= 0.0 <= high,
        "n": n,
    }


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """AUROC for ranking positive labels higher. None if undefined."""
    pairs = [(float(score), 1 if label else 0) for score, label in zip(scores, labels)]
    positives = sum(label for _score, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    for index, (_score, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += index
    # Mann–Whitney U formulation.
    u = rank_sum - positives * (positives + 1) / 2.0
    return round(u / (positives * negatives), 4)
