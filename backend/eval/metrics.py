"""Retrieval metrics that do not call a model."""

from typing import List, Sequence


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


def sweep_thresholds(scores: Sequence[float], thresholds: Sequence[float]) -> List[dict]:
    ordered = sorted(float(score) for score in scores)
    rows = []
    for threshold in thresholds:
        kept = [score for score in ordered if score >= threshold]
        rows.append({"threshold": threshold, "kept": len(kept), "best": max(kept) if kept else 0.0})
    return rows
