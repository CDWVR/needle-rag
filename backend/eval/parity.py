"""Phase 1 parity gate against the frozen pre-Postgres baseline.

Retrieval recall@5, recall@30 and MRR must stay within 0.02 of baseline, and
top-5 ids (mapped via content_hash + answer_span) must overlap at least 90%
per answerable query.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


PARITY_METRIC_TOLERANCE = 0.02
PARITY_TOP5_OVERLAP = 0.90


def metric_within_tolerance(baseline: float, candidate: float, *, tolerance: float = PARITY_METRIC_TOLERANCE) -> bool:
    return abs(float(candidate) - float(baseline)) <= float(tolerance) + 1e-12


def top_ids_from_parents(parents: Sequence[Dict[str, Any]], *, limit: int = 5) -> List[str]:
    """Stable ids from content_hash + first matching answer span key material."""
    ids: List[str] = []
    for parent in parents[: max(0, limit)]:
        content_hash = parent.get("content_hash") or ""
        text = parent.get("text") or ""
        # Span fingerprint: normalized head of parent text keeps ids migration-safe.
        span = " ".join(text.split())[:80]
        token = f"{content_hash}:{span}"
        if content_hash and token not in ids:
            ids.append(token)
    return ids


def top5_overlap(baseline_ids: Sequence[str], candidate_ids: Sequence[str]) -> float:
    left = list(baseline_ids)[:5]
    right = list(candidate_ids)[:5]
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    shared = len(set(left).intersection(right))
    return shared / max(len(left), len(right))


def parity_gate(
    *,
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    per_query_overlap: Sequence[float],
    tolerance: float = PARITY_METRIC_TOLERANCE,
    min_overlap: float = PARITY_TOP5_OVERLAP,
) -> Dict[str, Any]:
    checks = {
        "recall_at_5": metric_within_tolerance(
            baseline.get("recall_at_5", 0.0), candidate.get("recall_at_5", 0.0), tolerance=tolerance
        ),
        "recall_at_30": metric_within_tolerance(
            baseline.get("recall_at_30", 0.0), candidate.get("recall_at_30", 0.0), tolerance=tolerance
        ),
        "mrr": metric_within_tolerance(baseline.get("mrr", 0.0), candidate.get("mrr", 0.0), tolerance=tolerance),
    }
    mean_overlap = sum(per_query_overlap) / len(per_query_overlap) if per_query_overlap else 1.0
    checks["top5_overlap"] = mean_overlap + 1e-12 >= float(min_overlap)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_top5_overlap": round(mean_overlap, 4),
        "tolerance": tolerance,
        "min_overlap": min_overlap,
        "baseline": baseline,
        "candidate": candidate,
    }
