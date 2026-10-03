"""Cold-Jev candidate/concurrency/margin sweeps. Does not change defaults."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.metrics import hit_from_rank, mean_spread, mrr_from_rank, percentile, summarize_latencies
from eval.schema import first_match_rank, load_jsonl
from rag_engine import (
    RETRY_THRESHOLD,
    answer_for_eval,
    clear_jev_cache,
    retrieve_parents,
    set_jev_cache_enabled,
    set_jev_max_concurrency,
)


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "baseline_config.json")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sample(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    answerable = [row for row in rows if row.get("answerable", True)]
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in answerable:
        by_type.setdefault(row.get("type") or "factual", []).append(row)
    sample: List[Dict[str, Any]] = []
    if not by_type:
        return answerable[:limit]
    per = max(1, limit // len(by_type))
    for bucket in by_type.values():
        sample.extend(bucket[:per])
    return sample[:limit] or answerable[:limit]


def _retrieval_point(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    candidate_limit: int,
    concurrency: int,
    skip_margin: Optional[float] = None,
) -> Dict[str, Any]:
    set_jev_max_concurrency(concurrency)
    clear_jev_cache()
    recalls = []
    mrrs = []
    latency_rows = []
    for row in rows:
        result = retrieve_parents(
            row.get("question") or "",
            history=row.get("history") or [],
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=max(int(config.get("max_parents", 5)), 5),
            use_jev=True,
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
            retry_threshold=float(config.get("retry_threshold", RETRY_THRESHOLD)),
            jev_candidate_limit=candidate_limit,
            skip_jev_margin=skip_margin,
            allow_retry=False,
        )
        rank = first_match_rank(result.get("ordered_parents") or result.get("parents") or [], row.get("expected") or [])
        recalls.append(hit_from_rank(rank, 5))
        mrrs.append(mrr_from_rank(rank))
        latency_rows.append(result.get("latencies_ms") or {})
    rerank = summarize_latencies(latency_rows, ("rerank",)).get("rerank") or {}
    return {
        "recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else None,
        "rerank_p50_ms": rerank.get("p50_ms"),
        "rerank_p95_ms": rerank.get("p95_ms"),
        "n": len(rows),
        "skip_jev_margin": skip_margin,
        "jev_candidate_limit": candidate_limit,
        "concurrency": concurrency,
    }


def _abstention_point(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    candidate_limit: int,
    concurrency: int,
) -> Dict[str, float]:
    from eval.metrics import abstention_scores

    set_jev_max_concurrency(concurrency)
    clear_jev_cache()
    pred = []
    label = []
    # Cheap abstention probe: retrieval abstain only (no writer/checker).
    for row in rows:
        result = retrieve_parents(
            row.get("question") or "",
            history=row.get("history") or [],
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=int(config.get("max_parents", 5)),
            use_jev=True,
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
            retry_threshold=float(config.get("retry_threshold", RETRY_THRESHOLD)),
            jev_candidate_limit=candidate_limit,
            allow_retry=False,
        )
        pred.append(bool(result.get("abstained")))
        label.append(not bool(row.get("answerable", True)))
    return abstention_scores(pred, label)


def run_sweep(rows: List[Dict[str, Any]], config: Dict[str, Any], *, sample_size: int) -> Dict[str, Any]:
    set_jev_cache_enabled(False)
    clear_jev_cache()
    sample = _sample(rows, sample_size)
    # Include unanswerables for abstention on a small holdout.
    unans = [row for row in rows if not row.get("answerable", True)][: max(5, sample_size // 5)]
    abstain_rows = sample + unans

    base_concurrency = int(os.getenv("JEV_MAX_CONCURRENCY", "4") or 4)
    concurrencies = [base_concurrency, base_concurrency * 2, base_concurrency * 4]
    candidates = [10, 15, 20, 30]
    margins = [0.01, 0.02, 0.05]

    print(f"Cold baseline at candidates=30 concurrency={base_concurrency}", file=sys.stderr, flush=True)
    baseline = _retrieval_point(sample, config, candidate_limit=30, concurrency=base_concurrency)
    baseline_abs = _abstention_point(abstain_rows, config, candidate_limit=30, concurrency=base_concurrency)
    baseline["abstention"] = baseline_abs

    grid = []
    for cand in candidates:
        for conc in concurrencies:
            print(f"sweep candidates={cand} concurrency={conc}", file=sys.stderr, flush=True)
            point = _retrieval_point(sample, config, candidate_limit=cand, concurrency=conc)
            point["abstention"] = _abstention_point(abstain_rows, config, candidate_limit=cand, concurrency=conc)
            point["mrr_delta_vs_baseline"] = round((point["mrr"] or 0) - (baseline["mrr"] or 0), 4)
            point["within_noise"] = abs(point["mrr_delta_vs_baseline"]) <= 0.02 + 1e-12
            grid.append(point)

    margin_rows = []
    for margin in margins:
        print(f"margin sweep skip_jev_margin={margin}", file=sys.stderr, flush=True)
        point = _retrieval_point(sample, config, candidate_limit=30, concurrency=base_concurrency, skip_margin=margin)
        point["mrr_delta_vs_baseline"] = round((point["mrr"] or 0) - (baseline["mrr"] or 0), 4)
        point["within_noise"] = abs(point["mrr_delta_vs_baseline"]) <= 0.02 + 1e-12
        margin_rows.append(point)

    # Cheapest within noise: minimize candidates, then concurrency, prefer lower p95.
    affordable = [row for row in grid if row.get("within_noise")]
    affordable.sort(
        key=lambda row: (
            row.get("jev_candidate_limit") or 99,
            row.get("concurrency") or 99,
            row.get("rerank_p95_ms") or 1e12,
        )
    )
    recommendation = affordable[0] if affordable else baseline
    return {
        "sample_size": len(sample),
        "abstain_probe_n": len(abstain_rows),
        "jev_cache": "disabled",
        "baseline": baseline,
        "grid": grid,
        "skip_margin": margin_rows,
        "recommendation": {
            "jev_candidate_limit": recommendation.get("jev_candidate_limit"),
            "concurrency": recommendation.get("concurrency"),
            "mrr": recommendation.get("mrr"),
            "recall_at_5": recommendation.get("recall_at_5"),
            "rerank_p50_ms": recommendation.get("rerank_p50_ms"),
            "rerank_p95_ms": recommendation.get("rerank_p95_ms"),
            "mrr_delta_vs_baseline": recommendation.get("mrr_delta_vs_baseline", 0.0),
            "note": "Cheapest grid point within 0.02 MRR of cold baseline. Defaults not changed.",
        },
        "defaults_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold Jev rerank latency/quality sweep.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "baselines", "rerank_sweep.json"))
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    rows = load_jsonl(args.golden)
    started = time.perf_counter()
    try:
        report = run_sweep(rows, config, sample_size=args.sample)
    finally:
        set_jev_cache_enabled(True)
        set_jev_max_concurrency(None)
        clear_jev_cache()
    report["elapsed_s"] = round(time.perf_counter() - started, 2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
