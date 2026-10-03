"""Rewrite phase0.json freeze metadata (mean ± spread, no best-of)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import mean_spread

ROOT = Path(__file__).resolve().parents[2]
PHASE0 = Path(__file__).resolve().parent / "baselines" / "phase0.json"
GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"


def main() -> None:
    import hashlib

    data = json.loads(PHASE0.read_text(encoding="utf-8"))
    runs = data["runs"]
    abs_r = []
    abs_p = []
    for run in runs:
        abstention = run.get("abstention") or (run.get("answer_quality") or {}).get("abstention") or {}
        abs_r.append(float(abstention.get("recall") or 0))
        abs_p.append(float(abstention.get("precision") or 0))

    agg = {
        "recall_at_k": mean_spread([run.get("recall_at_k") or 0 for run in runs]),
        "recall_at_5": mean_spread([run.get("recall_at_5") or run.get("recall_at_k") or 0 for run in runs]),
        "mrr": mean_spread([run.get("mrr") or 0 for run in runs]),
        "faithfulness": mean_spread([run.get("faithfulness") or 0 for run in runs]),
        "abstention_recall": mean_spread(abs_r),
        "abstention_precision": mean_spread(abs_p),
        "mean_cost_usd": mean_spread([run.get("mean_cost_usd") or 0 for run in runs if "mean_cost_usd" in run]),
    }
    golden_sha = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    cfg = data["settings"]["config"]
    models = data["settings"]["models"]

    data["phase"] = "0.7-freeze"
    data["commit"] = None
    data["golden_file"] = "backend/eval/golden.jsonl"
    data["golden_sha256"] = golden_sha
    data["config_file"] = "backend/eval/baseline_config.json"
    data["config_snapshot"] = cfg
    data["model_slugs"] = models
    data["aggregate"] = {**(data.get("aggregate") or {}), **agg}
    data["recall_at_k"] = agg["recall_at_k"]
    data["recall_at_5"] = agg["recall_at_5"]
    data["mrr"] = agg["mrr"]
    data["faithfulness"] = agg["faithfulness"]
    data["abstention"] = {"recall": agg["abstention_recall"], "precision": agg["abstention_precision"]}
    data["mean_cost_usd"] = agg["mean_cost_usd"]
    data["parity_gate"] = {
        "module": "backend/eval/parity.py",
        "metric_tolerance": 0.02,
        "min_top5_overlap": 0.90,
        "metrics": ["recall_at_5", "recall_at_30", "mrr"],
        "id_mapping": "content_hash + answer_span / parent text head",
    }
    data["openrouter_at_freeze"] = {
        "limit": 1.25,
        "limit_remaining": 0.0,
        "usage": 1.250726619,
        "note": "Limit raised from 1.00 to 1.25; 0.7.2/0.7.3 consumed remaining budget. Fresh 3x full + contextual 3x blocked.",
    }
    data["phase07"] = {
        "infra_error": "OpenRouter 403/429/5xx/timeout → infra_error; excluded from faithfulness/abstention; fail if share > 0.02",
        "miss_analysis": "backend/eval/baselines/miss_analysis.json",
        "rerank_sweep": "backend/eval/baselines/rerank_sweep.json",
        "rerank_recommendation": {"jev_candidate_limit": 15, "concurrency": 4, "defaults_changed": False},
        "quote_span": "Paraphrase unless quoting exactly; quote check normalizes whitespace/Unicode/quotes/hyphenation/ellipses",
    }
    data.pop("targets", None)
    data.pop("git_commit", None)
    data["notes"] = [
        "Phase 0.7 freeze: mean ± spread only (no best-of reporting).",
        "Full 3-run faithfulness aggregates retained from prior cold-Jev harness (100 golden items).",
        "Fresh 3x full-set + contextual 3x not re-executed: OpenRouter limit_remaining=0 after miss analysis and rerank sweep (limit=1.25, usage≈1.251).",
        "Parity gate coded in backend/eval/parity.py (recall@5/30 and MRR within 0.02; top-5 id overlap ≥ 0.90).",
        "Tag baseline-pre-pg marks this freeze; Postgres not started.",
    ]
    PHASE0.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"golden_sha256": golden_sha, "aggregate": agg}, indent=2))


if __name__ == "__main__":
    main()
