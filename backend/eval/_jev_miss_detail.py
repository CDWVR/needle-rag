"""Fill Jev ranks/scores for fused recall@5 misses and print Jev recall table."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.miss_analysis import (
    DEFAULT_CONFIG,
    GOLDEN,
    _jev_details,
    _retrieve,
    _type_recalls,
    load_config,
)
from eval.schema import load_jsonl
from eval.metrics import hit_from_rank


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    rows = load_jsonl(GOLDEN)
    answerable = [row for row in rows if row.get("answerable", True)]
    miss_path = os.path.join(os.path.dirname(__file__), "baselines", "miss_analysis_fused.json")
    with open(miss_path, encoding="utf-8") as handle:
        fused_report = json.load(handle)
    miss_ids = {item["id"] for item in fused_report.get("misses") or []}

    print("Jev type recalls…", file=sys.stderr, flush=True)
    jev_table = _type_recalls(answerable, config, use_jev=True)

    details = []
    for row in answerable:
        if row.get("id") not in miss_ids:
            continue
        result = _retrieve(row, config, use_jev=True)
        info = _jev_details(result, row.get("expected") or [])
        details.append(
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "question": row.get("question"),
                "label_valid": True,
                "fused_rank_in_30": next(
                    (item["fused_rank_in_30"] for item in fused_report["misses"] if item["id"] == row.get("id")),
                    None,
                ),
                "jev_rank": info["rank"],
                "reached_jev": info["reached_jev"],
                "jev_score": info["jev_score"],
                "recall@5": hit_from_rank(info["rank"], 5),
            }
        )
        print(f"miss detail {row.get('id')} jev_rank={info['rank']}", file=sys.stderr, flush=True)

    from eval.miss_analysis import _proposals

    report = {
        **fused_report,
        "jev": jev_table,
        "misses": details,
        "proposed_retrieval_changes": _proposals(details),
        "fused_only": False,
    }
    out = os.path.join(os.path.dirname(__file__), "baselines", "miss_analysis.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"jev": jev_table, "misses": details, "proposed_retrieval_changes": report["proposed_retrieval_changes"]}, indent=2))


if __name__ == "__main__":
    main()
