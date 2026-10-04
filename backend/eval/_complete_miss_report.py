"""One Jev pass to list recall@5 misses; reuse fused table from prior fused-only run."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.metrics import hit_from_rank
from eval.miss_analysis import (
    DEFAULT_CONFIG,
    GOLDEN,
    _chunk_texts_by_hash,
    _jev_details,
    _label_in_index,
    _proposals,
    _rank_in_list,
    _retrieve,
    _type_recalls,
    load_config,
)
from eval.schema import load_jsonl
from rag_engine import list_parent_passages, set_jev_cache_enabled


def main() -> None:
    set_jev_cache_enabled(True)
    config = load_config(DEFAULT_CONFIG)
    rows = load_jsonl(GOLDEN)
    answerable = [row for row in rows if row.get("answerable", True)]
    passages = list_parent_passages()
    chunk_texts = _chunk_texts_by_hash()

    fused_path = os.path.join(os.path.dirname(__file__), "baselines", "miss_analysis_fused.json")
    with open(fused_path, encoding="utf-8") as handle:
        fused_report = json.load(handle)

    print("Jev type recalls (cached when possible)…", file=sys.stderr, flush=True)
    jev_table = _type_recalls(answerable, config, use_jev=True)

    misses = []
    invalid = []
    for index, row in enumerate(answerable, start=1):
        jev = _retrieve(row, config, use_jev=True)
        info = _jev_details(jev, row.get("expected") or [])
        if hit_from_rank(info["rank"], 5) == 1.0:
            continue
        fused_rank = _rank_in_list(jev.get("fused_parents") or [], row.get("expected") or [])
        if fused_rank is None:
            fused = _retrieve(row, config, use_jev=False)
            fused_rank = _rank_in_list(fused.get("fused_parents") or [], row.get("expected") or [])
        label = _label_in_index(row.get("expected") or [], passages, chunk_texts)
        item = {
            "id": row.get("id"),
            "type": row.get("type"),
            "question": row.get("question"),
            "label_valid": label["valid"],
            "label_location": label.get("location"),
            "fused_rank_in_30": fused_rank,
            "jev_rank": info["rank"],
            "reached_jev": info["reached_jev"],
            "jev_score": info["jev_score"],
            "expected": row.get("expected"),
        }
        misses.append(item)
        if not label["valid"]:
            invalid.append(item)
        print(
            f"miss {len(misses)} id={item['id']} fused={fused_rank} jev={info['rank']} reached={info['reached_jev']}",
            file=sys.stderr,
            flush=True,
        )

    report = {
        "answerable": len(answerable),
        "recall@5_misses": len(misses),
        "invalid_labels": invalid,
        "misses": misses,
        "fused": fused_report.get("fused"),
        "jev": jev_table,
        "proposed_retrieval_changes": _proposals(misses),
        "applied": False,
    }
    out = os.path.join(os.path.dirname(__file__), "baselines", "miss_analysis.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "recall@5_misses": len(misses),
                "invalid_label_count": len(invalid),
                "fused": report["fused"]["overall"],
                "jev": report["jev"]["overall"],
                "by_type_fused": report["fused"]["by_type"],
                "by_type_jev": report["jev"]["by_type"],
                "misses": [
                    {
                        "id": m["id"],
                        "type": m["type"],
                        "label_valid": m["label_valid"],
                        "fused_rank_in_30": m["fused_rank_in_30"],
                        "jev_rank": m["jev_rank"],
                        "reached_jev": m["reached_jev"],
                        "jev_score": m["jev_score"],
                    }
                    for m in misses
                ],
                "proposed_retrieval_changes": report["proposed_retrieval_changes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
