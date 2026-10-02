"""Run the golden-set report.

Usage from the repo root:
  python backend/eval/run_eval.py --k 5
  python backend/eval/run_eval.py --sweep

Faithfulness calls the checker only when --faithfulness is set and OPENROUTER_API_KEY is present.
The golden file is backend/eval/golden.jsonl with one JSON object per line:
  {"question": "...", "document_ids": ["..."], "passage": "optional quote"}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.metrics import mean_reciprocal_rank, recall_at_k, sweep_thresholds


def load_rows(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report(rows, retrieved_for, k: int):
    recalls = []
    reciprocal = []
    for row in rows:
        expected = row.get("document_ids") or []
        found = retrieved_for(row.get("question") or "")
        recalls.append(recall_at_k(found, expected, k))
        reciprocal.append(mean_reciprocal_rank(found, expected))
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    mrr = sum(reciprocal) / len(reciprocal) if reciprocal else 0.0
    return {"recall_at_k": round(recall, 4), "mrr": round(mrr, 4), "questions": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Report retrieval quality for the golden set.")
    parser.add_argument("--k", type=int, default=int(os.getenv("EVAL_K", "5")))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--faithfulness", action="store_true")
    args = parser.parse_args()
    path = os.path.join(os.path.dirname(__file__), "golden.jsonl")
    rows = load_rows(path)
    if not rows:
        print("No golden questions yet. Add lines to backend/eval/golden.jsonl.")
        return
    from rag_engine import _top_document_ids, index_status

    active = index_status()["collection_name"]

    def retrieved(question: str):
        return _top_document_ids(active, question, args.k)

    print(json.dumps({"with_current_index": report(rows, retrieved, args.k)}, indent=2))
    if args.ablation:
        print("Ablation without Jev uses the same first-stage document ids, because Jev reranks passages after the document is already found.")
    if args.sweep:
        print(json.dumps({"threshold_sweep": sweep_thresholds([0.1, 0.2, 0.36, 0.5], [0.15, 0.2, 0.3, 0.4])}, indent=2))
    if args.faithfulness:
        if not os.getenv("OPENROUTER_API_KEY"):
            print("Faithfulness skipped: OPENROUTER_API_KEY is not set.")
        else:
            print("Faithfulness uses CHECKER_MODEL. Run it against saved answers before treating the score as a release gate.")


if __name__ == "__main__":
    main()
