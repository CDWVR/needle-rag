"""Run answerable golden questions through the live /api/chat endpoint."""

from __future__ import annotations

import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.schema import load_jsonl

BASE = os.getenv("NEEDLE_BASE_URL", "http://127.0.0.1:8000")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def chat_once(client: httpx.Client, question: str) -> dict:
    with client.stream("POST", f"{BASE}/api/chat", json={"query": question}, timeout=180.0) as response:
        response.raise_for_status()
        validation = None
        answer_parts = []
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("type") == "validation":
                validation = payload
            elif payload.get("type") == "chunk":
                answer_parts.append(payload.get("content") or "")
            elif payload.get("type") == "error":
                return {"passed": False, "error": payload.get("content"), "answer": ""}
    return {
        "passed": bool((validation or {}).get("passed")),
        "grounded": bool((validation or {}).get("grounded")),
        "reason": (validation or {}).get("reason") or "",
        "answer": "".join(answer_parts),
        "validation": validation,
    }


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = [row for row in load_jsonl(GOLDEN) if row.get("answerable", True)][:limit]
    results = []
    with httpx.Client() as client:
        health = client.get(f"{BASE}/health", timeout=30)
        health.raise_for_status()
        print("models", health.json().get("answer_model"), health.json().get("checker_model"))
        for index, row in enumerate(rows, start=1):
            item = chat_once(client, row["question"])
            results.append({"id": row.get("id"), "passed": item["passed"], "grounded": item["grounded"], "reason": item["reason"][:120]})
            print(f"{index}/{len(rows)} passed={item['passed']} grounded={item['grounded']} {(item['reason'] or '')[:80]}")
    accepted = sum(1 for item in results if item["passed"])
    print(json.dumps({"n": len(results), "accepted": accepted, "accept_rate": round(accepted / len(results), 4) if results else 0}, indent=2))


if __name__ == "__main__":
    main()
