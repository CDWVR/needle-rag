"""Aggregate per-question records into metrics, apply gates, and write reports."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from eval import metrics as m

# Per-question scores kept in a baseline so later runs can be compared question by question.
PAIRED_SCORES = ("hit_at_5", "rr", "recall_at_k", "answered", "key_fact_recall", "cites_evidence", "judge_correct")


def _scores(records: List[Dict[str, Any]], key: str, *, where=None) -> List[float]:
    return [
        record["scores"][key]
        for record in records
        if key in (record.get("scores") or {}) and (where is None or where(record))
    ]


def _latency(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    stages: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        for stage, value in (record.get("latencies_ms") or {}).items():
            if value is not None:
                stages[stage].append(float(value))
    return {stage: {"p50": m.percentile(values, 50), "p95": m.percentile(values, 95), "n": len(values)}
            for stage, values in sorted(stages.items())}


def aggregate(records: List[Dict[str, Any]], *, generate: bool) -> Dict[str, Any]:
    scored = [r for r in records if not r.get("infra_error")]
    answerable = [r for r in scored if r["answerable"]]
    retrieval_keys = ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "recall_at_k", "rr", "ndcg_at_k", "page_hit_at_k")
    retrieval = {key: m.mean(_scores(scored, key)) for key in retrieval_keys}
    retrieval["mrr"] = retrieval.pop("rr")
    retrieval["n"] = len(_scores(scored, "hit_at_5"))
    retrieval["hit_at_5_ci"] = m.bootstrap_ci(_scores(scored, "hit_at_5"))

    by_type: Dict[str, Dict[str, Any]] = {}
    for qtype in sorted({r.get("type") for r in scored}):
        group = [r for r in scored if r.get("type") == qtype]
        entry = {"n": len(group), "hit_at_5": m.mean(_scores(group, "hit_at_5")), "mrr": m.mean(_scores(group, "rr"))}
        if generate:
            entry["answered"] = m.mean(_scores(group, "answered"))
            entry["key_fact_recall"] = m.mean(_scores(group, "key_fact_recall"))
            entry["abstained"] = m.mean([1.0 if r.get("abstained") else 0.0 for r in group])
        by_type[qtype] = entry

    infra = sum(1 for r in records if r.get("infra_error"))
    summary: Dict[str, Any] = {
        "questions": len(records),
        "retrieval": retrieval,
        "by_type": by_type,
        "latency_ms": _latency(scored),
        "cost": {
            "total_usd": round(sum(float(r.get("cost_usd") or 0) for r in records), 6),
            "per_question_usd": m.mean([float(r.get("cost_usd") or 0) for r in records]),
        },
        "infra": {"errors": infra, "share": round(infra / len(records), 4) if records else 0.0},
        "pipeline_errors": sum(1 for r in records if r.get("reject_category") == "pipeline_error"),
    }
    if not generate:
        return summary

    plain = [r for r in answerable if "injection" not in r["tags"]]
    released_answerable = [r for r in plain if r.get("released") and not r.get("declined")]
    unanswerable = [r for r in scored if not r["answerable"]]
    injection = [r for r in scored if "injection" in r["tags"]]
    abstain = m.binary_scores([bool(r.get("abstained")) for r in scored], [not r["answerable"] for r in scored])
    for tag in ("near_miss", "off_topic"):
        group = [r for r in unanswerable if tag in r["tags"]]
        abstain[f"recall_{tag}"] = m.mean([1.0 if r.get("abstained") else 0.0 for r in group])
    summary["answers"] = {
        "answer_rate": m.mean(_scores(plain, "answered")),
        "answer_rate_ci": m.bootstrap_ci(_scores(plain, "answered")),
        "false_refusal_rate": m.mean([1.0 if r.get("abstained") else 0.0 for r in plain]),
        "key_fact_recall": m.mean(_scores(released_answerable, "key_fact_recall")),
        "fully_correct_rate": m.mean(_scores(released_answerable, "fully_correct")),
        "citation_validity": m.mean(_scores(released_answerable, "citation_valid")),
        "evidence_citation_rate": m.mean(_scores(released_answerable, "cites_evidence")),
        "partial_rate": m.mean([1.0 if r.get("partially_supported") else 0.0 for r in released_answerable]),
        "released_unanswerable": sum(1 for r in unanswerable if r.get("released") and not r.get("declined")),
        "declined_rate": m.mean([1.0 if r.get("declined") else 0.0 for r in scored]),
        "n_answerable": len(plain),
    }
    if any("judge_correct" in (r.get("scores") or {}) for r in records):
        summary["answers"]["judge_correct"] = m.mean(_scores(released_answerable, "judge_correct"))
    summary["abstention"] = abstain
    summary["safety"] = {
        "injection_rows": len(injection),
        "injection_released": sum(1 for r in injection if r.get("released")),
        "injection_leaks": sum(1 for r in scored if r.get("leaked_phrases")),
    }
    summary["reject_categories"] = dict(Counter(r.get("reject_category") or "accepted" for r in records).most_common())
    return summary


def flatten(tree: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[path] = float(value)
    return flat


def per_question(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    return {
        record["id"]: {key: record["scores"][key] for key in PAIRED_SCORES if key in (record.get("scores") or {})}
        for record in records
        if not record.get("infra_error")
    }


def evaluate_gates(
    summary: Dict[str, Any],
    gates: Dict[str, Any],
    *,
    baseline: Optional[Dict[str, Any]],
    current_questions: Dict[str, Dict[str, float]],
    comparable: bool,
    deterministic: bool,
) -> Dict[str, Any]:
    flat = flatten(summary)
    checks: List[Dict[str, Any]] = []
    for key, floor in (gates.get("min") or {}).items():
        value = flat.get(key)
        checks.append({"check": f"{key} >= {floor}", "value": value, "passed": value is not None and value >= floor})
    for key, ceiling in (gates.get("max") or {}).items():
        value = flat.get(key)
        checks.append({"check": f"{key} <= {ceiling}", "value": value, "passed": value is not None and value <= ceiling})

    regressions: List[Dict[str, Any]] = []
    if baseline and comparable:
        old_flat = flatten(baseline.get("metrics") or {})
        old_questions = baseline.get("per_question") or {}
        for key, allowed in (gates.get("max_drop") or {}).items():
            new, old = flat.get(key), old_flat.get(key)
            if new is None or old is None:
                continue
            drop = old - new
            paired = None
            score_key = _paired_key(key)
            if score_key:
                paired = m.paired_delta(
                    {qid: s[score_key] for qid, s in current_questions.items() if score_key in s},
                    {qid: s[score_key] for qid, s in old_questions.items() if score_key in s},
                )
            # A deterministic tier has no run-to-run noise, so any drop past the margin is real.
            # A model-backed tier must also show a paired drop whose CI excludes zero.
            significant = deterministic or paired is None or paired["significant"]
            failed = drop > allowed + 1e-9 and significant
            regressions.append({
                "metric": key, "baseline": old, "current": new, "drop": round(drop, 4),
                "allowed_drop": allowed, "paired": paired, "passed": not failed,
            })
    passed = all(check["passed"] for check in checks) and all(item["passed"] for item in regressions)
    return {"passed": passed, "checks": checks, "regressions": regressions, "compared_to_baseline": bool(baseline and comparable)}


def _paired_key(metric: str) -> Optional[str]:
    return {
        "retrieval.hit_at_5": "hit_at_5",
        "retrieval.mrr": "rr",
        "retrieval.recall_at_k": "recall_at_k",
        "answers.answer_rate": "answered",
        "answers.key_fact_recall": "key_fact_recall",
        "answers.evidence_citation_rate": "cites_evidence",
        "answers.judge_correct": "judge_correct",
    }.get(metric)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:,.0f}"
    return str(value)


def markdown(report: Dict[str, Any]) -> str:
    summary = report["metrics"]
    gate = report["gate"]
    lines = [
        f"# Needle eval — {report['suite']} / {report['tier']}",
        "",
        f"- Finished: {report['finished_at']}  ·  commit `{(report.get('commit') or 'unknown')[:10]}`",
        f"- Questions: {summary['questions']}  ·  cost ${summary['cost']['total_usd']:.4f}  ·  "
        f"infra errors {summary['infra']['errors']}",
        f"- Gate: **{'PASS' if gate['passed'] else 'FAIL'}**"
        + ("" if gate["compared_to_baseline"] else " (absolute thresholds only; no comparable baseline)"),
        "",
        "## Retrieval",
        "",
        "| hit@1 | hit@3 | hit@5 | hit@10 | recall@k | MRR | nDCG@k | page hit@k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| " + " | ".join(_fmt(summary["retrieval"].get(key)) for key in
                         ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "recall_at_k", "mrr", "ndcg_at_k", "page_hit_at_k")) + " |",
    ]
    if "answers" in summary:
        answers, abstain, safety = summary["answers"], summary["abstention"], summary["safety"]
        lines += [
            "",
            "## Answers",
            "",
            "| answer rate | false refusals | key-fact recall | fully correct | citations valid | cites evidence | abstain P / R / F1 | injection leaks |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| {_fmt(answers['answer_rate'])} | {_fmt(answers['false_refusal_rate'])} | {_fmt(answers['key_fact_recall'])} | "
            f"{_fmt(answers['fully_correct_rate'])} | {_fmt(answers['citation_validity'])} | {_fmt(answers['evidence_citation_rate'])} | "
            f"{_fmt(abstain['precision'])} / {_fmt(abstain['recall'])} / {_fmt(abstain['f1'])} | {safety['injection_leaks']} |",
            "",
            "Reject categories: " + ", ".join(f"{key} {value}" for key, value in summary["reject_categories"].items()),
        ]
    lines += ["", "## By question type", "", "| type | n | hit@5 | MRR |" + (" answered | abstained |" if "answers" in summary else ""),
              "| --- | --- | --- | --- |" + (" --- | --- |" if "answers" in summary else "")]
    for qtype, entry in summary["by_type"].items():
        row = f"| {qtype} | {entry['n']} | {_fmt(entry['hit_at_5'])} | {_fmt(entry['mrr'])} |"
        if "answers" in summary:
            row += f" {_fmt(entry.get('answered'))} | {_fmt(entry.get('abstained'))} |"
        lines.append(row)
    total = summary["latency_ms"].get("total") or {}
    lines += ["", f"Latency p50 / p95: {_fmt(total.get('p50'))} / {_fmt(total.get('p95'))} ms total; "
              + ", ".join(f"{stage} {_fmt(v['p50'])}" for stage, v in summary["latency_ms"].items() if stage != "total")]
    lines += ["", "## Gate", ""]
    for check in gate["checks"]:
        lines.append(f"- {'✅' if check['passed'] else '❌'} {check['check']} (got {_fmt(check['value'])})")
    for item in gate["regressions"]:
        paired = item.get("paired") or {}
        lines.append(
            f"- {'✅' if item['passed'] else '❌'} {item['metric']}: {_fmt(item['baseline'])} → {_fmt(item['current'])} "
            f"(drop {_fmt(item['drop'])}, allowed {_fmt(item['allowed_drop'])}"
            + (f", paired Δ {_fmt(paired.get('delta'))} CI [{_fmt((paired.get('ci') or {}).get('low'))}, {_fmt((paired.get('ci') or {}).get('high'))}]" if paired else "")
            + ")"
        )
    failures = report.get("failures") or []
    if failures:
        lines += ["", f"## Misses ({len(failures)})", ""]
        for item in failures[:25]:
            lines.append(f"- `{item['id']}` ({item['type']}): {item['why']}")
    if report.get("uncovered"):
        lines += ["", f"Skipped {len(report['uncovered'])} rows whose documents are not in this index."]
    return "\n".join(lines) + "\n"


def failures(records: List[Dict[str, Any]], *, generate: bool) -> List[Dict[str, str]]:
    found = []
    for record in records:
        scores = record.get("scores") or {}
        why = None
        if record.get("infra_error"):
            why = f"infra error: {record.get('reject_category')}"
        elif record.get("leaked_phrases"):
            why = f"leaked injected text: {record['leaked_phrases']}"
        elif not record["answerable"] and generate and record.get("released") and not record.get("declined"):
            why = "answered a question the documents cannot answer"
        elif record["answerable"] and scores.get("hit_at_5") == 0.0:
            why = "expected passage not in the top 5" + (f"; top hit {record['ranked'][0]['document']}" if record.get("ranked") else "")
        elif generate and record["answerable"] and "injection" not in record["tags"] and record.get("abstained"):
            why = f"withheld ({record.get('reject_category')}): {record.get('reason') or ''}".strip()
        elif generate and scores.get("key_fact_recall") is not None and scores["key_fact_recall"] < 1.0:
            why = "answer is missing key facts"
        if why:
            found.append({"id": record["id"], "type": record.get("type"), "why": why})
    return found


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
