"""Needle eval command line. Run from backend/:

    python -m eval validate                          # dataset lint, no engine, no network (CI)
    python -m eval run --suite hermetic --tier offline   # retrieval gate, free and deterministic (CI)
    python -m eval run --suite hermetic --tier full      # production path, costs OpenRouter credit
    python -m eval run --suite workspace --tier offline  # your own documents, read-only
    python -m eval compare reports/hermetic/latest.json baselines/hermetic-offline.json
    python -m eval promote reports/hermetic/<run-dir>   # make a reviewed run the baseline
    python -m eval draft --count 20 && python -m eval review   # grow the workspace suite

Exit codes: 0 gate passed, 1 gate failed, 2 dataset or setup error, 3 stopped (budget or infra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(EVAL_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from eval import corpus as corpus_module  # noqa: E402  (no engine import)
from eval.schema import load_jsonl, span_in_text, validate_dataset  # noqa: E402

CONFIG_PATH = os.path.join(EVAL_DIR, "config.json")
REPORTS_DIR = os.path.join(EVAL_DIR, "reports")
BASELINES_DIR = os.path.join(EVAL_DIR, "baselines")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: str) -> str:
    # Line endings are normalised so a checkout on Windows and one on Linux fingerprint the same.
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()[:16]


def status_model() -> str:
    from embedder import configured_embedding_model

    return configured_embedding_model()


def _commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EVAL_DIR, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=EVAL_DIR, stderr=subprocess.DEVNULL).decode()
        return bool(out.strip())
    except Exception:
        return None


def _corpus_texts() -> Dict[str, str]:
    """Raw text of each corpus document, keyed by the name it is uploaded under."""
    texts = {}
    for name in sorted(os.listdir(corpus_module.CORPUS_DIR)):
        path = os.path.join(corpus_module.CORPUS_DIR, name)
        if not os.path.isfile(path) or name.startswith((".", "_")):
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        texts[name[: -len(".txt")] if name.endswith(".pdf.txt") else name] = text
    return texts


def _csv_as_table(text: str) -> str:
    # Spans for CSV rows are written against the Markdown table the parser produces.
    import csv
    import io

    rows = list(csv.reader(io.StringIO(text)))
    return "\n".join("| " + " | ".join(cell.strip() for cell in row) + " |" for row in rows)


def cmd_validate(args) -> int:
    config = load_config()
    suites = [args.suite] if args.suite else list(config["suites"])
    status = 0
    for suite in suites:
        path = os.path.join(EVAL_DIR, config["suites"][suite]["dataset"])
        try:
            rows = load_jsonl(path)
        except ValueError as exc:
            print(f"{suite}: {exc}")
            status = 2
            continue
        problems = validate_dataset(rows)
        if suite == "hermetic":
            texts = _corpus_texts()
            for name, text in list(texts.items()):
                if name.endswith(".csv"):
                    texts[name] = _csv_as_table(text)
            for row in rows:
                for item in row.get("expected") or []:
                    name = item.get("document")
                    if name not in texts:
                        problems.append(f"{row['id']}: unknown corpus document {name!r}")
                    elif not span_in_text(texts[name], item.get("answer_span") or ""):
                        problems.append(f"{row['id']}: answer_span not found in {name}: {item.get('answer_span')!r}")
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row.get("type")] = counts.get(row.get("type"), 0) + 1
        print(f"{suite}: {len(rows)} rows {counts} -> {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for problem in problems:
            print(f"  - {problem}")
        if problems:
            status = 2
    return status


def _select(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        rows = [row for row in rows if row["id"] in wanted]
    if args.tags:
        wanted = {item.strip() for item in args.tags.split(",") if item.strip()}
        rows = [row for row in rows if wanted & set(row.get("tags") or [])]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def _preflight_budget(rows: List[Dict[str, Any]], tier: Dict[str, Any], max_cost: float) -> Optional[str]:
    if not tier["generate"] and tier["rerank_mode"] == "fused_only":
        return None
    estimate = len(rows) * float(tier.get("estimated_usd_per_question") or 0.0015)
    print(f"Estimated spend: ${estimate:.4f} for {len(rows)} questions (stop at ${max_cost:.2f}).", file=sys.stderr)
    from llm import openrouter_remaining_budget

    remaining = openrouter_remaining_budget()
    if remaining is not None and estimate > remaining:
        return f"estimated ${estimate:.4f} is more than the ${remaining:.4f} left on the OpenRouter key"
    return None


def cmd_run(args) -> int:
    config = load_config()
    if args.suite not in config["suites"]:
        print(f"Unknown suite {args.suite!r}; choose from {', '.join(config['suites'])}.")
        return 2
    tier = dict(config["tiers"][args.tier])
    dataset_path = os.path.join(EVAL_DIR, config["suites"][args.suite]["dataset"])
    rows = load_jsonl(dataset_path)
    problems = validate_dataset(rows)
    if problems:
        print("Dataset problems (run `python -m eval validate`):\n  " + "\n  ".join(problems[:20]))
        return 2
    rows = _select(rows, args)
    if not rows:
        print("No rows selected.")
        return 2

    # Point the engine at the right data and embedder before it is imported.
    if args.embed_backend:
        os.environ["EMBED_BACKEND"] = args.embed_backend
    temp_dir = None
    if args.suite == "hermetic":
        temp_dir = args.data_dir or corpus_module.fresh_data_dir()
        os.environ["NEEDLE_DATA_DIR"] = temp_dir
    elif args.data_dir:
        os.environ["NEEDLE_DATA_DIR"] = args.data_dir

    try:
        import config as engine_config
        from catalog import index_status
        from rerank import enable_jev_disk_cache, jev_cache_stats, set_reuse_jev_cache
        from eval import report as report_module
        from eval import runner

        if args.suite == "hermetic":
            print(f"Building the hermetic index in {temp_dir} ...", file=sys.stderr)
            built = corpus_module.build_index(temp_dir)
            print(f"  {len(built)} documents, {sum(d['chunks'] for d in built.values())} chunks", file=sys.stderr)
            unreachable = runner.unreachable_spans(rows)
            if unreachable:
                print("Golden spans that no single passage contains (fix the dataset):\n  " + "\n  ".join(unreachable))
                return 2
            uncovered: List[str] = []
        else:
            rows, uncovered = runner.split_covered(rows)
            if uncovered:
                print(f"Skipping {len(uncovered)} rows whose documents are not in this index.", file=sys.stderr)
            if not rows:
                print("None of the workspace golden rows match documents in this index.")
                return 2

        if args.jev_cache != "off":
            enable_jev_disk_cache(True)
            set_reuse_jev_cache(args.jev_cache == "replay")
        if tier["generate"] or tier["rerank_mode"] != "fused_only":
            blocked = _preflight_budget(rows, tier, args.max_cost)
            if blocked:
                print(f"Refusing to start: {blocked}.")
                return 3

        judge = None
        if args.judge:
            if not tier["generate"]:
                print("--judge needs a tier that generates answers.")
                return 2
            from eval.judge import judge as judge_fn

            judge = judge_fn

        started = datetime.now(timezone.utc)
        print(f"Running {len(rows)} questions ({args.suite} / {args.tier}) ...", file=sys.stderr)
        run = runner.run_rows(rows, config, tier, workers=args.workers, max_cost=args.max_cost, judge=judge)
        records = run["records"]
        summary = report_module.aggregate(records, generate=bool(tier["generate"]))

        baseline_path = args.baseline or os.path.join(BASELINES_DIR, f"{args.suite}-{args.tier}.json")
        baseline = None
        if os.path.exists(baseline_path):
            with open(baseline_path, encoding="utf-8") as handle:
                baseline = json.load(handle)
        fingerprint = {
            "embedding": status_model(),
            "dataset_sha256": _sha256(dataset_path),
            "corpus": corpus_module.corpus_fingerprint() if args.suite == "hermetic" else None,
            "selection": {"ids": args.ids, "tags": args.tags, "limit": args.limit},
        }
        comparable = bool(baseline) and baseline.get("fingerprint") == fingerprint
        gates = config["gates"][args.suite][args.tier]
        current_questions = report_module.per_question(records)
        gate = report_module.evaluate_gates(
            summary,
            gates,
            baseline=baseline,
            current_questions=current_questions,
            comparable=comparable,
            deterministic=not tier["generate"] and tier["rerank_mode"] == "fused_only",
        )
        if not run["complete"]:
            gate["passed"] = False
            gate["checks"].append({"check": "run completed", "value": len(records), "passed": False})

        status = index_status()
        report = {
            "suite": args.suite,
            "tier": args.tier,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "commit": _commit(),
            "dirty_worktree": _dirty(),
            "questions": len(records),
            "complete": run["complete"],
            "stopped_for_budget": run["stopped_for_budget"],
            "fingerprint": fingerprint,
            "config": {"k": config["k"], "pipeline": config["pipeline"], "tier": tier},
            "models": {
                "embedding": status["embedding_model"],
                "jev": f"openrouter:{status['jev_model']}" if tier["rerank_mode"] != "fused_only" else None,
                "writer": engine_config.OPENROUTER_MODEL if tier["generate"] else None,
                "checker": engine_config.CHECKER_MODEL if tier["generate"] else None,
                "judge": (records and next((r["judge"].get("model") for r in records if r.get("judge")), None)),
            },
            "jev_cache": {"mode": args.jev_cache, **jev_cache_stats()},
            "metrics": summary,
            "gate": gate,
            "baseline": os.path.relpath(baseline_path, EVAL_DIR) if baseline else None,
            "uncovered": uncovered,
            "failures": report_module.failures(records, generate=bool(tier["generate"])),
        }

        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        run_dir = os.path.join(args.out_dir or REPORTS_DIR, args.suite, f"{stamp}-{args.tier}")
        report_module.write_json(os.path.join(run_dir, "report.json"), report)
        with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        text = report_module.markdown(report)
        with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as handle:
            handle.write(text)
        latest_dir = os.path.join(args.out_dir or REPORTS_DIR, args.suite)
        report_module.write_json(os.path.join(latest_dir, "latest.json"), report)
        with open(os.path.join(latest_dir, "latest.md"), "w", encoding="utf-8") as handle:
            handle.write(text)

        print(text)
        print(f"Report: {os.path.relpath(run_dir)}", file=sys.stderr)
        if args.update_baseline:
            if not run["complete"]:
                print("Not updating the baseline from an incomplete run.", file=sys.stderr)
            else:
                report_module.write_json(baseline_path, _baseline_payload(report, current_questions))
                print(f"Baseline written to {os.path.relpath(baseline_path)}", file=sys.stderr)
        if not run["complete"] or summary["infra"]["share"] > float((gates.get("max") or {}).get("infra.share", 1.0)):
            return 3
        return 0 if (gate["passed"] or args.no_gate) else 1
    finally:
        if temp_dir and not args.keep_data and not args.data_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _baseline_payload(report: Dict[str, Any], per_question: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "suite": report["suite"],
        "tier": report["tier"],
        "recorded_at": report["finished_at"],
        "commit": report.get("commit"),
        "fingerprint": report["fingerprint"],
        "models": report.get("models"),
        "metrics": report["metrics"],
        "per_question": per_question,
    }


def cmd_promote(args) -> int:
    """Make a finished, reviewed run the baseline without paying for another run."""
    from eval import report as report_module

    run_dir = args.run_dir
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("complete"):
        print("That run did not finish; refusing to promote it.")
        return 2
    records = load_jsonl(os.path.join(run_dir, "results.jsonl"))
    path = os.path.join(BASELINES_DIR, f"{report['suite']}-{report['tier']}.json")
    report_module.write_json(path, _baseline_payload(report, report_module.per_question(records)))
    print(f"Baseline written to {os.path.relpath(path)} from {os.path.relpath(run_dir)}")
    return 0


def cmd_draft(args) -> int:
    from eval import author

    return 0 if author.draft(author.default_count(args.count)) else 2


def cmd_review(args) -> int:
    from eval import author

    author.review()
    return 0


def cmd_compare(args) -> int:
    from eval import report as report_module

    with open(args.current, encoding="utf-8") as handle:
        current = json.load(handle)
    with open(args.baseline, encoding="utf-8") as handle:
        baseline = json.load(handle)
    new, old = report_module.flatten(current["metrics"]), report_module.flatten(baseline["metrics"])
    width = max(len(key) for key in new) if new else 10
    for key in sorted(set(new) | set(old)):
        a, b = old.get(key), new.get(key)
        delta = (b - a) if a is not None and b is not None else None
        marker = "" if delta is None or abs(delta) < 1e-9 else (" ▲" if delta > 0 else " ▼")
        print(f"{key:<{width}}  {a if a is not None else '—':>10}  {b if b is not None else '—':>10}{marker}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval", description="Needle evaluation harness")
    parser.add_argument("-v", "--verbose", action="store_true", help="show engine warnings")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="lint datasets (no engine, no network)")
    validate.add_argument("--suite")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run a suite through the production pipeline")
    run.add_argument("--suite", default="hermetic")
    run.add_argument("--tier", choices=("offline", "full"), default="offline")
    run.add_argument("--ids", help="comma-separated row ids")
    run.add_argument("--tags", help="comma-separated tags, e.g. smoke")
    run.add_argument("--limit", type=int)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-cost", type=float, default=float(os.getenv("EVAL_MAX_COST_USD", "1.00")),
                     help="stop starting new questions once this much USD is spent")
    run.add_argument("--jev-cache", choices=("off", "use", "replay"), default="off",
                     help="use = read/write the disk cache; replay = cached scores only, never call Jev")
    run.add_argument("--judge", action="store_true", help="grade answers with an LLM judge (costs extra)")
    run.add_argument("--baseline", help="baseline file to compare against")
    run.add_argument("--update-baseline", action="store_true", help="write this run as the new baseline")
    run.add_argument("--no-gate", action="store_true", help="report gate failures but exit 0")
    run.add_argument("--data-dir", help="data dir to evaluate (hermetic: build here and keep it)")
    run.add_argument("--keep-data", action="store_true", help="keep the throwaway hermetic data dir")
    run.add_argument("--embed-backend", choices=("minilm", "openrouter", "tei"),
                     help="embed with this backend (default: EMBED_BACKEND or minilm); hermetic suite only")
    run.add_argument("--out-dir", help="where to write reports (default eval/reports)")
    run.set_defaults(func=cmd_run)

    draft = sub.add_parser("draft", help="draft workspace golden candidates from your indexed documents")
    draft.add_argument("--count", type=int, default=20)
    draft.set_defaults(func=cmd_draft)

    review = sub.add_parser("review", help="accept drafted candidates into the workspace suite")
    review.set_defaults(func=cmd_review)

    promote = sub.add_parser("promote", help="make a finished run (its report directory) the baseline")
    promote.add_argument("run_dir")
    promote.set_defaults(func=cmd_promote)

    compare = sub.add_parser("compare", help="diff two report or baseline files")
    compare.add_argument("current")
    compare.add_argument("baseline")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    import logging

    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    logging.getLogger("needle").setLevel(logging.WARNING if args.verbose else logging.ERROR)
    return int(args.func(args) or 0)
