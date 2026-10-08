# Architecture review

Written after the pipeline unification, eval rebuild, and security pass. It says what to keep,
what to add, and what to leave out, with the evidence behind each call.

## Verdict

The retrieval architecture is sound and should not change. Hybrid search, a reranker that
decides what counts as evidence, parent-child chunks, and a two-stage answer check is the right
shape, and the hermetic eval backs it: recall@5 is 1.00 on the repo corpus and 0.985 on your
documents. The weak points were around the pipeline, not in it: no authentication, a monolithic
engine module, two diverging copies of the question path, and a metric system that could not
detect regressions. Those are fixed.

What is left is a short list of things that matter once more than one person, or more than a few
thousand passages, are involved.

## Keep

- **Hybrid retrieval with RRF.** Vector, BM25, and an exact-phrase list for codes such as E-104.
  The phrase list took exact-code MRR from 0.86 to 1.00.
- **Jev as an evidence filter, not just a ranker.** Dropping passages below a relevance floor is
  what makes "I don't know" reliable (abstention recall 0.95).
- **Parent-child chunking.** Small passages find the evidence, large ones give the writer context.
- **Deterministic checks before the checker model.** Citations, quotations, and numbers are
  verified without a model call, so the failure is cheap and explainable.
- **Versioned indexes with a recall gate.** Refresh builds a new collection, compares recall on
  the golden set, and publishes or refuses. Rollback is one click.
- **One pipeline for chat and eval.** Never fork it again; add options to `PipelineOptions`.

## Remove

Already removed in this round: the research harness (phase scripts and baselines), the members
and region/timezone settings (they did nothing server-side), the Gemini scripts and dependency,
the second retrieval implementation, the routing embedder, and dead helpers.

Still worth removing if unused by you:
- **`Index card summary` and `Fixed window` chunking.** Only Parent-child is evaluated. Either
  add them to the hermetic suite and compare, or drop them from Settings.
- **`jev_soft` and `fused_only` rerank modes.** `fused_only` is used by the offline eval and the
  refresh gate; `jev_soft` is not evaluated in production. Keep it only if you intend to test it.

## Add, in priority order

1. **Move the vector store to something with real concurrency before multi-user.** Chroma
   embedded plus SQLite works for one user and a few thousand documents. If several people
   upload at once, writes serialise behind one lock. The upgrade path is Postgres with pgvector
   (one store for vectors, keyword search, parents, and workspace data; this was the original
   "Phase Postgres" intent behind the `baseline-pre-pg` tag). Do this when you have a second
   real user, not before: the eval gives you the safety net to verify the migration.
2. **Per-user identity** if more than one person will sign in. Today there is one shared token.
   Put it behind an SSO proxy, or add accounts, before storing anything sensitive per user.
3. **Streaming the answer.** The writer and checker run to completion before any text appears
   (median 15 s). Streaming the draft while the checker runs, then retracting on failure, would
   cut perceived latency, but it changes the guarantee that nothing ungrounded is ever shown.
   Prefer reducing latency instead: Jev scoring is the biggest slice, and caching it per
   (query, passage) already helps repeat questions.
4. **Observability.** Structured logs with a request id, and counters for Jev/writer failures
   and circuit-breaker state. The Pipeline page shows some of this; an external sink would make
   alerts possible.
5. **Backups.** `backend/` data is one directory (`NEEDLE_DATA_DIR`). A nightly copy of
   `chroma_store/`, `workspace.db`, and `uploads/` is enough. Needle does not do this for you.
6. **Table-aware and scanned-PDF ingestion**, the two limitations the README already lists.
   Add a CSV/table question set to the hermetic suite first so you can see the improvement.

## Do not add yet

- A message queue or worker fleet. One ingestion thread handles the load; the lock around index
  writes would serialise a fleet anyway.
- A larger or fine-tuned embedder. See `docs/embedding-review.md`.
- Agentic multi-step retrieval. Multi-hop questions already reach recall@5 = 1.0 here; add it
  only when the eval shows multi-hop failures.
