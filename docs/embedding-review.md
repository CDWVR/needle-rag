# Embedding model review

**Recommendation: keep `all-MiniLM-L6-v2` as the default.** BGE-M3 is a supported option
(`EMBED_BACKEND=openrouter` or `tei`), but on this workload it does not earn its costs.

## What was measured

Offline tier (retrieval only, fused ranking, no model calls), same questions and chunking, only
the embedder changed. Indexes were built and queried by each model.

| Suite | Metric | MiniLM (384-d, local) | BGE-M3 (1024-d, OpenRouter) |
| --- | --- | --- | --- |
| Hermetic, 117 questions | hit@1 | 0.865 | 0.865 |
| | hit@3 | 0.979 | 0.990 |
| | hit@5 | 1.000 | 1.000 |
| | MRR | 0.924 | 0.924 |
| Your documents, 87 questions | hit@1 | 0.761 | 0.821 |
| | hit@5 | 0.985 | 0.985 |
| | MRR | 0.853 | 0.874 |
| | exact-code MRR (8) | 0.552 | 0.792 |
| | follow-up MRR (9) | 0.750 | 0.676 |

Both models put the evidence in the top five for essentially every answerable question. The
differences are one or two questions in either direction (the workspace suite has only 8
exact-code and 9 follow-up rows), so none of them is established. Treat BGE-M3's exact-code
advantage as a hypothesis worth a larger test, not a result.

## Why not switch

- **No measurable end-to-end gain.** Jev reranks the candidates afterwards, and recall@5 is
  already saturated, so a better first stage cannot raise answer quality here.
- **Privacy.** MiniLM runs on your machine. BGE-M3 through OpenRouter sends every passage and
  every query to a third party. TEI (`docker-compose.tei.yml`) keeps it local but needs a
  container and, for good speed, a GPU.
- **Latency.** Query embedding took about 0.5 s total retrieval locally versus about 1.8 s
  through the API (cold cache).
- **Storage.** 1024 dimensions is 2.7x the vector size of 384.
- **Cost.** Small: re-embedding the whole 4,500-passage index is about 0.3 million tokens, well
  under a cent at $0.01 per million. Cost is not the reason.

## When to revisit

- Your documents are mostly **not English**. MiniLM is English-centric; BGE-M3 is multilingual.
  This is the one clear case for switching.
- The workspace eval grows past a few hundred rows and shows **exact-code or paraphrase misses**
  at the first stage that Jev cannot recover (look at `deep_match_rank` in the results file).
- Passages get **longer**. MiniLM truncates around 256 word pieces; BGE-M3 handles 8k tokens.

## How to switch safely

1. Set `EMBED_BACKEND=openrouter` (or `tei`) and restart. The active index is now incompatible;
   searches and uploads are refused with a clear message instead of returning wrong results.
2. Run **Refresh index**. The new version is embedded with the new model, compared with the old
   one on the golden questions (each side is queried with its own model), and published only if
   recall did not drop. Rollback restores the old version, whose model is recorded with it.
3. Run `python -m eval run --suite workspace --tier offline` and compare with the baseline.

To compare models without touching your index:
`python -m eval run --suite hermetic --tier offline --embed-backend openrouter --no-gate`.
