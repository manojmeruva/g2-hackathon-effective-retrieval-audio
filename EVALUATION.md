# Evaluation

```bash
python -m app.cli evaluate
```

The queries are in [`data/golden_queries.json`](data/golden_queries.json): 48 queries, 6 of each of 8 types (see [README → Golden queries](README.md#golden-queries)). Each query lists the reference turns that answer it. A search result counts as a hit when it overlaps one of those turns by at least half of the shorter time span.

This runs all 48 queries against every configuration:

- keyword only
- semantic only
- weighted hybrid at α = 0.2, 0.4, 0.5, 0.6 and 0.8
- reciprocal rank fusion (RRF)
- the best hybrid configuration, with cross-encoder reranking on top (`cross-encoder/ms-marco-MiniLM-L-6-v2`, top 30 candidates)

All configurations use the same index: `faster-whisper small.en` transcripts, `pyannote/speaker-diarization-community-1` speaker labels and `BAAI/bge-base-en-v1.5` embeddings. The embedding model itself was chosen by a separate comparison (below).

α isn't picked by hand. The best configuration is the one with the highest measured **Recall@5**, with MRR as the tie-break.

Reported metrics:

| Metric | Meaning |
|---|---|
| **Recall@K** (1, 3, 5, 10) | Share of the relevant turns found in the top K. **Recall@5 is the primary metric.** |
| MRR | How high the first correct result appears |
| NDCG@10 | Ranking quality, rewarding correct results near the top |
| Precision@5 | Share of the top 5 that are relevant |
| p50 / p95 latency | Search time per query |

Results are broken down by query type and written to `reports/evaluation.md` and `reports/evaluation.json`.

## Results

Measured on the 48 golden queries with `bge-base-en-v1.5`. The full report, including per-query results, is in [`reports/evaluation.md`](reports/evaluation.md).

| Configuration | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | MRR | NDCG@10 | p95 latency |
|---|---|---|---|---|---|---|---|
| Keyword | 0.314 | 0.482 | 0.585 | 0.715 | 0.625 | 0.588 | 10 ms |
| Semantic | 0.377 | 0.649 | 0.745 | 0.853 | 0.754 | 0.720 | 47 ms |
| **Hybrid, weighted α = 0.2 (default)** | 0.387 | 0.674 | **0.781** | 0.857 | 0.784 | 0.740 | 47 ms |
| Hybrid, weighted α = 0.4 | 0.429 | 0.674 | 0.764 | 0.847 | 0.811 | 0.750 | 47 ms |
| Hybrid, weighted α = 0.5 | 0.436 | 0.649 | 0.764 | 0.819 | 0.814 | 0.740 | 50 ms |
| Hybrid, weighted α = 0.6 | 0.415 | 0.628 | 0.743 | 0.819 | 0.801 | 0.727 | 65 ms |
| Hybrid, weighted α = 0.8 | 0.405 | 0.628 | 0.722 | 0.809 | 0.782 | 0.711 | 43 ms |
| Hybrid, RRF | 0.366 | 0.628 | 0.715 | 0.835 | 0.754 | 0.709 | 46 ms |
| Hybrid α = 0.2 + rerank | **0.477** | **0.686** | 0.734 | **0.859** | **0.824** | **0.778** | 135 ms |

Recall@5 by query type:

| Query type | Keyword | Semantic | Weighted α = 0.2 | RRF | α = 0.2 + rerank |
|---|---|---|---|---|---|
| Exact keyword | 0.88 | 0.88 | 0.88 | 0.88 | 0.92 |
| Multi-word | 0.93 | 0.85 | 0.97 | 1.00 | 1.00 |
| Synonym | 0.29 | 0.71 | 0.71 | 0.54 | 0.54 |
| Paraphrase | 0.22 | 0.75 | 0.75 | 0.64 | 0.61 |
| Semantic | 0.58 | 0.83 | 0.83 | 0.72 | 0.78 |
| Speaker-specific | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Numeric | 0.67 | 0.83 | 0.83 | 0.75 | 0.83 |
| Conversational | 0.61 | 0.61 | 0.78 | 0.69 | 0.69 |

## Embedding model comparison

Each model was run through the same pipeline: re-embed the cached transcripts, then run the full evaluation.

| Model | Dimensions | Semantic only Recall@5 | Best hybrid (no rerank) | Recall@5 | Recall@10 | MRR | Synonym Recall@5 | p95 | With rerank Recall@5 |
|---|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | 384 | 0.666 | RRF | 0.733 | 0.800 | 0.773 | 0.54 | 32 ms | 0.734 |
| **bge-base-en-v1.5** | 768 | **0.745** | weighted α = 0.2 | **0.781** | **0.857** | 0.784 | 0.71 | 48 ms | 0.734 |
| bge-large-en-v1.5 | 1024 | 0.703 | weighted α = 0.2 | 0.744 | 0.798 | **0.806** | **0.79** | 71 ms | 0.724 |

The `bge-base` row matches the official report above. The `bge-small` and `bge-large` rows come from the same evaluation code, but their full reports are not kept. To reproduce one, set the model in `config.py`, then run `init-db --reset`, `ingest` and `evaluate`.

**What the numbers say**

- **Hybrid beats either method alone.** Recall@5 rises from 0.585 (keyword) and 0.745 (semantic) to 0.781 (hybrid, α = 0.2). Keyword search keeps exact terms and multi-word phrases strong; semantic search carries synonyms, paraphrases and numbers.
- **A better embedding model helped most.** Moving from `bge-small` to `bge-base` raised the best Recall@5 from 0.733 to 0.781. It also fixed "chatbot making things up" → "hallucinations" (synonym Recall@5 0.54 → 0.71). `bge-large` was not better overall: it matched synonyms best but was weaker on speaker queries, and it is slower.
- **With a stronger embedding model, less keyword weight is better.** The best α moved from 0.4 (bge-small) to 0.2 (bge-base). Leaning on keywords (α ≥ 0.6) hurts paraphrase queries most.
- **Weighted fusion beats RRF with bge-base** (0.781 vs 0.715). With bge-small the two were tied. RRF ignores how much stronger the semantic match is, which matters once the embeddings are good.
- **Reranking helps the top result but hurts Recall@5.** The cross-encoder raises Recall@1 (0.387 → 0.477) and MRR (0.784 → 0.824). It lowers Recall@5 (0.781 → 0.734) by pushing down synonym and paraphrase matches, and it triples the latency. With every embedding model it brings Recall@5 back to about 0.73. It is therefore **off by default** (`--mode hybrid_rerank` turns it on).
- **The default is bge-base + weighted hybrid α = 0.2, without reranking.** It has the highest Recall@5 (the primary metric) at 47 ms p95.
- **The remaining complete misses are not embedding problems.**
  - **"Why did David disagree with Sarah's plan?" (sp02) and "What did Bob suggest Alice should do?" (sp04):** speakers are anonymous, so these match segments where the names are said aloud instead.
  - **"How large is the Series A and at what valuation?" (nu05):** involves numbers, which Whisper writes as digits.

Ingestion quality, checked against the ground truth:
- **Speaker labels:** after mapping each anonymous speaker to the matching person, 99.9–100 % of words had the correct speaker in all six files.
- **Transcription:** Whisper produced 1,248–1,317 words per file, against 1,242–1,318 in the scripts.

---

## Requirements

### Evaluation

Retrieval quality must be measurable.

Support:

- Recall@1
- Recall@3
- Recall@5
- Recall@10

Do not hardcode evaluation results.

### Testing

Every important component must have focused tests.

Tests should verify:
- keyword retrieval
- semantic retrieval
- hybrid ranking
- Recall@K
- edge cases
