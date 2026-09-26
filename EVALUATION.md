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
- the best hybrid configuration, with cross-encoder reranking on top

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

Measured on the 48 golden queries. The full report, including per-query results, is in [`reports/evaluation.md`](reports/evaluation.md).

| Configuration | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | MRR | NDCG@10 | p95 latency |
|---|---|---|---|---|---|---|---|
| Keyword | 0.314 | 0.482 | 0.585 | 0.715 | 0.625 | 0.588 | 11 ms |
| Semantic | 0.413 | 0.613 | 0.666 | 0.824 | 0.780 | 0.712 | 112 ms |
| Hybrid, weighted α = 0.2 | 0.465 | 0.628 | 0.712 | 0.849 | 0.830 | 0.751 | 31 ms |
| Hybrid, weighted α = 0.4 | 0.472 | 0.656 | 0.719 | 0.842 | 0.838 | 0.754 | 32 ms |
| Hybrid, weighted α = 0.5 | 0.493 | 0.639 | 0.712 | 0.835 | 0.844 | 0.757 | 31 ms |
| Hybrid, weighted α = 0.6 | 0.483 | 0.628 | 0.691 | 0.835 | 0.832 | 0.749 | 31 ms |
| Hybrid, weighted α = 0.8 | 0.441 | 0.628 | 0.691 | 0.824 | 0.804 | 0.726 | 31 ms |
| Hybrid, RRF | 0.413 | 0.639 | 0.733 | 0.800 | 0.773 | 0.710 | 32 ms |
| **Hybrid, RRF + rerank** | 0.477 | 0.686 | **0.734** | **0.859** | 0.824 | **0.777** | 133 ms |

Recall@5 by query type:

| Query type | Keyword | Semantic | Weighted α = 0.4 | RRF | RRF + rerank |
|---|---|---|---|---|---|
| Exact keyword | 0.88 | 0.88 | 0.88 | 0.88 | 0.92 |
| Multi-word | 0.93 | 0.91 | 1.00 | 1.00 | 1.00 |
| Synonym | 0.29 | 0.54 | 0.54 | 0.54 | 0.54 |
| Paraphrase | 0.22 | 0.58 | 0.67 | 0.61 | 0.61 |
| Semantic | 0.58 | 0.72 | 0.72 | 0.72 | 0.78 |
| Speaker-specific | 0.50 | 0.25 | 0.33 | 0.50 | 0.50 |
| Numeric | 0.67 | 0.92 | 0.92 | 0.92 | 0.83 |
| Conversational | 0.61 | 0.53 | 0.69 | 0.69 | 0.69 |

**What the numbers say**

- **Hybrid beats either method alone.** Recall@5 rises from 0.585 (keyword) and 0.666 (semantic) to 0.72–0.73 (hybrid). Keyword search wins on exact terms and names; semantic search wins on synonyms, paraphrases and numbers. Combining them keeps most of both strengths.
- **α matters, but only moderately.** α = 0.4 is the best weighted setting. Leaning heavily on keywords (α ≥ 0.6) hurts paraphrase queries most, dropping them from 0.67 to 0.44.
- **RRF and weighted fusion are practically tied.** RRF has the highest Recall@5, but weighted α = 0.4–0.5 has a clearly better MRR and Recall@1. With 48 queries, one query moves Recall@5 by about 0.02, so a 0.014 difference is within noise.
- **Reranking improves the ordering, not the Recall@5.** Recall@5 barely moves, while Recall@3 (+0.05), Recall@10 (+0.06) and NDCG@10 (+0.07) all improve. It costs roughly 100 ms per query on CPU and slightly hurts numeric queries.
- **The default is RRF + rerank.** It has the highest Recall@5, the primary metric, and the best Recall@10 and NDCG. If latency matters more than ranking quality, `--mode hybrid` (RRF, about 32 ms) gives almost the same Recall@5.
- **The weakest query type is speaker-specific (0.50).** Speakers are anonymous (`SPEAKER_00`), so "What did Sarah propose?" only matches segments where someone says "Sarah" aloud. See Limitations.

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
