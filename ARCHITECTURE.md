# Architecture

## How it works

```
audio file
   │
   ├─ faster-whisper (small.en) ─────────────────────── words with timestamps
   ├─ pyannote (speaker-diarization-community-1) ────── who spoke when
   │
   ▼
speaker-labelled words ──► chunks of up to 40 s (one speaker each, 5 s overlap)
                                │
                                ▼
               BAAI/bge-base-en-v1.5 embeddings (768-dim, local)
                                │
                                ▼
              PostgreSQL: full-text index  +  pgvector HNSW index
```

At search time:

```
query ─┬─ keyword search (PostgreSQL full-text) ─────────── top 50 ─┐
       └─ semantic search (bge-base query → pgvector HNSW) ── top 50 ─┤
                                                                     ▼
          fusion: weighted α = 0.2 (default) or reciprocal rank fusion
                                                                     ▼
   optional: cross-encoder reranking of the top 30 (ms-marco-MiniLM-L-6-v2)
                                                                     ▼
             file · start–end time · speaker · text · all scores · context
```

Every result includes the file, the timestamps, the speaker, the text, the keyword, semantic and hybrid scores (plus the rerank score when reranking is on), and the lines spoken just before and after it.

## Design decisions

- **Short chunks, one speaker each.** One embedding per 10-minute file would make retrieval useless. A chunk closes at the first sentence end after 30 s and never exceeds 40 s. Because a chunk never mixes speakers, every result has one speaker and a precise timestamp. A 5 s overlap keeps ideas that cross a chunk boundary findable. In practice most speaking turns are shorter than 30 s, so the average chunk is about 12.6 s, roughly one turn (226 chunks across the six files).
- **PostgreSQL is the single store.** Segments, embeddings, the full-text index and the vector index all live in one database. There is no separate search engine to keep in sync.
- **HNSW vector index.** Search uses pgvector's approximate nearest-neighbour index instead of comparing the query against every row.
- **Score normalization before fusion.** Keyword and semantic scores are on different scales, so each list is rescaled to 0–1 before the α-weighted sum. RRF avoids the scale problem by using only ranks, and the evaluation decides which works better.
- **Reranking is optional.** The cross-encoder (`ms-marco-MiniLM-L-6-v2`) is too slow to run over everything, so it re-sorts only the top 30 hybrid results. It improved the top-1 result but lowered Recall@5, so it is off by default. See [How the cross-encoder reranker works](README.md#how-the-cross-encoder-reranker-works).
- **Context expansion.** Each result includes the segment before and after it, so you see the question as well as the answer.
