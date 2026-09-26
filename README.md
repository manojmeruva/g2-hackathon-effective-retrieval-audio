# Audio Retrieval

Search spoken conversations by **keyword** or by **meaning**, and jump straight to the moment it was said.

```
$ python -m app.cli search "What did they say about reducing infrastructure costs?"

1. conversation_03  06:26–06:42  SPEAKER_01  [conversation_03:26]
   keyword 0.00 · semantic 1.00 · hybrid 0.80
   · 06:23 SPEAKER_00: Where did we end up?
   ▶ 06:26 SPEAKER_01: Today our monthly infrastructure spend is about 18 % lower than the
     data center was and that's before counting the hardware refresh we no longer have to buy. ...
   · 06:43 SPEAKER_00: I think that's the most important lesson. The cloud doesn't make
     things cheaper by itself. ...

2. conversation_03  05:54–06:22  SPEAKER_01  [conversation_03:24]
   keyword 0.00 · semantic 0.82 · hybrid 0.66
   · 05:51 SPEAKER_00: And then what changed?
   ▶ 05:54 SPEAKER_01: We did a focused cost optimization project. We right -sized the
     instances based on actual usage, bought reserved instances for the baseline capacity ...
   · 06:23 SPEAKER_00: Where did we end up?
```

Output from a real run (top 2 of 3 shown, long lines shortened). Each result shows the file, time range, speaker, scores, and the lines spoken just before and after it (`▶` marks the match). Add `--html results.html` to get a page with a **▶ Jump to 06:26** audio player for each result.

Everything runs locally: transcription, speaker detection, embeddings and search. No hosted AI APIs are used.

---

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

---

## Models

All models run locally on CPU. They are downloaded once from Hugging Face and then cached.

| Step | Model | Settings |
|---|---|---|
| Transcription | `faster-whisper` **small.en** | int8, beam size 5, voice-activity filter, word timestamps |
| Speaker diarization | **pyannote/speaker-diarization-community-1** | exactly 2 speakers, non-overlapping turns (needs `HF_TOKEN`) |
| Embeddings | **BAAI/bge-base-en-v1.5** (sentence-transformers) | 768 dimensions, L2-normalized, cosine similarity |
| Reranking (optional) | **cross-encoder/ms-marco-MiniLM-L-6-v2** (sentence-transformers) | scores the top 30 hybrid candidates |

`bge-base` was chosen after comparing `bge-small`, `bge-base` and `bge-large` on the golden queries (see [EVALUATION.md](EVALUATION.md#embedding-model-comparison)). Models can be swapped in [`src/app/config.py`](src/app/config.py); after changing the embedding model, rebuild the index with `init-db --reset` and `ingest`.

### How the cross-encoder reranker works

- **Semantic search** embeds the query and each segment *separately* (a bi-encoder). Segment vectors are computed once at ingest, which makes search fast over the whole index.
- **The cross-encoder** reads the query and one segment *together* as a single input, `[query] [SEP] [segment]`, so every query word can attend to every segment word. It outputs one relevance score. This is more accurate, but nothing can be precomputed, so it runs only on a shortlist.
- **The model** is a 6-layer MiniLM trained on MS MARCO, a large dataset of real search questions paired with the passages that answer them.
- **In this project:**
  1. Hybrid search produces the top 30 candidates.
  2. The cross-encoder scores each `(query, segment text)` pair.
  3. The results are re-sorted by that score alone.
- **The rerank score** is the model's raw output, not a 0–1 value: higher means more relevant, and negative usually means a weak match. It is only meaningful for comparing results within the same query. It is shown next to the keyword, semantic and hybrid scores.
- **Limits:**
  - It can only reorder the 30 candidates; it cannot find a segment hybrid search missed.
  - It sees only text, not who is speaking.
  - It adds about 100 ms per search on CPU.
- **Measured effect:** it puts the best answer first more often (Recall@1 0.387 → 0.477, MRR 0.784 → 0.824). But it lowers Recall@5 (0.781 → 0.734) because it pushes down synonym and paraphrase matches, such as "chatbot making things up" → "hallucinations". It is therefore **off by default** and available with `--mode hybrid_rerank`. See [EVALUATION.md](EVALUATION.md).

---

## Quick start

**Requirements:** Python 3.12, Docker, and a free [Hugging Face](https://huggingface.co) account.

```bash
# 1. Install
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

# 2. Start PostgreSQL with pgvector
docker compose up -d

# 3. Allow speaker detection
#    Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1
#    Then create a token at https://huggingface.co/settings/tokens and add it to .env
echo "HF_TOKEN=hf_..." > .env

# 4. Index the audio (the first run downloads models and takes a while)
.venv/bin/python -m app.cli ingest ./data/audio

# 5. Search
.venv/bin/python -m app.cli search "Who disagreed about the VPN timeline?"

# 6. Measure retrieval quality
.venv/bin/python -m app.cli evaluate
```

Transcripts are cached in `data/transcripts/`, so re-running `ingest` only redoes the fast steps. Use `--force` to transcribe again from scratch.

---

## Searching

```bash
python -m app.cli search "QUERY" [OPTIONS]
```

| Option | What it does |
|---|---|
| `--mode keyword` | Exact words only (PostgreSQL full-text search) |
| `--mode semantic` | Meaning only (vector similarity) |
| `--mode hybrid` | Both, fused into one ranking (default) |
| `--mode hybrid_rerank` | Hybrid, then re-sorted by a cross-encoder |
| `--fusion weighted` / `rrf` | How hybrid modes combine the two lists (default `weighted`) |
| `--alpha 0.2` | Keyword weight for weighted fusion (0 = meaning only, 1 = keywords only; default 0.2) |
| `--top-k 5` | Number of results |
| `--html results.html` | Also write a page with an audio player and a **▶ Jump to 05:42** button per result |

---

## The dataset

Six synthetic podcast-style conversations. Each has two speakers, and no speaker appears in more than one conversation. They were written with deliberately controlled vocabulary so that retrieval can be measured:

| File | Speakers | Topic | Length |
|---|---|---|---|
| `conversation_01.wav` | Alice, Bob | Startup funding | 8.2 min |
| `conversation_02.wav` | Sarah, David | Cybersecurity | 8.1 min |
| `conversation_03.wav` | Priya, James | Cloud migration | 8.6 min |
| `conversation_04.wav` | Emma, Daniel | Hiring | 8.6 min |
| `conversation_05.wav` | Lisa, Mark | AI adoption | 9.2 min |
| `conversation_06.wav` | Nina, Alex | Product strategy | 9.1 min |

The conversations contain:
- exact terms, such as *PostgreSQL*, *zero trust* and *Kubernetes*
- paraphrases, such as *"cloud migration"* and *"moved our infrastructure to the cloud"*
- concepts that never use the obvious words, such as cutting a hosting bill without saying "reduce costs"
- statements tied to a speaker, such as *"Sarah proposed…"* and *"David disagreed…"*
- references back to earlier points, such as *"the deadline we discussed earlier"*
- decoys: the same term used in a different context in another conversation

`data/ground_truth/` holds the exact text, speaker and start/end time of every turn.

### Golden queries

`data/golden_queries.json` holds **48 labelled queries**, 6 of each type:

| Type | Example |
|---|---|
| Exact keyword | `Terraform` |
| Multi-word | `zero trust architecture` |
| Synonym | `customer attrition` (the audio says "churn") |
| Paraphrase | `How did they reduce cloud costs?` |
| Semantic | `systems that must keep working without internet` |
| Speaker-specific | `What did Sarah propose?` |
| Numeric | `What budget did they mention for the security keys?` |
| Conversational | `the lease deadline we talked about before` |

Each query lists the turns that answer it, e.g. `conversation_03:turn_07`. A search result counts as a hit when it overlaps one of those turns in time. The labels stay valid even if chunking or transcription changes.

---

## Evaluation

```bash
python -m app.cli evaluate
```

Method, results and findings are in **[EVALUATION.md](EVALUATION.md)**.

---

## Design decisions

- **Short chunks, one speaker each.** One embedding per 10-minute file would make retrieval useless. A chunk closes at the first sentence end after 30 s and never exceeds 40 s. Because a chunk never mixes speakers, every result has one speaker and a precise timestamp. A 5 s overlap keeps ideas that cross a chunk boundary findable. In practice most speaking turns are shorter than 30 s, so the average chunk is about 12.6 s, roughly one turn (226 chunks across the six files).
- **PostgreSQL is the single store.** Segments, embeddings, the full-text index and the vector index all live in one database. There is no separate search engine to keep in sync.
- **HNSW vector index.** Search uses pgvector's approximate nearest-neighbour index instead of comparing the query against every row.
- **Score normalization before fusion.** Keyword and semantic scores are on different scales, so each list is rescaled to 0–1 before the α-weighted sum. RRF avoids the scale problem by using only ranks, and the evaluation decides which works better.
- **Reranking is optional.** The cross-encoder (`ms-marco-MiniLM-L-6-v2`) is too slow to run over everything, so it re-sorts only the top 30 hybrid results. It improved the top-1 result but lowered Recall@5, so it is off by default. See [How the cross-encoder reranker works](#how-the-cross-encoder-reranker-works).
- **Context expansion.** Each result includes the segment before and after it, so you see the question as well as the answer.

## Metrics for production

This project measures **Recall@K**. A production system should also track:

| Area | Metric |
|---|---|
| Ranking quality | MRR, NDCG@K, Precision@K, measured on real user queries |
| Speed | p95 / p99 search latency, indexing throughput (audio hours per hour) |
| Transcription | Word error rate (WER) |
| Speaker detection | Diarization error rate (DER) |
| Cost and scale | Embedding generation cost, storage growth per audio hour, index build time |
| Usage | Click-through, "jump to" usage, queries with no useful result |

---

## Project structure

```
src/app/
├── cli.py                 # ingest, search, evaluate, init-db
├── config.py              # all settings (Pydantic)
├── wiring.py              # builds services from settings
├── domain/                # data models, protocols, errors
├── ingestion/             # audio, transcription, diarization, alignment, chunking, embeddings
├── retrieval/             # keyword, semantic, hybrid / RRF, reranking
├── application/           # ingestion, search and evaluation services
├── evaluation/            # golden set loading, metrics
├── infrastructure/        # PostgreSQL schema and repository
└── presentation/          # terminal output, HTML page, evaluation report
tests/
├── unit/                  # fast, no models or database
├── integration/           # real PostgreSQL and real models
└── evaluation/            # golden dataset checks and retrieval quality on the real index
```

## Development

```bash
.venv/bin/pytest -m "not integration"   # fast unit tests
.venv/bin/pytest                        # everything (needs Docker and HF_TOKEN)
.venv/bin/ruff check src tests
.venv/bin/mypy src tests
```

## Limitations

- The audio is synthetic text-to-speech, which is cleaner than real recordings.
- α, the fusion method and the embedding model are chosen on the same 48 queries they are reported on. A larger held-out query set would give a less optimistic estimate.
- Speakers are labelled `SPEAKER_00` / `SPEAKER_01`, not by name. Diarization tells voices apart but cannot say who they are, so "What did Sarah propose?" only matches segments where the name is spoken aloud.
- Whisper writes numbers as digits ("18 %", "$750,000"). A keyword query that spells a number out ("eighteen percent") will not match it exactly.
- 48 queries is a small evaluation set. A difference of less than 0.02 in Recall@5 is about one query.
