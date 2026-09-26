# Architectural requirements supplied to the coding agent
- `CLAUDE.md`: layered architecture, pragmatic SOLID, and the fixed stack (faster-whisper, pyannote, sentence-transformers, PostgreSQL + pgvector HNSW + full-text search, SQLAlchemy, Pydantic, Typer).
- `EVALUATION.md`: Recall@1/3/5/10, no hardcoded results, focused tests for every component.
- Written brief: short chunks, α-weighted hybrid search with an α sweep, reranking experiment, 40–60 golden queries, context expansion, local embeddings.

# Prompts used
- Check the folder structure and scaffold it; add the `data/` layout.
- Create 6 synthetic two-speaker conversations with controlled vocabulary.
- Implement `transcription.py`, then "complete all requirements" (planned in Claude Code plan mode).
- "Is RRF better for our use case?", then "implement"; "What exactly does the cross-encoder do?"
- Write the README, move evaluation into `EVALUATION.md`, fill `tests/evaluation/`, write this file.

# What the agent generated
- The implementation plan, all source code, the database schema, the CLI, the HTML results page and `docker-compose.yml`.
- Dialogue scripts, text-to-speech audio, ground-truth turn timings and the 48 golden query labels.
- All tests (unit, integration, evaluation), the evaluation report, and the README / `EVALUATION.md` text.

# What was reviewed/changed manually
- I have reviewed and approved the plan before implementation started.
- I have asked for the audio generator to be removed from the repo; only the dataset was kept.
- I have questioned the design (RRF versus weighted fusion, anonymous speaker labels), restructured the docs, and spotted the empty `tests/evaluation/` folder.

# Tests written 
- I have defined what the tests must cover and the edge cases (keyword retrieval, semantic retrieval, hybrid ranking, Recall@K, edge cases) and the dataset constraints (5–6 files, 8–10 min, 2 speakers, unique pairs). The agent wrote the test code to that specification.


