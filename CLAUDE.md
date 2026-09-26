# Engineering Rules

## Goal

Build a production-quality audio retrieval system.

The code must prioritize:

1. readability
2. correctness
3. testability
4. simple architecture
5. measurable retrieval quality


## SOLID

Apply SOLID principles pragmatically.

Use interfaces/Protocols only where they provide a real
seam for testing or replacing an implementation.

Do NOT create abstractions simply to demonstrate SOLID.

## Functions

- Functions should have one clear responsibility.
- Prefer functions under ~30 lines.
- Avoid functions over ~50 lines.
- Extract meaningful operations into named functions.
- Avoid deeply nested conditionals.
- Use early returns where appropriate.

## Classes

- Classes should have one responsibility.
- Prefer composition over inheritance.
- Avoid god classes.
- Avoid inheritance unless there is a genuine
  substitutable relationship.

## Dependencies

Prefer simple, mature libraries.

Do not introduce a framework/library when Python,
PostgreSQL, or an existing dependency can solve the problem
clearly.

Before adding a dependency, explain:
- why it is required
- what problem it solves
- why the standard library/current dependency is insufficient

# Tech Stack

## Language & Runtime
- Python 3.12+
- Type hints everywhere
- Prefer dataclasses/Pydantic models for structured data
- No unnecessary metaprogramming
- No global mutable state
- No magic constants
- Prefer explicit dependencies

## Audio Processing

### Transcription
- `faster-whisper`
- Use for local speech-to-text transcription.
- Preserve word/segment timestamps.

### Speaker Diarization
- `pyannote.audio`
- Use for identifying and separating the two speakers.
- Every transcript segment should retain speaker identity.

## Text Embeddings
- `sentence-transformers`
- Use local embedding models.
- Embeddings must be generated locally; do not depend on hosted embedding APIs.

## Database & Search
- PostgreSQL
- `pgvector` extension for vector similarity search.
- PostgreSQL Full-Text Search (FTS) for keyword search.
- Use PostgreSQL as the single source of truth for transcript segments and embeddings.

### Vector Index
- Use HNSW index through pgvector.

### Database Access
- `SQLAlchemy 2.x`
- Use SQLAlchemy for database access and schema management.
- Keep database-specific code inside the infrastructure/repository layer.

## Data Validation
- `Pydantic v2`
- Use Pydantic models for configuration and external/input data validation.
- Use Python `dataclasses` for simple internal domain objects where appropriate.

## CLI
- `Typer`
- Use a small CLI for ingestion, search, and evaluation.
- Do NOT introduce FastAPI unless an HTTP API is genuinely required.

Example commands:

    python -m app.cli ingest ./data/audio
    python -m app.cli search "What did they say about reducing infrastructure costs?"
    python -m app.cli evaluate

## Architecture

Use a simple layered architecture:

    CLI
      ↓
    Application Services
      ↓
    Domain
      ↓
    Infrastructure
      ↓
    PostgreSQL / pgvector

Keep business logic independent of  PostgreSQL and 
where practical.

## Retrieval

Keep these components independently testable:

- KeywordRetriever
- SemanticRetriever
- HybridRanker
- SearchService


## Code quality

Use:
- Ruff
- Pytest
- Pydantic
- type checking

No unused code.

No commented-out implementations.

No unnecessary TODOs.

## Agent behavior

Before implementing a significant feature:

1. inspect the existing architecture
2. explain the proposed change
3. identify affected files
4. implement the smallest clean solution
5. run tests
6. run lint/type checks
7. report what changed

Do not rewrite working code unnecessarily.