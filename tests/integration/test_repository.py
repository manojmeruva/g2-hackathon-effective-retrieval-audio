"""PostgresSegmentRepository against a real PostgreSQL + pgvector database.

Uses a separate database (TEST_DATABASE_URL) so tests never touch indexed data.
"""

import math
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, make_url, text
from sqlalchemy.exc import OperationalError

from app.domain.models import AudioTranscript, Chunk
from app.infrastructure.postgres import create_schema
from app.infrastructure.repositories import PostgresSegmentRepository
from app.infrastructure.schema import define_schema

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://audio:audio@localhost:5432/audio_retrieval_test",
)
DIMENSIONS = 3


def unit(*values: float) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def ensure_database(url: str) -> None:
    target = make_url(url)
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": target.database}
        ).scalar()
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{target.database}"'))
    admin.dispose()


@pytest.fixture
def engine() -> Iterator[Engine]:
    try:
        ensure_database(TEST_DATABASE_URL)
    except OperationalError:
        pytest.skip("PostgreSQL is not available; run `docker compose up -d`")
    engine = create_engine(TEST_DATABASE_URL)
    schema = define_schema(DIMENSIONS)
    schema.metadata.drop_all(engine)
    yield engine
    schema.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def repository(engine: Engine) -> PostgresSegmentRepository:
    schema = define_schema(DIMENSIONS)
    create_schema(engine, schema)
    return PostgresSegmentRepository(engine, schema)


def transcript(audio_id: str, texts: list[str], speakers: list[str]) -> AudioTranscript:
    chunks = tuple(
        Chunk(index, speaker, index * 10.0, index * 10.0 + 9.0, chunk_text)
        for index, (chunk_text, speaker) in enumerate(zip(texts, speakers, strict=True))
    )
    embeddings = tuple(unit(1.0, float(index), 0.5) for index in range(len(chunks)))
    return AudioTranscript(audio_id, f"{audio_id}.wav", 60.0, chunks, embeddings)


CLOUD = transcript(
    "conversation_03",
    [
        "Welcome to the show about infrastructure.",
        "We run PostgreSQL and moved it onto Kubernetes.",
        "The migration cut our monthly bill by eighteen percent.",
        "Thanks for listening.",
    ],
    ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_00"],
)


def test_keyword_search_matches_stemmed_terms_and_ranks(
    repository: PostgresSegmentRepository,
) -> None:
    repository.replace_audio(CLOUD)

    results = repository.keyword_search(["postgresql", "migrations"], limit=10)

    assert {r.segment.segment_index for r in results} == {1, 2}
    assert results[0].score >= results[1].score > 0
    assert results[0].segment.speaker == "SPEAKER_01"


def test_keyword_search_with_only_stop_words_finds_nothing(
    repository: PostgresSegmentRepository,
) -> None:
    repository.replace_audio(CLOUD)

    assert repository.keyword_search(["what", "did", "they"], limit=10) == []


def test_keyword_search_rejects_unsafe_terms(repository: PostgresSegmentRepository) -> None:
    with pytest.raises(ValueError):
        repository.keyword_search(["cloud | !x"], limit=10)


def test_vector_search_orders_by_cosine_similarity(
    repository: PostgresSegmentRepository,
) -> None:
    repository.replace_audio(CLOUD)

    results = repository.vector_search(unit(1.0, 2.0, 0.5), limit=3)

    assert results[0].segment.segment_index == 2
    assert results[0].score == pytest.approx(1.0)
    assert results[0].segment.start_seconds == 20.0
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_neighbors_returns_surrounding_segments_in_order(
    repository: PostgresSegmentRepository,
) -> None:
    repository.replace_audio(CLOUD)
    middle = repository.vector_search(unit(1.0, 1.0, 0.5), limit=1)[0].segment

    context = repository.neighbors(middle, window=1)

    assert [segment.segment_index for segment in context] == [0, 1, 2]


def test_replace_audio_is_idempotent_and_isolated_per_file(
    repository: PostgresSegmentRepository, engine: Engine
) -> None:
    other = transcript("conversation_01", ["We cut hosting costs."], ["SPEAKER_00"])
    repository.replace_audio(CLOUD)
    repository.replace_audio(other)
    repository.replace_audio(CLOUD)

    with engine.connect() as connection:
        counts: dict[str, int] = dict(
            connection.execute(
                text("SELECT audio_id, count(*) FROM transcript_segments GROUP BY audio_id")
            ).all()
        )
        speakers = connection.execute(text("SELECT count(*) FROM speakers")).scalar()
    assert counts == {"conversation_03": 4, "conversation_01": 1}
    assert speakers == 3


def test_vector_query_uses_hnsw_index(
    repository: PostgresSegmentRepository, engine: Engine
) -> None:
    repository.replace_audio(CLOUD)

    with engine.begin() as connection:
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        plan = connection.execute(
            text(
                "EXPLAIN SELECT id FROM transcript_segments "
                "ORDER BY embedding <=> '[1,0,0]' LIMIT 5"
            )
        ).all()
    assert "ix_segments_embedding_hnsw" in " ".join(row[0] for row in plan)
