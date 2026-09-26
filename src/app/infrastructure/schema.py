"""SQLAlchemy table definitions, HNSW and full-text indexes."""

from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR

TEXT_SEARCH_CONFIG = "english"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


@dataclass(frozen=True)
class Schema:
    metadata: MetaData
    audio_files: Table
    speakers: Table
    segments: Table


def define_schema(embedding_dimensions: int) -> Schema:
    metadata = MetaData()
    audio_files = Table(
        "audio_files",
        metadata,
        Column("id", Text, primary_key=True),
        Column("filename", Text, nullable=False),
        Column("duration_seconds", Float, nullable=False),
        Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    speakers = Table(
        "speakers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("audio_id", Text, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False),
        Column("label", Text, nullable=False),
        UniqueConstraint("audio_id", "label"),
    )
    segments = Table(
        "transcript_segments",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("audio_id", Text, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False),
        Column(
            "speaker_id", Integer, ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False
        ),
        Column("segment_index", Integer, nullable=False),
        Column("start_time", Float, nullable=False),
        Column("end_time", Float, nullable=False),
        Column("text", Text, nullable=False),
        Column("embedding", Vector(embedding_dimensions), nullable=False),
        Column(
            "text_search",
            TSVECTOR,
            Computed(f"to_tsvector('{TEXT_SEARCH_CONFIG}', text)", persisted=True),
        ),
        UniqueConstraint("audio_id", "segment_index"),
        Index("ix_segments_audio_start", "audio_id", "start_time"),
        Index("ix_segments_text_search", "text_search", postgresql_using="gin"),
        Index(
            "ix_segments_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    return Schema(metadata, audio_files, speakers, segments)
