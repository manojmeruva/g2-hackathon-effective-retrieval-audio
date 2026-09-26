"""PostgreSQL repository for storing and searching transcript segments."""

import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    ColumnElement,
    Connection,
    Engine,
    Row,
    Select,
    delete,
    func,
    insert,
    select,
    text,
)

from app.domain.models import AudioTranscript, ScoredSegment, TranscriptSegment
from app.infrastructure.schema import TEXT_SEARCH_CONFIG, Schema

SAFE_TERM = re.compile(r"^[a-z0-9]+$")
MIN_HNSW_EF_SEARCH = 100


class PostgresSegmentRepository:
    def __init__(self, engine: Engine, schema: Schema) -> None:
        self._engine = engine
        self._schema = schema

    def replace_audio(self, transcript: AudioTranscript) -> None:
        """Store one audio file's segments, replacing any earlier ingestion of it."""
        with self._engine.begin() as connection:
            connection.execute(
                delete(self._schema.audio_files).where(
                    self._schema.audio_files.c.id == transcript.audio_id
                )
            )
            connection.execute(
                insert(self._schema.audio_files).values(
                    id=transcript.audio_id,
                    filename=transcript.filename,
                    duration_seconds=transcript.duration_seconds,
                )
            )
            speaker_ids = self._insert_speakers(connection, transcript)
            self._insert_segments(connection, transcript, speaker_ids)

    def keyword_search(self, terms: Sequence[str], limit: int) -> list[ScoredSegment]:
        """Full-text search matching any term, ranked by cover density."""
        if not terms:
            return []
        unsafe = [term for term in terms if not SAFE_TERM.match(term)]
        if unsafe:
            raise ValueError(f"terms must be lowercase alphanumeric: {unsafe}")
        segments = self._schema.segments
        query = func.to_tsquery(TEXT_SEARCH_CONFIG, " | ".join(terms))
        score = func.ts_rank_cd(segments.c.text_search, query)
        statement = (
            self._select_segments(score.label("score"))
            .where(segments.c.text_search.op("@@")(query))
            .order_by(score.desc(), segments.c.audio_id, segments.c.segment_index)
            .limit(limit)
        )
        with self._engine.connect() as connection:
            return [to_scored_segment(row) for row in connection.execute(statement)]

    def vector_search(self, embedding: Sequence[float], limit: int) -> list[ScoredSegment]:
        """Approximate nearest neighbours by cosine similarity through the HNSW index."""
        segments = self._schema.segments
        distance = segments.c.embedding.cosine_distance(list(embedding))
        statement = (
            self._select_segments((1 - distance).label("score")).order_by(distance).limit(limit)
        )
        with self._engine.begin() as connection:
            # HNSW returns at most ef_search rows, so it must be at least the limit.
            ef_search = max(limit, MIN_HNSW_EF_SEARCH)
            connection.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
            return [to_scored_segment(row) for row in connection.execute(statement)]

    def neighbors(self, segment: TranscriptSegment, window: int) -> list[TranscriptSegment]:
        """The segment and up to `window` segments either side of it, in time order."""
        segments = self._schema.segments
        statement = (
            self._select_segments()
            .where(
                segments.c.audio_id == segment.audio_id,
                segments.c.segment_index.between(
                    segment.segment_index - window, segment.segment_index + window
                ),
            )
            .order_by(segments.c.segment_index)
        )
        with self._engine.connect() as connection:
            return [to_segment(row) for row in connection.execute(statement)]

    def _select_segments(self, *extra_columns: ColumnElement[Any]) -> Select[Any]:
        segments = self._schema.segments
        speakers = self._schema.speakers
        return select(
            segments.c.audio_id,
            segments.c.segment_index,
            speakers.c.label.label("speaker"),
            segments.c.start_time,
            segments.c.end_time,
            segments.c.text,
            *extra_columns,
        ).join_from(segments, speakers, segments.c.speaker_id == speakers.c.id)

    def _insert_speakers(
        self, connection: Connection, transcript: AudioTranscript
    ) -> dict[str, int]:
        labels = sorted({chunk.speaker for chunk in transcript.chunks})
        if not labels:
            return {}
        speakers = self._schema.speakers
        rows = connection.execute(
            insert(speakers).returning(speakers.c.id, speakers.c.label),
            [{"audio_id": transcript.audio_id, "label": label} for label in labels],
        )
        return {row.label: row.id for row in rows}

    def _insert_segments(
        self, connection: Connection, transcript: AudioTranscript, speaker_ids: dict[str, int]
    ) -> None:
        if not transcript.chunks:
            return
        rows = [
            {
                "audio_id": transcript.audio_id,
                "speaker_id": speaker_ids[chunk.speaker],
                "segment_index": chunk.segment_index,
                "start_time": chunk.start_seconds,
                "end_time": chunk.end_seconds,
                "text": chunk.text,
                "embedding": list(embedding),
            }
            for chunk, embedding in zip(transcript.chunks, transcript.embeddings, strict=True)
        ]
        connection.execute(insert(self._schema.segments), rows)


def to_segment(row: Row[Any]) -> TranscriptSegment:
    return TranscriptSegment(
        audio_id=row.audio_id,
        segment_index=row.segment_index,
        speaker=row.speaker,
        start_seconds=row.start_time,
        end_seconds=row.end_time,
        text=row.text,
    )


def to_scored_segment(row: Row[Any]) -> ScoredSegment:
    return ScoredSegment(segment=to_segment(row), score=float(row.score))
