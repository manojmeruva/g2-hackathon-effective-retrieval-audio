"""Orchestrates transcription, diarization, chunking, embedding and storage."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import ChunkingSettings
from app.domain.models import AudioTranscript, DiarizedTranscript
from app.domain.protocols import Embedder, SegmentStore, SpeechAnalyzer
from app.ingestion.chunking import chunk_words
from app.ingestion.transcript_cache import TranscriptCache


@dataclass(frozen=True)
class IngestionSummary:
    audio_id: str
    duration_seconds: float
    word_count: int
    segment_count: int
    from_cache: bool


class IngestionService:
    def __init__(
        self,
        analyzer_factory: Callable[[], SpeechAnalyzer],
        cache: TranscriptCache,
        embedder: Embedder,
        store: SegmentStore,
        chunking: ChunkingSettings,
    ) -> None:
        self._analyzer_factory = analyzer_factory
        self._analyzer: SpeechAnalyzer | None = None
        self._cache = cache
        self._embedder = embedder
        self._store = store
        self._chunking = chunking

    def ingest_file(self, audio_path: Path, force: bool = False) -> IngestionSummary:
        audio_id = audio_path.stem
        transcript, from_cache = self._diarized_transcript(audio_id, audio_path, force)
        chunks = tuple(chunk_words(transcript.words, self._chunking))
        embeddings = tuple(self._embedder.embed_documents([chunk.text for chunk in chunks]))
        self._store.replace_audio(
            AudioTranscript(
                audio_id=audio_id,
                filename=audio_path.name,
                duration_seconds=transcript.duration_seconds,
                chunks=chunks,
                embeddings=embeddings,
            )
        )
        return IngestionSummary(
            audio_id=audio_id,
            duration_seconds=transcript.duration_seconds,
            word_count=len(transcript.words),
            segment_count=len(chunks),
            from_cache=from_cache,
        )

    def _diarized_transcript(
        self, audio_id: str, audio_path: Path, force: bool
    ) -> tuple[DiarizedTranscript, bool]:
        if not force:
            cached = self._cache.load(audio_id)
            if cached is not None:
                return cached, True
        transcript = self._speech_analyzer().analyze(audio_path)
        self._cache.save(audio_id, transcript)
        return transcript, False

    def _speech_analyzer(self) -> SpeechAnalyzer:
        """Load the speech models only when a file is not already cached."""
        if self._analyzer is None:
            self._analyzer = self._analyzer_factory()
        return self._analyzer
