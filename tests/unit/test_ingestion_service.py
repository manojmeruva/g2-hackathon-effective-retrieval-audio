from collections.abc import Sequence
from pathlib import Path

import pytest

from app.application.ingestion_service import IngestionService
from app.config import ChunkingSettings
from app.domain.models import AudioTranscript, DiarizedTranscript
from app.ingestion.transcript_cache import TranscriptCache
from tests.factories import word

TRANSCRIPT = DiarizedTranscript(
    duration_seconds=12.0,
    words=(word("Hello.", 0.0, 1.0, "A"), word("Hi.", 2.0, 3.0, "B")),
)


class FakeAnalyzer:
    def __init__(self) -> None:
        self.analyzed: list[Path] = []

    def analyze(self, audio_path: Path) -> DiarizedTranscript:
        self.analyzed.append(audio_path)
        return TRANSCRIPT


class FakeEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(float(len(text)),) for text in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (0.0,)


class FakeStore:
    def __init__(self) -> None:
        self.stored: list[AudioTranscript] = []

    def replace_audio(self, transcript: AudioTranscript) -> None:
        self.stored.append(transcript)


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.analyzer = FakeAnalyzer()
        self.factory_calls = 0
        self.cache = TranscriptCache(tmp_path / "transcripts")
        self.store = FakeStore()
        self.service = IngestionService(
            self._factory, self.cache, FakeEmbedder(), self.store, ChunkingSettings()
        )
        self.audio = tmp_path / "conversation_07.wav"

    def _factory(self) -> FakeAnalyzer:
        self.factory_calls += 1
        return self.analyzer


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_ingest_stores_chunks_with_embeddings(harness: Harness) -> None:
    summary = harness.service.ingest_file(harness.audio)

    stored = harness.store.stored[0]
    assert stored.audio_id == "conversation_07"
    assert stored.filename == "conversation_07.wav"
    assert [chunk.text for chunk in stored.chunks] == ["Hello.", "Hi."]
    assert stored.embeddings == ((6.0,), (3.0,))
    assert summary.segment_count == 2
    assert summary.word_count == 2
    assert not summary.from_cache


def test_second_ingest_uses_cache_without_loading_models(harness: Harness) -> None:
    harness.service.ingest_file(harness.audio)

    fresh = IngestionService(
        harness._factory, harness.cache, FakeEmbedder(), harness.store, ChunkingSettings()
    )
    summary = fresh.ingest_file(harness.audio)

    assert summary.from_cache
    assert harness.factory_calls == 1
    assert harness.store.stored[0] == harness.store.stored[1]


def test_force_reanalyzes_even_when_cached(harness: Harness) -> None:
    harness.service.ingest_file(harness.audio)

    summary = harness.service.ingest_file(harness.audio, force=True)

    assert not summary.from_cache
    assert len(harness.analyzer.analyzed) == 2


def test_models_load_once_for_many_files(harness: Harness, tmp_path: Path) -> None:
    harness.service.ingest_file(tmp_path / "a.wav")
    harness.service.ingest_file(tmp_path / "b.wav")

    assert harness.factory_calls == 1


def test_transcript_cache_round_trip(tmp_path: Path) -> None:
    cache = TranscriptCache(tmp_path)

    cache.save("conversation_01", TRANSCRIPT)

    assert cache.load("conversation_01") == TRANSCRIPT
    assert cache.load("conversation_02") is None
