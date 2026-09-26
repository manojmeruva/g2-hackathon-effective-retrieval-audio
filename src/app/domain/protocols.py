"""Protocols for replaceable components (retrievers, embedder, repositories)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from app.domain.models import (
    AudioTranscript,
    DiarizedTranscript,
    RankedSegment,
    ScoredSegment,
    SpeakerTurn,
    TranscribedSegment,
    TranscriptSegment,
)

AudioSamples = npt.NDArray[np.float32]


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[TranscribedSegment]: ...


class Diarizer(Protocol):
    def diarize(self, samples: AudioSamples, sample_rate: int) -> list[SpeakerTurn]: ...


class SpeechAnalyzer(Protocol):
    def analyze(self, audio_path: Path) -> DiarizedTranscript: ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[RankedSegment]) -> list[RankedSegment]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, limit: int) -> list[ScoredSegment]: ...


class SegmentStore(Protocol):
    def replace_audio(self, transcript: AudioTranscript) -> None: ...


class KeywordIndex(Protocol):
    def keyword_search(self, terms: Sequence[str], limit: int) -> list[ScoredSegment]: ...


class VectorIndex(Protocol):
    def vector_search(self, embedding: Sequence[float], limit: int) -> list[ScoredSegment]: ...


class SegmentContext(Protocol):
    def neighbors(self, segment: TranscriptSegment, window: int) -> list[TranscriptSegment]: ...
