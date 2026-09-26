"""Domain objects: transcript segments, chunks, search results."""

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class TranscribedWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TranscribedSegment:
    """A stretch of speech recognized as one unit, before speakers are assigned."""

    text: str
    start_seconds: float
    end_seconds: float
    words: tuple[TranscribedWord, ...]


@dataclass(frozen=True)
class SpeakerTurn:
    """A time span attributed to one speaker by diarization."""

    speaker: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SpokenWord:
    text: str
    start_seconds: float
    end_seconds: float
    speaker: str


@dataclass(frozen=True)
class DiarizedTranscript:
    """Speaker-attributed words for one audio file; the expensive part of ingestion."""

    duration_seconds: float
    words: tuple[SpokenWord, ...]


@dataclass(frozen=True)
class Chunk:
    """A speaker-homogeneous span of words that is indexed as one searchable unit."""

    segment_index: int
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class AudioTranscript:
    """Everything ingestion derives from one audio file before it is stored."""

    audio_id: str
    filename: str
    duration_seconds: float
    chunks: tuple[Chunk, ...]
    embeddings: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class TranscriptSegment:
    """A stored, searchable chunk of one audio file."""

    audio_id: str
    segment_index: int
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str

    @property
    def segment_id(self) -> str:
        return f"{self.audio_id}:{self.segment_index}"


@dataclass(frozen=True)
class ScoredSegment:
    segment: TranscriptSegment
    score: float


@dataclass(frozen=True)
class RankedSegment:
    """A candidate with its normalized component scores and fused hybrid score."""

    segment: TranscriptSegment
    keyword_score: float
    semantic_score: float
    hybrid_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class SearchResult:
    ranked: RankedSegment
    context: tuple[TranscriptSegment, ...]

    @property
    def segment(self) -> TranscriptSegment:
        return self.ranked.segment


class SearchMode(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class FusionMethod(StrEnum):
    """How hybrid modes combine the keyword and semantic result lists."""

    WEIGHTED = "weighted"
    RRF = "rrf"
