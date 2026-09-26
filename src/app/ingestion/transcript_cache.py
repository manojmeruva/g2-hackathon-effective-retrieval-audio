"""File cache of speaker-attributed transcripts.

Transcription and diarization take minutes per file; caching their output lets
chunking and embedding be re-run cheaply. PostgreSQL remains the source of
truth for everything that is searched.
"""

from pathlib import Path

from pydantic import BaseModel

from app.domain.models import DiarizedTranscript, SpokenWord


class CachedWord(BaseModel):
    text: str
    start_seconds: float
    end_seconds: float
    speaker: str


class CachedTranscript(BaseModel):
    audio_id: str
    duration_seconds: float
    words: list[CachedWord]


class TranscriptCache:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self, audio_id: str) -> DiarizedTranscript | None:
        path = self._path(audio_id)
        if not path.is_file():
            return None
        cached = CachedTranscript.model_validate_json(path.read_text())
        words = tuple(SpokenWord(**word.model_dump()) for word in cached.words)
        return DiarizedTranscript(duration_seconds=cached.duration_seconds, words=words)

    def save(self, audio_id: str, transcript: DiarizedTranscript) -> None:
        cached = CachedTranscript(
            audio_id=audio_id,
            duration_seconds=transcript.duration_seconds,
            words=[CachedWord(**vars(word)) for word in transcript.words],
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        self._path(audio_id).write_text(cached.model_dump_json(indent=1) + "\n")

    def _path(self, audio_id: str) -> Path:
        return self._directory / f"{audio_id}.json"
