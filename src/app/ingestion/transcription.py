"""Speech-to-text with faster-whisper, preserving timestamps."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, Word

from app.config import TranscriptionSettings
from app.domain.exceptions import TranscriptionError
from app.domain.models import TranscribedSegment, TranscribedWord


class SpeechToTextModel(Protocol):
    """The subset of faster-whisper's WhisperModel that the transcriber uses."""

    def transcribe(
        self,
        audio: str,
        *,
        language: str,
        beam_size: int,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> tuple[Iterable[Segment], Any]: ...


class WhisperTranscriber:
    """Transcribes an audio file into segments with segment and word timestamps."""

    def __init__(self, model: SpeechToTextModel, settings: TranscriptionSettings) -> None:
        self._model = model
        self._settings = settings

    @classmethod
    def from_settings(cls, settings: TranscriptionSettings) -> WhisperTranscriber:
        model = WhisperModel(
            settings.model_size, device=settings.device, compute_type=settings.compute_type
        )
        return cls(model, settings)

    def transcribe(self, audio_path: Path) -> list[TranscribedSegment]:
        if not audio_path.is_file():
            raise TranscriptionError(f"audio file not found: {audio_path}")
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self._settings.language,
            beam_size=self._settings.beam_size,
            word_timestamps=True,
            vad_filter=self._settings.vad_filter,
        )
        transcribed = (to_transcribed_segment(segment) for segment in segments)
        return [segment for segment in transcribed if segment.text]


def to_transcribed_segment(segment: Segment) -> TranscribedSegment:
    words = tuple(to_transcribed_word(word) for word in segment.words or ())
    return TranscribedSegment(
        text=segment.text.strip(),
        start_seconds=segment.start,
        end_seconds=segment.end,
        words=tuple(word for word in words if word.text),
    )


def to_transcribed_word(word: Word) -> TranscribedWord:
    return TranscribedWord(text=word.word.strip(), start_seconds=word.start, end_seconds=word.end)
