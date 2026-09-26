from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from faster_whisper.transcribe import Segment, Word

from app.config import TranscriptionSettings
from app.domain.exceptions import TranscriptionError
from app.domain.models import TranscribedSegment, TranscribedWord
from app.ingestion.transcription import WhisperTranscriber


def make_segment(text: str, start: float, end: float, words: list[Word] | None) -> Segment:
    return Segment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[],
        avg_logprob=0.0,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=words,
        temperature=0.0,
    )


class FakeWhisperModel:
    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio: str,
        *,
        language: str,
        beam_size: int,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> tuple[Iterable[Segment], Any]:
        self.calls.append(
            {
                "audio": audio,
                "language": language,
                "beam_size": beam_size,
                "word_timestamps": word_timestamps,
                "vad_filter": vad_filter,
            }
        )
        return iter(self._segments), None


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "conversation.wav"
    path.write_bytes(b"")
    return path


def test_transcribe_converts_segments_and_words(audio_file: Path) -> None:
    words = [
        Word(start=0.0, end=0.4, word=" Zero", probability=0.9),
        Word(start=0.4, end=0.9, word=" trust.", probability=0.8),
    ]
    model = FakeWhisperModel([make_segment(" Zero trust. ", 0.0, 0.9, words)])

    segments = WhisperTranscriber(model, TranscriptionSettings()).transcribe(audio_file)

    assert segments == [
        TranscribedSegment(
            text="Zero trust.",
            start_seconds=0.0,
            end_seconds=0.9,
            words=(TranscribedWord("Zero", 0.0, 0.4), TranscribedWord("trust.", 0.4, 0.9)),
        )
    ]


def test_transcribe_passes_settings_and_requests_word_timestamps(audio_file: Path) -> None:
    model = FakeWhisperModel([])
    settings = TranscriptionSettings(language="de", beam_size=2, vad_filter=False)

    WhisperTranscriber(model, settings).transcribe(audio_file)

    assert model.calls == [
        {
            "audio": str(audio_file),
            "language": "de",
            "beam_size": 2,
            "word_timestamps": True,
            "vad_filter": False,
        }
    ]


def test_transcribe_drops_blank_segments_and_words(audio_file: Path) -> None:
    words = [
        Word(start=1.0, end=1.2, word=" ", probability=0.1),
        Word(start=1.2, end=1.5, word=" Kubernetes", probability=0.9),
    ]
    model = FakeWhisperModel(
        [make_segment("   ", 0.0, 1.0, []), make_segment(" Kubernetes", 1.0, 1.5, words)]
    )

    segments = WhisperTranscriber(model, TranscriptionSettings()).transcribe(audio_file)

    assert [segment.text for segment in segments] == ["Kubernetes"]
    assert segments[0].words == (TranscribedWord("Kubernetes", 1.2, 1.5),)


def test_transcribe_handles_segments_without_word_timestamps(audio_file: Path) -> None:
    model = FakeWhisperModel([make_segment(" PostgreSQL", 2.0, 3.0, None)])

    segments = WhisperTranscriber(model, TranscriptionSettings()).transcribe(audio_file)

    assert segments[0].words == ()


def test_transcribe_rejects_missing_audio_file(tmp_path: Path) -> None:
    transcriber = WhisperTranscriber(FakeWhisperModel([]), TranscriptionSettings())

    with pytest.raises(TranscriptionError, match="audio file not found"):
        transcriber.transcribe(tmp_path / "missing.wav")


def test_settings_reject_non_positive_beam_size() -> None:
    with pytest.raises(ValueError):
        TranscriptionSettings(beam_size=0)
