import wave
from pathlib import Path

import pytest

from app.config import TranscriptionSettings
from app.ingestion.transcription import WhisperTranscriber

GOLDEN_AUDIO = Path("data/audio/conversation_02.wav")
CLIP_SECONDS = 30

pytestmark = pytest.mark.integration


def write_leading_clip(source: Path, destination: Path, seconds: int) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(seconds * reader.getframerate())
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


@pytest.mark.skipif(not GOLDEN_AUDIO.is_file(), reason="golden audio not available")
def test_whisper_transcribes_golden_clip_with_timestamps(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    write_leading_clip(GOLDEN_AUDIO, clip, CLIP_SECONDS)
    transcriber = WhisperTranscriber.from_settings(TranscriptionSettings())

    segments = transcriber.transcribe(clip)

    text = " ".join(segment.text for segment in segments).lower()
    assert "inside engineering" in text
    assert "sarah" in text
    words = [word for segment in segments for word in segment.words]
    assert words
    assert all(0 <= word.start_seconds <= word.end_seconds <= CLIP_SECONDS for word in words)
    starts = [word.start_seconds for word in words]
    assert starts == sorted(starts)
