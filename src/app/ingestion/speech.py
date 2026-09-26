"""Combines transcription and diarization into speaker-attributed words."""

from pathlib import Path

from app.domain.models import DiarizedTranscript
from app.domain.protocols import Diarizer, Transcriber
from app.ingestion.alignment import assign_speakers
from app.ingestion.audio import SAMPLE_RATE_HZ, load_audio


class SpeechAnalyzer:
    def __init__(self, transcriber: Transcriber, diarizer: Diarizer) -> None:
        self._transcriber = transcriber
        self._diarizer = diarizer

    def analyze(self, audio_path: Path) -> DiarizedTranscript:
        samples = load_audio(audio_path)
        segments = self._transcriber.transcribe(audio_path)
        turns = self._diarizer.diarize(samples, SAMPLE_RATE_HZ)
        return DiarizedTranscript(
            duration_seconds=len(samples) / SAMPLE_RATE_HZ,
            words=tuple(assign_speakers(segments, turns)),
        )
