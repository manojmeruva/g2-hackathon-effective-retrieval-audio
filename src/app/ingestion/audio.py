"""Audio decoding shared by transcription and diarization."""

from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

from app.domain.protocols import AudioSamples

SAMPLE_RATE_HZ = 16_000
SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg"})


def load_audio(path: Path) -> AudioSamples:
    """Decode any supported file to mono float32 samples at SAMPLE_RATE_HZ.

    faster-whisper bundles PyAV, so this needs no system ffmpeg installation.
    """
    return np.asarray(decode_audio(str(path), sampling_rate=SAMPLE_RATE_HZ), dtype=np.float32)


def find_audio_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
