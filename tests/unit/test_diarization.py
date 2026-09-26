from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.core import Annotation, Segment

from app.config import DiarizationSettings
from app.domain.exceptions import ConfigurationError, DiarizationError
from app.domain.models import SpeakerTurn
from app.ingestion.diarization import PyannoteDiarizer


def annotation(*turns: tuple[str, float, float]) -> Annotation:
    result = Annotation()
    for label, start, end in turns:
        result[Segment(start, end)] = label
    return result


class FakePipeline:
    def __init__(self, output: Any) -> None:
        self._output = output
        self.calls: list[tuple[Mapping[str, Any], int]] = []

    def __call__(self, file: Mapping[str, Any], *, num_speakers: int) -> Any:
        self.calls.append((file, num_speakers))
        return self._output


SAMPLES = np.zeros(16_000, dtype=np.float32)


def test_uses_exclusive_diarization_sorted_by_time() -> None:
    exclusive = annotation(("SPEAKER_01", 5.0, 9.0), ("SPEAKER_00", 0.0, 5.0))
    overlapping = annotation(("SPEAKER_00", 0.0, 6.0), ("SPEAKER_01", 4.0, 9.0))
    pipeline = FakePipeline(DiarizeOutput(overlapping, exclusive))

    turns = PyannoteDiarizer(pipeline, DiarizationSettings()).diarize(SAMPLES, 16_000)

    assert turns == [SpeakerTurn("SPEAKER_00", 0.0, 5.0), SpeakerTurn("SPEAKER_01", 5.0, 9.0)]


def test_passes_waveform_and_speaker_count() -> None:
    pipeline = FakePipeline(annotation(("SPEAKER_00", 0.0, 1.0)))

    PyannoteDiarizer(pipeline, DiarizationSettings(num_speakers=2)).diarize(SAMPLES, 16_000)

    file, num_speakers = pipeline.calls[0]
    assert num_speakers == 2
    assert file["sample_rate"] == 16_000
    assert tuple(file["waveform"].shape) == (1, 16_000)


def test_no_speech_is_an_error() -> None:
    diarizer = PyannoteDiarizer(FakePipeline(Annotation()), DiarizationSettings())

    with pytest.raises(DiarizationError):
        diarizer.diarize(SAMPLES, 16_000)


def test_loading_without_token_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="HF_TOKEN"):
        PyannoteDiarizer.from_settings(DiarizationSettings(hf_token=None))
