"""Speaker diarization with pyannote.audio."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.core import Annotation

from app.config import DiarizationSettings
from app.domain.exceptions import ConfigurationError, DiarizationError
from app.domain.models import SpeakerTurn
from app.domain.protocols import AudioSamples


class DiarizationPipeline(Protocol):
    def __call__(self, file: Mapping[str, Any], *, num_speakers: int) -> Any: ...


class PyannoteDiarizer:
    """Splits audio into speaker turns; turns never overlap, so each instant has one speaker."""

    def __init__(self, pipeline: DiarizationPipeline, settings: DiarizationSettings) -> None:
        self._pipeline = pipeline
        self._settings = settings

    @classmethod
    def from_settings(cls, settings: DiarizationSettings) -> PyannoteDiarizer:
        if not settings.hf_token:
            raise ConfigurationError(
                f"HF_TOKEN is required to download {settings.model_name}; add it to .env"
            )
        pipeline = Pipeline.from_pretrained(settings.model_name, token=settings.hf_token)
        if pipeline is None:
            raise ConfigurationError(f"could not load diarization model {settings.model_name}")
        pipeline.to(torch.device(settings.device))
        return cls(pipeline, settings)

    def diarize(self, samples: AudioSamples, sample_rate: int) -> list[SpeakerTurn]:
        waveform = torch.from_numpy(samples).unsqueeze(0)
        output = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=self._settings.num_speakers,
        )
        turns = to_speaker_turns(exclusive_annotation(output))
        if not turns:
            raise DiarizationError("no speech found during diarization")
        return turns


def exclusive_annotation(output: DiarizeOutput | Annotation) -> Annotation:
    if isinstance(output, DiarizeOutput):
        return output.exclusive_speaker_diarization
    return output


def to_speaker_turns(annotation: Annotation) -> list[SpeakerTurn]:
    turns = [
        SpeakerTurn(speaker=str(label), start_seconds=segment.start, end_seconds=segment.end)
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda turn: turn.start_seconds)
