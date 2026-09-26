"""Application settings validated with Pydantic."""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.models import FusionMethod

DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DATABASE_URL = "postgresql+psycopg://audio:audio@localhost:5432/audio_retrieval"


class FrozenSettings(BaseModel):
    model_config = ConfigDict(frozen=True)


class TranscriptionSettings(FrozenSettings):
    model_size: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True


class DiarizationSettings(FrozenSettings):
    model_name: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int = Field(default=2, ge=1)
    device: str = "cpu"
    hf_token: str | None = None


class ChunkingSettings(FrozenSettings):
    target_seconds: float = Field(default=30.0, gt=0)
    max_seconds: float = Field(default=40.0, gt=0)
    overlap_seconds: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def check_durations(self) -> "ChunkingSettings":
        if not self.overlap_seconds < self.target_seconds <= self.max_seconds:
            raise ValueError("expected overlap_seconds < target_seconds <= max_seconds")
        return self


class EmbeddingSettings(FrozenSettings):
    model_name: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = Field(default=384, gt=0)
    batch_size: int = Field(default=32, gt=0)
    query_prefix: str = "Represent this sentence for searching relevant passages: "


class RerankerSettings(FrozenSettings):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SearchSettings(FrozenSettings):
    candidate_count: int = Field(default=50, gt=0)
    rerank_candidate_count: int = Field(default=30, gt=0)
    # Defaults chosen by `evaluate` on the golden set (see reports/evaluation.md).
    fusion: FusionMethod = FusionMethod.RRF
    alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, gt=0)
    context_window: int = Field(default=1, ge=0)


class EvaluationSettings(FrozenSettings):
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    primary_k: int = 5
    alpha_grid: tuple[float, ...] = (0.2, 0.4, 0.5, 0.6, 0.8)

    @model_validator(mode="after")
    def check_primary_k(self) -> "EvaluationSettings":
        if self.primary_k not in self.k_values:
            raise ValueError("primary_k must be one of k_values")
        if not all(0.0 <= alpha <= 1.0 for alpha in self.alpha_grid):
            raise ValueError("every alpha must be between 0 and 1")
        return self


class DatabaseSettings(FrozenSettings):
    url: str = DEFAULT_DATABASE_URL


class AppSettings(FrozenSettings):
    transcription: TranscriptionSettings = TranscriptionSettings()
    diarization: DiarizationSettings = DiarizationSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    search: SearchSettings = SearchSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    database: DatabaseSettings = DatabaseSettings()


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines; blank lines and # comments are ignored."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def load_settings(
    environ: Mapping[str, str] = os.environ, env_file: Path = DEFAULT_ENV_FILE
) -> AppSettings:
    """Build settings; process environment variables take precedence over the .env file."""
    values = {**read_env_file(env_file), **environ}
    return AppSettings(
        diarization=DiarizationSettings(hf_token=values.get("HF_TOKEN") or None),
        database=DatabaseSettings(url=values.get("DATABASE_URL", DEFAULT_DATABASE_URL)),
    )
