from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_DATABASE_URL,
    ChunkingSettings,
    EvaluationSettings,
    load_settings,
    read_env_file,
)


def test_read_env_file_skips_comments_and_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\n\nHF_TOKEN="abc"\nDATABASE_URL = postgres://x\nnot a pair\n')

    assert read_env_file(env_file) == {"HF_TOKEN": "abc", "DATABASE_URL": "postgres://x"}


def test_read_env_file_returns_empty_when_missing(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / ".env") == {}


def test_environment_overrides_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from-file\nDATABASE_URL=file-url\n")

    settings = load_settings({"HF_TOKEN": "from-env"}, env_file)

    assert settings.diarization.hf_token == "from-env"
    assert settings.database.url == "file-url"


def test_defaults_without_env(tmp_path: Path) -> None:
    settings = load_settings({}, tmp_path / ".env")

    assert settings.diarization.hf_token is None
    assert settings.database.url == DEFAULT_DATABASE_URL


def test_chunking_requires_overlap_below_target_below_max() -> None:
    with pytest.raises(ValidationError):
        ChunkingSettings(target_seconds=30, max_seconds=20)
    with pytest.raises(ValidationError):
        ChunkingSettings(target_seconds=10, overlap_seconds=10)


def test_evaluation_primary_k_must_be_evaluated() -> None:
    with pytest.raises(ValidationError):
        EvaluationSettings(k_values=(1, 3), primary_k=5)
