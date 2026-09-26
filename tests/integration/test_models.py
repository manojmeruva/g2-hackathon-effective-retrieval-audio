"""Real local models: embeddings, cross-encoder reranking and diarization."""

import math
from pathlib import Path

import pytest
from huggingface_hub.errors import GatedRepoError

from app.config import EmbeddingSettings, RerankerSettings, load_settings
from app.domain.exceptions import ConfigurationError
from app.ingestion.audio import SAMPLE_RATE_HZ, load_audio
from app.ingestion.diarization import PyannoteDiarizer
from app.ingestion.embeddings import SentenceTransformerEmbedder
from app.retrieval.reranking import CrossEncoderReranker
from tests.factories import ranked

pytestmark = pytest.mark.integration

GOLDEN_AUDIO = Path("data/audio/conversation_02.wav")
CLIP_SECONDS = 60


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_embeddings_are_normalized_and_capture_meaning() -> None:
    settings = EmbeddingSettings()
    embedder = SentenceTransformerEmbedder.from_settings(settings)

    migration, hiring = embedder.embed_documents(
        [
            "We moved our servers out of the data center and into AWS.",
            "We are raising the referral bonus for engineering candidates.",
        ]
    )
    query = embedder.embed_query("cloud migration")

    assert len(migration) == settings.dimensions
    assert math.isclose(math.sqrt(cosine(migration, migration)), 1.0, rel_tol=1e-4)
    assert cosine(query, migration) > cosine(query, hiring)


def test_cross_encoder_prefers_the_answering_passage() -> None:
    reranker = CrossEncoderReranker.from_settings(RerankerSettings())
    candidates = [
        ranked(1, 0.9, "We raised the referral bonus to five thousand dollars."),
        ranked(2, 0.5, "Our hosting bill dropped by thirty five percent after resizing servers."),
    ]

    reranked = reranker.rerank("How did they cut infrastructure costs?", candidates)

    assert reranked[0].segment.segment_index == 2


@pytest.mark.skipif(not GOLDEN_AUDIO.is_file(), reason="golden audio not available")
def test_pyannote_finds_two_speakers_in_golden_clip() -> None:
    try:
        diarizer = PyannoteDiarizer.from_settings(load_settings().diarization)
    except (ConfigurationError, GatedRepoError) as error:
        pytest.skip(f"diarization model unavailable: {error}")
    samples = load_audio(GOLDEN_AUDIO)[: CLIP_SECONDS * SAMPLE_RATE_HZ]

    turns = diarizer.diarize(samples, SAMPLE_RATE_HZ)

    assert {turn.speaker for turn in turns} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(0 <= t.start_seconds < t.end_seconds <= CLIP_SECONDS + 1 for t in turns)
