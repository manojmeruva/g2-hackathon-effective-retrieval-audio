"""Cross-encoder reranking of hybrid candidates.

A cross-encoder reads the query and a candidate together, which is more
accurate than comparing independent embeddings but too slow to run over the
whole index; it is therefore applied only to the top hybrid candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.config import RerankerSettings
from app.domain.models import RankedSegment

ScorePairs = Callable[[list[tuple[str, str]]], Iterable[float]]


class CrossEncoderReranker:
    def __init__(self, score_pairs: ScorePairs) -> None:
        self._score_pairs = score_pairs

    @classmethod
    def from_settings(cls, settings: RerankerSettings) -> CrossEncoderReranker:
        model = CrossEncoder(settings.model_name)

        def score_pairs(pairs: list[tuple[str, str]]) -> Iterable[float]:
            scores = model.predict(pairs, show_progress_bar=False)
            return [float(score) for score in scores]

        return cls(score_pairs)

    def rerank(self, query: str, candidates: Sequence[RankedSegment]) -> list[RankedSegment]:
        if not candidates:
            return []
        pairs = [(query, candidate.segment.text) for candidate in candidates]
        scores = [float(score) for score in self._score_pairs(pairs)]
        rescored = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        order = sorted(range(len(rescored)), key=lambda index: -scores[index])
        return [rescored[index] for index in order]
