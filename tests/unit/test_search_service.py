from collections.abc import Sequence
from dataclasses import replace

import pytest

from app.application.search_service import SearchService
from app.config import SearchSettings
from app.domain.exceptions import ConfigurationError
from app.domain.models import (
    FusionMethod,
    RankedSegment,
    ScoredSegment,
    SearchMode,
    TranscriptSegment,
)
from tests.factories import scored, segment


class FakeRetriever:
    def __init__(self, results: list[ScoredSegment]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, limit: int) -> list[ScoredSegment]:
        self.calls.append((query, limit))
        return self._results


class FakeContext:
    def neighbors(self, target: TranscriptSegment, window: int) -> list[TranscriptSegment]:
        return [
            segment(target.audio_id, index)
            for index in range(target.segment_index - window, target.segment_index + window + 1)
        ]


class ReverseReranker:
    def __init__(self) -> None:
        self.received: list[RankedSegment] = []

    def rerank(self, query: str, candidates: Sequence[RankedSegment]) -> list[RankedSegment]:
        self.received = list(candidates)
        return [replace(item, rerank_score=1.0) for item in reversed(candidates)]


KEYWORD_HITS = [scored(1, 3.0), scored(2, 1.0)]
SEMANTIC_HITS = [scored(3, 0.9), scored(1, 0.5)]


def make_service(
    settings: SearchSettings | None = None, reranker: ReverseReranker | None = None
) -> tuple[SearchService, FakeRetriever, FakeRetriever]:
    keyword = FakeRetriever(KEYWORD_HITS)
    semantic = FakeRetriever(SEMANTIC_HITS)
    service = SearchService(
        keyword, semantic, FakeContext(), settings or SearchSettings(), reranker
    )
    return service, keyword, semantic


def indexes(results: Sequence[RankedSegment]) -> list[int]:
    return [item.segment.segment_index for item in results]


def test_keyword_mode_uses_only_keyword_results() -> None:
    service, _keyword, semantic = make_service()

    assert indexes(service.rank("q", SearchMode.KEYWORD)) == [1, 2]
    assert semantic.calls == []


def test_semantic_mode_uses_only_semantic_results() -> None:
    service, keyword, _semantic = make_service()

    assert indexes(service.rank("q", SearchMode.SEMANTIC)) == [3, 1]
    assert keyword.calls == []


def test_hybrid_mode_fuses_both_lists_with_candidate_limit() -> None:
    service, keyword, semantic = make_service(
        SearchSettings(candidate_count=20, alpha=0.5, fusion=FusionMethod.WEIGHTED)
    )

    ranked = service.rank("q", SearchMode.HYBRID)

    assert indexes(ranked) == [1, 3, 2]
    assert keyword.calls == [("q", 20)] and semantic.calls == [("q", 20)]


def test_alpha_argument_overrides_settings() -> None:
    service, _keyword, _semantic = make_service(
        SearchSettings(alpha=0.5, fusion=FusionMethod.WEIGHTED)
    )

    assert indexes(service.rank("q", SearchMode.HYBRID, alpha=0.0))[0] == 3


def test_rerank_mode_reranks_only_the_top_candidates() -> None:
    reranker = ReverseReranker()
    service, _keyword, _semantic = make_service(SearchSettings(rerank_candidate_count=2), reranker)

    ranked = service.rank("q", SearchMode.HYBRID_RERANK)

    assert indexes(reranker.received) == [1, 3]
    assert indexes(ranked) == [3, 1]


def test_rerank_mode_without_reranker_is_a_configuration_error() -> None:
    service, _keyword, _semantic = make_service()

    with pytest.raises(ConfigurationError):
        service.rank("q", SearchMode.HYBRID_RERANK)


def test_search_cuts_to_top_k_and_adds_context() -> None:
    service, _keyword, _semantic = make_service(SearchSettings(context_window=1))

    results = service.search("q", top_k=2, mode=SearchMode.HYBRID)

    assert len(results) == 2
    assert [s.segment_index for s in results[0].context] == [0, 1, 2]


def test_context_window_zero_skips_context() -> None:
    service, _keyword, _semantic = make_service(SearchSettings(context_window=0))

    assert service.search("q", top_k=1)[0].context == ()


def test_blank_query_returns_nothing_without_searching() -> None:
    service, keyword, semantic = make_service()

    assert service.search("   ", top_k=5) == []
    assert keyword.calls == [] and semantic.calls == []


def test_rrf_fusion_ranks_by_reciprocal_rank() -> None:
    service, _keyword, _semantic = make_service(SearchSettings(rrf_k=60))

    ranked = service.rank("q", SearchMode.HYBRID)

    assert indexes(ranked) == [1, 3, 2]
    assert ranked[0].hybrid_score == pytest.approx(1 / 61 + 1 / 62)


def test_fusion_argument_overrides_configured_fusion() -> None:
    service, _keyword, _semantic = make_service(SearchSettings(fusion=FusionMethod.RRF))

    ranked = service.rank("q", SearchMode.HYBRID, alpha=0.5, fusion=FusionMethod.WEIGHTED)

    assert ranked[0].hybrid_score == pytest.approx(0.5)


def test_rerank_reorders_rrf_candidates() -> None:
    reranker = ReverseReranker()
    service, _keyword, _semantic = make_service(reranker=reranker)

    service.rank("q", SearchMode.HYBRID_RERANK, fusion=FusionMethod.RRF)

    assert indexes(reranker.received) == [1, 3, 2]
