"""Runs keyword and semantic retrieval, fuses the lists, and optionally reranks."""

from app.config import SearchSettings
from app.domain.exceptions import ConfigurationError
from app.domain.models import (
    FusionMethod,
    RankedSegment,
    SearchMode,
    SearchResult,
    TranscriptSegment,
)
from app.domain.protocols import Reranker, Retriever, SegmentContext
from app.retrieval.hybrid import HybridRanker, ReciprocalRankFusion

KEYWORD_ONLY_ALPHA = 1.0
SEMANTIC_ONLY_ALPHA = 0.0


class SearchService:
    def __init__(
        self,
        keyword: Retriever,
        semantic: Retriever,
        context: SegmentContext,
        settings: SearchSettings,
        reranker: Reranker | None = None,
    ) -> None:
        self._keyword = keyword
        self._semantic = semantic
        self._context = context
        self._settings = settings
        self._reranker = reranker

    def search(
        self,
        query: str,
        top_k: int,
        mode: SearchMode = SearchMode.HYBRID,
        alpha: float | None = None,
        fusion: FusionMethod | None = None,
    ) -> list[SearchResult]:
        """Top-k results; `alpha` and `fusion` override the configured hybrid settings."""
        ranked = self.rank(query, mode, alpha, fusion)[:top_k]
        return [SearchResult(item, self._context_for(item.segment)) for item in ranked]

    def rank(
        self,
        query: str,
        mode: SearchMode,
        alpha: float | None = None,
        fusion: FusionMethod | None = None,
    ) -> list[RankedSegment]:
        if not query.strip():
            return []
        limit = self._settings.candidate_count
        if mode is SearchMode.KEYWORD:
            return HybridRanker(KEYWORD_ONLY_ALPHA).rank(self._keyword.retrieve(query, limit), [])
        if mode is SearchMode.SEMANTIC:
            return HybridRanker(SEMANTIC_ONLY_ALPHA).rank([], self._semantic.retrieve(query, limit))
        fused = self._fuser(alpha, fusion).rank(
            self._keyword.retrieve(query, limit), self._semantic.retrieve(query, limit)
        )
        if mode is SearchMode.HYBRID:
            return fused
        return self._rerank(query, fused)

    def _fuser(
        self, alpha: float | None, fusion: FusionMethod | None
    ) -> HybridRanker | ReciprocalRankFusion:
        method = self._settings.fusion if fusion is None else fusion
        if method is FusionMethod.RRF:
            return ReciprocalRankFusion(self._settings.rrf_k)
        return HybridRanker(self._settings.alpha if alpha is None else alpha)

    def _rerank(self, query: str, fused: list[RankedSegment]) -> list[RankedSegment]:
        if self._reranker is None:
            raise ConfigurationError("hybrid_rerank mode requires a reranker")
        return self._reranker.rerank(query, fused[: self._settings.rerank_candidate_count])

    def _context_for(self, segment: TranscriptSegment) -> tuple[TranscriptSegment, ...]:
        if self._settings.context_window == 0:
            return ()
        return tuple(self._context.neighbors(segment, self._settings.context_window))
