"""Fusion of keyword and semantic result lists.

Two strategies are provided so they can be compared on the golden set:

- HybridRanker (score-based): raw keyword (ts_rank_cd) and semantic (cosine)
  scores are on different scales, so each list is min-max normalized to
  [0, 1] and combined as

      hybrid_score = alpha * keyword_score + (1 - alpha) * semantic_score

- ReciprocalRankFusion (rank-based): ignores raw scores and sums
  1 / (k + rank) over the lists that contain a segment.

In both, a segment found by only one retriever scores 0 for the other.
Every ranked segment also carries its normalized keyword and semantic scores
so results can show why they were retrieved.
"""

from collections.abc import Mapping, Sequence

from app.domain.models import RankedSegment, ScoredSegment, TranscriptSegment


def min_max_normalize(scored: Sequence[ScoredSegment]) -> dict[str, float]:
    """Map segment IDs to scores rescaled to [0, 1]; equal scores all become 1."""
    if not scored:
        return {}
    scores = [item.score for item in scored]
    low, high = min(scores), max(scores)
    spread = high - low
    return {
        item.segment.segment_id: (item.score - low) / spread if spread > 0 else 1.0
        for item in scored
    }


def unique_segments(*result_lists: Sequence[ScoredSegment]) -> dict[str, TranscriptSegment]:
    return {item.segment.segment_id: item.segment for results in result_lists for item in results}


def build_ranking(
    keyword: Sequence[ScoredSegment],
    semantic: Sequence[ScoredSegment],
    fused_scores: Mapping[str, float],
) -> list[RankedSegment]:
    """Attach component scores to fused scores and sort, best first."""
    keyword_scores = min_max_normalize(keyword)
    semantic_scores = min_max_normalize(semantic)
    ranked = [
        RankedSegment(
            segment=segment,
            keyword_score=keyword_scores.get(segment_id, 0.0),
            semantic_score=semantic_scores.get(segment_id, 0.0),
            hybrid_score=fused_scores[segment_id],
        )
        for segment_id, segment in unique_segments(keyword, semantic).items()
    ]
    return sorted(
        ranked,
        key=lambda item: (-item.hybrid_score, item.segment.audio_id, item.segment.segment_index),
    )


class HybridRanker:
    """Score-based fusion: alpha-weighted sum of min-max normalized scores."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self._alpha = alpha

    def rank(
        self, keyword: Sequence[ScoredSegment], semantic: Sequence[ScoredSegment]
    ) -> list[RankedSegment]:
        keyword_scores = min_max_normalize(keyword)
        semantic_scores = min_max_normalize(semantic)
        fused = {
            segment_id: self._alpha * keyword_scores.get(segment_id, 0.0)
            + (1 - self._alpha) * semantic_scores.get(segment_id, 0.0)
            for segment_id in unique_segments(keyword, semantic)
        }
        return build_ranking(keyword, semantic, fused)


class ReciprocalRankFusion:
    """Rank-based fusion; each input list must already be ordered best first."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._k = k

    def rank(
        self, keyword: Sequence[ScoredSegment], semantic: Sequence[ScoredSegment]
    ) -> list[RankedSegment]:
        fused = dict.fromkeys(unique_segments(keyword, semantic), 0.0)
        for results in (keyword, semantic):
            for rank, item in enumerate(results, start=1):
                fused[item.segment.segment_id] += 1.0 / (self._k + rank)
        return build_ranking(keyword, semantic, fused)
