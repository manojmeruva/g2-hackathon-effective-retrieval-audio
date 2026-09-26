import pytest

from app.retrieval.hybrid import HybridRanker, ReciprocalRankFusion, min_max_normalize
from tests.factories import scored


def test_min_max_normalize_rescales_to_unit_interval() -> None:
    normalized = min_max_normalize([scored(1, 2.0), scored(2, 4.0), scored(3, 3.0)])

    assert normalized == {
        "conversation_01:1": 0.0,
        "conversation_01:2": 1.0,
        "conversation_01:3": 0.5,
    }


def test_min_max_normalize_equal_scores_become_one() -> None:
    assert set(min_max_normalize([scored(1, 0.3), scored(2, 0.3)]).values()) == {1.0}


def test_min_max_normalize_empty() -> None:
    assert min_max_normalize([]) == {}


def test_alpha_weights_keyword_against_semantic() -> None:
    keyword = [scored(1, 10.0), scored(2, 0.0)]
    semantic = [scored(2, 0.9), scored(1, 0.1)]

    keyword_heavy = HybridRanker(0.8).rank(keyword, semantic)
    semantic_heavy = HybridRanker(0.2).rank(keyword, semantic)

    assert keyword_heavy[0].segment.segment_index == 1
    assert semantic_heavy[0].segment.segment_index == 2
    assert keyword_heavy[0].hybrid_score == pytest.approx(0.8)


def test_segment_missing_from_one_list_scores_zero_there() -> None:
    ranked = HybridRanker(0.5).rank([scored(1, 1.0), scored(2, 0.5)], [scored(3, 0.7)])

    only_semantic = next(item for item in ranked if item.segment.segment_index == 3)
    assert only_semantic.keyword_score == 0.0
    assert only_semantic.semantic_score == 1.0
    assert len(ranked) == 3


def test_alpha_one_is_pure_keyword_order() -> None:
    ranked = HybridRanker(1.0).rank([scored(1, 0.2), scored(2, 0.9)], [scored(1, 0.99)])

    assert [item.segment.segment_index for item in ranked] == [2, 1]


def test_ties_break_deterministically_by_position() -> None:
    ranked = HybridRanker(0.5).rank([scored(3, 1.0), scored(1, 1.0)], [])

    assert [item.segment.segment_index for item in ranked] == [1, 3]


def test_empty_inputs_give_empty_ranking() -> None:
    assert HybridRanker(0.5).rank([], []) == []


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_outside_unit_interval_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        HybridRanker(alpha)


def test_rrf_sums_reciprocal_ranks_across_lists() -> None:
    keyword = [scored(1, 9.0), scored(2, 5.0)]
    semantic = [scored(2, 0.9), scored(3, 0.8)]

    ranked = ReciprocalRankFusion(k=60).rank(keyword, semantic)

    scores = {item.segment.segment_index: item.hybrid_score for item in ranked}
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert [item.segment.segment_index for item in ranked] == [2, 1, 3]


def test_rrf_ignores_score_magnitudes() -> None:
    close = ReciprocalRankFusion(k=60).rank([scored(1, 1.01), scored(2, 1.0)], [])
    far = ReciprocalRankFusion(k=60).rank([scored(1, 900.0), scored(2, 1.0)], [])

    assert [item.hybrid_score for item in close] == [item.hybrid_score for item in far]


def test_rrf_keeps_normalized_component_scores_for_display() -> None:
    ranked = ReciprocalRankFusion(k=60).rank([scored(1, 3.0), scored(2, 1.0)], [scored(2, 0.7)])

    by_index = {item.segment.segment_index: item for item in ranked}
    assert (by_index[1].keyword_score, by_index[1].semantic_score) == (1.0, 0.0)
    assert (by_index[2].keyword_score, by_index[2].semantic_score) == (0.0, 1.0)


def test_smaller_k_rewards_top_ranks_more() -> None:
    """Segment 1 is first in one list; segment 2 is fourth in both."""
    keyword = [scored(1, 1.0), scored(5, 0.9), scored(6, 0.8), scored(2, 0.7)]
    semantic = [scored(7, 0.9), scored(8, 0.8), scored(9, 0.7), scored(2, 0.6)]

    def scores(k: int) -> dict[int, float]:
        ranked = ReciprocalRankFusion(k).rank(keyword, semantic)
        return {item.segment.segment_index: item.hybrid_score for item in ranked}

    assert scores(60)[2] > scores(60)[1]
    assert scores(1)[1] > scores(1)[2]


def test_rrf_empty_inputs() -> None:
    assert ReciprocalRankFusion(k=60).rank([], []) == []


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=0)
