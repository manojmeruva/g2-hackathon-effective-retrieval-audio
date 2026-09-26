import math

import pytest

from app.evaluation.golden import RelevantTurn
from app.evaluation.metrics import (
    hits,
    hits_by_rank,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)
from tests.factories import segment

TURN_A = RelevantTurn("conversation_01:turn_03", "conversation_01", 20.0, 40.0)
TURN_B = RelevantTurn("conversation_01:turn_09", "conversation_01", 100.0, 110.0)

MISS: frozenset[str] = frozenset()
HIT_A = frozenset({TURN_A.turn_id})
HIT_B = frozenset({TURN_B.turn_id})


def test_segment_inside_turn_hits() -> None:
    assert hits(segment(start=25.0, end=35.0), TURN_A)


def test_segment_from_another_audio_misses() -> None:
    assert not hits(segment(audio_id="conversation_02", start=25.0, end=35.0), TURN_A)


def test_small_boundary_overlap_misses() -> None:
    assert not hits(segment(start=0.0, end=21.0), TURN_A)


def test_overlap_of_half_the_shorter_span_hits() -> None:
    assert hits(segment(start=15.0, end=25.0), TURN_A)


def test_hits_by_rank_lists_turns_per_result() -> None:
    retrieved = [segment(start=0, end=5), segment(start=101, end=109), segment(start=22, end=30)]

    assert hits_by_rank(retrieved, [TURN_A, TURN_B]) == [MISS, HIT_B, HIT_A]


def test_recall_counts_distinct_turns_in_top_k() -> None:
    ranked = [HIT_A, HIT_A, MISS, HIT_B]

    assert recall_at_k(ranked, 2, 1) == 0.5
    assert recall_at_k(ranked, 2, 3) == 0.5
    assert recall_at_k(ranked, 2, 4) == 1.0


def test_recall_with_k_beyond_results() -> None:
    assert recall_at_k([HIT_A], 2, 10) == 0.5


def test_recall_without_relevant_turns_is_zero() -> None:
    assert recall_at_k([MISS], 0, 5) == 0.0


def test_precision_divides_by_k_including_missing_positions() -> None:
    assert precision_at_k([HIT_A, MISS, HIT_B], 3) == pytest.approx(2 / 3)
    assert precision_at_k([HIT_A], 5) == pytest.approx(1 / 5)


def test_reciprocal_rank_of_first_hit() -> None:
    assert reciprocal_rank([MISS, MISS, HIT_A]) == pytest.approx(1 / 3)
    assert reciprocal_rank([MISS, MISS]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert ndcg_at_k([HIT_A, HIT_B, MISS], 2, 3) == pytest.approx(1.0)


def test_ndcg_discounts_late_hits_and_ignores_duplicates() -> None:
    ideal = 1 + 1 / math.log2(3)
    actual = 1 / math.log2(3) + 1 / math.log2(5)

    assert ndcg_at_k([MISS, HIT_A, HIT_A, HIT_B], 2, 4) == pytest.approx(actual / ideal)


def test_ndcg_without_hits_is_zero() -> None:
    assert ndcg_at_k([MISS, MISS], 1, 2) == 0.0


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 100) == 40.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_score_query_computes_all_metrics() -> None:
    retrieved = [segment(start=0, end=5), segment(start=22, end=30)]

    metrics = score_query(retrieved, [TURN_A, TURN_B], k_values=(1, 5))

    assert metrics.recall == {1: 0.0, 5: 0.5}
    assert metrics.precision == {1: 0.0, 5: pytest.approx(0.2)}
    assert metrics.reciprocal_rank == 0.5
