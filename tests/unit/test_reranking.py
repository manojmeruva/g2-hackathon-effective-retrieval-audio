from collections.abc import Iterable

from app.retrieval.reranking import CrossEncoderReranker
from tests.factories import ranked


def score_by_keyword(pairs: list[tuple[str, str]]) -> Iterable[float]:
    return [1.0 if "cloud" in text else 0.0 for _query, text in pairs]


def test_rerank_orders_by_cross_encoder_score_and_records_it() -> None:
    candidates = [ranked(1, 0.9, "hiring plan"), ranked(2, 0.5, "cloud migration")]

    reranked = CrossEncoderReranker(score_by_keyword).rerank("move to cloud", candidates)

    assert [item.segment.segment_index for item in reranked] == [2, 1]
    assert [item.rerank_score for item in reranked] == [1.0, 0.0]
    assert reranked[0].hybrid_score == 0.5


def test_rerank_keeps_hybrid_order_on_equal_scores() -> None:
    candidates = [ranked(1, 0.9), ranked(2, 0.5)]

    reranked = CrossEncoderReranker(lambda pairs: [0.3 for _ in pairs]).rerank("q", candidates)

    assert [item.segment.segment_index for item in reranked] == [1, 2]


def test_rerank_passes_query_text_pairs() -> None:
    seen: list[tuple[str, str]] = []

    def record(pairs: list[tuple[str, str]]) -> Iterable[float]:
        seen.extend(pairs)
        return [0.0 for _ in pairs]

    CrossEncoderReranker(record).rerank("budget", [ranked(1, 0.9, "twenty thousand")])

    assert seen == [("budget", "twenty thousand")]


def test_rerank_empty_candidates_skips_model() -> None:
    def fail(pairs: list[tuple[str, str]]) -> Iterable[float]:
        raise AssertionError("model should not be called")

    assert CrossEncoderReranker(fail).rerank("q", []) == []
