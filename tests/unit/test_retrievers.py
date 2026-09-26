from collections.abc import Sequence

import pytest

from app.domain.models import ScoredSegment
from app.retrieval.keyword import KeywordRetriever, extract_terms
from app.retrieval.semantic import SemanticRetriever
from tests.factories import scored


class FakeIndex:
    def __init__(self, results: list[ScoredSegment]) -> None:
        self._results = results
        self.keyword_calls: list[tuple[list[str], int]] = []
        self.vector_calls: list[tuple[list[float], int]] = []

    def keyword_search(self, terms: Sequence[str], limit: int) -> list[ScoredSegment]:
        self.keyword_calls.append((list(terms), limit))
        return self._results

    def vector_search(self, embedding: Sequence[float], limit: int) -> list[ScoredSegment]:
        self.vector_calls.append((list(embedding), limit))
        return self._results


class FakeEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(1.0,) for _ in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (float(len(text)), 1.0)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Kubernetes", ["kubernetes"]),
        ("zero trust architecture", ["zero", "trust", "architecture"]),
        ("SOC 2 Type Two", ["soc", "2", "type", "two"]),
        ("What's the plan? The plan!", ["what", "s", "the", "plan"]),
        ("a & b | !c:*", ["a", "b", "c"]),
        ("  ?!  ", []),
    ],
)
def test_extract_terms(query: str, expected: list[str]) -> None:
    assert extract_terms(query) == expected


def test_keyword_retriever_passes_terms_and_limit() -> None:
    index = FakeIndex([scored(1, 0.5)])

    results = KeywordRetriever(index).retrieve("PostgreSQL replication", limit=7)

    assert results == [scored(1, 0.5)]
    assert index.keyword_calls == [(["postgresql", "replication"], 7)]


def test_keyword_retriever_skips_search_without_terms() -> None:
    index = FakeIndex([scored(1, 0.5)])

    assert KeywordRetriever(index).retrieve("?!", limit=7) == []
    assert index.keyword_calls == []


def test_semantic_retriever_embeds_query_and_searches() -> None:
    index = FakeIndex([scored(2, 0.9)])

    results = SemanticRetriever(FakeEmbedder(), index).retrieve("cloud", limit=5)

    assert results == [scored(2, 0.9)]
    assert index.vector_calls == [([5.0, 1.0], 5)]


def test_semantic_retriever_ignores_blank_query() -> None:
    index = FakeIndex([scored(2, 0.9)])

    assert SemanticRetriever(FakeEmbedder(), index).retrieve("   ", limit=5) == []
    assert index.vector_calls == []
