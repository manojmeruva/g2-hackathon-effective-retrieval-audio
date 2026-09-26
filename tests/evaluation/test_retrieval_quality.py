"""Retrieval quality on the real index, measured with the golden query set.

These are relative checks, not hardcoded scores: fusing keyword and semantic
search must not be worse than either alone, and evaluation must be reproducible.
Requires `docker compose up -d` and `python -m app.cli ingest ./data/audio`.
"""

from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from app.application.evaluation_service import (
    ConfigResult,
    EvaluationConfig,
    EvaluationService,
)
from app.application.search_service import SearchService
from app.config import load_settings
from app.domain.models import SearchMode
from app.evaluation.golden import load_golden_set
from app.wiring import build_search_service

pytestmark = pytest.mark.integration

QUERIES_PATH = Path("data/golden_queries.json")
GROUND_TRUTH_DIR = Path("data/ground_truth")
PROBE_QUERY = "Kubernetes"


@pytest.fixture(scope="module")
def search_service() -> SearchService:
    service = build_search_service(load_settings(), with_reranker=False)
    try:
        indexed = service.search(PROBE_QUERY, top_k=1, mode=SearchMode.KEYWORD)
    except OperationalError:
        pytest.skip("PostgreSQL is not available; run `docker compose up -d`")
    if not indexed:
        pytest.skip("the index is empty; run `python -m app.cli ingest ./data/audio`")
    return service


@pytest.fixture(scope="module")
def evaluator(search_service: SearchService) -> EvaluationService:
    settings = load_settings().evaluation
    return EvaluationService(
        search_service, load_golden_set(QUERIES_PATH, GROUND_TRUTH_DIR), settings
    )


def primary_recall(result: ConfigResult) -> float:
    return result.overall.recall[load_settings().evaluation.primary_k]


def test_hybrid_is_at_least_as_good_as_either_retriever(evaluator: EvaluationService) -> None:
    keyword = evaluator.evaluate(EvaluationConfig("keyword", SearchMode.KEYWORD))
    semantic = evaluator.evaluate(EvaluationConfig("semantic", SearchMode.SEMANTIC))
    hybrid = evaluator.evaluate(EvaluationConfig("hybrid", SearchMode.HYBRID))

    assert primary_recall(hybrid) >= primary_recall(keyword)
    assert primary_recall(hybrid) >= primary_recall(semantic)


def test_evaluation_is_reproducible(evaluator: EvaluationService) -> None:
    config = EvaluationConfig("hybrid", SearchMode.HYBRID)

    first = evaluator.evaluate(config)
    second = evaluator.evaluate(config)

    assert [o.retrieved for o in first.outcomes] == [o.retrieved for o in second.outcomes]
    assert first.overall == second.overall
