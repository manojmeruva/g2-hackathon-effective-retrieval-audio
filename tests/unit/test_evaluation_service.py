from itertools import count

import pytest

from app.application.evaluation_service import EvaluationReport, EvaluationService
from app.config import EvaluationSettings
from app.domain.models import FusionMethod, RankedSegment, SearchMode, SearchResult
from app.evaluation.golden import LabeledQuery, QueryType, RelevantTurn
from tests.factories import segment

RELEVANT = RelevantTurn("conversation_01:turn_05", "conversation_01", 50.0, 60.0)
QUERIES = [
    LabeledQuery("q1", "cloud costs", QueryType.PARAPHRASE, (RELEVANT,)),
    LabeledQuery("q2", "Kubernetes", QueryType.EXACT_KEYWORD, (RELEVANT,)),
]
SETTINGS = EvaluationSettings(k_values=(1, 5), primary_k=5, alpha_grid=(0.2, 0.8))


def result(start: float) -> SearchResult:
    return SearchResult(RankedSegment(segment(start=start, end=start + 8), 0.5, 0.5, 0.5), ())


class ScriptedSearcher:
    """Finds the relevant turn at a rank that depends on the configuration.

    Weighted alpha=0.2 ranks it first; everything else (including RRF) second,
    except keyword search, which never finds it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, SearchMode, float | None, FusionMethod | None]] = []

    def search(
        self,
        query: str,
        top_k: int,
        mode: SearchMode,
        alpha: float | None = None,
        fusion: FusionMethod | None = None,
    ) -> list[SearchResult]:
        self.calls.append((top_k, mode, alpha, fusion))
        miss = result(200.0)
        hit = result(51.0)
        if mode is SearchMode.KEYWORD:
            return [miss] * top_k
        if fusion is FusionMethod.WEIGHTED and alpha == 0.2:
            return [hit, miss]
        return [miss, hit]


Run = tuple[EvaluationReport, ScriptedSearcher]
MILLISECOND = 0.001


@pytest.fixture
def run() -> Run:
    searcher = ScriptedSearcher()
    ticks = count(start=0.0, step=MILLISECOND)
    service = EvaluationService(searcher, QUERIES, SETTINGS, clock=lambda: next(ticks))
    return service.run(), searcher


def test_runs_baselines_alpha_sweep_and_rerank(run: Run) -> None:
    report, _searcher = run

    names = [r.config.name for r in report.results]
    assert names == [
        "keyword",
        "semantic",
        "hybrid a=0.2",
        "hybrid a=0.8",
        "hybrid rrf",
        "hybrid a=0.2 + rerank",
    ]


def test_selects_alpha_by_primary_recall_then_mrr(
    run: Run,
) -> None:
    report, _searcher = run

    assert report.selected_alpha == 0.2
    assert report.selected_hybrid == "hybrid a=0.2"


def test_metrics_and_latency_come_from_measured_runs(
    run: Run,
) -> None:
    report, _searcher = run
    keyword, semantic, best = report.results[0], report.results[1], report.results[2]

    assert keyword.overall.recall[5] == 0.0
    assert semantic.overall.recall == {1: 0.0, 5: 1.0}
    assert semantic.overall.mrr == 0.5
    assert best.overall.mrr == 1.0
    assert best.latency_p50_ms == pytest.approx(1.0)
    assert set(best.by_query_type) == {"paraphrase", "exact_keyword"}


def test_searches_top_max_k_with_one_untimed_warm_up(
    run: Run,
) -> None:
    _report, searcher = run

    configs = 6
    assert len(searcher.calls) == configs * (len(QUERIES) + 1)
    assert {call[0] for call in searcher.calls} == {5}


def test_rerank_can_be_skipped() -> None:
    service = EvaluationService(ScriptedSearcher(), QUERIES, SETTINGS)

    report = service.run(include_rerank=False)

    assert all(r.config.mode is not SearchMode.HYBRID_RERANK for r in report.results)


def test_needs_queries() -> None:
    with pytest.raises(ValueError):
        EvaluationService(ScriptedSearcher(), [], SETTINGS)


def test_rerank_runs_on_the_best_hybrid_fusion() -> None:
    class RrfWins(ScriptedSearcher):
        def search(
            self,
            query: str,
            top_k: int,
            mode: SearchMode,
            alpha: float | None = None,
            fusion: FusionMethod | None = None,
        ) -> list[SearchResult]:
            self.calls.append((top_k, mode, alpha, fusion))
            if fusion is FusionMethod.RRF:
                return [result(51.0)]
            return [result(200.0)]

    searcher = RrfWins()

    report = EvaluationService(searcher, QUERIES, SETTINGS).run()

    assert report.selected_hybrid == "hybrid rrf"
    assert report.results[-1].config.name == "hybrid rrf + rerank"
    assert (SearchMode.HYBRID_RERANK, FusionMethod.RRF) in {
        (mode, fusion) for _k, mode, _alpha, fusion in searcher.calls
    }
