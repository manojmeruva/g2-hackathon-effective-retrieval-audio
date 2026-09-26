"""Runs labelled queries through SearchService and reports retrieval quality.

Every configuration is evaluated on the same golden query set: keyword,
semantic, weighted hybrid for each alpha, reciprocal rank fusion, and
reranking on top of the best hybrid. "Best" means highest measured
Recall@primary_k, with MRR as the tie-break.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from app.config import EvaluationSettings
from app.domain.models import FusionMethod, SearchMode, SearchResult
from app.evaluation.golden import LabeledQuery
from app.evaluation.metrics import QueryMetrics, percentile, score_query

MILLISECONDS_PER_SECOND = 1000.0
MEDIAN = 50
TAIL_PERCENTILE = 95


class Searcher(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        mode: SearchMode,
        alpha: float | None = None,
        fusion: FusionMethod | None = None,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class EvaluationConfig:
    name: str
    mode: SearchMode
    alpha: float | None = None
    fusion: FusionMethod | None = None


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    query_type: str
    metrics: QueryMetrics
    latency_ms: float
    retrieved: tuple[str, ...]


@dataclass(frozen=True)
class MetricSummary:
    query_count: int
    recall: dict[int, float]
    precision: dict[int, float]
    ndcg: dict[int, float]
    mrr: float


@dataclass(frozen=True)
class ConfigResult:
    config: EvaluationConfig
    overall: MetricSummary
    by_query_type: dict[str, MetricSummary]
    latency_p50_ms: float
    latency_p95_ms: float
    outcomes: tuple[QueryOutcome, ...]


@dataclass(frozen=True)
class EvaluationReport:
    query_count: int
    k_values: tuple[int, ...]
    primary_k: int
    selected_alpha: float
    selected_hybrid: str
    results: tuple[ConfigResult, ...]


def summarize(metrics: Sequence[QueryMetrics], k_values: Sequence[int]) -> MetricSummary:
    return MetricSummary(
        query_count=len(metrics),
        recall={k: fmean(m.recall[k] for m in metrics) for k in k_values},
        precision={k: fmean(m.precision[k] for m in metrics) for k in k_values},
        ndcg={k: fmean(m.ndcg[k] for m in metrics) for k in k_values},
        mrr=fmean(m.reciprocal_rank for m in metrics),
    )


def rerank_config(hybrid: EvaluationConfig) -> EvaluationConfig:
    """Rerank the candidates produced by an already-evaluated hybrid configuration."""
    return EvaluationConfig(
        f"{hybrid.name} + rerank", SearchMode.HYBRID_RERANK, hybrid.alpha, hybrid.fusion
    )


class EvaluationService:
    def __init__(
        self,
        searcher: Searcher,
        queries: Sequence[LabeledQuery],
        settings: EvaluationSettings,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not queries:
            raise ValueError("evaluation needs at least one query")
        self._searcher = searcher
        self._queries = queries
        self._settings = settings
        self._clock = clock

    def run(self, include_rerank: bool = True) -> EvaluationReport:
        baselines = [
            self.evaluate(EvaluationConfig("keyword", SearchMode.KEYWORD)),
            self.evaluate(EvaluationConfig("semantic", SearchMode.SEMANTIC)),
        ]
        weighted = [
            self.evaluate(
                EvaluationConfig(
                    f"hybrid a={alpha}", SearchMode.HYBRID, alpha, FusionMethod.WEIGHTED
                )
            )
            for alpha in self._settings.alpha_grid
        ]
        rrf = self.evaluate(
            EvaluationConfig("hybrid rrf", SearchMode.HYBRID, fusion=FusionMethod.RRF)
        )
        best_weighted = self.select_best(weighted)
        best_hybrid = self.select_best([*weighted, rrf])
        results = [*baselines, *weighted, rrf]
        if include_rerank:
            results.append(self.evaluate(rerank_config(best_hybrid.config)))
        return EvaluationReport(
            query_count=len(self._queries),
            k_values=self._settings.k_values,
            primary_k=self._settings.primary_k,
            selected_alpha=best_weighted.config.alpha or 0.0,
            selected_hybrid=best_hybrid.config.name,
            results=tuple(results),
        )

    def select_best(self, results: Sequence[ConfigResult]) -> ConfigResult:
        """Highest Recall@primary_k, then highest MRR; earlier configs win exact ties."""
        primary_k = self._settings.primary_k
        return max(
            results, key=lambda result: (result.overall.recall[primary_k], result.overall.mrr)
        )

    def evaluate(self, config: EvaluationConfig) -> ConfigResult:
        self._warm_up(config)
        outcomes = tuple(self._evaluate_query(query, config) for query in self._queries)
        k_values = self._settings.k_values
        latencies = [outcome.latency_ms for outcome in outcomes]
        return ConfigResult(
            config=config,
            overall=summarize([outcome.metrics for outcome in outcomes], k_values),
            by_query_type=self._summarize_by_type(outcomes),
            latency_p50_ms=percentile(latencies, MEDIAN),
            latency_p95_ms=percentile(latencies, TAIL_PERCENTILE),
            outcomes=outcomes,
        )

    def _warm_up(self, config: EvaluationConfig) -> None:
        """Run one untimed search so model loading does not distort latency."""
        self._search(self._queries[0].query, config)

    def _evaluate_query(self, query: LabeledQuery, config: EvaluationConfig) -> QueryOutcome:
        started = self._clock()
        results = self._search(query.query, config)
        latency_ms = (self._clock() - started) * MILLISECONDS_PER_SECOND
        segments = [result.segment for result in results]
        return QueryOutcome(
            query_id=query.id,
            query_type=query.query_type.value,
            metrics=score_query(segments, query.relevant, self._settings.k_values),
            latency_ms=latency_ms,
            retrieved=tuple(segment.segment_id for segment in segments),
        )

    def _search(self, query: str, config: EvaluationConfig) -> list[SearchResult]:
        top_k = max(self._settings.k_values)
        return self._searcher.search(query, top_k, config.mode, config.alpha, config.fusion)

    def _summarize_by_type(self, outcomes: Sequence[QueryOutcome]) -> dict[str, MetricSummary]:
        query_types = sorted({outcome.query_type for outcome in outcomes})
        return {
            query_type: summarize(
                [o.metrics for o in outcomes if o.query_type == query_type],
                self._settings.k_values,
            )
            for query_type in query_types
        }
