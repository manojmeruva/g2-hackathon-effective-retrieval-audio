"""Renders an EvaluationReport as Markdown and JSON."""

import json
from dataclasses import asdict

from app.application.evaluation_service import EvaluationReport

NDCG_K = 10
PRECISION_K = 5


def to_json(report: EvaluationReport) -> str:
    return json.dumps(asdict(report), indent=2, default=str) + "\n"


def to_markdown(report: EvaluationReport) -> str:
    sections = [
        "# Retrieval evaluation",
        f"Queries: {report.query_count}. Primary metric: Recall@{report.primary_k}. "
        f"Best weighted alpha (Recall@{report.primary_k}, MRR tie-break): "
        f"**{report.selected_alpha}**. Best hybrid overall: **{report.selected_hybrid}**.",
        overall_table(report),
        query_type_table(report),
        missed_queries(report),
    ]
    return "\n\n".join(sections) + "\n"


def overall_table(report: EvaluationReport) -> str:
    recall_headers = [f"R@{k}" for k in report.k_values]
    headers = [
        "Configuration",
        *recall_headers,
        "MRR",
        f"NDCG@{NDCG_K}",
        f"P@{PRECISION_K}",
        "p50 ms",
        "p95 ms",
    ]
    rows = [
        [
            result.config.name,
            *(f"{result.overall.recall[k]:.3f}" for k in report.k_values),
            f"{result.overall.mrr:.3f}",
            f"{result.overall.ndcg.get(NDCG_K, 0.0):.3f}",
            f"{result.overall.precision.get(PRECISION_K, 0.0):.3f}",
            f"{result.latency_p50_ms:.1f}",
            f"{result.latency_p95_ms:.1f}",
        ]
        for result in report.results
    ]
    return "## Overall\n\n" + markdown_table(headers, rows)


def query_type_table(report: EvaluationReport) -> str:
    k = report.primary_k
    query_types = sorted({qt for result in report.results for qt in result.by_query_type})
    headers = ["Query type", *(result.config.name for result in report.results)]
    rows = [
        [
            query_type,
            *(f"{result.by_query_type[query_type].recall[k]:.2f}" for result in report.results),
        ]
        for query_type in query_types
    ]
    return f"## Recall@{k} by query type\n\n" + markdown_table(headers, rows)


def missed_queries(report: EvaluationReport) -> str:
    final = report.results[-1]
    k = report.primary_k
    missed = [outcome.query_id for outcome in final.outcomes if outcome.metrics.recall[k] == 0.0]
    listing = ", ".join(missed) if missed else "none"
    return f"## Queries with Recall@{k} = 0 ({final.config.name})\n\n{listing}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)
