"""Retrieval metrics over reference turns: Recall@K, Precision@K, MRR, NDCG@K.

A retrieved segment "hits" a relevant turn when both come from the same audio
file and their time spans overlap by at least half of the shorter span. Every
metric is computed from `hits_by_rank`: for each ranked result, the set of
relevant turn IDs it hits (empty when the result is not relevant).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.models import TranscriptSegment
from app.evaluation.golden import RelevantTurn

MIN_OVERLAP_FRACTION = 0.5
HitsByRank = Sequence[frozenset[str]]


def hits(segment: TranscriptSegment, turn: RelevantTurn) -> bool:
    if segment.audio_id != turn.audio_id:
        return False
    overlap = min(segment.end_seconds, turn.end_seconds) - max(
        segment.start_seconds, turn.start_seconds
    )
    shorter = min(
        segment.end_seconds - segment.start_seconds, turn.end_seconds - turn.start_seconds
    )
    return overlap > 0 and overlap >= MIN_OVERLAP_FRACTION * shorter


def hits_by_rank(
    retrieved: Sequence[TranscriptSegment], relevant: Sequence[RelevantTurn]
) -> list[frozenset[str]]:
    return [
        frozenset(turn.turn_id for turn in relevant if hits(segment, turn)) for segment in retrieved
    ]


def recall_at_k(ranked_hits: HitsByRank, relevant_count: int, k: int) -> float:
    """Share of relevant turns found anywhere in the top k."""
    if relevant_count == 0:
        return 0.0
    found = frozenset().union(*ranked_hits[:k])
    return len(found) / relevant_count


def precision_at_k(ranked_hits: HitsByRank, k: int) -> float:
    """Share of the top k positions holding a relevant result; empty positions count as misses."""
    return sum(1 for turn_ids in ranked_hits[:k] if turn_ids) / k


def reciprocal_rank(ranked_hits: HitsByRank) -> float:
    for rank, turn_ids in enumerate(ranked_hits, start=1):
        if turn_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_hits: HitsByRank, relevant_count: int, k: int) -> float:
    """Binary-gain NDCG; a result gains only if it finds a turn not already credited."""
    if relevant_count == 0:
        return 0.0
    credited: set[str] = set()
    dcg = 0.0
    for position, turn_ids in enumerate(ranked_hits[:k]):
        new_turns = turn_ids - credited
        if new_turns:
            credited |= new_turns
            dcg += 1.0 / math.log2(position + 2)
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(k, relevant_count)))
    return dcg / ideal


def percentile(values: Sequence[float], percent: float) -> float:
    """Linear-interpolated percentile, e.g. percent=95 for p95."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class QueryMetrics:
    recall: dict[int, float]
    precision: dict[int, float]
    ndcg: dict[int, float]
    reciprocal_rank: float


def score_query(
    retrieved: Sequence[TranscriptSegment],
    relevant: Sequence[RelevantTurn],
    k_values: Sequence[int],
) -> QueryMetrics:
    ranked_hits = hits_by_rank(retrieved, relevant)
    count = len(relevant)
    return QueryMetrics(
        recall={k: recall_at_k(ranked_hits, count, k) for k in k_values},
        precision={k: precision_at_k(ranked_hits, k) for k in k_values},
        ndcg={k: ndcg_at_k(ranked_hits, count, k) for k in k_values},
        reciprocal_rank=reciprocal_rank(ranked_hits),
    )
