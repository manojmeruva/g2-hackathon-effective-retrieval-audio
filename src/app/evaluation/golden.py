"""Loads the golden query set and resolves its labels against the ground truth.

Queries are labelled with reference turn IDs such as "conversation_03:turn_07".
Those IDs come from the ground-truth transcripts, not from ingestion, so the
labels stay valid when chunking or transcription changes.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from app.domain.exceptions import GoldenDatasetError


class QueryType(StrEnum):
    EXACT_KEYWORD = "exact_keyword"
    MULTI_WORD = "multi_word"
    SYNONYM = "synonym"
    PARAPHRASE = "paraphrase"
    SEMANTIC = "semantic"
    SPEAKER_SPECIFIC = "speaker_specific"
    NUMERIC = "numeric"
    CONVERSATIONAL = "conversational"


class GoldenQuery(BaseModel):
    id: str
    query: str = Field(min_length=1)
    type: QueryType
    relevant: list[str] = Field(min_length=1)


class GroundTruthTurn(BaseModel):
    index: int
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


class GroundTruthConversation(BaseModel):
    conversation_id: str
    turns: list[GroundTruthTurn]


@dataclass(frozen=True)
class RelevantTurn:
    turn_id: str
    audio_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    query: str
    query_type: QueryType
    relevant: tuple[RelevantTurn, ...]


def turn_id(conversation_id: str, index: int) -> str:
    return f"{conversation_id}:turn_{index:02d}"


def load_turns(ground_truth_dir: Path) -> dict[str, RelevantTurn]:
    turns: dict[str, RelevantTurn] = {}
    for path in sorted(ground_truth_dir.glob("*.json")):
        conversation = GroundTruthConversation.model_validate_json(path.read_text())
        for turn in conversation.turns:
            identifier = turn_id(conversation.conversation_id, turn.index)
            turns[identifier] = RelevantTurn(
                identifier, conversation.conversation_id, turn.start_seconds, turn.end_seconds
            )
    return turns


def load_golden_set(queries_path: Path, ground_truth_dir: Path) -> list[LabeledQuery]:
    queries = TypeAdapter(list[GoldenQuery]).validate_json(queries_path.read_text())
    if not queries:
        raise GoldenDatasetError(f"{queries_path} contains no queries")
    duplicates = {query.id for query in queries if [q.id for q in queries].count(query.id) > 1}
    if duplicates:
        raise GoldenDatasetError(f"duplicate query ids: {sorted(duplicates)}")
    turns = load_turns(ground_truth_dir)
    return [resolve_labels(query, turns) for query in queries]


def resolve_labels(query: GoldenQuery, turns: dict[str, RelevantTurn]) -> LabeledQuery:
    unknown = [label for label in query.relevant if label not in turns]
    if unknown:
        raise GoldenDatasetError(f"query {query.id} references unknown turns: {unknown}")
    return LabeledQuery(
        id=query.id,
        query=query.query,
        query_type=query.type,
        relevant=tuple(turns[label] for label in dict.fromkeys(query.relevant)),
    )
