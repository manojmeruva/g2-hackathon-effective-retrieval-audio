"""The committed golden dataset meets the evaluation requirements.

Five or six conversations of 8–10 minutes, exactly two speakers each, a unique
speaker pair per file, and 40–60 labelled queries covering every query type.
"""

import wave
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.golden import (
    GroundTruthConversation,
    LabeledQuery,
    QueryType,
    load_golden_set,
)

AUDIO_DIR = Path("data/audio")
GROUND_TRUTH_DIR = Path("data/ground_truth")
QUERIES_PATH = Path("data/golden_queries.json")

MIN_CONVERSATIONS, MAX_CONVERSATIONS = 5, 6
MIN_MINUTES, MAX_MINUTES = 8.0, 10.0
SPEAKERS_PER_CONVERSATION = 2
MIN_QUERIES, MAX_QUERIES = 40, 60
MIN_QUERIES_PER_TYPE = 5
SECONDS_PER_MINUTE = 60


def conversations() -> list[GroundTruthConversation]:
    return [
        GroundTruthConversation.model_validate_json(path.read_text())
        for path in sorted(GROUND_TRUTH_DIR.glob("*.json"))
    ]


def audio_minutes(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / reader.getframerate() / SECONDS_PER_MINUTE


@pytest.fixture(scope="module")
def golden_set() -> list[LabeledQuery]:
    return load_golden_set(QUERIES_PATH, GROUND_TRUTH_DIR)


def test_every_audio_file_has_ground_truth() -> None:
    audio_ids = {path.stem for path in AUDIO_DIR.glob("*.wav")}

    assert MIN_CONVERSATIONS <= len(audio_ids) <= MAX_CONVERSATIONS
    assert audio_ids == {conversation.conversation_id for conversation in conversations()}


@pytest.mark.parametrize("path", sorted(AUDIO_DIR.glob("*.wav")), ids=lambda path: path.stem)
def test_audio_is_8_to_10_minutes(path: Path) -> None:
    assert MIN_MINUTES <= audio_minutes(path) <= MAX_MINUTES


def test_each_conversation_has_exactly_two_speakers() -> None:
    for conversation in conversations():
        speakers = {turn.speaker for turn in conversation.turns}
        assert len(speakers) == SPEAKERS_PER_CONVERSATION, conversation.conversation_id


def test_speaker_pairs_are_unique() -> None:
    pairs = [frozenset(turn.speaker for turn in c.turns) for c in conversations()]

    assert len(set(pairs)) == len(pairs)


def test_query_count_and_every_type_is_well_covered(golden_set: list[LabeledQuery]) -> None:
    per_type = Counter(query.query_type for query in golden_set)

    assert MIN_QUERIES <= len(golden_set) <= MAX_QUERIES
    assert set(per_type) == set(QueryType)
    assert min(per_type.values()) >= MIN_QUERIES_PER_TYPE


def test_every_conversation_is_queried(golden_set: list[LabeledQuery]) -> None:
    queried = {turn.audio_id for query in golden_set for turn in query.relevant}

    assert queried == {conversation.conversation_id for conversation in conversations()}
