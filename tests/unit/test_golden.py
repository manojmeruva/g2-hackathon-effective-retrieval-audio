import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.domain.exceptions import GoldenDatasetError
from app.evaluation.golden import QueryType, load_golden_set


@pytest.fixture
def ground_truth_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "ground_truth"
    directory.mkdir()
    conversation = {
        "conversation_id": "conversation_01",
        "title": "ignored",
        "turns": [
            {"index": 0, "speaker": "Alice", "start_seconds": 0.0, "end_seconds": 4.0, "text": "a"},
            {"index": 1, "speaker": "Bob", "start_seconds": 5.0, "end_seconds": 9.0, "text": "b"},
        ],
    }
    (directory / "conversation_01.json").write_text(json.dumps(conversation))
    return directory


def write_queries(tmp_path: Path, queries: list[dict[str, object]]) -> Path:
    path = tmp_path / "golden_queries.json"
    path.write_text(json.dumps(queries))
    return path


def query(query_id: str = "q01", relevant: list[str] | None = None) -> dict[str, object]:
    return {
        "id": query_id,
        "query": "What did Bob say?",
        "type": "speaker_specific",
        "relevant": relevant or ["conversation_01:turn_01"],
    }


def test_labels_resolve_to_turn_time_spans(tmp_path: Path, ground_truth_dir: Path) -> None:
    labeled = load_golden_set(write_queries(tmp_path, [query()]), ground_truth_dir)

    assert labeled[0].query_type is QueryType.SPEAKER_SPECIFIC
    turn = labeled[0].relevant[0]
    assert (turn.audio_id, turn.start_seconds, turn.end_seconds) == ("conversation_01", 5.0, 9.0)


def test_duplicate_labels_are_counted_once(tmp_path: Path, ground_truth_dir: Path) -> None:
    labels = ["conversation_01:turn_01", "conversation_01:turn_01"]

    labeled = load_golden_set(write_queries(tmp_path, [query(relevant=labels)]), ground_truth_dir)

    assert len(labeled[0].relevant) == 1


def test_unknown_turn_is_an_error(tmp_path: Path, ground_truth_dir: Path) -> None:
    path = write_queries(tmp_path, [query(relevant=["conversation_01:turn_09"])])

    with pytest.raises(GoldenDatasetError, match="turn_09"):
        load_golden_set(path, ground_truth_dir)


def test_duplicate_query_ids_are_an_error(tmp_path: Path, ground_truth_dir: Path) -> None:
    with pytest.raises(GoldenDatasetError, match="duplicate"):
        load_golden_set(write_queries(tmp_path, [query(), query()]), ground_truth_dir)


def test_empty_query_set_is_an_error(tmp_path: Path, ground_truth_dir: Path) -> None:
    with pytest.raises(GoldenDatasetError):
        load_golden_set(write_queries(tmp_path, []), ground_truth_dir)


def test_query_without_labels_is_invalid(tmp_path: Path, ground_truth_dir: Path) -> None:
    with pytest.raises(ValidationError):
        load_golden_set(write_queries(tmp_path, [query() | {"relevant": []}]), ground_truth_dir)
