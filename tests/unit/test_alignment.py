import pytest

from app.domain.exceptions import DiarizationError
from app.domain.models import SpeakerTurn, TranscribedSegment, TranscribedWord
from app.ingestion.alignment import assign_speakers


def transcribed(*words: tuple[str, float, float]) -> TranscribedSegment:
    items = tuple(TranscribedWord(text, start, end) for text, start, end in words)
    return TranscribedSegment(
        " ".join(w.text for w in items), items[0].start_seconds, items[-1].end_seconds, items
    )


TURNS = [SpeakerTurn("A", 0.0, 5.0), SpeakerTurn("B", 5.0, 10.0)]


def test_each_word_gets_the_speaker_of_the_overlapping_turn() -> None:
    words = assign_speakers([transcribed(("hello", 1.0, 1.5), ("there", 6.0, 6.5))], TURNS)

    assert [(w.text, w.speaker) for w in words] == [("hello", "A"), ("there", "B")]


def test_word_straddling_a_boundary_goes_to_the_larger_overlap() -> None:
    words = assign_speakers([transcribed(("across", 4.0, 5.5))], TURNS)

    assert words[0].speaker == "A"


def test_word_in_a_gap_goes_to_the_nearest_turn() -> None:
    turns = [SpeakerTurn("A", 0.0, 2.0), SpeakerTurn("B", 8.0, 10.0)]

    words = assign_speakers(
        [transcribed(("early", 2.5, 3.0)), transcribed(("late", 7.0, 7.5))], turns
    )

    assert [w.speaker for w in words] == ["A", "B"]


def test_word_timestamps_are_preserved() -> None:
    words = assign_speakers([transcribed(("hello", 1.0, 1.5))], TURNS)

    assert (words[0].start_seconds, words[0].end_seconds) == (1.0, 1.5)


def test_no_turns_is_an_error() -> None:
    with pytest.raises(DiarizationError):
        assign_speakers([transcribed(("hello", 1.0, 1.5))], [])


def test_no_segments_gives_no_words() -> None:
    assert assign_speakers([], TURNS) == []
