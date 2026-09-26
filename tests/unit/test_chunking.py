from app.config import ChunkingSettings
from app.domain.models import SpokenWord
from app.ingestion.chunking import chunk_words
from tests.factories import word

SETTINGS = ChunkingSettings(target_seconds=10.0, max_seconds=15.0, overlap_seconds=3.0)


def steady_words(count: int, speaker: str = "A", start: float = 0.0) -> list[SpokenWord]:
    """One word per second; every fifth word ends a sentence."""
    return [
        word(f"w{i}." if i % 5 == 4 else f"w{i}", start + i, start + i + 0.8, speaker)
        for i in range(count)
    ]


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_words([], SETTINGS) == []


def test_short_turn_becomes_one_chunk() -> None:
    chunks = chunk_words(steady_words(4), SETTINGS)

    assert len(chunks) == 1
    assert chunks[0].text == "w0 w1 w2 w3"
    assert (chunks[0].start_seconds, chunks[0].end_seconds) == (0.0, 3.8)


def test_chunks_never_mix_speakers() -> None:
    words = steady_words(3, "A") + steady_words(3, "B", start=3.0) + steady_words(2, "A", start=6.0)

    chunks = chunk_words(words, SETTINGS)

    assert [chunk.speaker for chunk in chunks] == ["A", "B", "A"]
    assert [chunk.segment_index for chunk in chunks] == [0, 1, 2]


def test_long_turn_closes_at_sentence_end_after_target() -> None:
    chunks = chunk_words(steady_words(30), SETTINGS)

    first = chunks[0]
    assert first.text.endswith(".")
    assert first.end_seconds - first.start_seconds >= SETTINGS.target_seconds


def test_no_chunk_exceeds_max_duration_without_sentence_ends() -> None:
    words = [word(f"w{i}", i, i + 0.8) for i in range(60)]

    chunks = chunk_words(words, SETTINGS)

    assert len(chunks) > 1
    assert all(c.end_seconds - c.start_seconds <= SETTINGS.max_seconds for c in chunks)


def test_consecutive_chunks_of_a_turn_overlap() -> None:
    chunks = chunk_words(steady_words(30), SETTINGS)

    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.start_seconds < previous.end_seconds
        assert previous.end_seconds - following.start_seconds <= SETTINGS.overlap_seconds


def test_every_word_is_covered() -> None:
    words = steady_words(40)

    chunks = chunk_words(words, SETTINGS)

    covered = {token for chunk in chunks for token in chunk.text.split()}
    assert covered == {w.text for w in words}


def test_single_word_longer_than_max_still_forms_a_chunk() -> None:
    chunks = chunk_words([word("long", 0.0, 20.0)], SETTINGS)

    assert [chunk.text for chunk in chunks] == ["long"]
