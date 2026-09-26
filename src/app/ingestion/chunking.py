"""Groups speaker-labelled words into retrievable chunks.

Chunks never mix speakers, so every search result has one speaker. Within a
speaker's run of words, a chunk closes at the first sentence end after the
target duration, or before it would exceed the maximum duration. Consecutive
chunks of the same run overlap by roughly `overlap_seconds` so that an idea
spanning a boundary is still retrievable from one chunk.
"""

from collections.abc import Sequence

from app.config import ChunkingSettings
from app.domain.models import Chunk, SpokenWord

SENTENCE_ENDINGS = (".", "?", "!")


def chunk_words(words: Sequence[SpokenWord], settings: ChunkingSettings) -> list[Chunk]:
    windows = [window for run in split_speaker_runs(words) for window in window_run(run, settings)]
    return [to_chunk(index, window) for index, window in enumerate(windows)]


def split_speaker_runs(words: Sequence[SpokenWord]) -> list[list[SpokenWord]]:
    runs: list[list[SpokenWord]] = []
    for word in words:
        if runs and runs[-1][-1].speaker == word.speaker:
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def window_run(run: Sequence[SpokenWord], settings: ChunkingSettings) -> list[Sequence[SpokenWord]]:
    windows: list[Sequence[SpokenWord]] = []
    start = 0
    while start < len(run):
        end = find_window_end(run, start, settings)
        windows.append(run[start:end])
        if end == len(run):
            break
        start = find_overlap_start(run, start, end, settings.overlap_seconds)
    return windows


def find_window_end(run: Sequence[SpokenWord], start: int, settings: ChunkingSettings) -> int:
    """Return the exclusive end index of the window that begins at `start`."""
    window_start = run[start].start_seconds
    for index in range(start + 1, len(run)):
        if run[index].end_seconds - window_start > settings.max_seconds:
            return index
        previous = run[index - 1]
        reached_target = previous.end_seconds - window_start >= settings.target_seconds
        if reached_target and previous.text.endswith(SENTENCE_ENDINGS):
            return index
    return len(run)


def find_overlap_start(
    run: Sequence[SpokenWord], start: int, end: int, overlap_seconds: float
) -> int:
    """First word of the next window: the earliest word within the overlap of the last window."""
    window_end = run[end - 1].end_seconds
    for index in range(start + 1, end):
        if window_end - run[index].start_seconds <= overlap_seconds:
            return index
    return end


def to_chunk(segment_index: int, words: Sequence[SpokenWord]) -> Chunk:
    return Chunk(
        segment_index=segment_index,
        speaker=words[0].speaker,
        start_seconds=words[0].start_seconds,
        end_seconds=words[-1].end_seconds,
        text=" ".join(word.text for word in words),
    )
