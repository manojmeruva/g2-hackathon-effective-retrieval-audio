"""Assigns a diarized speaker to every transcribed word."""

from collections.abc import Sequence

from app.domain.exceptions import DiarizationError
from app.domain.models import SpeakerTurn, SpokenWord, TranscribedSegment, TranscribedWord


def assign_speakers(
    segments: Sequence[TranscribedSegment], turns: Sequence[SpeakerTurn]
) -> list[SpokenWord]:
    """Label each word with the speaker whose turn overlaps it most.

    Words that fall into a gap between turns get the nearest turn's speaker.
    """
    if not turns:
        raise DiarizationError("diarization produced no speaker turns")
    return [
        SpokenWord(word.text, word.start_seconds, word.end_seconds, speaker_for(word, turns))
        for segment in segments
        for word in segment.words
    ]


def speaker_for(word: TranscribedWord, turns: Sequence[SpeakerTurn]) -> str:
    best_turn = max(turns, key=lambda turn: overlap_seconds(word, turn))
    if overlap_seconds(word, best_turn) > 0:
        return best_turn.speaker
    return min(turns, key=lambda turn: gap_seconds(word, turn)).speaker


def overlap_seconds(word: TranscribedWord, turn: SpeakerTurn) -> float:
    overlap = min(word.end_seconds, turn.end_seconds) - max(word.start_seconds, turn.start_seconds)
    return max(0.0, overlap)


def gap_seconds(word: TranscribedWord, turn: SpeakerTurn) -> float:
    if word.end_seconds < turn.start_seconds:
        return turn.start_seconds - word.end_seconds
    return max(0.0, word.start_seconds - turn.end_seconds)
