"""Small builders for domain objects used across tests."""

from app.domain.models import RankedSegment, ScoredSegment, SpokenWord, TranscriptSegment


def segment(
    audio_id: str = "conversation_01",
    index: int = 0,
    start: float = 0.0,
    end: float = 10.0,
    speaker: str = "SPEAKER_00",
    text: str = "text",
) -> TranscriptSegment:
    return TranscriptSegment(audio_id, index, speaker, start, end, text)


def scored(index: int, score: float, audio_id: str = "conversation_01") -> ScoredSegment:
    return ScoredSegment(segment(audio_id, index, start=index * 10.0, end=index * 10.0 + 9), score)


def ranked(index: int, hybrid: float, text: str = "text") -> RankedSegment:
    return RankedSegment(
        segment=segment(index=index, text=text),
        keyword_score=hybrid,
        semantic_score=hybrid,
        hybrid_score=hybrid,
    )


def word(text: str, start: float, end: float, speaker: str = "SPEAKER_00") -> SpokenWord:
    return SpokenWord(text, start, end, speaker)
