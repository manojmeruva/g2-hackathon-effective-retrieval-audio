"""Human-readable formatting shared by the CLI and the HTML page."""

from app.domain.models import SearchResult

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


def format_timestamp(seconds: float) -> str:
    """mm:ss, or h:mm:ss for audio longer than an hour."""
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, SECONDS_PER_HOUR)
    minutes, secs = divmod(remainder, SECONDS_PER_MINUTE)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_span(start_seconds: float, end_seconds: float) -> str:
    return f"{format_timestamp(start_seconds)}–{format_timestamp(end_seconds)}"


def format_result(rank: int, result: SearchResult) -> list[str]:
    """Terminal lines for one result: location, scores, and the surrounding conversation."""
    segment = result.segment
    ranked = result.ranked
    scores = (
        f"keyword {ranked.keyword_score:.2f} · semantic {ranked.semantic_score:.2f} · "
        f"hybrid {ranked.hybrid_score:.2f}"
    )
    if ranked.rerank_score is not None:
        scores += f" · rerank {ranked.rerank_score:.2f}"
    header = (
        f"{rank}. {segment.audio_id}  {format_span(segment.start_seconds, segment.end_seconds)}"
        f"  {segment.speaker}  [{segment.segment_id}]"
    )
    context = result.context or (segment,)
    lines = [
        f"   {'▶' if line == segment else '·'} {format_timestamp(line.start_seconds)} "
        f"{line.speaker}: {line.text}"
        for line in context
    ]
    return [header, f"   {scores}", *lines]
