from pathlib import Path

import pytest

from app.domain.models import RankedSegment, SearchResult
from app.presentation.formatting import format_result, format_span, format_timestamp
from app.presentation.html_report import render_results_page
from tests.factories import segment

MATCH = segment("conversation_03", 12, 321.4, 348.0, "SPEAKER_01", "We run <PostgreSQL> & more")
BEFORE = segment("conversation_03", 11, 300.0, 320.0, "SPEAKER_00", "What do you run?")
RESULT = SearchResult(RankedSegment(MATCH, 0.74, 0.91, 0.86, rerank_score=5.12), (BEFORE, MATCH))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00"), (59.9, "00:59"), (321.4, "05:21"), (3725.0, "1:02:05"), (-3.0, "00:00")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_format_span() -> None:
    assert format_span(321.4, 348.0) == "05:21–05:48"


def test_format_result_shows_location_scores_and_context() -> None:
    lines = format_result(1, RESULT)

    assert lines[0] == "1. conversation_03  05:21–05:48  SPEAKER_01  [conversation_03:12]"
    assert "keyword 0.74 · semantic 0.91 · hybrid 0.86 · rerank 5.12" in lines[1]
    assert lines[2].startswith("   · 05:00 SPEAKER_00")
    assert lines[3].startswith("   ▶ 05:21 SPEAKER_01")


def test_html_page_escapes_text_and_links_audio_at_timestamp(tmp_path: Path) -> None:
    audio = tmp_path / "data" / "audio" / "conversation_03.wav"
    output = tmp_path / "results.html"

    page = render_results_page("<cloud>", [RESULT], {"conversation_03": audio}, output)

    assert "&lt;PostgreSQL&gt; &amp; more" in page
    assert "“&lt;cloud&gt;”" in page
    assert 'src="data/audio/conversation_03.wav#t=321.4"' in page
    assert 'data-start="321.4"' in page
    assert "Jump to 05:21" in page
    assert page.count('class="line match"') == 1


def test_html_page_without_results_or_audio(tmp_path: Path) -> None:
    assert "No results." in render_results_page("q", [], {}, tmp_path / "r.html")
    assert "Audio file not found." in render_results_page("q", [RESULT], {}, tmp_path / "r.html")
