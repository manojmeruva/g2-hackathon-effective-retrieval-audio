"""Static HTML page for search results with an audio player per result."""

import os
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path

from app.domain.models import SearchResult, TranscriptSegment
from app.presentation.formatting import format_span, format_timestamp

PAGE_STYLE = """
:root { color-scheme: light dark; --bg: #fafafa; --card: #fff; --text: #1b1b1f;
  --muted: #5f6368; --accent: #1a56db; --border: #e2e4e8; --match: #eef3ff; }
@media (prefers-color-scheme: dark) { :root { --bg: #121316; --card: #1c1d21;
  --text: #e8e8ea; --muted: #a0a4ab; --accent: #8ab4ff; --border: #2e3036; --match: #1f2940; } }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width: 820px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.query { color: var(--muted); margin: 0 0 24px; }
.result { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; margin-bottom: 16px; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--muted); }
.meta strong { color: var(--text); }
.scores { font-variant-numeric: tabular-nums; }
.context { margin: 12px 0; }
.line { padding: 4px 8px; border-radius: 6px; }
.line.match { background: var(--match); }
.line .who { font-size: 12px; color: var(--muted); }
audio { width: 100%; margin-top: 8px; }
button { background: var(--accent); color: var(--bg); border: 0; border-radius: 6px;
  padding: 6px 12px; font: inherit; cursor: pointer; }
"""

JUMP_SCRIPT = """
document.querySelectorAll("button[data-start]").forEach((button) => {
  button.addEventListener("click", () => {
    const player = document.getElementById(button.dataset.player);
    player.currentTime = Number(button.dataset.start);
    player.play();
  });
});
"""


def render_results_page(
    query: str,
    results: Sequence[SearchResult],
    audio_paths: Mapping[str, Path],
    output_path: Path,
) -> str:
    cards = "\n".join(
        render_result(rank, result, audio_paths, output_path.parent)
        for rank, result in enumerate(results, start=1)
    )
    body = cards or "<p>No results.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Search Results</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<main>
<h1>Audio search results</h1>
<p class="query">Query: “{escape(query)}”</p>
{body}
</main>
<script>{JUMP_SCRIPT}</script>
</body>
</html>
"""


def render_result(
    rank: int, result: SearchResult, audio_paths: Mapping[str, Path], page_dir: Path
) -> str:
    segment = result.segment
    ranked = result.ranked
    player_id = f"player-{rank}"
    rerank = "" if ranked.rerank_score is None else f" · rerank {ranked.rerank_score:.2f}"
    return f"""<section class="result">
<div class="meta">
<strong>#{rank} {escape(segment.audio_id)}</strong>
<span>{format_span(segment.start_seconds, segment.end_seconds)}</span>
<span>{escape(segment.speaker)}</span>
<span>segment {escape(segment.segment_id)}</span>
</div>
<div class="meta scores">keyword {ranked.keyword_score:.2f} · semantic {ranked.semantic_score:.2f}
 · hybrid {ranked.hybrid_score:.2f}{rerank}</div>
<div class="context">{render_context(result)}</div>
{render_player(player_id, segment, audio_paths, page_dir)}
</section>"""


def render_context(result: SearchResult) -> str:
    lines = result.context or (result.segment,)
    return "\n".join(render_line(line, line == result.segment) for line in lines)


def render_line(segment: TranscriptSegment, is_match: bool) -> str:
    css_class = "line match" if is_match else "line"
    who = f"{format_timestamp(segment.start_seconds)} · {escape(segment.speaker)}"
    return f'<div class="{css_class}"><div class="who">{who}</div>{escape(segment.text)}</div>'


def render_player(
    player_id: str, segment: TranscriptSegment, audio_paths: Mapping[str, Path], page_dir: Path
) -> str:
    audio_path = audio_paths.get(segment.audio_id)
    if audio_path is None:
        return "<p>Audio file not found.</p>"
    source = Path(os.path.relpath(audio_path.resolve(), page_dir.resolve())).as_posix()
    start = f"{segment.start_seconds:.1f}"
    return f"""<button data-player="{player_id}" data-start="{start}">▶ Jump to \
{format_timestamp(segment.start_seconds)}</button>
<audio id="{player_id}" controls preload="none" src="{escape(source)}#t={start}"></audio>"""
