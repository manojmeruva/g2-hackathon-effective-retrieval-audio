"""Typer CLI entry point: ingest, search, evaluate."""

from pathlib import Path
from typing import Annotated

import typer

from app.application.evaluation_service import EvaluationService
from app.config import load_settings
from app.domain.models import FusionMethod, SearchMode
from app.evaluation.golden import load_golden_set
from app.ingestion.audio import find_audio_files
from app.presentation.evaluation_report import to_json, to_markdown
from app.presentation.formatting import format_result
from app.presentation.html_report import render_results_page
from app.wiring import build_ingestion_service, build_repository, build_search_service

DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")
DEFAULT_QUERIES_PATH = Path("data/golden_queries.json")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_REPORT_DIR = Path("reports")

app = typer.Typer(help="Search spoken conversations by keyword and meaning.", no_args_is_help=True)


@app.command("init-db")
def init_db() -> None:
    """Create the pgvector extension, tables and indexes."""
    build_repository(load_settings(), create_tables=True)
    typer.echo("Database schema is ready.")


@app.command()
def ingest(
    audio_dir: Annotated[
        Path, typer.Argument(help="Directory of audio files.")
    ] = DEFAULT_AUDIO_DIR,
    force: Annotated[
        bool, typer.Option(help="Re-run transcription and diarization even if cached.")
    ] = False,
) -> None:
    """Transcribe, diarize, chunk, embed and index every audio file in a directory."""
    service = build_ingestion_service(load_settings(), DEFAULT_TRANSCRIPT_DIR)
    for audio_path in find_audio_files(audio_dir):
        summary = service.ingest_file(audio_path, force)
        source = "cache" if summary.from_cache else "models"
        typer.echo(
            f"{summary.audio_id}: {summary.duration_seconds / 60:.1f} min, "
            f"{summary.word_count} words, {summary.segment_count} segments "
            f"(transcript from {source})"
        )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    top_k: Annotated[int, typer.Option(min=1, help="Number of results.")] = 5,
    mode: Annotated[
        SearchMode, typer.Option(help="Retrieval strategy.")
    ] = SearchMode.HYBRID_RERANK,
    alpha: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Keyword weight for weighted fusion.")
    ] = None,
    fusion: Annotated[
        FusionMethod | None, typer.Option(help="How hybrid modes combine the two lists.")
    ] = None,
    html: Annotated[Path | None, typer.Option(help="Also write an HTML results page.")] = None,
    audio_dir: Annotated[
        Path, typer.Option(help="Where the audio files live.")
    ] = DEFAULT_AUDIO_DIR,
) -> None:
    """Search the indexed conversations."""
    service = build_search_service(load_settings(), with_reranker=mode is SearchMode.HYBRID_RERANK)
    results = service.search(query, top_k, mode, alpha, fusion)
    if not results:
        typer.echo("No results.")
    for rank, result in enumerate(results, start=1):
        typer.echo("\n".join(format_result(rank, result)) + "\n")
    if html is not None:
        audio_paths = {path.stem: path for path in find_audio_files(audio_dir)}
        html.write_text(render_results_page(query, results, audio_paths, html))
        typer.echo(f"Wrote {html}")


@app.command()
def evaluate(
    queries: Annotated[Path, typer.Option(help="Golden query file.")] = DEFAULT_QUERIES_PATH,
    ground_truth: Annotated[
        Path, typer.Option(help="Reference transcripts.")
    ] = DEFAULT_GROUND_TRUTH_DIR,
    output: Annotated[Path, typer.Option(help="Directory for the reports.")] = DEFAULT_REPORT_DIR,
    rerank: Annotated[bool, typer.Option(help="Include the reranking experiment.")] = True,
) -> None:
    """Measure Recall@K, MRR, NDCG, precision and latency on the golden query set."""
    settings = load_settings()
    service = EvaluationService(
        build_search_service(settings, with_reranker=rerank),
        load_golden_set(queries, ground_truth),
        settings.evaluation,
    )
    report = service.run(include_rerank=rerank)
    markdown = to_markdown(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.md").write_text(markdown)
    (output / "evaluation.json").write_text(to_json(report))
    typer.echo(markdown)
    typer.echo(f"Wrote {output / 'evaluation.md'} and {output / 'evaluation.json'}")


if __name__ == "__main__":
    app()
