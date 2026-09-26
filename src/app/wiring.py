"""Composition root: builds services from settings with their real dependencies."""

from pathlib import Path

from app.application.ingestion_service import IngestionService
from app.application.search_service import SearchService
from app.config import AppSettings
from app.infrastructure.postgres import create_engine_from_settings, create_schema
from app.infrastructure.repositories import PostgresSegmentRepository
from app.infrastructure.schema import define_schema
from app.ingestion.diarization import PyannoteDiarizer
from app.ingestion.embeddings import SentenceTransformerEmbedder
from app.ingestion.speech import SpeechAnalyzer
from app.ingestion.transcript_cache import TranscriptCache
from app.ingestion.transcription import WhisperTranscriber
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.reranking import CrossEncoderReranker
from app.retrieval.semantic import SemanticRetriever


def build_repository(
    settings: AppSettings, create_tables: bool = False
) -> PostgresSegmentRepository:
    engine = create_engine_from_settings(settings.database)
    schema = define_schema(settings.embedding.dimensions)
    if create_tables:
        create_schema(engine, schema)
    return PostgresSegmentRepository(engine, schema)


def build_speech_analyzer(settings: AppSettings) -> SpeechAnalyzer:
    return SpeechAnalyzer(
        WhisperTranscriber.from_settings(settings.transcription),
        PyannoteDiarizer.from_settings(settings.diarization),
    )


def build_ingestion_service(settings: AppSettings, transcript_dir: Path) -> IngestionService:
    return IngestionService(
        analyzer_factory=lambda: build_speech_analyzer(settings),
        cache=TranscriptCache(transcript_dir),
        embedder=SentenceTransformerEmbedder.from_settings(settings.embedding),
        store=build_repository(settings, create_tables=True),
        chunking=settings.chunking,
    )


def build_search_service(settings: AppSettings, with_reranker: bool) -> SearchService:
    repository = build_repository(settings)
    embedder = SentenceTransformerEmbedder.from_settings(settings.embedding)
    reranker = CrossEncoderReranker.from_settings(settings.reranker) if with_reranker else None
    return SearchService(
        keyword=KeywordRetriever(repository),
        semantic=SemanticRetriever(embedder, repository),
        context=repository,
        settings=settings.search,
        reranker=reranker,
    )
