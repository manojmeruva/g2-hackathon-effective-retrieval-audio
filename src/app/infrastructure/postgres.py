"""SQLAlchemy engine creation and schema setup."""

from sqlalchemy import Engine, create_engine, text

from app.config import DatabaseSettings
from app.infrastructure.schema import Schema


def create_engine_from_settings(settings: DatabaseSettings) -> Engine:
    return create_engine(settings.url, pool_pre_ping=True)


def create_schema(engine: Engine, schema: Schema) -> None:
    """Create the pgvector extension, tables and indexes if they do not exist."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    schema.metadata.create_all(engine)
