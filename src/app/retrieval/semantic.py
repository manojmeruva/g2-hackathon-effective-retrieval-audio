"""SemanticRetriever: pgvector similarity search."""

from app.domain.models import ScoredSegment
from app.domain.protocols import Embedder, VectorIndex


class SemanticRetriever:
    def __init__(self, embedder: Embedder, index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = index

    def retrieve(self, query: str, limit: int) -> list[ScoredSegment]:
        if not query.strip():
            return []
        return self._index.vector_search(self._embedder.embed_query(query), limit)
