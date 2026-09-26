"""Local text embeddings with sentence-transformers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from sentence_transformers import SentenceTransformer

from app.config import EmbeddingSettings

EncodeTexts = Callable[[list[str]], Iterable[Iterable[float]]]


class SentenceTransformerEmbedder:
    """Produces L2-normalized embeddings, so cosine similarity equals the dot product."""

    def __init__(self, encode: EncodeTexts, query_prefix: str) -> None:
        self._encode = encode
        self._query_prefix = query_prefix

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> SentenceTransformerEmbedder:
        model = SentenceTransformer(settings.model_name)

        def encode(texts: list[str]) -> Iterable[Iterable[float]]:
            return model.encode(
                texts,
                batch_size=settings.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return cls(encode, settings.query_prefix)

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [tuple(float(value) for value in vector) for vector in self._encode(texts)]
