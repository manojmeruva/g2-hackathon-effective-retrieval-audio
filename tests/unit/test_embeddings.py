from collections.abc import Iterable

from app.ingestion.embeddings import SentenceTransformerEmbedder


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> Iterable[Iterable[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


def test_embed_documents_returns_one_tuple_per_text() -> None:
    encoder = RecordingEncoder()

    vectors = SentenceTransformerEmbedder(encoder, "q: ").embed_documents(["ab", "abc"])

    assert vectors == [(2.0, 1.0), (3.0, 1.0)]
    assert encoder.calls == [["ab", "abc"]]


def test_embed_query_adds_the_query_prefix() -> None:
    encoder = RecordingEncoder()

    SentenceTransformerEmbedder(encoder, "q: ").embed_query("cloud")

    assert encoder.calls == [["q: cloud"]]


def test_no_documents_skips_the_model() -> None:
    encoder = RecordingEncoder()

    assert SentenceTransformerEmbedder(encoder, "").embed_documents([]) == []
    assert encoder.calls == []
