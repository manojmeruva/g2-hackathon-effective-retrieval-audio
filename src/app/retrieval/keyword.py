"""KeywordRetriever: PostgreSQL full-text search."""

import re

from app.domain.models import ScoredSegment
from app.domain.protocols import KeywordIndex

TERM_PATTERN = re.compile(r"[a-z0-9]+")


def extract_terms(query: str) -> list[str]:
    """Lowercase alphanumeric terms in first-seen order.

    Restricting terms to [a-z0-9] also keeps user input from injecting
    tsquery operators. Stop words are removed later by PostgreSQL.
    """
    return list(dict.fromkeys(TERM_PATTERN.findall(query.lower())))


class KeywordRetriever:
    def __init__(self, index: KeywordIndex) -> None:
        self._index = index

    def retrieve(self, query: str, limit: int) -> list[ScoredSegment]:
        terms = extract_terms(query)
        if not terms:
            return []
        return self._index.keyword_search(terms, limit)
