"""Shared types used across multiple modules in muenster4you."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any, TypedDict


class BaseWikiPage(TypedDict):
    id: int
    namespace: int
    title: str
    content: str
    rev_id: int
    rev_timestamp: datetime
    rev_actor: str


class WikiPage(BaseWikiPage):
    embedding: list[float]


class RetrievalSource(StrEnum):
    WIKI = auto()
    WEBSEARCH = auto()


@dataclass
class RetrievalResult:
    content: str
    score: float
    source: RetrievalSource
    url: str
    # Optional numpy.ndarray populated by retrievers that already have it
    # (e.g. wiki/LanceDB) so the bi-encoder rerank can skip re-embedding.
    # Typed as `Any` so Pydantic skips schema generation; excluded from API
    # responses via Field(exclude=...) on the response models.
    embedding: Any = field(default=None, repr=False, compare=False)
