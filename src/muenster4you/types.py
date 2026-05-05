"""Shared types used across multiple modules in muenster4you."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict


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


@dataclass
class RetrievalResult:
    page_id: int
    page_title: str
    content_text: str
    similarity_score: float
    page_len: int
    source: str = "wiki"
    source_url: str | None = None  # URL for web results
