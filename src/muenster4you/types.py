"""Shared types used across multiple modules in muenster4you."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
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


class RetrievalSource(StrEnum):
    WIKI = auto()
    WEBSEARCH = auto()


@dataclass
class RetrievalResult:
    content: str
    score: float
    source: RetrievalSource
    url: str
