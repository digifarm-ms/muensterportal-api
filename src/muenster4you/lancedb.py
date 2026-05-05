from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from datetime import datetime

from lancedb import connect
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from muenster4you.rag.retrieval import RetrievalResult


class WikiPageData(TypedDict):
    id: int
    namespace: int
    title: str
    content: str
    rev_id: int
    rev_timestamp: datetime
    rev_actor: str


model = (
    get_registry()
    .get("sentence-transformers")
    .create(name="jinaai/jina-embeddings-v5-text-small-retrieval")
)


class LanceDBWikiPage(LanceModel):
    id: int
    namespace: int
    title: str
    content: str = model.SourceField()
    rev_id: int
    rev_timestamp: datetime
    rev_actor: str
    embedding: Vector(model.ndims()) = model.VectorField()


class LanceDBMediaWiki:
    def __init__(self, path: Path) -> None:
        self.db = connect(path)
        self.table = self.db.create_table(
            "mediawiki_pages", schema=LanceDBWikiPage.to_arrow_schema(), exist_ok=True
        )

    def upsert_pages(self, pages: Sequence[WikiPageData]) -> None:
        if len(pages) == 0:
            return
        (
            self.table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(pages)
        )

    def search(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        rows = (
            self.table.search(query)
            .metric("cosine")
            .select(["id", "title", "content"])
            .limit(limit)
            .to_list()
        )
        return [
            RetrievalResult(
                page_id=r["id"],
                page_title=r["title"],
                content_text=r["content"],
                similarity_score=1.0 - r["_distance"],
                page_len=len(r["content"]),
                source="wiki",
            )
            for r in rows
        ]
