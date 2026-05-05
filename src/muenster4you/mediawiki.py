"""MediaWiki SQLite database access layer."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from sqlite3 import Row, connect
from typing import Iterator, Self


@dataclass
class RawMediaWikiPage:
    id: int
    namespace: int
    title: str
    length: int


@dataclass
class MediaWikiPage:
    id: int
    namespace: int
    title: str
    content: str
    rev_id: int
    rev_timestamp: datetime
    rev_actor: str

    @classmethod
    def from_db_row(cls, row: Row) -> Self:
        id_, namespace, title, content, rev_id, rev_timestamp_str, rev_actor = row
        rev_timestamp = datetime.strptime(rev_timestamp_str, "%Y%m%d%H%M%S")
        return cls(
            id=id_,
            namespace=namespace,
            title=title,
            content=content,
            rev_id=rev_id,
            rev_timestamp=rev_timestamp,
            rev_actor=rev_actor,
        )


class SQLiteMediaWiki:
    def __init__(self, db_path: Path) -> None:
        self.conn = connect(db_path)

    def get_all_pages(self, namespace: int = 0) -> Iterator[RawMediaWikiPage]:
        """Get all pages in a namespace"""
        query = """
        SELECT
            page_id,
            page_namespace,
            page_title,
            page_len
        FROM page
        WHERE page_namespace = ?
        """
        with self.conn as conn:
            cur = conn.cursor()
            cur.row_factory = lambda cursor, row: RawMediaWikiPage(*row)
            yield from cur.execute(query, (namespace,))

    def get_page_content_by_id(
        self, page_id: int, namespace: int = 0
    ) -> MediaWikiPage | None:
        query = """
        SELECT
            p.page_id AS id,
            p.page_namespace AS namespace,
            p.page_title AS title,
            t.old_text AS content,
            r.rev_id,
            r.rev_timestamp,
            a.actor_name AS rev_actor
        FROM page p
        JOIN revision r ON p.page_id = r.rev_page
        JOIN actor a ON r.rev_actor = a.actor_id
        JOIN slots s ON r.rev_id = s.slot_revision_id
        JOIN content c ON s.slot_content_id = c.content_id
        JOIN text t ON CAST(substr(c.content_address, 4) AS INTEGER) = t.old_id
        WHERE (
            p.page_id = ?
            AND p.page_namespace = ?
            AND s.slot_role_id = 1
            AND c.content_model = 1
        )
        """
        with self.conn as conn:
            cur = conn.cursor()
            cur.row_factory = lambda cursor, row: MediaWikiPage.from_db_row(row)
            return cur.execute(query, (page_id, namespace)).fetchone()
