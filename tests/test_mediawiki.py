import sqlite3
from pathlib import Path

import pytest

from muenster4you.mediawiki import SQLiteMediaWiki

SCHEMA = """
CREATE TABLE page (page_id INTEGER PRIMARY KEY, page_namespace INTEGER, page_title TEXT,
                   page_len INTEGER, page_latest INTEGER);
CREATE TABLE revision (rev_id INTEGER PRIMARY KEY, rev_page INTEGER, rev_timestamp TEXT);
CREATE TABLE slots (slot_revision_id INTEGER, slot_role_id INTEGER, slot_content_id INTEGER);
CREATE TABLE content (content_id INTEGER PRIMARY KEY, content_address TEXT, content_model INTEGER);
CREATE TABLE text (old_id INTEGER PRIMARY KEY, old_text TEXT);
"""


def _add_revision(conn: sqlite3.Connection, rev_id: int, page_id: int, ts: str, text: str) -> None:
    conn.execute("INSERT INTO text VALUES (?, ?)", (rev_id, text))
    conn.execute("INSERT INTO content VALUES (?, ?, 1)", (rev_id, f"tt:{rev_id}"))
    conn.execute("INSERT INTO slots VALUES (?, 1, ?)", (rev_id, rev_id))
    conn.execute("INSERT INTO revision VALUES (?, ?, ?)", (rev_id, page_id, ts))


@pytest.fixture
def wiki_db(tmp_path: Path) -> Path:
    path = tmp_path / "wiki.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    _add_revision(conn, 1, 10, "20240101000000", "alter Text")
    _add_revision(conn, 2, 10, "20250101000000", "neuer Text")
    _add_revision(conn, 3, 10, "20241231000000", "mittlerer Text")
    conn.execute("INSERT INTO page VALUES (10, 0, 'Aasee', 10, 2)")
    conn.commit()
    conn.close()
    return path


def test_page_content_comes_from_the_latest_revision(wiki_db: Path):
    page = SQLiteMediaWiki(wiki_db).get_page_content_by_id(10, 0)

    assert page is not None
    assert page.rev_id == 2
    assert page.content == "neuer Text"
    assert page.title == "Aasee"


def test_get_all_pages_lists_namespace(wiki_db: Path):
    pages = list(SQLiteMediaWiki(wiki_db).get_all_pages(namespace=0))

    assert [(p.id, p.title) for p in pages] == [(10, "Aasee")]
    assert list(SQLiteMediaWiki(wiki_db).get_all_pages(namespace=1)) == []
