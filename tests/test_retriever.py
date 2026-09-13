"""Smoke test for the LanceDB retriever using an in-memory database."""

from datetime import datetime

import numpy as np

import lancedb
from muenster4you.lancedb import WIKIPAGE_TABLE_NAME


def test_search_returns_most_similar_page_first(retriever_with_pages):
    results = retriever_with_pages.search("Was ist Münster?", top_k=2)

    assert len(results) == 2


def test_search_respects_top_k(retriever_with_pages):
    results = retriever_with_pages.search("Münster", top_k=1)

    assert len(results) == 1


def test_delete_pages_not_in_removes_stale_rows(retriever_with_pages):
    from muenster4you.lancedb import LanceDBMediaWiki

    store = LanceDBMediaWiki.__new__(LanceDBMediaWiki)
    store.table = retriever_with_pages.conn.open_table("mediawiki_pages")

    deleted = store.delete_pages_not_in([1])

    assert deleted == 1
    assert [r["id"] for r in store.table.search().select(["id"]).to_list()] == [1]


def test_store_drops_columns_the_model_no_longer_has(tmp_path):
    import pyarrow as pa

    from muenster4you.lancedb import EMBEDDING_DIM, LanceDBMediaWiki, LanceDBWikiPage

    old_schema = LanceDBWikiPage.to_arrow_schema().append(pa.field("rev_actor", pa.string()))
    db = lancedb.connect(tmp_path)
    db.create_table(WIKIPAGE_TABLE_NAME, schema=old_schema).add(
        [
            {
                "id": 1,
                "namespace": 0,
                "title": "Aasee",
                "content": "x",
                "rev_id": 1,
                "rev_timestamp": datetime(2024, 1, 1),
                "rev_actor": "someone",
                "embedding": np.zeros(EMBEDDING_DIM, dtype=np.float32),
            }
        ]
    )

    store = LanceDBMediaWiki(tmp_path)

    assert "rev_actor" not in store.table.schema.names
    assert store.table.count_rows() == 1
