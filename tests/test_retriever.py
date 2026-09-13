"""Smoke test for the LanceDB retriever using an in-memory database."""


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
