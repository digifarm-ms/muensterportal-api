"""Smoke test for the LanceDB retriever using an in-memory database."""


def test_search_returns_most_similar_page_first(retriever_with_pages):
    results = retriever_with_pages.search("Was ist Münster?", top_k=2)

    assert len(results) == 2


def test_search_respects_top_k(retriever_with_pages):
    results = retriever_with_pages.search("Münster", top_k=1)

    assert len(results) == 1
