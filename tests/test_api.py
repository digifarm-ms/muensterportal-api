from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from muenster4you.api import app, get_reranker, get_retriever, get_web_searcher
from muenster4you.retriever import LanceDBRetriever
from muenster4you.types import RetrievalResult


class _PassThroughReranker:
    """Test double: returns the first top_k candidates unchanged."""

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        return candidates[:top_k]


@pytest.fixture
def test_client(retriever_with_pages: LanceDBRetriever) -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = lambda: retriever_with_pages
    app.dependency_overrides[get_web_searcher] = lambda: None
    app.dependency_overrides[get_reranker] = lambda: _PassThroughReranker()

    with TestClient(app=app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_smoke_test_test_client(test_client: TestClient) -> None:
    pass


def test_search_endpoint(test_client: TestClient) -> None:
    params = {"query": "Wieso ist Münster cool?"}

    response = test_client.get("/search", params=params)

    assert len(response.json()["results"]) == 2


def test_search_endpoint_returns_422_for_empty_query(test_client: TestClient) -> None:
    params = {"query": "ab"}

    response = test_client.get("/search", params=params)

    assert response.status_code == 422
