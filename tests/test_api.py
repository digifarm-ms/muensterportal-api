from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from muenster4you.api import app, get_model, get_reranker, get_retriever, get_web_searcher
from muenster4you.retriever import LanceDBRetriever
from muenster4you.types import RetrievalResult


class _PassThroughReranker:
    """Test double: returns the first top_k candidates unchanged."""

    def rerank(
        self, _query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        return candidates[:top_k]


class _NoWebSearcher:
    """Test double: web search that finds nothing."""

    def search(self, query: str, max_results: int) -> list[RetrievalResult]:  # noqa: ARG002
        return []

    def __call__(self) -> "_NoWebSearcher":
        return self


@pytest.fixture
def test_client(retriever_with_pages: LanceDBRetriever) -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = lambda: retriever_with_pages
    app.dependency_overrides[get_web_searcher] = _NoWebSearcher()
    app.dependency_overrides[get_model] = lambda: TestModel(custom_output_text="Testantwort")
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


def test_chat_endpoint_answers_with_injected_model(test_client: TestClient) -> None:
    response = test_client.post("/chat", json={"message": "Was ist der Aasee?"})

    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Testantwort"
    assert [m["role"] for m in body["history"]] == ["user", "assistant"]
    assert len(body["sources"]) == 2


def test_chat_endpoint_continues_conversation(test_client: TestClient) -> None:
    first = test_client.post("/chat", json={"message": "Was ist der Aasee?"}).json()

    second = test_client.post(
        "/chat", json={"message": "Und wie groß?", "conversation_id": first["conversation_id"]}
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert [m["role"] for m in second["history"]] == ["user", "assistant", "user", "assistant"]
    assert second["remaining_followups"] == first["remaining_followups"] - 1


def test_chat_endpoint_passes_language_into_system_prompt(test_client: TestClient) -> None:
    response = test_client.post(
        "/chat", json={"message": "Where is the Aasee?", "language": "Englisch"}
    )

    assert response.status_code == 200
