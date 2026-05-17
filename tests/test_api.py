from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from muenster4you.api import app, get_retriever
from muenster4you.retriever import LanceDBRetriever


@pytest.fixture
def test_client(retriever_with_pages: LanceDBRetriever) -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = lambda: retriever_with_pages

    with TestClient(app=app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_smoke_test_test_client(test_client: TestClient) -> None:
    pass


def test_search_endpoint(test_client: TestClient) -> None:
    params = {"query": "Wieso ist Münster cool?"}

    response = test_client.get("/search", params=params)

    assert len(response.json()["results"]) == 2
