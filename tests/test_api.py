import pytest
from fastapi.testclient import TestClient
from muenster4you import app
from urllib.parse import quote

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from muenster4you!"}


def test_search_endpoint_with_query():
    response = client.get("/search?q=test")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Search results for: test"
    assert data["query"] == "test"
    assert len(data["results"]) == 3
    assert all("test" in result for result in data["results"])


def test_search_endpoint_without_query():
    response = client.get("/search")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Please provide a search query"
    assert data["query"] is None


def test_search_endpoint_with_empty_query():
    response = client.get("/search?q=")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Please provide a search query"
    assert data["query"] is None


def test_search_endpoint_with_special_characters():
    query = "hello world & special chars!"
    response = client.get("/search", params={"q": query})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == query
    assert all(query in result for result in data["results"])


def test_search_endpoint_with_unicode():
    query = "müenster"
    response = client.get("/search", params={"q": query})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == query
    assert all(query in result for result in data["results"])