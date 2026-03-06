import pytest
from fastapi.testclient import TestClient
from muenster4you.api import app
from urllib.parse import quote

client = TestClient(app, raise_server_exceptions=False)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "muenster4you"
    assert data["version"] == "0.1.0"
    assert data["status"] == "ok"


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
