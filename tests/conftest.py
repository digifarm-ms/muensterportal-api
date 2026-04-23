from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muenster4you.api import app, get_generator, get_retriever


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.generate.return_value = "Mock answer"
    return generator


@pytest.fixture
def client(mock_retriever, mock_generator):
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()
