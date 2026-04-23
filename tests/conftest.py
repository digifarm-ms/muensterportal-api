from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from muenster4you.api import app, get_generator, get_retriever, get_session_manager
from muenster4you.rag.sessions import ChatSessionManager


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    generator.generate.return_value = "Mock answer"
    generator.chat.return_value = "Mock chat answer"
    generator.build_system_message.return_value = {
        "role": "system",
        "content": "System prompt",
    }
    return generator


@pytest.fixture
def session_manager():
    return ChatSessionManager(ttl=300, max_followups=3)


@pytest.fixture
def client(mock_retriever, mock_generator, session_manager):
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_session_manager] = lambda: session_manager
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()
