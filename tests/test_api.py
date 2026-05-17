from collections.abc import Iterator
from dataclasses import dataclass

from fastapi.testclient import TestClient
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
import pytest

from muenster4you.api import app, get_retriever
from muenster4you.lancedb import EMBEDDING_DIM
from muenster4you.retriever import LanceDBRetriever


@dataclass
class FakeEmbedder:
    rng: Generator

    def encode(self, text: str) -> NDArray[np.float32]:
        return self.rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture
def numpy_rng() -> Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def get_in_memory_retriever(numpy_rng: Generator) -> LanceDBRetriever:
    retriever = LanceDBRetriever(
        db_path="memory://", embedder=FakeEmbedder(rng=numpy_rng)
    )
    return retriever


@pytest.fixture
def test_client() -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = get_in_memory_retriever
    with TestClient(app=app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_smoke_test_test_client(test_client: TestClient) -> None:
    pass
