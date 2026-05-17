"""Smoke test for the LanceDB retriever using an in-memory database."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.random import Generator
import pytest
from numpy.typing import NDArray

from muenster4you.lancedb import EMBEDDING_DIM, LanceDBWikiPage, WIKIPAGE_TABLE_NAME
from muenster4you.retriever import LanceDBRetriever



@dataclass
class FakeEmbedder:
    """Returns a predetermined vector regardless of input."""

    rng: Generator

    def encode(self, text: str) -> NDArray[np.float32]:
        return self.rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture
def numpy_rng() -> Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def retriever_with_pages(numpy_rng: Generator):
    retriever = LanceDBRetriever(
        db_path="memory://", embedder=FakeEmbedder(rng=numpy_rng)
    )

    table = retriever.conn.create_table(
        WIKIPAGE_TABLE_NAME, schema=LanceDBWikiPage.to_arrow_schema()
    )
    table.add(
        [
            {
                "id": 1,
                "namespace": 0,
                "title": "Münster",
                "content": "Münster ist eine Stadt in Nordrhein-Westfalen.",
                "rev_id": 1,
                "rev_timestamp": datetime(2024, 1, 1),
                "rev_actor": "tester",
                "embedding": numpy_rng.standard_normal(EMBEDDING_DIM, dtype=np.float32),
            },
            {
                "id": 2,
                "namespace": 0,
                "title": "Aasee",
                "content": "Der Aasee ist ein See im Süden von Münster.",
                "rev_id": 1,
                "rev_timestamp": datetime(2024, 1, 1),
                "rev_actor": "tester",
                "embedding": numpy_rng.standard_normal(EMBEDDING_DIM, dtype=np.float32),
            },
        ]
    )

    return retriever


def test_search_returns_most_similar_page_first(retriever_with_pages):
    results = retriever_with_pages.search("Was ist Münster?", top_k=2)

    assert len(results) == 2


def test_search_respects_top_k(retriever_with_pages):
    results = retriever_with_pages.search("Münster", top_k=1)

    assert len(results) == 1
