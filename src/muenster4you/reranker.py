"""Cross-encoder reranking over RetrievalResult candidates."""

from dataclasses import replace
from typing import Protocol

from sentence_transformers import CrossEncoder

from muenster4you.types import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]: ...


class CrossEncoderReranker:
    def __init__(self, model_id: str):
        self.model = CrossEncoder(model_id, trust_remote_code=True)

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.content) for c in candidates])
        rescored = [replace(c, score=float(s)) for c, s in zip(candidates, scores)]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
