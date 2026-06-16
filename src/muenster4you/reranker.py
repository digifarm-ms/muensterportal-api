"""Reranking over RetrievalResult candidates."""

from dataclasses import replace
from typing import Protocol

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

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
        rescored = [replace(c, score=float(s)) for c, s in zip(candidates, scores, strict=True)]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]


class BiEncoderReranker:
    """Score candidates by cosine similarity in the retrieval embedder's space.

    Reuses cached embeddings on candidates that already carry them (wiki
    hits from LanceDB), so only candidates without an embedding (web hits)
    pay the forward-pass cost. All scores land in the same normalized
    space, which makes cross-source ranking meaningful.
    """

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        q = self.model.encode(
            query,
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        missing_idx = [i for i, c in enumerate(candidates) if c.embedding is None]
        if missing_idx:
            fresh = self.model.encode(
                [candidates[i].content for i in missing_idx],
                prompt_name="document",
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            fresh = np.asarray(fresh, dtype=np.float32)
            for slot, i in enumerate(missing_idx):
                candidates[i] = replace(candidates[i], embedding=fresh[slot])

        docs = np.stack([np.asarray(c.embedding, dtype=np.float32) for c in candidates])
        # Fresh-encoded vectors were normalized via `normalize_embeddings=True`, but
        # LanceDB-stored ones are raw model outputs — normalize them here so the
        # dot product gives cosine similarity regardless of source.
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        docs = docs / np.where(norms == 0.0, 1.0, norms)
        scores = docs @ np.asarray(q, dtype=np.float32)
        rescored = [replace(c, score=float(s)) for c, s in zip(candidates, scores, strict=True)]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
