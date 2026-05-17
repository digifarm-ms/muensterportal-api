"""Orchestrates wiki + web retrieval and cross-encoder reranking."""

from muenster4you.reranker import Reranker
from muenster4you.retriever import LanceDBRetriever
from muenster4you.types import RetrievalResult
from muenster4you.websearch import TavilySearcher


class RetrievalOrchestrator:
    def __init__(
        self,
        wiki_retriever: LanceDBRetriever,
        web_searcher: TavilySearcher | None,
        reranker: Reranker | None,
        rerank_top_k: int,
        oversample_factor: int,
    ):
        self.wiki_retriever = wiki_retriever
        self.web_searcher = web_searcher
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.oversample_factor = oversample_factor

    def retrieve(self, query: str) -> list[RetrievalResult]:
        candidate_k = self.rerank_top_k * self.oversample_factor
        wiki = self.wiki_retriever.search(query, top_k=candidate_k)
        web = (
            self.web_searcher.search(query, max_results=candidate_k)
            if self.web_searcher is not None
            else []
        )
        pool = wiki + web
        if self.reranker is None or not pool:
            pool.sort(key=lambda r: r.score, reverse=True)
            return pool[: self.rerank_top_k]
        return self.reranker.rerank(query, pool, top_k=self.rerank_top_k)
