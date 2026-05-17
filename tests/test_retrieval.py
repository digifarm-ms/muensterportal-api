from dataclasses import dataclass, field

from muenster4you.retrieval import RetrievalOrchestrator
from muenster4you.types import RetrievalResult, RetrievalSource


def _result(content: str, score: float, source: RetrievalSource) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source=source,
        url=f"/{source.value}/{content}",
    )


@dataclass
class _StubWikiRetriever:
    results: list[RetrievalResult]
    calls: list[int] = field(default_factory=list)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append(top_k)
        return list(self.results[:top_k])


@dataclass
class _StubWebSearcher:
    results: list[RetrievalResult]
    calls: list[int] = field(default_factory=list)

    def search(self, query: str, max_results: int) -> list[RetrievalResult]:
        self.calls.append(max_results)
        return list(self.results[:max_results])


@dataclass
class _StubReranker:
    """Reranks by content-length descending — deterministic, no model."""

    calls: list[tuple[str, int, int]] = field(default_factory=list)

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        self.calls.append((query, len(candidates), top_k))
        ranked = sorted(candidates, key=lambda r: len(r.content), reverse=True)
        return ranked[:top_k]


def test_oversample_factor_drives_candidate_k():
    wiki = _StubWikiRetriever(results=[])
    web = _StubWebSearcher(results=[])
    reranker = _StubReranker()
    orchestrator = RetrievalOrchestrator(
        wiki_retriever=wiki,
        web_searcher=web,
        reranker=reranker,
        rerank_top_k=3,
        oversample_factor=4,
    )

    orchestrator.retrieve("anything")

    assert wiki.calls == [12]
    assert web.calls == [12]


def test_retrieve_pools_wiki_and_web_then_reranks():
    wiki = _StubWikiRetriever(
        results=[
            _result("short", 0.9, RetrievalSource.WIKI),
            _result("medium length", 0.5, RetrievalSource.WIKI),
        ]
    )
    web = _StubWebSearcher(
        results=[
            _result("this is the longest content of all", 0.1, RetrievalSource.WEBSEARCH),
            _result("xy", 0.8, RetrievalSource.WEBSEARCH),
        ]
    )
    reranker = _StubReranker()
    orchestrator = RetrievalOrchestrator(
        wiki_retriever=wiki,
        web_searcher=web,
        reranker=reranker,
        rerank_top_k=2,
        oversample_factor=10,
    )

    out = orchestrator.retrieve("query")

    assert reranker.calls == [("query", 4, 2)]
    assert [r.content for r in out] == [
        "this is the longest content of all",
        "medium length",
    ]


def test_no_web_searcher_only_uses_wiki():
    wiki = _StubWikiRetriever(
        results=[_result("only wiki", 0.7, RetrievalSource.WIKI)]
    )
    reranker = _StubReranker()
    orchestrator = RetrievalOrchestrator(
        wiki_retriever=wiki,
        web_searcher=None,
        reranker=reranker,
        rerank_top_k=5,
        oversample_factor=2,
    )

    out = orchestrator.retrieve("q")

    assert wiki.calls == [10]
    assert [r.content for r in out] == ["only wiki"]
    assert reranker.calls == [("q", 1, 5)]


def test_no_reranker_falls_back_to_score_sort_and_truncates():
    wiki = _StubWikiRetriever(
        results=[
            _result("a", 0.3, RetrievalSource.WIKI),
            _result("b", 0.9, RetrievalSource.WIKI),
        ]
    )
    web = _StubWebSearcher(
        results=[
            _result("c", 0.5, RetrievalSource.WEBSEARCH),
            _result("d", 0.7, RetrievalSource.WEBSEARCH),
        ]
    )
    orchestrator = RetrievalOrchestrator(
        wiki_retriever=wiki,
        web_searcher=web,
        reranker=None,
        rerank_top_k=2,
        oversample_factor=10,
    )

    out = orchestrator.retrieve("q")

    assert [r.content for r in out] == ["b", "d"]
    assert [r.score for r in out] == [0.9, 0.7]


def test_empty_pool_returns_empty_without_calling_reranker():
    wiki = _StubWikiRetriever(results=[])
    reranker = _StubReranker()
    orchestrator = RetrievalOrchestrator(
        wiki_retriever=wiki,
        web_searcher=None,
        reranker=reranker,
        rerank_top_k=5,
        oversample_factor=2,
    )

    out = orchestrator.retrieve("q")

    assert out == []
    assert reranker.calls == []
