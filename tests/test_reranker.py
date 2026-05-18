import pytest

from muenster4you import reranker as reranker_module
from muenster4you.reranker import CrossEncoderReranker
from muenster4you.types import RetrievalResult, RetrievalSource


def _result(content: str, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source=RetrievalSource.WIKI,
        url=f"/wiki/{content}",
    )


class _SpyCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder: canned scores + call recording."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        pairs = list(pairs)
        self.calls.append(pairs)
        return self._scores[: len(pairs)]


def _patch_cross_encoder(monkeypatch, spy: _SpyCrossEncoder) -> None:
    """Make CrossEncoderReranker(model_id=...) return this spy, no I/O."""
    monkeypatch.setattr(reranker_module, "CrossEncoder", lambda *_a, **_kw: spy)


def test_rerank_sorts_by_new_scores_descending(monkeypatch):
    spy = _SpyCrossEncoder(scores=[0.1, 0.9, 0.5])
    _patch_cross_encoder(monkeypatch, spy)
    reranker = CrossEncoderReranker(model_id="spy-model")

    candidates = [_result("a"), _result("b"), _result("c")]
    out = reranker.rerank(query="q", candidates=candidates, top_k=3)

    assert [r.content for r in out] == ["b", "c", "a"]
    assert [r.score for r in out] == [0.9, 0.5, 0.1]


def test_rerank_truncates_to_top_k(monkeypatch):
    spy = _SpyCrossEncoder(scores=[0.1, 0.9, 0.5, 0.7])
    _patch_cross_encoder(monkeypatch, spy)
    reranker = CrossEncoderReranker(model_id="spy-model")

    candidates = [_result("a"), _result("b"), _result("c"), _result("d")]
    out = reranker.rerank(query="q", candidates=candidates, top_k=2)

    assert [r.content for r in out] == ["b", "d"]
    assert len(out) == 2


def test_rerank_replaces_score_field(monkeypatch):
    spy = _SpyCrossEncoder(scores=[0.42])
    _patch_cross_encoder(monkeypatch, spy)
    reranker = CrossEncoderReranker(model_id="spy-model")

    candidates = [_result("only", score=0.0001)]
    out = reranker.rerank(query="q", candidates=candidates, top_k=1)

    assert out[0].score == pytest.approx(0.42)
    assert out[0].content == "only"
    assert out[0].source is RetrievalSource.WIKI
    assert out[0].url == "/wiki/only"


def test_rerank_empty_candidates_short_circuits(monkeypatch):
    spy = _SpyCrossEncoder(scores=[])
    _patch_cross_encoder(monkeypatch, spy)
    reranker = CrossEncoderReranker(model_id="spy-model")

    out = reranker.rerank(query="q", candidates=[], top_k=5)

    assert out == []
    assert spy.calls == []


def test_rerank_passes_query_and_content_pairs_to_model(monkeypatch):
    spy = _SpyCrossEncoder(scores=[0.0, 0.0])
    _patch_cross_encoder(monkeypatch, spy)
    reranker = CrossEncoderReranker(model_id="spy-model")

    candidates = [_result("alpha"), _result("beta")]
    reranker.rerank(query="my query", candidates=candidates, top_k=2)

    assert spy.calls == [[("my query", "alpha"), ("my query", "beta")]]
