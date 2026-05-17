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


class _StubCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder used by the unit tests."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        pairs = list(pairs)
        self.calls.append(pairs)
        return self._scores[: len(pairs)]


@pytest.fixture
def patched_cross_encoder(monkeypatch):
    """Replace the real CrossEncoder so constructing the reranker does no I/O."""
    holder: dict[str, _StubCrossEncoder] = {}

    def _factory(model_id: str):
        stub = _StubCrossEncoder(scores=holder.get("scores", []))
        holder["stub"] = stub
        holder["model_id"] = model_id
        return stub

    monkeypatch.setattr(reranker_module, "CrossEncoder", _factory)
    return holder


def test_rerank_sorts_by_new_scores_descending(patched_cross_encoder):
    patched_cross_encoder["scores"] = [0.1, 0.9, 0.5]
    reranker = CrossEncoderReranker(model_id="stub-model")

    candidates = [_result("a"), _result("b"), _result("c")]
    out = reranker.rerank(query="q", candidates=candidates, top_k=3)

    assert [r.content for r in out] == ["b", "c", "a"]
    assert [r.score for r in out] == [0.9, 0.5, 0.1]


def test_rerank_truncates_to_top_k(patched_cross_encoder):
    patched_cross_encoder["scores"] = [0.1, 0.9, 0.5, 0.7]
    reranker = CrossEncoderReranker(model_id="stub-model")

    candidates = [_result("a"), _result("b"), _result("c"), _result("d")]
    out = reranker.rerank(query="q", candidates=candidates, top_k=2)

    assert [r.content for r in out] == ["b", "d"]
    assert len(out) == 2


def test_rerank_replaces_score_field(patched_cross_encoder):
    patched_cross_encoder["scores"] = [0.42]
    reranker = CrossEncoderReranker(model_id="stub-model")

    candidates = [_result("only", score=0.0001)]
    out = reranker.rerank(query="q", candidates=candidates, top_k=1)

    assert out[0].score == pytest.approx(0.42)
    assert out[0].content == "only"
    assert out[0].source is RetrievalSource.WIKI
    assert out[0].url == "/wiki/only"


def test_rerank_empty_candidates_short_circuits(patched_cross_encoder):
    patched_cross_encoder["scores"] = []
    reranker = CrossEncoderReranker(model_id="stub-model")

    out = reranker.rerank(query="q", candidates=[], top_k=5)

    assert out == []
    assert patched_cross_encoder["stub"].calls == []


def test_rerank_passes_query_and_content_pairs_to_model(patched_cross_encoder):
    patched_cross_encoder["scores"] = [0.0, 0.0]
    reranker = CrossEncoderReranker(model_id="stub-model")

    candidates = [_result("alpha"), _result("beta")]
    reranker.rerank(query="my query", candidates=candidates, top_k=2)

    assert patched_cross_encoder["stub"].calls == [
        [("my query", "alpha"), ("my query", "beta")]
    ]
