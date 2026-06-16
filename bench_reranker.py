"""Bench bi-encoder vs cross-encoder reranking on real local data.

Loads ~20 wiki candidates from LanceDB for a German query, reranks them
with each strategy, prints timing and the top-5 ordering. Excludes any
HF model download from the timer.
"""

import time
from contextlib import contextmanager

from sentence_transformers import SentenceTransformer

from muenster4you.config import AppConfig
from muenster4you.embedder import SentenceTransformerEmbedder
from muenster4you.reranker import BiEncoderReranker, CrossEncoderReranker
from muenster4you.retriever import LanceDBRetriever
from muenster4you.types import RetrievalResult

QUERIES = [
    "Was ist der Aasee?",
    "Wo finde ich Informationen zur Stadtgeschichte Münsters?",
    "Welche Hofläden gibt es in Münster?",
    "Wie melde ich mich beim Bürgeramt an?",
]

RERANKERS_TO_BENCH = [
    ("bi/jina-v5-nano (reuse embedder)", "BI"),
    ("ce/BAAI/bge-reranker-base (prod)", "BAAI/bge-reranker-base"),
    ("ce/Alibaba-NLP/gte-multilingual-reranker-base (config default)",
     "Alibaba-NLP/gte-multilingual-reranker-base"),
]

CANDIDATES_PER_QUERY = 20


@contextmanager
def timer(label: str, results: dict):
    t0 = time.perf_counter()
    yield
    results[label] = time.perf_counter() - t0


def fmt_result(r: RetrievalResult) -> str:
    return f"{r.score:+.4f}  {r.url[:60]}"


def main() -> None:
    cfg = AppConfig()  # type: ignore[call-arg]
    print(f"embedding_model={cfg.embedding_model}")
    print(f"lancedb_fp={cfg.lancedb_fp}")
    print()

    # Shared embedder (loaded once, reused by retriever + bi-encoder rerank)
    print("Loading embedder model (one-time, excluded from rerank timing)...")
    t0 = time.perf_counter()
    st_model = SentenceTransformer(
        model_name_or_path=cfg.embedding_model, trust_remote_code=True
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    retriever = LanceDBRetriever(
        db_path=cfg.lancedb_fp,
        embedder=SentenceTransformerEmbedder(model=st_model),
    )

    # Build candidate pools once per query (wiki-only — web search would
    # add ~10s of network noise, no rerank-relevant difference).
    pools: dict[str, list[RetrievalResult]] = {}
    for q in QUERIES:
        pools[q] = retriever.search(q, top_k=CANDIDATES_PER_QUERY)
        print(f"[{q!r}] retrieved {len(pools[q])} candidates")

    # Load each reranker once, time per-query rerank.
    for label, key in RERANKERS_TO_BENCH:
        print(f"\n=== {label} ===")
        t0 = time.perf_counter()
        if key == "BI":
            rr = BiEncoderReranker(model=st_model)
        else:
            rr = CrossEncoderReranker(model_id=key)
        print(f"  init: {time.perf_counter() - t0:.1f}s")

        # Warmup on the first query (cold path is misleading).
        _ = rr.rerank(QUERIES[0], pools[QUERIES[0]], top_k=5)

        for q in QUERIES:
            t0 = time.perf_counter()
            top = rr.rerank(q, pools[q], top_k=5)
            dt = time.perf_counter() - t0
            print(f"\n  [{q!r}]  rerank {len(pools[q])} → 5 in {dt:.2f}s")
            for r in top:
                print(f"    {fmt_result(r)}")


if __name__ == "__main__":
    main()
