from pathlib import Path

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    lancedb_fp: Path

    embedding_model: str

    reranker_model: str = "Alibaba-NLP/gte-multilingual-reranker-base"
    rerank_top_k: int = 5
    retrieval_oversample_factor: int = 4
