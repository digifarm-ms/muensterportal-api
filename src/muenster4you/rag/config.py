"""Configuration for RAG system using Pydantic Settings."""

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """Configuration for RAG system with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="M4U_RAG_",
        extra="ignore"
    )

    # Paths
    wiki_db_path: str = "wiki.sqlite"
    embeddings_path: str = "data/wiki_embeddings.parquet"

    # Model configuration
    embedding_model: str = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    generation_model: str = "qwen3:30b"
    ollama_url: str = "http://localhost:11434"

    # Extraction parameters
    min_page_length: int = 100

    # Retrieval parameters
    default_top_k: int = 5
    min_similarity_score: float = 0.3

    # Generation parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 2048

    # Performance
    embedding_batch_size: int = 32
    use_mps: bool = True  # Apple Silicon acceleration

    # Web search (DuckDuckGo - no API key required)
    websearch_enabled: bool = False
    websearch_site_filters: List[str] = Field(
        default_factory=lambda: ["muenster.de", "stadt-muenster.de", "muensterland.de"]
    )
    websearch_max_results: int = 5

    @property
    def wiki_db_path_resolved(self) -> Path:
        """Return resolved path for wiki database."""
        return Path(self.wiki_db_path).resolve()

    @property
    def embeddings_path_resolved(self) -> Path:
        """Return resolved path for embeddings parquet file."""
        return Path(self.embeddings_path).resolve()


# Default config instance
config = RAGConfig()
