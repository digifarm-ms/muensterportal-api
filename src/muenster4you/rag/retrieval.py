"""Retrieval layer for similarity search over wiki embeddings."""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sentence_transformers import util

from .config import config
from .embeddings import GermanEmbedder
from .storage import load_embeddings


@dataclass
class RetrievalResult:
    """Result from similarity search."""

    page_id: int
    page_title: str
    content_text: str
    similarity_score: float
    page_len: int
    source: str = "wiki"  # "wiki" or "web"
    source_url: str | None = None  # URL for web results


class WikiRetriever:
    """Retriever for semantic search over wiki pages."""

    def __init__(
        self,
        embeddings_path: str = None,
        embedder: GermanEmbedder = None
    ):
        """
        Initialize the retriever.

        Args:
            embeddings_path: Path to parquet file with embeddings
            embedder: Optional pre-initialized embedder
        """
        if embeddings_path is None:
            embeddings_path = str(config.embeddings_path_resolved)

        # Load embeddings and metadata
        print(f"Loading embeddings from {embeddings_path}...")
        self.df, self.doc_embeddings = load_embeddings(embeddings_path)

        # Initialize or use provided embedder
        if embedder is None:
            print("Initializing embedding model...")
            self.embedder = GermanEmbedder()
        else:
            self.embedder = embedder

        # Convert to torch tensor for sentence-transformers util
        self.doc_embeddings_tensor = torch.from_numpy(self.doc_embeddings)

        print(f"Retriever ready with {len(self.df)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k most similar documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold

        Returns:
            List of RetrievalResult objects sorted by similarity score
        """
        if top_k is None:
            top_k = config.default_top_k

        if min_score is None:
            min_score = config.min_similarity_score

        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        query_embedding_tensor = torch.from_numpy(query_embedding)

        # Compute cosine similarity using sentence-transformers util
        similarities = util.cos_sim(query_embedding_tensor, self.doc_embeddings_tensor)[0]

        # Convert to numpy for easier manipulation
        similarities_np = similarities.cpu().numpy()

        # Get top-k indices
        top_indices = np.argsort(similarities_np)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities_np[idx])

            # Filter by minimum score
            if score < min_score:
                continue

            row = self.df.row(idx, named=True)
            results.append(
                RetrievalResult(
                    page_id=row["page_id"],
                    page_title=row["page_title"],
                    content_text=row["content_text"],
                    similarity_score=score,
                    page_len=row["page_len"],
                )
            )

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = None,
        min_score: float = None
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries (batch processing).

        Args:
            queries: List of query texts
            top_k: Number of documents to retrieve per query
            min_score: Minimum similarity score threshold

        Returns:
            List of result lists (one per query)
        """
        return [self.retrieve(q, top_k, min_score) for q in queries]


if __name__ == "__main__":
    # Test retrieval
    retriever = WikiRetriever()

    test_queries = [
        "Wo finde ich Hofläden in Münster?",
        "Was kann man in der Freizeit machen?",
        "Geschichte von Münster"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        results = retriever.retrieve(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.page_title} (score: {result.similarity_score:.4f})")
            print(f"   {result.content_text[:200]}...")
