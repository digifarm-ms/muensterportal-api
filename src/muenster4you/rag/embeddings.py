"""German embedding model wrapper using sentence-transformers."""

from typing import List, Union

import numpy as np
import torch
from rich.progress import track
from sentence_transformers import SentenceTransformer

from .config import config


class GermanEmbedder:
    """Wrapper for German embedding model with MPS acceleration support."""

    def __init__(self, model_name: str = None, use_mps: bool = None):
        """
        Initialize the German embedding model.

        Args:
            model_name: Hugging Face model name (defaults to config)
            use_mps: Use Apple Silicon MPS acceleration (defaults to config)
        """
        self.model_name = model_name or config.embedding_model
        self.use_mps = use_mps if use_mps is not None else config.use_mps

        # Determine device
        if self.use_mps and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading embedding model: {self.model_name}")
        print(f"Using device: {self.device}")

        # Load model
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        # Get embedding dimension
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self._embedding_dim}")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text documents
            batch_size: Batch size for processing (defaults to config)
            show_progress: Show progress bar
            normalize: Normalize embeddings for cosine similarity

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if batch_size is None:
            batch_size = config.embedding_batch_size

        if not texts:
            return np.array([])

        # Generate embeddings with progress bar
        if show_progress:
            print(f"Generating embeddings for {len(texts)} documents...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def embed_query(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            text: Query text
            normalize: Normalize embedding for cosine similarity

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embedding

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim


if __name__ == "__main__":
    # Test embedder
    embedder = GermanEmbedder()

    # Test single query
    query = "Wo finde ich Hofläden in Münster?"
    query_emb = embedder.embed_query(query)
    print(f"\nQuery embedding shape: {query_emb.shape}")

    # Test batch of documents
    docs = [
        "Hofläden in Münster verkaufen frisches Gemüse.",
        "Der Aasee ist ein beliebter Ort für Freizeit.",
        "Münster hat viele interessante Sehenswürdigkeiten."
    ]
    doc_embs = embedder.embed_documents(docs)
    print(f"Document embeddings shape: {doc_embs.shape}")
