"""Storage layer for embeddings using Polars and Parquet format."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl

from .extraction import WikiPage


def save_embeddings(
    pages: List[WikiPage],
    embeddings: np.ndarray,
    output_path: str,
    compression: str = "snappy"
) -> None:
    """
    Save pages and their embeddings to a Parquet file.

    Args:
        pages: List of WikiPage objects
        embeddings: Numpy array of embeddings (n_pages, embedding_dim)
        output_path: Path to save parquet file
        compression: Compression algorithm (snappy, zstd, etc.)
    """
    if len(pages) != len(embeddings):
        raise ValueError(
            f"Number of pages ({len(pages)}) must match number of embeddings ({len(embeddings)})"
        )

    embedding_dim = embeddings.shape[1]

    # Create DataFrame with proper schema
    df = pl.DataFrame({
        "page_id": [p.page_id for p in pages],
        "page_title": [p.title for p in pages],
        "content_text": [p.content_text for p in pages],
        "embedding": embeddings.tolist(),  # Convert to list for Polars
        "page_len": [p.page_len for p in pages],
    })

    # Cast embedding to fixed-size array
    df = df.with_columns(
        pl.col("embedding").cast(pl.Array(pl.Float32, embedding_dim))
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to parquet with compression
    df.write_parquet(output_path, compression=compression)

    print(f"Saved {len(pages)} pages with embeddings to {output_path}")
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")


def load_embeddings(input_path: str) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Load embeddings and metadata from a Parquet file.

    Args:
        input_path: Path to parquet file

    Returns:
        Tuple of (DataFrame with metadata, embeddings as numpy array)
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Embeddings file not found: {input_path}")

    # Load parquet file
    df = pl.read_parquet(input_path)

    # Extract embeddings as numpy array
    embeddings_list = df["embedding"].to_list()
    embeddings = np.array(embeddings_list, dtype=np.float32)

    print(f"Loaded {len(df)} pages from {input_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    return df, embeddings


def get_embedding_stats(df: pl.DataFrame) -> dict:
    """
    Get statistics about the embeddings dataset.

    Args:
        df: DataFrame with embeddings

    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_pages": len(df),
        "avg_content_length": df["page_len"].mean(),
        "min_content_length": df["page_len"].min(),
        "max_content_length": df["page_len"].max(),
        "total_characters": df["page_len"].sum(),
    }

    return stats


if __name__ == "__main__":
    # Test with dummy data
    from .extraction import WikiPage

    test_pages = [
        WikiPage(1, "TestPage1", "This is test content for page 1", 30),
        WikiPage(2, "TestPage2", "This is test content for page 2", 30),
    ]

    # Create dummy embeddings
    test_embeddings = np.random.rand(2, 1024).astype(np.float32)

    # Save
    test_path = "data/test_embeddings.parquet"
    save_embeddings(test_pages, test_embeddings, test_path)

    # Load
    df, embeddings = load_embeddings(test_path)
    print("\nLoaded data:")
    print(df.head())

    # Stats
    stats = get_embedding_stats(df)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Clean up
    Path(test_path).unlink()
    print("\nTest completed successfully!")
