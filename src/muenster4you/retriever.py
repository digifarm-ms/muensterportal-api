"""Similarity search over wiki and web sources."""

from pathlib import Path
from urllib.parse import quote

import numpy as np
from lancedb import connect
from muenster4you.embedder import TextEmbedder
from muenster4you.lancedb import WIKIPAGE_TABLE_NAME
from muenster4you.types import RetrievalResult, RetrievalSource


class LanceDBRetriever:
    def __init__(self, db_path: Path | str, embedder: TextEmbedder):
        self.conn = connect(db_path)
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search the LanceDB for relevant wiki pages."""
        query_vector = self.embedder.encode(query)
        table = self.conn.open_table(WIKIPAGE_TABLE_NAME)
        results = (
            table.search(query_vector)
            .distance_type("cosine")  # type: ignore[reportAttributeAccessIssue]
            .limit(top_k)
            .to_list()
        )
        return [
            RetrievalResult(
                content=r["content"],
                score=1.0 - r["_distance"],
                source=RetrievalSource.WIKI,
                url=f"/wiki/{quote(r['title'].replace(' ', '_'))}",
                embedding=np.asarray(r["embedding"], dtype=np.float32),
            )
            for r in results
        ]
