"""Web search layer using DuckDuckGo Search (no API key required)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np
from ddgs import DDGS
from sentence_transformers import util

from rag.config import config
from rag.retrieval import RetrievalResult

if TYPE_CHECKING:
    from rag.embeddings import GermanEmbedder


@dataclass
class WebSearchResult:
    """Raw result from DuckDuckGo Search."""

    title: str
    url: str
    description: str
    rank: int


class DuckDuckGoSearcher:
    """Web searcher using DuckDuckGo (free, no API key required)."""

    def __init__(
        self,
        site_filters: List[str] | None = None,
        max_results: int | None = None,
        embedder: "GermanEmbedder | None" = None,
    ):
        """
        Initialize the DuckDuckGo web searcher.

        Args:
            site_filters: List of domains to restrict search to
            max_results: Maximum number of results to return
            embedder: Optional embedder for computing semantic similarity scores
        """
        self.site_filters = site_filters or config.websearch_site_filters
        self.max_results = max_results or config.websearch_max_results
        self.embedder = embedder

    def _build_query(self, query: str) -> str:
        """
        Build search query for Münster-related content.

        Adds "Münster" to query if not present to improve relevance.

        Args:
            query: User query

        Returns:
            Query optimized for Münster-related results
        """
        # Add Münster to query if not already present for better relevance
        if "münster" not in query.lower() and "muenster" not in query.lower():
            return f"{query} Münster"
        return query

    def _matches_site_filter(self, url: str) -> bool:
        """Check if URL matches any of the site filters."""
        if not self.site_filters:
            return True
        return any(site in url for site in self.site_filters)

    def search(self, query: str) -> List[WebSearchResult]:
        """
        Search using DuckDuckGo with optional domain filtering.

        Note: DuckDuckGo API doesn't support site: operator reliably,
        so we filter results by domain after fetching.

        Args:
            query: User query

        Returns:
            List of WebSearchResult objects
        """
        search_query = self._build_query(query)

        try:
            # Fetch more results than needed to allow for filtering
            fetch_count = self.max_results * 3 if self.site_filters else self.max_results

            with DDGS() as ddgs:
                # Note: region filter can cause unreliable results, so we omit it
                raw_results = list(ddgs.text(
                    search_query,
                    safesearch="moderate",
                    max_results=fetch_count,
                ))

        except Exception as e:
            print(f"Web search error: {e}")
            return []

        if not raw_results:
            # DuckDuckGo may return empty results due to rate limiting
            print("Web search returned no results (possible rate limiting)")
            return []

        # Parse and filter results
        results = []
        rank = 0
        for item in raw_results:
            url = item.get("href", "")

            # Apply site filter if configured
            if self.site_filters and not self._matches_site_filter(url):
                continue

            rank += 1
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=url,
                    description=item.get("body", ""),
                    rank=rank,
                )
            )

            if rank >= self.max_results:
                break

        return results

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve web results as RetrievalResult objects for generator compatibility.

        Converts web search results to the same format as wiki retrieval results,
        allowing them to be merged and processed by the RAG generator.

        If an embedder is provided, computes real semantic similarity scores.
        Otherwise, falls back to rank-based scoring.

        Args:
            query: User query

        Returns:
            List of RetrievalResult objects with source="web"
        """
        web_results = self.search(query)

        if not web_results:
            return []

        # Compute similarity scores
        if self.embedder is not None:
            scores = self._compute_embedding_scores(query, web_results)
        else:
            # Fall back to rank-based scoring
            scores = [max(0.5, 1.0 - (r.rank - 1) * 0.05) for r in web_results]

        retrieval_results = []
        for result, score in zip(web_results, scores):
            retrieval_results.append(
                RetrievalResult(
                    page_id=-result.rank,  # Negative ID to distinguish from wiki
                    page_title=result.title,
                    content_text=result.description,
                    similarity_score=score,
                    page_len=len(result.description),
                    source="web",
                    source_url=result.url,
                )
            )

        return retrieval_results

    def _compute_embedding_scores(
        self, query: str, results: List[WebSearchResult]
    ) -> List[float]:
        """
        Compute semantic similarity scores using embeddings.

        Args:
            query: User query
            results: List of web search results

        Returns:
            List of similarity scores (0-1) for each result
        """
        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Create text for each result (title + description for richer context)
        result_texts = [
            f"{r.title}. {r.description}" for r in results
        ]

        # Embed all results
        result_embeddings = self.embedder.embed_documents(
            result_texts, show_progress=False
        )

        # Compute cosine similarities
        similarities = util.cos_sim(
            query_embedding.reshape(1, -1),
            result_embeddings
        )[0]

        # Convert to list of floats
        return [float(s) for s in similarities.cpu().numpy()]


if __name__ == "__main__":
    # Test web search
    searcher = DuckDuckGoSearcher()

    test_query = "Veranstaltungen in Münster"
    print(f"Query: {test_query}")
    print(f"Site filters: {searcher.site_filters}")
    print(f"Built query: {searcher._build_query(test_query)}")
    print()

    results = searcher.retrieve(test_query)

    for result in results:
        print(f"[{result.source}] {result.page_title}")
        print(f"  URL: {result.source_url}")
        print(f"  Score: {result.similarity_score:.2f}")
        print(f"  {result.content_text[:100]}...")
        print()
