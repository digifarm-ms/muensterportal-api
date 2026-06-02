"""Web search layer using Tavily (sanctioned API, free tier 1k credits/month)."""

from dataclasses import dataclass
from typing import Literal

from tavily import TavilyClient

from muenster4you.types import RetrievalResult, RetrievalSource

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


@dataclass
class TavilySearcher:
    """Web searcher using Tavily's structured include_domains filtering."""

    client: TavilyClient
    site_filters: list[str]
    search_depth: SearchDepth = "basic"

    def search(self, query: str, max_results: int = 20) -> list[RetrievalResult]:
        """
        Search Tavily, restricted to configured Münster domains.

        Args:
            query: User query
            max_results: Maximum number of results to return

        Returns:
            List of RetrievalResult objects with source=WEBSEARCH
        """

        response = self.client.search(
            query=query,
            include_domains=self.site_filters,
            max_results=max_results,
            search_depth=self.search_depth,
        )

        return [
            RetrievalResult(
                content=item["content"],
                score=float(item["score"]),
                source=RetrievalSource.WEBSEARCH,
                url=item["url"],
            )
            for item in response["results"]
        ]
