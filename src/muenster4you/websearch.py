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

    def search(
        self, query: str, max_results: int = 20, search_depth: SearchDepth = "basic"
    ) -> list[RetrievalResult]:
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
            search_depth=search_depth,
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


if __name__ == "__main__":
    import os

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    searcher = TavilySearcher(
        client=client,
        site_filters=[
            "muenster.org",
            "muenster4you.de",
            "stadt-muenster.de",
            "muensterland.com",
        ],
    )
    results = searcher.search("best restaurants in Münster", max_results=5)
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.url} - {result.content} [Score: {result.score}]")
