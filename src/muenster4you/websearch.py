"""Web search layer using Tavily (sanctioned API, free tier 1k credits/month)."""

from dataclasses import dataclass
from typing import Literal

from tavily import TavilyClient


SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


@dataclass
class WebSearchResult:
    """Raw result from Tavily search."""

    title: str
    url: str
    description: str
    score: float
    rank: int


@dataclass
class TavilySearcher:
    """Web searcher using Tavily's structured include_domains filtering."""

    client: TavilyClient
    site_filters: list[str]

    def search(
        self, query: str, max_results: int = 20, search_depth: SearchDepth = "basic"
    ) -> list[WebSearchResult]:
        """
        Search Tavily, restricted to configured Münster domains.

        Args:
            query: User query
            max_results: Maximum number of results to return

        Returns:
            List of WebSearchResult objects (empty on API error)
        """

        response = self.client.search(
            query=query,
            include_domains=self.site_filters,
            max_results=max_results,
            search_depth=search_depth,
        )

        results = []
        for rank, item in enumerate(response["results"], start=1):
            results.append(
                WebSearchResult(
                    title=item["title"],
                    url=item["url"],
                    description=item["content"],
                    score=float(item["score"]),
                    rank=rank,
                )
            )
        return results


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
    for result in results:
        print(
            f"{result.rank}. {result.title} ({result.url}) - {result.description} [Score: {result.score}]"
        )
