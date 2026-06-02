"""Web search layer using Tavily (sanctioned API, free tier 1k credits/month)."""

from dataclasses import dataclass
from typing import Literal

from tavily import TavilyClient

from muenster4you.types import RetrievalResult, RetrievalSource

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


@dataclass
class TavilySearcher:
    """Web searcher using Tavily's structured include_domains filtering.

    `site_filters` is passed verbatim to Tavily's `include_domains`. Entries
    may be bare hostnames or include path prefixes (e.g.
    `"de.wikipedia.org/wiki/Portal:Münster"`); Tavily honors path prefixes,
    so curator path choices in the allowlist act as section-level filters.
    """

    client: TavilyClient
    site_filters: list[str]
    search_depth: SearchDepth = "basic"
    location_keyword: str = "Münster"

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
            query=self._inject_location(query),
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

    def _inject_location(self, query: str) -> str:
        """Append the location keyword if missing.

        The chatbot is implicitly about Münster, but ~20% of user queries
        don't repeat the city name. Without the keyword, Tavily can't tell
        a "Krankenversicherung" question is local — and our allowlist is
        not enough of a location signal on its own.
        """
        kw_lower = self.location_keyword.lower()
        kw_ascii = (
            kw_lower.replace("ü", "ue")
            .replace("ö", "oe")
            .replace("ä", "ae")
            .replace("ß", "ss")
        )
        q_lower = query.lower()
        if kw_lower in q_lower or kw_ascii in q_lower:
            return query
        return f"{query} {self.location_keyword}"
