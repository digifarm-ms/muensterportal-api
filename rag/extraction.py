"""Extract and clean wiki pages from MediaWiki SQLite database."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mwparserfromhell

# Add parent directory to path to import MediaWikiDB
sys.path.insert(0, str(Path(__file__).parent.parent))
from get_page_content import MediaWikiDB

from rag.config import config


@dataclass
class WikiPage:
    """Represents a cleaned wiki page."""

    page_id: int
    title: str
    content_text: str
    page_len: int


def clean_wikitext(content: str) -> str:
    """
    Clean MediaWiki markup to extract plain text.

    Args:
        content: Raw wikitext content

    Returns:
        Cleaned plain text suitable for embedding
    """
    if not content:
        return ""

    try:
        # Parse wikitext
        wikicode = mwparserfromhell.parse(content)

        # Remove templates ({{...}})
        for template in wikicode.filter_templates():
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        # Remove HTML tags
        for tag in wikicode.filter_tags():
            try:
                wikicode.remove(tag)
            except ValueError:
                pass

        # Convert to plain text
        text = wikicode.strip_code()

    except Exception:
        # Fallback to simple regex cleaning if parsing fails
        text = content

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove display_map and widget tags
    text = re.sub(r"<display_map[^>]*>.*?</display_map>", "", text, flags=re.DOTALL)
    text = re.sub(r"<widget[^>]*>.*?</widget>", "", text, flags=re.DOTALL)

    # Remove other remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Clean up whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # Multiple newlines to double newline
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single space
    text = text.strip()

    return text


def extract_all_pages(db_path: str = None, min_length: int = None) -> List[WikiPage]:
    """
    Extract all non-redirect pages from the wiki database.

    Args:
        db_path: Path to SQLite database (defaults to config)
        min_length: Minimum page length in characters (defaults to config)

    Returns:
        List of cleaned WikiPage objects
    """
    if db_path is None:
        db_path = config.wiki_db_path

    if min_length is None:
        min_length = config.min_page_length

    db = MediaWikiDB(db_path)
    pages = []

    try:
        # List all pages from namespace 0 (main namespace)
        raw_pages = db.list_all_pages(namespace=0, limit=1000)

        for raw_page in raw_pages:
            # Skip redirects
            if raw_page["page_is_redirect"]:
                continue

            # Get full page content
            page_data = db.get_page_content(raw_page["page_title"], namespace=0)

            if not page_data or not page_data.get("content"):
                continue

            # Clean the wikitext
            cleaned_text = clean_wikitext(page_data["content"])

            # Filter by minimum length
            if len(cleaned_text) < min_length:
                continue

            pages.append(
                WikiPage(
                    page_id=page_data["page_id"],
                    title=page_data["page_title"],
                    content_text=cleaned_text,
                    page_len=len(cleaned_text),
                )
            )

    finally:
        db.close()

    return pages


if __name__ == "__main__":
    # Test extraction
    pages = extract_all_pages()
    print(f"Extracted {len(pages)} pages")
    print(f"\nSample page:")
    if pages:
        sample = pages[0]
        print(f"Title: {sample.title}")
        print(f"Length: {sample.page_len} chars")
        print(f"Content preview: {sample.content_text[:200]}...")
