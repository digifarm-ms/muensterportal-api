"""Ingest MediaWiki pages from SQLite into LanceDB."""

import argparse
import re
from collections.abc import Iterable
from itertools import batched
from pathlib import Path

import mwparserfromhell
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import get_device_name
from tqdm import tqdm

from muenster4you.lancedb import (
    EMBEDDING_MODEL_NAME,
    LanceDBMediaWiki,
)
from muenster4you.mediawiki import SQLiteMediaWiki
from muenster4you.types import BaseWikiPage, WikiPage


def clean_wikitext(content: str) -> str:
    """Clean MediaWiki markup to extract plain text suitable for embedding."""
    if not content:
        return ""

    try:
        wikicode = mwparserfromhell.parse(content)

        for template in wikicode.filter_templates():
            try:
                wikicode.remove(template)
            except ValueError:
                pass

        for tag in wikicode.filter_tags():
            try:
                wikicode.remove(tag)
            except ValueError:
                pass

        text = wikicode.strip_code()

    except Exception:
        text = content

    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<display_map[^>]*>.*?</display_map>", "", text, flags=re.DOTALL)
    text = re.sub(r"<widget[^>]*>.*?</widget>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text


def load_pages_from_media_wiki(
    sqlite: SQLiteMediaWiki, namespace: int
) -> Iterable[BaseWikiPage]:
    for raw in tqdm(sqlite.get_all_pages(namespace=namespace)):
        page = sqlite.get_page_content_by_id(raw.id, raw.namespace)
        if page is None:
            continue
        yield {
            "id": page.id,
            "namespace": page.namespace,
            "title": page.title,
            "content": page.content,
            "rev_id": page.rev_id,
            "rev_timestamp": page.rev_timestamp,
            "rev_actor": page.rev_actor,
        }


def clean_pages(pages: Iterable[BaseWikiPage]) -> Iterable[BaseWikiPage]:
    for page in pages:
        cleaned_content = clean_wikitext(page["content"])
        if cleaned_content == "":
            continue
        yield {
            **page,
            "content": cleaned_content,
        }


def add_embeddings(
    pages: Iterable[BaseWikiPage], embedder: SentenceTransformer, batch_size: int = 32
) -> Iterable[WikiPage]:
    for batch in batched(pages, batch_size):
        texts = [page["content"] for page in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        for page, embedding in zip(batch, embeddings):
            yield {
                **page,
                "embedding": embedding.tolist(),
            }


def ingest(
    sqlite_path: Path,
    lance_path: Path,
    namespace: int = 0,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 32,
) -> None:
    sqlite = SQLiteMediaWiki(sqlite_path)
    lance = LanceDBMediaWiki(lance_path)
    embedder = SentenceTransformer(
        embedding_model, trust_remote_code=True, device=get_device_name()
    )

    raw_pages = load_pages_from_media_wiki(sqlite, namespace)
    pages = clean_pages(raw_pages)
    pages_with_embeddings = add_embeddings(pages, embedder, batch_size)
    for batch in batched(pages_with_embeddings, 50):
        lance.upsert_pages(list(batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, default=Path("wiki.sqlite"))
    parser.add_argument("--lance", type=Path, default=Path("lancedb"))
    parser.add_argument("--namespace", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL_NAME)
    args = parser.parse_args()

    print(
        f"ingesting from {args.sqlite} -> {args.lance} "
        f"(namespace={args.namespace}, batch_size={args.batch_size}, embedding_model={args.embedding_model})"
    )
    ingest(
        args.sqlite, args.lance, args.namespace, args.embedding_model, args.batch_size
    )
    print("done!")


if __name__ == "__main__":
    main()
