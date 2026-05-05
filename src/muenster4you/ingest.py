"""Ingest MediaWiki pages from SQLite into LanceDB."""

import argparse
from pathlib import Path

from tqdm import tqdm

from muenster4you.lancedb import LanceDBMediaWiki
from muenster4you.mediawiki import SQLiteMediaWiki


def ingest(
    sqlite_path: Path,
    lance_path: Path,
    namespace: int = 0,
) -> None:
    sqlite = SQLiteMediaWiki(sqlite_path)
    lance = LanceDBMediaWiki(lance_path)

    for raw in tqdm(sqlite.get_all_pages(namespace=namespace)):
        page = sqlite.get_page_content_by_id(raw.id, raw.namespace)
        if page is None:
            continue
        lance.upsert_page(
            {
                "id": page.id,
                "namespace": page.namespace,
                "title": page.title,
                "content": page.content,
                "rev_id": page.rev_id,
                "rev_timestamp": page.rev_timestamp,
                "rev_actor": page.rev_actor,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, default=Path("wiki.sqlite"))
    parser.add_argument("--lance", type=Path, default=Path("lancedb"))
    parser.add_argument("--namespace", type=int, default=0)
    args = parser.parse_args()

    print(f"ingesting from {args.sqlite} -> {args.lance} (namespace={args.namespace})")
    ingest(args.sqlite, args.lance, args.namespace)
    print("done!")


if __name__ == "__main__":
    main()
