#!/usr/bin/env python3
"""CLI tool for inspecting MediaWiki SQLite database content."""

import sys

from muenster4you.db.mediawiki import MediaWikiDB, format_timestamp


def main():
    db_path = "wiki.sqlite"

    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} list                    # List all pages")
        print(f"  {sys.argv[0]} search <term>           # Search for pages")
        print(f"  {sys.argv[0]} get <PageTitle>         # Get page content")
        print(f"  {sys.argv[0]} history <PageTitle>     # Get revision history")
        sys.exit(1)

    db = MediaWikiDB(db_path)
    command = sys.argv[1].lower()

    try:
        if command == "list":
            print("\n=== All Pages ===\n")
            pages = db.list_all_pages(limit=100)
            for page in pages:
                redirect = " (redirect)" if page["page_is_redirect"] else ""
                print(
                    f"{page['page_id']:4d}. {page['page_title']:40s} ({page['page_len']:6d} bytes){redirect}"
                )

        elif command == "search":
            if len(sys.argv) < 3:
                print("Error: Please provide a search term")
                sys.exit(1)

            search_term = sys.argv[2]
            print(f"\n=== Search Results for '{search_term}' ===\n")
            pages = db.search_pages(search_term)

            if not pages:
                print("No pages found")
            else:
                for page in pages:
                    print(
                        f"{page['page_id']:4d}. {page['page_title']:40s} ({page['page_len']:6d} bytes)"
                    )

        elif command == "get":
            if len(sys.argv) < 3:
                print("Error: Please provide a page title")
                sys.exit(1)

            page_title = sys.argv[2].replace(" ", "_")
            page = db.get_page_content(page_title)

            if not page:
                print(f"Page '{page_title}' not found")
                sys.exit(1)

            print(f"\n{'=' * 80}")
            print(f"Page: {page['page_title']}")
            print(f"{'=' * 80}")
            print(f"Page ID:        {page['page_id']}")
            print(f"Namespace:      {page['page_namespace']}")
            print(f"Content Length: {page['page_len']} bytes")
            print(f"Content Model:  {page['content_model']}")
            print(f"Is Redirect:    {bool(page['page_is_redirect'])}")
            print("\nCurrent Revision:")
            print(f"  Revision ID:  {page['rev_id']}")
            print(f"  Timestamp:    {format_timestamp(page['rev_timestamp'])}")
            print(f"  Editor:       {page['editor']}")
            print(f"  Edit Summary: {page['edit_summary'] or '(no summary)'}")
            print(f"  Content ID:   {page['slot_content_id']}")
            print(f"  Origin Rev:   {page['slot_origin']}")
            print("\nStorage Info:")
            print(f"  Address:      {page['content_address']}")
            print(f"  Flags:        {page['storage_flags']}")
            print(f"  SHA1:         {page['content_sha1']}")
            print(f"\n{'=' * 80}")
            print("CONTENT:")
            print(f"{'=' * 80}\n")
            print(page["content"])
            print(f"\n{'=' * 80}\n")

        elif command == "history":
            if len(sys.argv) < 3:
                print("Error: Please provide a page title")
                sys.exit(1)

            page_title = sys.argv[2].replace(" ", "_")
            revisions = db.get_page_revisions(page_title)

            if not revisions:
                print(f"Page '{page_title}' not found")
                sys.exit(1)

            print(f"\n=== Revision History for '{page_title}' ===\n")
            print(
                f"{'Rev ID':<10} {'Timestamp':<20} {'Editor':<20} {'Size':<10} {'New?':<6} Edit Summary"
            )
            print("-" * 100)

            for rev in revisions:
                new_content = "NEW" if rev["is_new_content"] else "reuse"
                summary = (rev["edit_summary"] or "")[:30]
                timestamp = format_timestamp(rev["rev_timestamp"])
                print(
                    f"{rev['rev_id']:<10} {timestamp:<20} {rev['editor']:<20} {rev['content_size']:<10} {new_content:<6} {summary}"
                )

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    main()
