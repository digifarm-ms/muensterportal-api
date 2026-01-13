#!/usr/bin/env python3
"""
MediaWiki Page Content Retriever
Demonstrates how to extract and display page content from MediaWiki SQLite database
"""

import sqlite3
import sys
from typing import Optional, Dict, List

class MediaWikiDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name

    def get_page_content(self, page_title: str, namespace: int = 0) -> Optional[Dict]:
        """
        Get the current content of a page by following the complete chain:
        page -> revision -> slots -> content -> text
        """
        query = """
        SELECT
            p.page_id,
            p.page_namespace,
            p.page_title,
            p.page_latest as current_rev_id,
            p.page_len,
            p.page_is_redirect,
            r.rev_id,
            r.rev_timestamp,
            r.rev_actor,
            a.actor_name as editor,
            c.comment_text as edit_summary,
            s.slot_role_id,
            sr.role_name as slot_role,
            s.slot_content_id,
            s.slot_origin,
            co.content_size,
            co.content_sha1,
            cm.model_name as content_model,
            co.content_address,
            t.old_text as content,
            t.old_flags as storage_flags
        FROM page p
        JOIN revision r ON p.page_latest = r.rev_id
        JOIN actor a ON r.rev_actor = a.actor_id
        LEFT JOIN comment c ON r.rev_comment_id = c.comment_id
        JOIN slots s ON r.rev_id = s.slot_revision_id
        JOIN slot_roles sr ON s.slot_role_id = sr.role_id
        JOIN content co ON s.slot_content_id = co.content_id
        JOIN content_models cm ON co.content_model = cm.model_id
        JOIN text t ON CAST(substr(co.content_address, 4) AS INTEGER) = t.old_id
        WHERE p.page_title = ? AND p.page_namespace = ?
        """

        cursor = self.conn.execute(query, (page_title, namespace))
        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def get_page_revisions(self, page_title: str, namespace: int = 0) -> List[Dict]:
        """Get all revisions for a page"""
        query = """
        SELECT
            r.rev_id,
            r.rev_timestamp,
            a.actor_name as editor,
            c.comment_text as edit_summary,
            co.content_size,
            s.slot_origin,
            CASE
                WHEN s.slot_origin = r.rev_id THEN 1
                ELSE 0
            END as is_new_content
        FROM page p
        JOIN revision r ON p.page_id = r.rev_page
        JOIN actor a ON r.rev_actor = a.actor_id
        LEFT JOIN comment c ON r.rev_comment_id = c.comment_id
        JOIN slots s ON r.rev_id = s.slot_revision_id
        JOIN content co ON s.slot_content_id = co.content_id
        WHERE p.page_title = ? AND p.page_namespace = ?
        ORDER BY r.rev_timestamp DESC
        """

        cursor = self.conn.execute(query, (page_title, namespace))
        return [dict(row) for row in cursor.fetchall()]

    def list_all_pages(self, namespace: int = 0, limit: int = 50) -> List[Dict]:
        """List all pages in a namespace"""
        query = """
        SELECT
            page_id,
            page_namespace,
            page_title,
            page_len,
            page_is_redirect
        FROM page
        WHERE page_namespace = ?
        ORDER BY page_title
        LIMIT ?
        """

        cursor = self.conn.execute(query, (namespace, limit))
        return [dict(row) for row in cursor.fetchall()]

    def search_pages(self, search_term: str) -> List[Dict]:
        """Search for pages by title"""
        query = """
        SELECT
            page_id,
            page_namespace,
            page_title,
            page_len
        FROM page
        WHERE page_title LIKE ?
        ORDER BY page_title
        LIMIT 20
        """

        cursor = self.conn.execute(query, (f"%{search_term}%",))
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()


def format_timestamp(timestamp: str) -> str:
    """Format MediaWiki timestamp (YYYYMMDDHHmmSS) to readable format"""
    if not timestamp or len(timestamp) < 14:
        return timestamp
    return f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"


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
                redirect = " (redirect)" if page['page_is_redirect'] else ""
                print(f"{page['page_id']:4d}. {page['page_title']:40s} ({page['page_len']:6d} bytes){redirect}")

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
                    print(f"{page['page_id']:4d}. {page['page_title']:40s} ({page['page_len']:6d} bytes)")

        elif command == "get":
            if len(sys.argv) < 3:
                print("Error: Please provide a page title")
                sys.exit(1)

            page_title = sys.argv[2].replace(" ", "_")
            page = db.get_page_content(page_title)

            if not page:
                print(f"Page '{page_title}' not found")
                sys.exit(1)

            print(f"\n{'='*80}")
            print(f"Page: {page['page_title']}")
            print(f"{'='*80}")
            print(f"Page ID:        {page['page_id']}")
            print(f"Namespace:      {page['page_namespace']}")
            print(f"Content Length: {page['page_len']} bytes")
            print(f"Content Model:  {page['content_model']}")
            print(f"Is Redirect:    {bool(page['page_is_redirect'])}")
            print(f"\nCurrent Revision:")
            print(f"  Revision ID:  {page['rev_id']}")
            print(f"  Timestamp:    {format_timestamp(page['rev_timestamp'])}")
            print(f"  Editor:       {page['editor']}")
            print(f"  Edit Summary: {page['edit_summary'] or '(no summary)'}")
            print(f"  Content ID:   {page['slot_content_id']}")
            print(f"  Origin Rev:   {page['slot_origin']}")
            print(f"\nStorage Info:")
            print(f"  Address:      {page['content_address']}")
            print(f"  Flags:        {page['storage_flags']}")
            print(f"  SHA1:         {page['content_sha1']}")
            print(f"\n{'='*80}")
            print("CONTENT:")
            print(f"{'='*80}\n")
            print(page['content'])
            print(f"\n{'='*80}\n")

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
            print(f"{'Rev ID':<10} {'Timestamp':<20} {'Editor':<20} {'Size':<10} {'New?':<6} Edit Summary")
            print("-" * 100)

            for rev in revisions:
                new_content = "NEW" if rev['is_new_content'] else "reuse"
                summary = (rev['edit_summary'] or '')[:30]
                timestamp = format_timestamp(rev['rev_timestamp'])
                print(f"{rev['rev_id']:<10} {timestamp:<20} {rev['editor']:<20} {rev['content_size']:<10} {new_content:<6} {summary}")

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    main()
