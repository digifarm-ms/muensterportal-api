"""MediaWiki SQLite database access layer."""

import sqlite3
from typing import Dict, List, Optional


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
