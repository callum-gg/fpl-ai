"""Bluesky via the public AT Protocol API. No key needed for reads. docs/02 tier 5.

Cheap and unrestricted, so it is worth doing properly: follow the football journalists
who migrated plus the FPL community accounts, and search the same handles by keyword.
"""

from __future__ import annotations

import logging

from ..db.engine import query_one
from ..db.settings_store import global_settings
from .base import Connector, ParsedBatch, RawDoc, fetch_url
from .rss_news import chunk_text

log = logging.getLogger(__name__)

BASE = "https://public.api.bsky.app/xrpc"


class BlueskyConnector(Connector):
    id = "bluesky"
    category = "social"
    default_cadence = "*/30 * * * *"
    rate_limit_per_min = 60
    parser_version = 1

    async def fetch(self, ctx):
        settings = global_settings()
        queries = settings.get("bluesky.queries") or [
            "premier league injury", "FPL team news", "predicted lineup premier league"
        ]
        for q in queries:
            try:
                r = await fetch_url(
                    f"{BASE}/app.bsky.feed.searchPosts",
                    self.id,
                    params={"q": q, "limit": 40, "sort": "latest"},
                    per_min=self.rate_limit_per_min,
                )
            except Exception:  # noqa: BLE001
                log.info("bluesky search failed for %r", q)
                continue
            for post in r.json().get("posts", []):
                yield _to_doc(post)

        for handle in settings.get("bluesky.handles", []):
            try:
                r = await fetch_url(
                    f"{BASE}/app.bsky.feed.getAuthorFeed",
                    self.id,
                    params={"actor": handle, "limit": 30},
                    per_min=self.rate_limit_per_min,
                )
            except Exception:  # noqa: BLE001 - handles get renamed and deleted
                continue
            for item in r.json().get("feed", []):
                if item.get("post"):
                    yield _to_doc(item["post"])

    def parse(self, doc: RawDoc) -> ParsedBatch:
        return _parse_social(doc, "bluesky", "api", source_id="bluesky")


def _to_doc(post: dict) -> RawDoc:
    record = post.get("record", {})
    text = record.get("text", "")
    author = post.get("author", {})
    return RawDoc(
        "bluesky_post",
        {
            "external_id": post.get("uri", ""),
            "handle": author.get("handle"),
            "display": author.get("displayName"),
            "body": text,
            "posted_at": record.get("createdAt"),
            "likes": post.get("likeCount"),
            "reposts": post.get("repostCount"),
            "replies": post.get("replyCount"),
        },
        external_id=post.get("uri"),
        url=f"https://bsky.app/profile/{author.get('handle')}",
        published_at=record.get("createdAt"),
        text_for_simhash=text,
    )


def _parse_social(doc: RawDoc, platform: str, method: str, source_id: str | None = None) -> ParsedBatch:
    p = doc.payload
    b = ParsedBatch()

    def _post(conn) -> int:
        row = query_one(
            "SELECT id FROM raw_documents WHERE source_id=? AND external_id=? "
            "ORDER BY id DESC LIMIT 1",
            (source_id or platform, p["external_id"]),
        )
        if not row:
            return 0
        conn.execute(
            "INSERT OR REPLACE INTO social_posts(raw_doc_id,platform,external_id,author_handle,"
            "author_display,body_text,posted_at,likes,reposts,replies,retrieval_method) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["id"], platform, p["external_id"], p.get("handle"), p.get("display"),
                p["body"], p.get("posted_at"), p.get("likes"), p.get("reposts"), p.get("replies"),
                p.get("retrieval_method", method),
            ),
        )
        for ordinal, chunk in enumerate(chunk_text(p["body"])):
            conn.execute(
                "INSERT OR IGNORE INTO doc_chunks(raw_doc_id,ordinal,text,token_count) "
                "VALUES(?,?,?,?)",
                (row["id"], ordinal, chunk, len(chunk.split())),
            )
        return 1

    b.defer(_post)
    return b
