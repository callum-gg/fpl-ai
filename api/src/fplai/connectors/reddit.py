"""r/FantasyPL megathreads and team-news posts. docs/02 tier 5.

Uses the public JSON endpoints when no OAuth app is configured — reads work either way,
and PRAW is only reached for if credentials exist.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..db.engine import query_one
from ..db.settings_store import global_settings
from .base import Connector, ParsedBatch, RawDoc, fetch_url
from .rss_news import chunk_text

log = logging.getLogger(__name__)


class RedditConnector(Connector):
    id = "reddit"
    category = "social"
    default_cadence = "*/30 * * * *"
    rate_limit_per_min = 30
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        for sub in global_settings().get("reddit.subreddits", ["FantasyPL"]):
            for sort in ("new", "hot"):
                try:
                    r = await fetch_url(
                        f"https://www.reddit.com/r/{sub}/{sort}.json",
                        self.id,
                        params={"limit": 50},
                        per_min=self.rate_limit_per_min,
                    )
                except Exception:  # noqa: BLE001 - reddit rate-limits aggressively
                    log.info("reddit r/%s/%s unavailable", sub, sort)
                    continue
                for child in r.json().get("data", {}).get("children", []):
                    d = child.get("data", {})
                    body = (d.get("selftext") or d.get("title") or "").strip()
                    if len(body) < 60:
                        continue
                    yield RawDoc(
                        "reddit_post",
                        {
                            "external_id": d["id"],
                            "author": d.get("author"),
                            "title": d.get("title"),
                            "body": body,
                            "score": d.get("score"),
                            "num_comments": d.get("num_comments"),
                            "subreddit": sub,
                            "permalink": f"https://reddit.com{d.get('permalink', '')}",
                            "created_utc": d.get("created_utc"),
                        },
                        external_id=d["id"],
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        published_at=_ts(d.get("created_utc")),
                        text_for_simhash=body,
                    )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        p = doc.payload
        b = ParsedBatch()

        def _post(conn) -> int:
            row = query_one(
                "SELECT id FROM raw_documents WHERE source_id='reddit' AND external_id=? "
                "ORDER BY id DESC LIMIT 1",
                (p["external_id"],),
            )
            if not row:
                return 0
            conn.execute(
                "INSERT OR REPLACE INTO social_posts(raw_doc_id,platform,external_id,author_handle,"
                "body_text,posted_at,score,replies,retrieval_method) "
                "VALUES(?,'reddit',?,?,?,?,?,?,'api')",
                (
                    row["id"], p["external_id"], p.get("author"),
                    f"{p.get('title', '')}\n\n{p['body']}".strip(), _ts(p.get("created_utc")),
                    p.get("score"), p.get("num_comments"),
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


def _ts(epoch) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="seconds")
