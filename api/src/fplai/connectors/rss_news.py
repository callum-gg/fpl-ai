"""News RSS/Atom + full-text extraction. Free, no keys. docs/02 tier 4.

~60 feeds with conditional GET. Syndicated wire copy appears on six sites at once, so
the SimHash near-dupe layer in `base.archive` collapses them into one fact — without
that, every consensus signal downstream is garbage.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from ..db.engine import query_one
from ..db.settings_store import global_settings
from .base import Connector, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

_ETAGS: dict[str, tuple[str | None, str | None]] = {}
GOOGLE_NEWS = "https://news.google.com/rss/search"


def _published(entry) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        t = getattr(entry, key, None) or entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc).isoformat(timespec="seconds")
    return None


def extract_text(html: str, url: str | None = None) -> tuple[str, str | None]:
    """trafilatura where installed, else a selectolax paragraph join. Returns (text, title)."""
    try:
        import trafilatura

        text = trafilatura.extract(html, url=url, include_comments=False, favor_precision=True)
        meta = trafilatura.extract_metadata(html)
        if text:
            return text, (meta.title if meta else None)
    except Exception:  # noqa: BLE001 - extraction is best-effort by nature
        pass
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    for tag in ("script", "style", "nav", "header", "footer", "aside"):
        for node in tree.css(tag):
            node.decompose()
    paras = [p.text(strip=True) for p in tree.css("p")]
    title = tree.css_first("title")
    return "\n\n".join(x for x in paras if len(x) > 40), (title.text(strip=True) if title else None)


class RssNewsConnector(Connector):
    id = "rss_news"
    category = "news"
    default_cadence = "*/20 * * * *"
    rate_limit_per_min = 120
    parser_version = 1

    async def fetch(self, ctx):
        import feedparser

        settings = global_settings()
        feeds = list(settings.get("rss.feeds", []))
        if settings.get("rss.google_news_watchlist_query"):
            feeds += _watchlist_queries(settings.get("watchlist", []))

        for feed_url in feeds:
            etag, modified = _ETAGS.get(feed_url, (None, None))
            headers = {}
            if etag:
                headers["If-None-Match"] = etag
            if modified:
                headers["If-Modified-Since"] = modified
            try:
                r = await fetch_url(feed_url, self.id, headers=headers,
                                    per_min=self.rate_limit_per_min)
            except Exception:  # noqa: BLE001 - one dead feed must not stop the sweep
                log.info("feed unreachable: %s", feed_url)
                continue
            if r.status_code == 304:
                continue
            _ETAGS[feed_url] = (r.headers.get("ETag"), r.headers.get("Last-Modified"))

            parsed = feedparser.parse(r.text)
            outlet = (parsed.feed.get("title") if parsed.feed else None) or feed_url
            for entry in parsed.entries[:40]:
                link = entry.get("link")
                if not link or _already_have(link):
                    continue
                try:
                    page = await fetch_url(link, self.id, per_min=self.rate_limit_per_min)
                    body, title = extract_text(page.text, link)
                except Exception:  # noqa: BLE001 - paywalls and 403s are routine
                    body, title = entry.get("summary", ""), entry.get("title")
                if len(body) < 200:
                    continue
                yield RawDoc(
                    "article",
                    {
                        "title": title or entry.get("title"),
                        "body": body,
                        "outlet": outlet,
                        "author": entry.get("author"),
                        "url": link,
                        "published_at": _published(entry),
                    },
                    external_id=link,
                    url=link,
                    published_at=_published(entry),
                    text_for_simhash=body,
                )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        p = doc.payload
        b = ParsedBatch()

        def _article(conn) -> int:
            row = query_one(
                "SELECT id, meta_json FROM raw_documents WHERE source_id='rss_news' AND url=? "
                "ORDER BY id DESC LIMIT 1",
                (doc.url,),
            )
            if not row:
                return 0
            import json

            near = json.loads(row["meta_json"]).get("near_dupe_of")
            group = near or row["id"]
            conn.execute(
                "INSERT OR REPLACE INTO articles(raw_doc_id,title,author,outlet,published_at,url,"
                "body_text,word_count,near_dupe_group) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    row["id"], p.get("title"), p.get("author"), p.get("outlet"),
                    p.get("published_at") or utcnow(), p.get("url"), p["body"],
                    len(p["body"].split()), group,
                ),
            )
            _chunk(conn, row["id"], p["body"])
            return 1

        b.defer(_article)
        return b


def _already_have(url: str) -> bool:
    return query_one(
        "SELECT 1 FROM raw_documents WHERE source_id='rss_news' AND url=? LIMIT 1", (url,)
    ) is not None


def _watchlist_queries(player_ids: list[int]) -> list[str]:
    """One Google News RSS query per watchlist player. Cheap, no key, surprisingly complete."""
    from urllib.parse import quote_plus

    out = []
    for pid in player_ids[:40]:
        row = query_one("SELECT canonical_name FROM players WHERE id=?", (pid,))
        if row:
            q = quote_plus(f'"{row["canonical_name"]}" injury')
            out.append(f"{GOOGLE_NEWS}?q={q}&hl=en-GB&gl=GB&ceid=GB:en")
    return out


CHUNK_WORDS = 550
CHUNK_OVERLAP = 60


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if not words:
        return []
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


def _chunk(conn, raw_doc_id: int, text: str) -> None:
    for ordinal, chunk in enumerate(chunk_text(text)):
        conn.execute(
            "INSERT OR IGNORE INTO doc_chunks(raw_doc_id,ordinal,text,token_count) "
            "VALUES(?,?,?,?)",
            (raw_doc_id, ordinal, chunk, len(chunk.split())),
        )
