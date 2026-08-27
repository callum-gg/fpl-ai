"""X, via a layered unauthenticated fallback chain. docs/02 tier 5 — FLAKY, ToS-adjacent.

Order: syndication endpoint -> Nitter pool -> twscrape -> (playwright, not shipped).
Every layer writes into `social_posts` with a `retrieval_method` so you can see what is
actually working. Nothing downstream may hard-depend on X: if all layers fail, the app
logs it and carries on. Honours the SCRAPE_ENABLED kill switch.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from ..db.settings_store import global_settings
from .base import Connector, ParsedBatch, RawDoc, fetch_url
from .bluesky import _parse_social

log = logging.getLogger(__name__)

SYNDICATION = "https://cdn.syndication.twimg.com/timeline/profile"


class TwitterScrapeConnector(Connector):
    id = "twitter_scrape"
    category = "social"
    default_cadence = "*/30 * * * *"
    rate_limit_per_min = 10
    scrapey = True
    parser_version = 1

    def unavailable_reason(self, settings):
        if not settings.x_enabled:
            return "X_ENABLED=false"
        return super().unavailable_reason(settings)

    async def fetch(self, ctx):
        handles = global_settings().get("x.handles", [])
        methods = ctx.settings.x_methods
        for handle in handles:
            handle = handle.lstrip("@")
            posts: list[dict] = []
            used = None
            for method in methods:
                fn = {
                    "syndication": self._syndication,
                    "nitter": self._nitter,
                    "twscrape": self._twscrape,
                }.get(method)
                if fn is None:
                    continue
                try:
                    posts = await fn(handle, ctx)
                except Exception as e:  # noqa: BLE001 - each layer is expected to break
                    log.info("x/%s via %s failed: %s", handle, method, e)
                    posts = []
                if posts:
                    used = method
                    break
            if not posts:
                log.warning("x/%s: every retrieval method failed", handle)
                continue
            for p in posts:
                p["retrieval_method"] = used
                yield RawDoc(
                    "x_post", p, external_id=p["external_id"],
                    url=f"https://x.com/{handle}/status/{p['external_id']}",
                    published_at=p.get("posted_at"), text_for_simhash=p["body"],
                )

    async def _syndication(self, handle: str, ctx) -> list[dict]:
        r = await fetch_url(
            SYNDICATION, self.id,
            params={"screen_name": handle, "suppress_response_codes": "true"},
            headers={"Referer": "https://platform.twitter.com/"},
            per_min=self.rate_limit_per_min,
        )
        try:
            data = r.json()
        except json.JSONDecodeError:
            return []
        out = []
        for entry in (data.get("timeline", {}).get("entries") or data.get("props", {}).values()):
            tweet = _dig(entry, "tweet") or (entry if isinstance(entry, dict) else None)
            if not isinstance(tweet, dict) or "id_str" not in tweet:
                continue
            out.append(_tweet_dict(tweet, handle))
        return out

    async def _nitter(self, handle: str, ctx) -> list[dict]:
        import feedparser

        for instance in ctx.settings.nitter_instances:
            try:
                r = await fetch_url(f"{instance.rstrip('/')}/{handle}/rss", self.id,
                                    per_min=self.rate_limit_per_min)
            except Exception:  # noqa: BLE001 - most public instances are dead
                continue
            feed = feedparser.parse(r.text)
            out = []
            for e in feed.entries[:30]:
                body = re.sub(r"<[^>]+>", "", e.get("description", "")).strip()
                if not body:
                    continue
                out.append(
                    {
                        "external_id": (e.get("link") or "").rstrip("#m").split("/")[-1],
                        "handle": handle,
                        "display": handle,
                        "body": body,
                        "posted_at": _rss_time(e),
                    }
                )
            if out:
                return out
        return []

    async def _twscrape(self, handle: str, ctx) -> list[dict]:
        """Burner-account route. Best coverage, highest ban risk, off unless configured."""
        from pathlib import Path

        if not Path(ctx.settings.twscrape_accounts_file).exists():
            return []
        from twscrape import API

        api = API()
        await api.pool.load_accounts(ctx.settings.twscrape_accounts_file)
        user = await api.user_by_login(handle)
        out = []
        async for tweet in api.user_tweets(user.id, limit=30):
            out.append(
                {
                    "external_id": str(tweet.id),
                    "handle": handle,
                    "display": tweet.user.displayname,
                    "body": tweet.rawContent,
                    "posted_at": tweet.date.isoformat(),
                    "likes": tweet.likeCount,
                    "reposts": tweet.retweetCount,
                    "replies": tweet.replyCount,
                }
            )
        return out

    def parse(self, doc: RawDoc) -> ParsedBatch:
        return _parse_social(doc, "x", doc.payload.get("retrieval_method", "syndication"),
                             source_id="twitter_scrape")


def _tweet_dict(tweet: dict, handle: str) -> dict:
    user = tweet.get("user", {})
    return {
        "external_id": tweet["id_str"],
        "handle": user.get("screen_name") or handle,
        "display": user.get("name"),
        "body": tweet.get("full_text") or tweet.get("text", ""),
        "posted_at": _twitter_time(tweet.get("created_at")),
        "likes": tweet.get("favorite_count"),
        "reposts": tweet.get("retweet_count"),
        "replies": tweet.get("reply_count"),
    }


def _dig(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _dig(v, key)
            if found is not None:
                return found
    return None


def _twitter_time(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").isoformat(timespec="seconds")
    except ValueError:
        return None


def _rss_time(entry) -> str | None:
    t = entry.get("published_parsed")
    return datetime(*t[:6], tzinfo=timezone.utc).isoformat(timespec="seconds") if t else None
