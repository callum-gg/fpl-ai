"""YouTube Data API v3: tracked channels + discovery + transcripts. docs/02 tier 5.

10,000 quota units/day is plenty. Transcripts come from youtube-transcript-api first
(manual captions preferred over auto), then a paid fallback if a key is configured.
`transcript_source` is an enum column so Whisper slots in later without a migration.
"""

from __future__ import annotations

import logging
import re
from datetime import timezone

from ..db.engine import query, query_one
from ..db.settings_store import global_settings
from .base import Connector, ParsedBatch, RawDoc, fetch_url
from .fpl_official import current_gameweek
from .rss_news import chunk_text

log = logging.getLogger(__name__)

BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeConnector(Connector):
    id = "youtube"
    category = "video"
    requires_keys = ["youtube_api_key"]
    default_cadence = "40 */3 * * *"
    rate_limit_per_min = 60
    parser_version = 1

    async def fetch(self, ctx):
        key = ctx.settings.youtube_api_key
        mode = ctx.params.get("mode", "tracked")
        gw = current_gameweek(ctx.season_id)
        settings = global_settings()

        if mode == "tracked":
            for ch in settings.get("youtube.channels", []):
                if not ch.get("tracked"):
                    continue
                channel_id = ch.get("channel_id") or await self._resolve_channel(key, ch["title"])
                if not channel_id:
                    continue
                async for doc in self._channel_uploads(key, channel_id, ch.get("title")):
                    yield doc

        elif mode == "discovery":
            min_views = settings.get("youtube.discovery_min_views", 2000)
            for template in settings.get("youtube.discovery_queries", []):
                q = template.format(gw=gw)
                r = await fetch_url(
                    f"{BASE}/search",
                    self.id,
                    params={"part": "snippet", "q": q, "type": "video", "maxResults": 15,
                            "order": "relevance", "publishedAfter": _days_ago_iso(10), "key": key},
                    per_min=self.rate_limit_per_min,
                )
                ids = [i["id"]["videoId"] for i in r.json().get("items", [])]
                async for doc in self._videos(key, ids, "search_discovery", min_views):
                    yield doc

    async def _resolve_channel(self, key: str, title: str) -> str | None:
        r = await fetch_url(
            f"{BASE}/search",
            self.id,
            params={"part": "snippet", "q": title, "type": "channel", "maxResults": 1, "key": key},
            per_min=self.rate_limit_per_min,
        )
        items = r.json().get("items", [])
        return items[0]["snippet"]["channelId"] if items else None

    async def _channel_uploads(self, key: str, channel_id: str, title: str | None):
        r = await fetch_url(
            f"{BASE}/channels",
            self.id,
            params={"part": "contentDetails,statistics", "id": channel_id, "key": key},
            per_min=self.rate_limit_per_min,
        )
        items = r.json().get("items", [])
        if not items:
            return
        uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        yield RawDoc(
            "channel",
            {"channel_id": channel_id, "title": title,
             "subscriber_count": _i(items[0].get("statistics", {}).get("subscriberCount"))},
            external_id=channel_id,
        )
        r = await fetch_url(
            f"{BASE}/playlistItems",
            self.id,
            params={"part": "contentDetails", "playlistId": uploads, "maxResults": 15, "key": key},
            per_min=self.rate_limit_per_min,
        )
        ids = [i["contentDetails"]["videoId"] for i in r.json().get("items", [])]
        async for doc in self._videos(key, ids, "tracked_channel", 0):
            yield doc

    async def _videos(self, key: str, video_ids: list[str], discovered_via: str, min_views: int):
        fresh = [v for v in video_ids if not _have_video(v)]
        for batch_start in range(0, len(fresh), 50):
            batch = fresh[batch_start:batch_start + 50]
            if not batch:
                continue
            r = await fetch_url(
                f"{BASE}/videos",
                self.id,
                params={"part": "snippet,contentDetails,statistics", "id": ",".join(batch),
                        "key": key},
                per_min=self.rate_limit_per_min,
            )
            for item in r.json().get("items", []):
                stats = item.get("statistics", {})
                views = _i(stats.get("viewCount")) or 0
                if views < min_views:
                    continue
                sn = item["snippet"]
                yield RawDoc(
                    "video",
                    {
                        "youtube_id": item["id"],
                        "channel_id": sn["channelId"],
                        "channel_title": sn.get("channelTitle"),
                        "title": sn.get("title"),
                        "description": sn.get("description", "")[:4000],
                        "published_at": sn.get("publishedAt"),
                        "duration_s": _duration(item.get("contentDetails", {}).get("duration")),
                        "view_count": views,
                        "like_count": _i(stats.get("likeCount")),
                        "discovered_via": discovered_via,
                    },
                    external_id=item["id"],
                    url=f"https://youtube.com/watch?v={item['id']}",
                    published_at=sn.get("publishedAt"),
                )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        if doc.doc_type == "channel":
            p = doc.payload
            b.add(
                "channels",
                [{"channel_id": p["channel_id"], "title": p.get("title") or p["channel_id"],
                  "tracked": 1, "subscriber_count": p.get("subscriber_count"),
                  "added_by": "seed"}],
                ["channel_id"],
            )
            return b
        if doc.doc_type != "video":
            return b
        p = doc.payload

        def _video(conn) -> int:
            row = query_one(
                "SELECT id FROM raw_documents WHERE source_id='youtube' AND doc_type='video' "
                "AND external_id=? ORDER BY id DESC LIMIT 1",
                (p["youtube_id"],),
            )
            if not row:
                return 0
            conn.execute(
                "INSERT OR IGNORE INTO channels(channel_id,title,tracked,added_by) "
                "VALUES(?,?,?,?)",
                (p["channel_id"], p.get("channel_title") or p["channel_id"],
                 int(p["discovered_via"] == "tracked_channel"),
                 "discovery" if p["discovered_via"] == "search_discovery" else "seed"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO videos(raw_doc_id,youtube_id,channel_id,channel_title,"
                "title,description,published_at,duration_s,view_count,like_count,gameweek_hint,"
                "discovered_via) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["id"], p["youtube_id"], p["channel_id"], p.get("channel_title"),
                    p.get("title"), p.get("description"), p.get("published_at"),
                    p.get("duration_s"), p.get("view_count"), p.get("like_count"),
                    _gw_hint(p.get("title", "")), p["discovered_via"],
                ),
            )
            return 1

        b.defer(_video)
        return b


# --- transcripts ---------------------------------------------------------------


def pending_transcripts(limit: int = 20) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT id, youtube_id, raw_doc_id FROM videos WHERE transcript_text IS NULL "
            "ORDER BY published_at DESC LIMIT ?",
            (limit,),
        )
    ]


def fetch_transcript(youtube_id: str) -> tuple[str, list[dict], str] | None:
    """(text, segments, source). Manual captions preferred; en-GB before en."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        listing = YouTubeTranscriptApi.list_transcripts(youtube_id)
        for finder, label in (
            (lambda: listing.find_manually_created_transcript(["en-GB", "en"]), "youtube_manual"),
            (lambda: listing.find_generated_transcript(["en-GB", "en"]), "youtube_auto"),
        ):
            try:
                segs = finder().fetch()
                return " ".join(s["text"] for s in segs), segs, label
            except Exception:  # noqa: BLE001 - try the next transcript kind
                continue
    except Exception:  # noqa: BLE001 - library missing or the video is blocked
        log.info("no free transcript for %s", youtube_id)
    return None


def store_transcript(video_id: int, raw_doc_id: int, text: str, segs: list[dict], source: str):
    from ..db.engine import jdump, writer

    with writer() as conn:
        conn.execute(
            "UPDATE videos SET transcript_text=?, transcript_json=?, transcript_source=? WHERE id=?",
            (text, jdump(segs), source, video_id),
        )
        # Chunk with timestamps so the evidence panel can deep-link to the moment.
        cursor = 0
        for ordinal, chunk in enumerate(chunk_text(text)):
            start_s = segs[min(cursor, len(segs) - 1)].get("start") if segs else None
            cursor += max(1, len(chunk.split()) // 8)
            end_s = segs[min(cursor, len(segs) - 1)].get("start") if segs else None
            conn.execute(
                "INSERT OR IGNORE INTO doc_chunks(raw_doc_id,ordinal,text,start_s,end_s,token_count)"
                " VALUES(?,?,?,?,?,?)",
                (raw_doc_id, ordinal, chunk, start_s, end_s, len(chunk.split())),
            )


def _have_video(youtube_id: str) -> bool:
    return query_one("SELECT 1 FROM videos WHERE youtube_id=?", (youtube_id,)) is not None


def _gw_hint(title: str) -> int | None:
    m = re.search(r"\b(?:gw|gameweek)\s*(\d{1,2})\b", title, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _duration(iso: str | None) -> int | None:
    if not iso:
        return None
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _days_ago_iso(days: int) -> str:
    from datetime import datetime, timedelta

    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
