"""One place that knows every connector. Imported by the CLI, scheduler and API."""

from __future__ import annotations

from ..config import get_settings
from ..db.settings_store import source_enabled
from .base import Connector
from .bluesky import BlueskyConnector
from .euro_fixtures import EuroFixturesConnector
from .fpl_official import FplOfficialConnector
from .lineups import ApiFootballConnector, SportmonksConnector
from .livefpl import LiveFplConnector
from .misc_stats import (
    FbrefConnector,
    FootballDataOrgConnector,
    RefereeConnector,
    SofascoreConnector,
    TransfermarktConnector,
    WhoScoredConnector,
)
from .odds_api import OddsApiConnector
from .physioroom import PhysioRoomConnector
from .premier_injuries import PremierInjuriesConnector
from .reddit import RedditConnector
from .rss_news import RssNewsConnector
from .setpieces import SetPiecesConnector
from .twitter_scrape import TwitterScrapeConnector
from .understat import UnderstatConnector
from .vaastav_history import VaastavConnector
from .weather import WeatherConnector
from .youtube import YouTubeConnector

_CLASSES: list[type[Connector]] = [
    FplOfficialConnector,
    VaastavConnector,
    LiveFplConnector,
    UnderstatConnector,
    FbrefConnector,
    SofascoreConnector,
    WhoScoredConnector,
    TransfermarktConnector,
    FootballDataOrgConnector,
    ApiFootballConnector,
    SportmonksConnector,
    OddsApiConnector,
    PremierInjuriesConnector,
    PhysioRoomConnector,
    RssNewsConnector,
    SetPiecesConnector,
    EuroFixturesConnector,
    WeatherConnector,
    RefereeConnector,
    YouTubeConnector,
    RedditConnector,
    BlueskyConnector,
    TwitterScrapeConnector,
]

CONNECTORS: dict[str, Connector] = {c.id: c() for c in _CLASSES}


def get(source_id: str) -> Connector:
    if source_id not in CONNECTORS:
        raise KeyError(f"unknown connector {source_id!r}; known: {', '.join(sorted(CONNECTORS))}")
    return CONNECTORS[source_id]


def available(category: str | None = None) -> list[Connector]:
    """Enabled in settings, keys present, and not blocked by the scrape kill switch."""
    s = get_settings()
    return [
        c
        for c in CONNECTORS.values()
        if (category is None or c.category == category)
        and source_enabled(c.id)
        and c.is_available(s)
    ]


def status() -> list[dict]:
    """Powers GET /api/sources and the Sources screen."""
    from ..db.engine import query_one

    s = get_settings()
    out = []
    for c in CONNECTORS.values():
        last = query_one(
            "SELECT started_at, finished_at, status, docs_new, rows_upserted, error_text "
            "FROM ingest_runs WHERE source_id=? ORDER BY started_at DESC LIMIT 1",
            (c.id,),
        )
        out.append(
            {
                "id": c.id,
                "category": c.category,
                "enabled": source_enabled(c.id),
                "available": c.is_available(s),
                "unavailable_reason": c.unavailable_reason(s),
                "requires_keys": [k.upper() for k in c.requires_keys],
                "cadence": c.default_cadence,
                "last_run": dict(last) if last else None,
            }
        )
    return sorted(out, key=lambda r: (r["category"], r["id"]))
