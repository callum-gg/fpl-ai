"""Understat — xG/xA/npxG/shots per player per match since 2014. Free scrape. docs/02 tier 2.

Data is embedded in the page inside JSON.parse('...') blocks with hex escapes.
"""

from __future__ import annotations

import codecs
import json
import logging
import re
from collections.abc import AsyncIterator

from ..db.engine import query_one
from ..resolve.entities import by_external_id, link_external_id, resolve_name
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

BASE = "https://understat.com"
_JSON_BLOCK = re.compile(r"JSON\.parse\('([^']+)'\)")


def extract_json_blocks(html: str) -> list:
    """Every JSON.parse('\\x7b...') payload on an Understat page, decoded."""
    out = []
    for raw in _JSON_BLOCK.findall(html):
        try:
            out.append(json.loads(codecs.decode(raw, "unicode_escape")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return out


def _season_start_year(season_id: str) -> str:
    return season_id.split("-")[0]


class UnderstatConnector(Connector):
    id = "understat"
    category = "stats"
    default_cadence = "0 3 * * 2"
    rate_limit_per_min = 20
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        seasons = ctx.params.get("seasons") or [ctx.season_id]
        for season in seasons:
            year = _season_start_year(season)
            r = await fetch_url(f"{BASE}/league/EPL/{year}", self.id,
                                per_min=self.rate_limit_per_min)
            blocks = extract_json_blocks(r.text)
            # league page: [datesData, teamsData, playersData]
            if len(blocks) >= 3:
                yield RawDoc("understat_players", {"season": season, "players": blocks[2]},
                             external_id=f"players:{season}")
            if len(blocks) >= 1:
                yield RawDoc("understat_matches", {"season": season, "matches": blocks[0]},
                             external_id=f"matches:{season}")

            for pid in ctx.params.get("player_ids", [])[:200]:
                r = await fetch_url(f"{BASE}/player/{pid}", self.id,
                                    per_min=self.rate_limit_per_min)
                blocks = extract_json_blocks(r.text)
                if blocks:
                    yield RawDoc("understat_player_matches",
                                 {"season": season, "player_id": pid, "matches": blocks[0]},
                                 external_id=f"pm:{pid}")

    def parse(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        if doc.doc_type == "understat_players":
            season = doc.payload["season"]
            players = doc.payload["players"]

            def _link(conn) -> int:
                n = 0
                for p in players:
                    if by_external_id("understat", p["id"]):
                        continue
                    res = resolve_name(p["player_name"], p.get("team_title"), season)
                    if res.player_id:
                        link_external_id(conn, res.player_id, "understat", p["id"],
                                         res.method, res.confidence)
                        n += 1
                return n

            b.defer(_link)

        elif doc.doc_type == "understat_player_matches":
            matches = doc.payload["matches"]
            us_pid = str(doc.payload["player_id"])
            season = doc.payload["season"]

            def _stats(conn) -> int:
                pid = by_external_id("understat", us_pid)
                if pid is None:
                    return 0
                n = 0
                for m in matches:
                    fx = _match_fixture(season, m)
                    if fx is None:
                        continue
                    conn.execute(
                        "UPDATE player_fixture_stats SET xg=?, xa=?, npxg=?, shots=?, "
                        "key_passes=?, updated_at=? WHERE player_id=? AND fixture_id=?",
                        (
                            _f(m.get("xG")), _f(m.get("xA")), _f(m.get("npxG")),
                            _i(m.get("shots")), _i(m.get("key_passes")), utcnow(), pid, fx,
                        ),
                    )
                    n += 1
                return n

            b.defer(_stats)
        return b


def _match_fixture(season: str, m: dict) -> int | None:
    date = (m.get("date") or "")[:10]
    if not date:
        return None
    row = query_one(
        "SELECT id FROM fixtures WHERE season_id=? AND substr(kickoff_utc,1,10)=? LIMIT 1",
        (season, date),
    )
    return row["id"] if row else None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
