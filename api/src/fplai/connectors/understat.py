"""Understat — xG/xA/npxG/shots per player per match since 2014. Free scrape. docs/02 tier 2.

The HTML pages stopped embedding their data (one JSON.parse block left, no players); the
site now loads it from `getLeagueData/EPL/{year}` and `getPlayerData/{id}`, which want an
XHR header. A player's JSON carries his whole career, so one request per player backfills
every season at once.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import date, timedelta

from ..resolve.entities import by_external_id, link_external_id, resolve_name
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

BASE = "https://understat.com"
# ponytail: Opta's "big chance" is a judgement call understat doesn't publish; xG >= 0.3
# is the usual stand-in. Swap in a real source if one ever lands.
BIG_CHANCE_XG = 0.3
ON_TARGET = {"Goal", "SavedShot"}


def _season_start_year(season_id: str) -> str:
    return season_id.split("-")[0]


def _season_id(year: str) -> str:
    y = int(year)
    return f"{y}-{(y + 1) % 100:02d}"


def _xhr(referer: str) -> dict:
    return {"X-Requested-With": "XMLHttpRequest", "Referer": referer}


class UnderstatConnector(Connector):
    id = "understat"
    category = "stats"
    default_cadence = "0 3 * * 2"
    rate_limit_per_min = 30
    scrapey = True
    parser_version = 3

    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        seasons = ctx.params.get("seasons") or [ctx.season_id]
        player_ids: set[str] = set()
        for season in seasons:
            year = _season_start_year(season)
            r = await fetch_url(f"{BASE}/getLeagueData/EPL/{year}", self.id,
                                per_min=self.rate_limit_per_min,
                                headers=_xhr(f"{BASE}/league/EPL/{year}"))
            players = r.json().get("players") or []
            player_ids.update(str(p["id"]) for p in players)
            # Parsed before the player docs below, so their links exist by then.
            yield RawDoc("understat_players", {"season": season, "players": players},
                         external_id=f"players:{season}")

        for pid in sorted(player_ids, key=int):
            r = await fetch_url(f"{BASE}/getPlayerData/{pid}", self.id,
                                per_min=self.rate_limit_per_min,
                                headers=_xhr(f"{BASE}/player/{pid}"))
            data = r.json()
            # Only what parse reads: the full shot map is ~700 rows of coordinates per player.
            shots = [{k: s.get(k) for k in ("match_id", "result", "situation", "xG")}
                     for s in data.get("shots") or []]
            yield RawDoc("understat_player_matches",
                         {"player_id": pid, "matches": data.get("matches") or [],
                          "shots": shots},
                         external_id=f"pm:{pid}")

    def parse(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        if doc.doc_type == "understat_players":
            season = doc.payload["season"]
            players = doc.payload["players"]

            # Keyed per season, like fpl_element ids: the same footballer can be two
            # `players` rows split across seasons (vaastav's older record vs the FPL API's),
            # and one career-wide link reached only one of them — Caicedo got no 2022-26.
            def _link(conn) -> int:
                n = 0
                for p in players:
                    if by_external_id("understat", f"{season}:{p['id']}"):
                        continue
                    # Last club only for a mid-season mover ("Chelsea,Everton").
                    team = (p.get("team_title") or "").split(",")[-1] or None
                    res = resolve_name(p["player_name"], team, season, allow_review=False)
                    if res.player_id:
                        link_external_id(conn, res.player_id, "understat",
                                         f"{season}:{p['id']}", res.method, res.confidence)
                        n += 1
                return n

            b.defer(_link)

        elif doc.doc_type == "understat_player_matches":
            us_pid = str(doc.payload["player_id"])
            matches = doc.payload["matches"]
            per_match = shot_stats(doc.payload.get("shots") or [])

            def _stats(conn) -> int:
                n = 0
                for m in matches:
                    if not m.get("date") or not m.get("season"):
                        continue
                    season = _season_id(m["season"])
                    pid = by_external_id("understat", f"{season}:{us_pid}")
                    if pid is None:
                        continue
                    s = per_match.get(str(m.get("id")), {})
                    n += update_fixture_stats(conn, pid, season, m["date"], m, s)
                return n

            b.defer(_stats)
        return b


def shot_stats(shots: list[dict]) -> dict[str, dict[str, int]]:
    """Per understat match id: shots on target, big chances (proxy), penalties taken."""
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {"shots_on_target": 0, "big_chances": 0, "penalties_taken": 0})
    for s in shots:
        row = out[str(s.get("match_id"))]
        row["shots_on_target"] += s.get("result") in ON_TARGET
        row["big_chances"] += (_f(s.get("xG")) or 0) >= BIG_CHANCE_XG
        row["penalties_taken"] += s.get("situation") == "Penalty"
    return out


def update_fixture_stats(conn, pid: int, season: str, day: str, m: dict, s: dict) -> int:
    """Write one understat match onto the player's own fixture that day.

    Matching on the date alone took `LIMIT 1` of whatever fixture kicked off first that
    day, so on a ten-game Saturday most rows landed nowhere. The player's own stats row is
    the key; a cup or foreign-league match simply finds no row and writes nothing.
    xG/xA only fill gaps — FPL's Opta numbers stay authoritative where they exist.
    """
    start = date.fromisoformat(day[:10])
    cur = conn.execute(
        "UPDATE player_fixture_stats SET xg=COALESCE(xg, ?), xa=COALESCE(xa, ?), npxg=?, "
        "shots=?, key_passes=?, shots_on_target=?, big_chances=?, penalties_taken=?, "
        "updated_at=? WHERE player_id=? AND fixture_id IN (SELECT id FROM fixtures "
        "WHERE season_id=? AND kickoff_utc >= ? AND kickoff_utc < ?)",
        (
            _f(m.get("xG")), _f(m.get("xA")), _f(m.get("npxG")), _i(m.get("shots")),
            _i(m.get("key_passes")), s.get("shots_on_target", 0), s.get("big_chances", 0),
            s.get("penalties_taken", 0), utcnow(), pid, season,
            start.isoformat(), (start + timedelta(days=1)).isoformat(),
        ),
    )
    return cur.rowcount


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
