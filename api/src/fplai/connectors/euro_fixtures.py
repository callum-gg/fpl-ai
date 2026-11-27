"""European fixtures (UCL, UEL, UECL). Feeds the congestion and rotation-risk features.

Source: ESPN's public JSON, no key. football-data.org's free tier only carries the
Champions League, so Thursday Europa/Conference games were invisible: GW12 rated
Sunderland and Bournemouth six days rested for a Sunday kickoff three days after
Europa League games.

Non-PL rows land in `fixtures` with `competition` set, which is what the congestion
features read. One doc holds a whole competition-season for every PL club in it, and
parsing it *replaces* that competition-season's rows. There is no external key to upsert
on (fpl_fixture_id is NULL), so INSERT OR IGNORE re-added every match on each new doc
(120 rows for 40 UCL matches by GW12) and never updated `finished`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from ..db.engine import query_one
from ..resolve.entities import resolve_team
from .base import Connector, ParsedBatch, RawDoc, fetch_url

log = logging.getLogger(__name__)

CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
COMPETITIONS = {"uefa.champions": "UCL", "uefa.europa": "UEL", "uefa.europa.conf": "UECL"}


class EuroFixturesConnector(Connector):
    id = "euro_fixtures"
    category = "meta"
    requires_keys: list[str] = []
    # Daily: a Thursday result has to land before a Friday or Saturday deadline.
    default_cadence = "45 4 * * *"
    rate_limit_per_min = 30
    # 4: ESPN, all three competitions; a parse replaces its competition-season.
    parser_version = 4

    async def fetch(self, ctx):
        for season in ctx.params.get("seasons", [ctx.season_id]):
            year = season.split("-")[0]
            pl = await self._team_ids("eng.1", year)
            if not pl:
                # An empty PL list would turn every doc below into "delete all rows".
                log.warning("euro fixtures %s: ESPN returned no PL clubs, skipping", season)
                continue
            for league, comp in COMPETITIONS.items():
                try:
                    events = await self._events(league, year, pl)
                except Exception as e:  # noqa: BLE001 - a partial doc would delete good rows
                    log.warning("euro fixtures %s %s unavailable: %s", comp, season, e)
                    continue
                yield RawDoc("euro_matches", {"comp": comp, "events": events},
                             external_id=f"{comp}:{season}")

    async def _team_ids(self, league: str, year: str) -> set[int]:
        r = await fetch_url(f"{CORE}/{league}/seasons/{year}/teams", self.id,
                            params={"limit": 200}, per_min=self.rate_limit_per_min)
        return {int(m.group(1)) for i in r.json().get("items", [])
                if (m := re.search(r"/teams/(\d+)", i.get("$ref", "")))}

    async def _events(self, league: str, year: str, pl: set[int]) -> list[dict]:
        """Every match of every PL club in the competition, once each: Arsenal v City
        is in both clubs' schedules. `fixture=true` is the upcoming half, and it ignores
        `season` (2023 returns this season's games), hence the July-June window."""
        lo, hi = f"{year}-07-01", f"{int(year) + 1}-07-01"
        events: dict[str, dict] = {}
        for team in sorted(pl & await self._team_ids(league, year)):
            for extra in ({}, {"fixture": "true"}):
                r = await fetch_url(f"{SITE}/{league}/teams/{team}/schedule", self.id,
                                    params={"season": year, **extra},
                                    per_min=self.rate_limit_per_min)
                for e in r.json().get("events", []):
                    if lo <= e["date"] < hi:
                        events[e["id"]] = _slim(e)
        return sorted(events.values(), key=lambda e: (e["date"], e["id"]))

    def parse(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        if "comp" not in doc.payload:
            return b  # a football-data.org doc (parser <= 3), superseded by ESPN
        comp, events = doc.payload["comp"], doc.payload["events"]
        season = doc.external_id.split(":", 1)[1]

        def _rows(conn) -> int:
            old = "SELECT id FROM fixtures WHERE season_id=? AND competition=?"
            # The weather job used to forecast European rows too; those (and any referee
            # row) would block the delete on their foreign key.
            for table in ("weather_observations", "referee_appointments"):
                conn.execute(f"DELETE FROM {table} WHERE fixture_id IN ({old})", (season, comp))
            conn.execute(f"DELETE FROM fixtures WHERE id IN ({old})", (season, comp))
            n = 0
            for e in events:
                h, a = _team_id(season, e["home"]), _team_id(season, e["away"])
                if h is None and a is None:
                    continue  # neither side is a PL club, so it cannot drive rotation
                # away_team_id is NOT NULL and foreign clubs aren't in `teams`, so a PL club
                # against one is stored against itself. Every reader matches either side,
                # and scores stay NULL: team-form features read PL rows only.
                conn.execute(
                    "INSERT INTO fixtures(season_id,fpl_fixture_id,gameweek,kickoff_utc,"
                    "home_team_id,away_team_id,finished,competition) VALUES(?,NULL,NULL,?,?,?,?,?)",
                    (season, _kickoff(e["date"]), h or a, a or h, int(e["completed"]), comp),
                )
                n += 1
            return n

        b.defer(_rows)
        return b


def _slim(e: dict) -> dict:
    """Just what parse needs, so the stored doc only changes when a match does."""
    c = e["competitions"][0]
    sides = {x["homeAway"]: x["team"]["displayName"] for x in c["competitors"]}
    status = (c.get("status") or e.get("status") or {}).get("type", {})
    return {"id": e["id"], "date": e["date"], "home": sides.get("home"),
            "away": sides.get("away"), "completed": bool(status.get("completed"))}


def _kickoff(espn_date: str) -> str:
    """ESPN's "2026-11-26T20:00Z" -> "2026-11-26T20:00:00+00:00", the PL fixtures' form.
    Kickoffs compare as strings, and "20:00+" sorts before "20:00:00+"."""
    return datetime.fromisoformat(espn_date.replace("Z", "+00:00")).isoformat()


def _team_id(season: str, name: str | None) -> int | None:
    key = resolve_team(name)
    if not key:
        return None
    row = query_one(
        "SELECT id FROM teams WHERE season_id=? AND (name=? OR short_name=?)", (season, key, key)
    )
    return row["id"] if row else None
