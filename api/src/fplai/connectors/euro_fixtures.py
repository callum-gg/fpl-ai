"""European + domestic-cup dates. Feeds the congestion and rotation-risk features.

Non-PL rows land in `fixtures` with a `competition` other than 'PL', which is exactly
what `team_matches_next_14_days` counts.
"""

from __future__ import annotations

import logging

from ..db.engine import query_one
from ..resolve.entities import resolve_team
from .base import Connector, ParsedBatch, RawDoc, fetch_url

log = logging.getLogger(__name__)

BASE = "https://api.football-data.org/v4"
COMPETITIONS = {"CL": "UCL", "EC": "UEL", "FAC": "FAC", "EFL": "EFL"}


class EuroFixturesConnector(Connector):
    id = "euro_fixtures"
    category = "meta"
    requires_keys = ["football_data_org_key"]
    default_cadence = "0 5 * * 1"
    rate_limit_per_min = 10
    parser_version = 1

    async def fetch(self, ctx):
        headers = {"X-Auth-Token": ctx.settings.football_data_org_key}
        for code in ctx.params.get("competitions", ["CL"]):
            try:
                r = await fetch_url(
                    f"{BASE}/competitions/{code}/matches",
                    self.id,
                    headers=headers,
                    per_min=self.rate_limit_per_min,
                )
                yield RawDoc(
                    "euro_matches",
                    {"code": code, "data": r.json()},
                    external_id=f"{code}:{ctx.season_id}",
                )
            except Exception:  # noqa: BLE001 - free-tier keys cover fewer competitions
                log.info("euro fixtures %s unavailable on this key", code)

    def parse(self, doc: RawDoc) -> ParsedBatch:
        comp = COMPETITIONS.get(doc.payload["code"], "UCL")
        matches = doc.payload["data"].get("matches", [])
        season = doc.external_id.split(":", 1)[1]
        b = ParsedBatch()

        def _rows(conn) -> int:
            n = 0
            for m in matches:
                h = _team_id(season, (m.get("homeTeam") or {}).get("name"))
                a = _team_id(season, (m.get("awayTeam") or {}).get("name"))
                if h is None and a is None:
                    continue  # neither side is a PL club, so it cannot drive rotation
                conn.execute(
                    "INSERT OR IGNORE INTO fixtures(season_id,fpl_fixture_id,gameweek,kickoff_utc,"
                    "home_team_id,away_team_id,finished,competition) VALUES(?,NULL,NULL,?,?,?,?,?)",
                    (season, m.get("utcDate"), h or a, a or h,
                     int(m.get("status") == "FINISHED"), comp),
                )
                n += 1
            return n

        b.defer(_rows)
        return b


def _team_id(season: str, name: str | None) -> int | None:
    key = resolve_team(name)
    if not key:
        return None
    row = query_one(
        "SELECT id FROM teams WHERE season_id=? AND (name=? OR short_name=?)", (season, key, key)
    )
    return row["id"] if row else None
