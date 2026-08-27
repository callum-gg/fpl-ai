"""Open-Meteo forecast at the stadium, at kickoff hour. Free, no key. docs/02 tier 4.

Deliberately low-weight: high wind suppresses xG conversion slightly, and that is all.
"""

from __future__ import annotations

from datetime import datetime

from ..db.engine import query
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow

BASE = "https://api.open-meteo.com/v1/forecast"


class WeatherConnector(Connector):
    id = "weather"
    category = "meta"
    default_cadence = "0 7 * * *"
    rate_limit_per_min = 60
    parser_version = 1

    async def fetch(self, ctx: IngestContext):
        fixtures = query(
            "SELECT f.id, f.kickoff_utc, t.stadium_lat lat, t.stadium_lon lon "
            "FROM fixtures f JOIN teams t ON t.id = f.home_team_id "
            "WHERE f.season_id=? AND f.finished=0 AND f.kickoff_utc IS NOT NULL "
            "AND t.stadium_lat IS NOT NULL AND f.kickoff_utc < datetime('now','+14 day')",
            (ctx.season_id,),
        )
        for f in fixtures:
            r = await fetch_url(
                BASE,
                self.id,
                params={
                    "latitude": f["lat"],
                    "longitude": f["lon"],
                    "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
                    "start_date": f["kickoff_utc"][:10],
                    "end_date": f["kickoff_utc"][:10],
                },
                per_min=self.rate_limit_per_min,
            )
            yield RawDoc(
                "weather",
                {"fixture_id": f["id"], "kickoff": f["kickoff_utc"], "data": r.json()},
                external_id=str(f["id"]),
            )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        p = doc.payload
        hourly = p["data"].get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return ParsedBatch()
        target = datetime.fromisoformat(p["kickoff"]).strftime("%Y-%m-%dT%H:00")
        idx = times.index(target) if target in times else len(times) // 2
        b = ParsedBatch()
        b.add(
            "weather_observations",
            [
                {
                    "fixture_id": p["fixture_id"],
                    "temp_c": _at(hourly, "temperature_2m", idx),
                    "wind_kph": (_at(hourly, "wind_speed_10m", idx) or 0) * 3.6,
                    "precip_mm": _at(hourly, "precipitation", idx),
                    "humidity": _at(hourly, "relative_humidity_2m", idx),
                    "is_forecast": 1,
                    "observed_at": utcnow(),
                }
            ],
            ["fixture_id"],
        )
        return b


def _at(hourly: dict, key: str, idx: int):
    vals = hourly.get(key) or []
    return vals[idx] if idx < len(vals) else None
