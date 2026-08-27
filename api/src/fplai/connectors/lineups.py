"""Predicted + confirmed lineups behind one interface. docs/02 tier 2.

API-Football and Sportmonks both provide them; whichever key exists wins. Predicted
lineups days before a deadline are the single biggest pre-deadline edge available, and
confirmed lineups collapse the minutes model to the truth about an hour before kickoff.

If neither key is present the LineupProvider yields nothing and the minutes model falls
back to `predicted_lineup_prob` derived from LLM claim extraction (docs/08).
"""

from __future__ import annotations

import logging

from ..db.engine import query, query_one
from ..resolve.entities import by_external_id, link_external_id, resolve_name, resolve_team
from .base import Connector, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)


class ApiFootballConnector(Connector):
    id = "api_football"
    category = "stats"
    requires_keys = ["api_football_key"]
    default_cadence = "20 * * * *"
    rate_limit_per_min = 30
    parser_version = 1

    async def fetch(self, ctx):
        host = ctx.settings.api_football_host
        headers = {"x-apisports-key": ctx.settings.api_football_key}
        for fx in _upcoming_fixtures(ctx.season_id):
            ext = _external_fixture_id(fx["id"], "apifootball")
            if not ext:
                continue
            for endpoint, doc_type in (("fixtures/lineups", "lineup"), ("injuries", "injury")):
                try:
                    r = await fetch_url(
                        f"https://{host}/{endpoint}", self.id,
                        params={"fixture": ext}, headers=headers,
                        per_min=self.rate_limit_per_min,
                    )
                except Exception:  # noqa: BLE001 - quota exhaustion is routine on free tiers
                    log.info("api_football %s unavailable for fixture %s", endpoint, ext)
                    continue
                yield RawDoc(
                    doc_type, {"fixture_id": fx["id"], "kind": "confirmed", "data": r.json()},
                    external_id=f"{fx['id']}:{endpoint}",
                )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        if doc.doc_type != "lineup":
            return ParsedBatch()
        return _parse_lineups(doc, self.id)


class SportmonksConnector(Connector):
    id = "sportmonks"
    category = "stats"
    requires_keys = ["sportmonks_api_key"]
    default_cadence = "25 * * * *"
    rate_limit_per_min = 30
    parser_version = 1

    async def fetch(self, ctx):
        for fx in _upcoming_fixtures(ctx.season_id):
            ext = _external_fixture_id(fx["id"], "sportmonks")
            if not ext:
                continue
            try:
                r = await fetch_url(
                    f"https://api.sportmonks.com/v3/football/fixtures/{ext}", self.id,
                    params={"api_token": ctx.settings.sportmonks_api_key,
                            "include": "lineups;lineups.player"},
                    per_min=self.rate_limit_per_min,
                )
            except Exception:  # noqa: BLE001
                continue
            yield RawDoc(
                "lineup", {"fixture_id": fx["id"], "kind": "predicted", "data": r.json()},
                external_id=f"{fx['id']}:sportmonks",
            )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        return _parse_lineups(doc, self.id)


def _parse_lineups(doc: RawDoc, source_id: str) -> ParsedBatch:
    """Both providers land in the same shape: a list of {team, players[]}."""
    payload = doc.payload
    fixture_id = payload["fixture_id"]
    kind = payload.get("kind", "predicted")
    blocks = payload["data"].get("response") or payload["data"].get("data") or []
    observed = utcnow()
    b = ParsedBatch()

    def _rows(conn) -> int:
        from ..config import get_settings

        season = get_settings().current_season
        n = 0
        for block in blocks if isinstance(blocks, list) else [blocks]:
            team_name = (block.get("team") or {}).get("name")
            formation = block.get("formation")
            for group, starting in (("startXI", 1), ("substitutes", 0)):
                for item in block.get(group, []) or []:
                    p = item.get("player", item)
                    name = p.get("name") or p.get("display_name")
                    if not name:
                        continue
                    ext = str(p.get("id")) if p.get("id") else None
                    pid = by_external_id(source_id, ext) if ext else None
                    if pid is None:
                        res = resolve_name(name, team_name or resolve_team(team_name), season)
                        pid = res.player_id
                        if pid and ext:
                            link_external_id(conn, pid, source_id, ext, res.method, res.confidence)
                    if pid is None:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO lineups(fixture_id,player_id,source_id,kind,"
                        "is_starting,formation,position_slot,observed_at) VALUES(?,?,?,?,?,?,?,?)",
                        (fixture_id, pid, source_id, kind, starting, formation,
                         p.get("pos") or p.get("position"), observed),
                    )
                    n += 1
        return n

    b.defer(_rows)
    return b


def _upcoming_fixtures(season_id: str) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT id, kickoff_utc FROM fixtures WHERE season_id=? AND finished=0 "
            "AND kickoff_utc BETWEEN datetime('now') AND datetime('now','+5 day') "
            "ORDER BY kickoff_utc",
            (season_id,),
        )
    ]


def _external_fixture_id(fixture_id: int, system: str) -> str | None:
    row = query_one(
        "SELECT external_id FROM team_external_ids WHERE system=? AND team_key=?",
        (f"{system}_fixture", str(fixture_id)),
    )
    return row["external_id"] if row else None


def predicted_lineup_prob(fixture_id: int, player_id: int) -> float | None:
    """Share of the most recent predicted XIs, across providers, that include the player."""
    rows = query(
        "SELECT source_id, MAX(observed_at) obs FROM lineups WHERE fixture_id=? AND kind='predicted'"
        " GROUP BY source_id",
        (fixture_id,),
    )
    if not rows:
        return None
    hits = total = 0
    for r in rows:
        total += 1
        got = query_one(
            "SELECT is_starting FROM lineups WHERE fixture_id=? AND player_id=? AND source_id=? "
            "AND observed_at=?",
            (fixture_id, player_id, r["source_id"], r["obs"]),
        )
        hits += int(bool(got and got["is_starting"]))
    return hits / total if total else None


def confirmed_start(fixture_id: int, player_id: int) -> bool | None:
    """Truth once the sheet lands. None means no confirmed lineup exists yet."""
    any_confirmed = query_one(
        "SELECT 1 FROM lineups WHERE fixture_id=? AND kind='confirmed' LIMIT 1", (fixture_id,)
    )
    if not any_confirmed:
        return None
    row = query_one(
        "SELECT is_starting FROM lineups WHERE fixture_id=? AND player_id=? AND kind='confirmed' "
        "ORDER BY observed_at DESC LIMIT 1",
        (fixture_id, player_id),
    )
    return bool(row["is_starting"]) if row else False
