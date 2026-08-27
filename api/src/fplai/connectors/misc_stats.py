"""The remaining tier-2/4 sources, each thin enough that a file apiece would be noise.

- fbref        — DefCon raw components (tackles/interceptions/blocks/clearances/recoveries),
                 the single most valuable non-FPL dataset now DefCon points exist. Aggressive
                 rate limits: 15/min with a hard 3s gap, weekly, never in a loop.
- sofascore    — player ratings. Cloudflare-protected; a soft feature, never a dependency.
- whoscored    — same signal as SofaScore but harder (Incapsula). Opt-in, default off.
- transfermarkt— market value: the only usable prior for promoted-club players with no PL history.
- football_data_org — fixture/result cross-check.
- referees     — appointments drive card rates, which are direct negative points (docs/14, grade A).
"""

from __future__ import annotations

import logging
import re

from ..db.engine import query, query_one
from ..resolve.entities import by_external_id, link_external_id, resolve_name
from .base import Connector, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

DEFCON_COLS = ("tackles", "interceptions", "blocks", "clearances", "recoveries")


class FbrefConnector(Connector):
    """Via `soccerdata`, which handles FBref's HTML churn so we do not have to."""

    id = "fbref"
    category = "stats"
    default_cadence = "0 4 * * 2"
    rate_limit_per_min = 15
    min_gap_s = 3.0
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        try:
            import soccerdata as sd
        except ImportError:
            log.warning("fbref: install the `extras` group for soccerdata")
            return
        seasons = ctx.params.get("seasons") or [ctx.season_id]
        for season in seasons:
            try:
                fb = sd.FBref(leagues="ENG-Premier League", seasons=season.replace("-", ""))
                df = fb.read_player_match_stats(stat_type="defense")
            except Exception as e:  # noqa: BLE001 - FBref bans aggressively; shrug and move on
                log.warning("fbref unavailable for %s: %s", season, e)
                continue
            yield RawDoc(
                "fbref_defense",
                {"season": season, "rows": df.reset_index().to_dict("records")},
                external_id=f"defense:{season}",
            )

    def parse(self, doc: RawDoc) -> ParsedBatch:
        season = doc.payload["season"]
        rows = doc.payload["rows"]
        b = ParsedBatch()

        def _stats(conn) -> int:
            n = 0
            for r in rows:
                res = resolve_name(str(r.get("player", "")), str(r.get("team", "")), season)
                if not res.player_id:
                    continue
                fx = _fixture_by_date(season, str(r.get("date", ""))[:10])
                if fx is None:
                    continue
                conn.execute(
                    "UPDATE player_fixture_stats SET tackles=?, interceptions=?, blocks=?, "
                    "clearances=?, recoveries=?, progressive_carries=?, sca=?, gca=?, "
                    "touches_in_box=?, updated_at=? WHERE player_id=? AND fixture_id=?",
                    (
                        _i(r.get("Tkl")), _i(r.get("Int")), _i(r.get("Blocks")),
                        _i(r.get("Clr")), _i(r.get("Recov")), _i(r.get("PrgC")),
                        _i(r.get("SCA")), _i(r.get("GCA")), _i(r.get("Touches_Att Pen")),
                        utcnow(), res.player_id, fx,
                    ),
                )
                n += 1
            return n

        b.defer(_stats)
        return b


class SofascoreConnector(Connector):
    id = "sofascore"
    category = "stats"
    default_cadence = "0 5 * * 2"
    rate_limit_per_min = 30
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        headers = {"Accept": "application/json", "Referer": "https://www.sofascore.com/"}
        for fx in query(
            "SELECT id, kickoff_utc FROM fixtures WHERE season_id=? AND finished=1 "
            "AND kickoff_utc > datetime('now','-10 day')",
            (ctx.season_id,),
        ):
            ext = _sofascore_event_id(fx["id"])
            if not ext:
                continue
            try:
                r = await fetch_url(
                    f"https://api.sofascore.com/api/v1/event/{ext}/lineups", self.id,
                    headers=headers, per_min=self.rate_limit_per_min,
                )
            except Exception:  # noqa: BLE001 - Cloudflare; ratings are a soft feature
                continue
            yield RawDoc("sofascore_ratings", {"fixture_id": fx["id"], "data": r.json()},
                         external_id=str(fx["id"]))

    def parse(self, doc: RawDoc) -> ParsedBatch:
        fixture_id = doc.payload["fixture_id"]
        data = doc.payload["data"]
        b = ParsedBatch()

        def _ratings(conn) -> int:
            n = 0
            for side in ("home", "away"):
                for p in (data.get(side) or {}).get("players", []):
                    ext = str((p.get("player") or {}).get("id", ""))
                    rating = (p.get("statistics") or {}).get("rating")
                    if not ext or rating is None:
                        continue
                    pid = by_external_id("sofascore", ext)
                    if pid is None:
                        res = resolve_name((p.get("player") or {}).get("name", ""))
                        pid = res.player_id
                        if pid:
                            link_external_id(conn, pid, "sofascore", ext, res.method, res.confidence)
                    if pid is None:
                        continue
                    conn.execute(
                        "UPDATE player_fixture_stats SET sofascore_rating=? "
                        "WHERE player_id=? AND fixture_id=?",
                        (float(rating), pid, fixture_id),
                    )
                    n += 1
            return n

        b.defer(_ratings)
        return b


class TransfermarktConnector(Connector):
    """Market value + contract data. The usable prior for Coventry/Ipswich/Hull players."""

    id = "transfermarkt"
    category = "stats"
    default_cadence = "0 6 * * 3"
    rate_limit_per_min = 10
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        url = ("https://www.transfermarkt.co.uk/premier-league/marktwerteverein/"
               "wettbewerb/GB1")
        r = await fetch_url(url, self.id, per_min=self.rate_limit_per_min)
        yield RawDoc("market_values", r.text, external_id=ctx.season_id, url=url)

    def parse(self, doc: RawDoc) -> ParsedBatch:
        from selectolax.parser import HTMLParser

        tree = HTMLParser(doc.payload)
        season = doc.external_id
        entries = []
        for row in tree.css("table.items tbody tr"):
            cells = [re.sub(r"\s+", " ", c.text(strip=True)) for c in row.css("td")]
            if len(cells) < 4:
                continue
            name = cells[1] if len(cells) > 1 else None
            value = next((c for c in cells if "€" in c or "£" in c), None)
            if name and value:
                entries.append({"name": name, "value": _money(value)})

        b = ParsedBatch()

        def _values(conn) -> int:
            n = 0
            for e in entries:
                res = resolve_name(e["name"], None, season)
                if not res.player_id or e["value"] is None:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO feature_values(season_id,gameweek,player_id,"
                    "fixture_key,name,value,computed_at,feature_version) "
                    "VALUES(?,0,?,0,'transfermarkt_value_eur',?,?,1)",
                    (season, res.player_id, e["value"], utcnow()),
                )
                n += 1
            return n

        b.defer(_values)
        return b


class FootballDataOrgConnector(Connector):
    id = "football_data_org"
    category = "stats"
    requires_keys = ["football_data_org_key"]
    default_cadence = "0 6 * * *"
    rate_limit_per_min = 10
    parser_version = 1

    async def fetch(self, ctx):
        r = await fetch_url(
            "https://api.football-data.org/v4/competitions/PL/matches", self.id,
            headers={"X-Auth-Token": ctx.settings.football_data_org_key},
            per_min=self.rate_limit_per_min,
        )
        yield RawDoc("pl_matches", r.json(), external_id=ctx.season_id)


class WhoScoredConnector(Connector):
    """Opt-in, default off. SofaScore covers the same feature; if this breaks, shrug."""

    id = "whoscored"
    category = "stats"
    default_cadence = "0 6 * * 2"
    rate_limit_per_min = 6
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        try:
            import soccerdata as sd
        except ImportError:
            log.info("whoscored: soccerdata not installed, skipping")
            return
        try:
            ws = sd.WhoScored(leagues="ENG-Premier League",
                              seasons=ctx.season_id.replace("-", ""))
            df = ws.read_events()
        except Exception as e:  # noqa: BLE001 - Incapsula; opt-in for a reason
            log.warning("whoscored unavailable: %s", e)
            return
        yield RawDoc("whoscored_events", {"rows": df.head(5000).to_dict("records")},
                     external_id=ctx.season_id)


class RefereeConnector(Connector):
    """Referee appointments. Card rates vary meaningfully by official (docs/14, grade A)."""

    id = "referees"
    category = "meta"
    default_cadence = "0 9 * * 4"
    rate_limit_per_min = 5
    scrapey = True
    parser_version = 1

    async def fetch(self, ctx):
        url = "https://www.premierleague.com/referees"
        try:
            r = await fetch_url(url, self.id, per_min=self.rate_limit_per_min)
        except Exception:  # noqa: BLE001
            log.info("referee appointments unavailable")
            return
        yield RawDoc("referee_page", r.text, external_id=ctx.season_id, url=url)


# --- helpers ---------------------------------------------------------------------


def _fixture_by_date(season: str, date: str) -> int | None:
    if not date:
        return None
    row = query_one(
        "SELECT id FROM fixtures WHERE season_id=? AND substr(kickoff_utc,1,10)=? LIMIT 1",
        (season, date),
    )
    return row["id"] if row else None


def _sofascore_event_id(fixture_id: int) -> str | None:
    row = query_one(
        "SELECT external_id FROM team_external_ids WHERE system='sofascore_event' AND team_key=?",
        (str(fixture_id),),
    )
    return row["external_id"] if row else None


def _money(s: str) -> float | None:
    m = re.search(r"([\d.,]+)\s*([mk])?", s.replace("€", "").replace("£", ""), re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    mult = {"m": 1e6, "k": 1e3}.get((m.group(2) or "").lower(), 1)
    return val * mult


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
