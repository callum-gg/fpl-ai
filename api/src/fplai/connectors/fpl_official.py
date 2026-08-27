"""The core FPL connector. Free, no key, undocumented but stable. docs/02 tier 1.

Every endpoint in the tier-1 table is covered. Cadence adapts to deadline proximity;
`fpl_post_lockdown_reconcile` re-pulls `event/{gw}/live` after the 09:00 next-day
lockdown, because from 2026/27 full-time data is provisional until Opta review lands.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from ..db.engine import jdump, query, query_one
from ..defaults import POSITION_BY_ELEMENT_TYPE, STADIUM_COORDS
from ..resolve.entities import by_external_id, link_external_id, upsert_player
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api"


def _iso(ts: str | None) -> str | None:
    if not ts:
        return None
    return ts.replace("Z", "+00:00")


def _status_from_bootstrap(el: dict) -> tuple[str, int | None]:
    """FPL's `status` letter -> our availability vocabulary."""
    chance = el.get("chance_of_playing_next_round")
    status = {
        "a": "available", "d": "doubt", "i": "injured",
        "s": "suspended", "u": "unknown", "n": "unknown",
    }.get(el.get("status", "a"), "unknown")
    if status == "available" and chance is not None and chance < 100:
        status = "doubt"
    return status, chance


class FplOfficialConnector(Connector):
    id = "fpl_official"
    category = "fpl"
    default_cadence = "0 * * * *"
    rate_limit_per_min = 60
    parser_version = 1

    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        mode = ctx.params.get("mode", "core")

        if mode in ("core", "bootstrap"):
            r = await fetch_url(f"{BASE}/bootstrap-static/", self.id, per_min=self.rate_limit_per_min)
            yield RawDoc("bootstrap", r.json(), external_id=ctx.season_id)

        if mode in ("core", "fixtures"):
            r = await fetch_url(f"{BASE}/fixtures/", self.id, per_min=self.rate_limit_per_min)
            yield RawDoc("fixtures", r.json(), external_id=ctx.season_id)

        if mode == "element_summaries":
            for pid in ctx.params.get("element_ids") or _all_element_ids(ctx.season_id):
                try:
                    r = await fetch_url(f"{BASE}/element-summary/{pid}/", self.id,
                                        per_min=self.rate_limit_per_min)
                    yield RawDoc("element_summary", r.json(), external_id=str(pid))
                except Exception:  # noqa: BLE001 - one player must not kill the sweep
                    log.warning("element-summary %s failed", pid)

        if mode == "live":
            gw = ctx.params["gameweek"]
            r = await fetch_url(f"{BASE}/event/{gw}/live/", self.id, per_min=self.rate_limit_per_min)
            yield RawDoc("event_live", r.json(), external_id=f"{ctx.season_id}:{gw}")

        if mode == "entry":
            for entry_id in ctx.params.get("entry_ids", []):
                for path, doc_type in (
                    (f"entry/{entry_id}/", "entry"),
                    (f"entry/{entry_id}/history/", "entry_history"),
                    (f"entry/{entry_id}/transfers/", "entry_transfers"),
                ):
                    r = await fetch_url(f"{BASE}/{path}", self.id, per_min=self.rate_limit_per_min)
                    yield RawDoc(doc_type, r.json(), external_id=str(entry_id))
                gw = ctx.params.get("gameweek")
                if gw:
                    try:
                        r = await fetch_url(f"{BASE}/entry/{entry_id}/event/{gw}/picks/", self.id,
                                            per_min=self.rate_limit_per_min)
                        yield RawDoc("entry_picks", r.json(), external_id=f"{entry_id}:{gw}")
                    except Exception:  # noqa: BLE001 - picks 404 before the GW starts
                        log.info("no picks yet for entry %s gw %s", entry_id, gw)

        if mode == "league":
            for lid in ctx.params.get("league_ids", []):
                page = 1
                while page <= ctx.params.get("max_pages", 3):
                    r = await fetch_url(
                        f"{BASE}/leagues-classic/{lid}/standings/", self.id,
                        params={"page_standings": page}, per_min=self.rate_limit_per_min,
                    )
                    data = r.json()
                    yield RawDoc("league_standings", data, external_id=f"{lid}:{page}")
                    if not data.get("standings", {}).get("has_next"):
                        break
                    page += 1

        if mode == "extras":
            for path, doc_type in (
                ("team/set-piece-notes/", "set_piece_notes"),
                ("element-types/", "element_types"),
            ):
                r = await fetch_url(f"{BASE}/{path}", self.id, per_min=self.rate_limit_per_min)
                yield RawDoc(doc_type, r.json(), external_id=ctx.season_id)

    # --- parsers ---------------------------------------------------------------

    def parse(self, doc: RawDoc) -> ParsedBatch:
        handler = {
            "bootstrap": self._parse_bootstrap,
            "fixtures": self._parse_fixtures,
            "element_summary": self._parse_element_summary,
            "event_live": self._parse_live,
            "entry": self._parse_entry,
            "entry_picks": self._parse_entry_picks,
            "league_standings": self._parse_league,
            "set_piece_notes": self._parse_set_pieces,
        }.get(doc.doc_type)
        return handler(doc) if handler else ParsedBatch()

    def _parse_bootstrap(self, doc: RawDoc) -> ParsedBatch:
        data = doc.payload
        season_id = doc.external_id or _season_from_events(data.get("events", []))
        b = ParsedBatch()
        now = utcnow()

        b.add("seasons", [{"id": season_id, "is_current": 1}], ["id"])
        b.add(
            "gameweeks",
            [
                {
                    "season_id": season_id,
                    "gameweek": e["id"],
                    "deadline_utc": _iso(e["deadline_time"]),
                    "is_current": int(bool(e.get("is_current"))),
                    "is_next": int(bool(e.get("is_next"))),
                    "finished": int(bool(e.get("finished"))),
                    # data_checked flips only after the 09:00 next-day lockdown.
                    "data_checked": int(bool(e.get("data_checked"))),
                    "average_score": e.get("average_entry_score"),
                    "highest_score": e.get("highest_score"),
                    "chip_plays_json": jdump(e.get("chip_plays", [])),
                }
                for e in data.get("events", [])
            ],
            ["season_id", "gameweek"],
        )

        def _teams_and_players(conn) -> int:
            gw_now = current_gameweek(season_id)
            # Set-piece duty is current state, not a time series, and `_set_pieces` takes
            # MIN(rank) over everything before the cutoff — so leaving old rows behind would
            # keep a demoted taker on his old rank forever. Replace the snapshot each run.
            # ponytail: one live snapshot, no history. If duty changes ever need to be
            # backtested, keep one row per gameweek instead of one per season.
            conn.execute(
                "DELETE FROM set_piece_roles WHERE season_id=? AND source_id='fpl_official'",
                (season_id,),
            )
            n = 0
            team_by_fpl: dict[int, int] = {}
            for t in data.get("teams", []):
                stadium, lat, lon = STADIUM_COORDS.get(t["name"], (None, None, None))
                conn.execute(
                    "INSERT INTO teams(season_id,fpl_team_id,name,short_name,code,"
                    "strength_overall_home,strength_overall_away,strength_attack_home,"
                    "strength_attack_away,strength_defence_home,strength_defence_away,"
                    "stadium_name,stadium_lat,stadium_lon) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(season_id,fpl_team_id) DO UPDATE SET name=excluded.name,"
                    "short_name=excluded.short_name,code=excluded.code,"
                    "strength_overall_home=excluded.strength_overall_home,"
                    "strength_overall_away=excluded.strength_overall_away,"
                    "strength_attack_home=excluded.strength_attack_home,"
                    "strength_attack_away=excluded.strength_attack_away,"
                    "strength_defence_home=excluded.strength_defence_home,"
                    "strength_defence_away=excluded.strength_defence_away,"
                    "stadium_name=excluded.stadium_name,stadium_lat=excluded.stadium_lat,"
                    "stadium_lon=excluded.stadium_lon",
                    (
                        season_id, t["id"], t["name"], t.get("short_name"), t.get("code"),
                        t.get("strength_overall_home"), t.get("strength_overall_away"),
                        t.get("strength_attack_home"), t.get("strength_attack_away"),
                        t.get("strength_defence_home"), t.get("strength_defence_away"),
                        stadium, lat, lon,
                    ),
                )
                row = query_one(
                    "SELECT id FROM teams WHERE season_id=? AND fpl_team_id=?", (season_id, t["id"])
                )
                team_by_fpl[t["id"]] = row["id"]
                n += 1

            for el in data.get("elements", []):
                # Position is re-read every season — 11 players were reclassified for 2026/27.
                pos = POSITION_BY_ELEMENT_TYPE.get(el["element_type"], "MID")
                full = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
                pid = by_external_id("fpl", str(el["code"]))
                if pid is None:
                    pid = upsert_player(
                        conn, full or el["web_name"], el.get("first_name"), el.get("second_name"),
                        el.get("web_name"), el.get("birth_date"), None,
                    )
                    link_external_id(conn, pid, "fpl", str(el["code"]), "exact")
                link_external_id(conn, pid, "fpl_element", f"{season_id}:{el['id']}", "exact")

                conn.execute(
                    "INSERT INTO player_seasons(player_id,season_id,team_id,fpl_element_id,"
                    "position,start_price) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(player_id,season_id) DO UPDATE SET team_id=excluded.team_id,"
                    "fpl_element_id=excluded.fpl_element_id,position=excluded.position",
                    (
                        pid, season_id, team_by_fpl.get(el["team"]), el["id"], pos,
                        el.get("now_cost"),
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO player_prices(player_id,season_id,observed_at,price,"
                    "selected_by_percent,transfers_in_event,transfers_out_event,net_transfers) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        pid, season_id, now, el.get("now_cost"),
                        float(el.get("selected_by_percent") or 0),
                        el.get("transfers_in_event"), el.get("transfers_out_event"),
                        (el.get("transfers_in_event") or 0) - (el.get("transfers_out_event") or 0),
                    ),
                )
                # FPL publishes overall ownership itself, so the differential and
                # template logic works from GW1 without waiting on LiveFPL — whose
                # top-10k EO does not exist until a gameweek has actually been played.
                conn.execute(
                    "INSERT OR REPLACE INTO ownership_snapshots(player_id,gameweek,season_id,"
                    "scope,owned_pct,captained_pct,effective_ownership,observed_at) "
                    "VALUES(?,?,?,'overall',?,NULL,NULL,?)",
                    (pid, gw_now, season_id, float(el.get("selected_by_percent") or 0), now),
                )
                # FPL publishes set-piece order itself: no community sheet to configure,
                # no scraping, and it is already in the payload we fetch hourly. Penalty
                # duty alone is worth roughly half a point a game to a forward.
                for field, role in (
                    ("penalties_order", "penalties"),
                    ("corners_and_indirect_freekicks_order", "corners_left"),
                    ("direct_freekicks_order", "direct_fk"),
                ):
                    order = el.get(field)
                    if order is None:
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO set_piece_roles(player_id,season_id,role,rank,"
                        "source_id,observed_at) VALUES(?,?,?,?,'fpl_official',?)",
                        (pid, season_id, role, int(order), now),
                    )
                status, chance = _status_from_bootstrap(el)
                conn.execute(
                    "INSERT OR IGNORE INTO availability(player_id,source_id,observed_at,status,"
                    "chance_pct,issue,note) VALUES(?,'fpl_official',?,?,?,NULL,?)",
                    (pid, now, status, chance, (el.get("news") or "")[:400]),
                )
                n += 1
            return n

        b.defer(_teams_and_players)
        return b

    def _parse_fixtures(self, doc: RawDoc) -> ParsedBatch:
        rows = doc.payload
        season_id = doc.external_id
        b = ParsedBatch()

        def _fixtures(conn) -> int:
            teams = {
                r["fpl_team_id"]: r["id"]
                for r in query("SELECT id, fpl_team_id FROM teams WHERE season_id=?", (season_id,))
            }
            n = 0
            for f in rows:
                h, a = teams.get(f["team_h"]), teams.get(f["team_a"])
                if h is None or a is None:
                    continue
                conn.execute(
                    "INSERT INTO fixtures(season_id,fpl_fixture_id,gameweek,kickoff_utc,"
                    "home_team_id,away_team_id,finished,home_score,away_score,fdr_home,fdr_away,"
                    "competition) VALUES(?,?,?,?,?,?,?,?,?,?,?,'PL') "
                    "ON CONFLICT(season_id,fpl_fixture_id) DO UPDATE SET "
                    "gameweek=excluded.gameweek,kickoff_utc=excluded.kickoff_utc,"
                    "finished=excluded.finished,home_score=excluded.home_score,"
                    "away_score=excluded.away_score,fdr_home=excluded.fdr_home,"
                    "fdr_away=excluded.fdr_away",
                    (
                        season_id, f["id"], f.get("event"), _iso(f.get("kickoff_time")), h, a,
                        int(bool(f.get("finished"))), f.get("team_h_score"), f.get("team_a_score"),
                        f.get("team_h_difficulty"), f.get("team_a_difficulty"),
                    ),
                )
                n += 1
            return n

        b.defer(_fixtures)
        return b

    def _parse_element_summary(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        element_id = int(doc.external_id)
        history = doc.payload.get("history", [])

        def _stats(conn) -> int:
            return _upsert_history_rows(conn, element_id, history)

        b.defer(_stats)
        return b

    def _parse_live(self, doc: RawDoc) -> ParsedBatch:
        """Post-lockdown reconcile: overwrite stats with Opta-reviewed truth."""
        b = ParsedBatch()
        season_id, gw = doc.external_id.split(":")
        elements = doc.payload.get("elements", [])

        def _live(conn) -> int:
            n = 0
            for el in elements:
                stats = el.get("stats") or {}
                pid = by_external_id("fpl_element", f"{season_id}:{el['id']}")
                if pid is None:
                    continue
                fx = query_one(
                    "SELECT f.id, f.home_team_id, f.away_team_id, ps.team_id FROM fixtures f "
                    "JOIN player_seasons ps ON ps.player_id=? AND ps.season_id=? "
                    "WHERE f.season_id=? AND f.gameweek=? "
                    "AND (f.home_team_id=ps.team_id OR f.away_team_id=ps.team_id) LIMIT 1",
                    (pid, season_id, season_id, int(gw)),
                )
                if not fx:
                    continue
                conn.execute(
                    "INSERT INTO player_fixture_stats(player_id,fixture_id,team_id,was_home,"
                    "minutes,goals_scored,assists,clean_sheets,goals_conceded,own_goals,"
                    "penalties_saved,penalties_missed,yellow_cards,red_cards,saves,bonus,bps,"
                    "defensive_contribution,total_points,starts,source_ids_json,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'[\"fpl_official\"]',?) "
                    "ON CONFLICT(player_id,fixture_id) DO UPDATE SET minutes=excluded.minutes,"
                    "goals_scored=excluded.goals_scored,assists=excluded.assists,"
                    "clean_sheets=excluded.clean_sheets,goals_conceded=excluded.goals_conceded,"
                    "own_goals=excluded.own_goals,penalties_saved=excluded.penalties_saved,"
                    "penalties_missed=excluded.penalties_missed,yellow_cards=excluded.yellow_cards,"
                    "red_cards=excluded.red_cards,saves=excluded.saves,bonus=excluded.bonus,"
                    "bps=excluded.bps,defensive_contribution=excluded.defensive_contribution,"
                    "total_points=excluded.total_points,starts=excluded.starts,"
                    "updated_at=excluded.updated_at",
                    (
                        pid, fx["id"], fx["team_id"], int(fx["home_team_id"] == fx["team_id"]),
                        stats.get("minutes"), stats.get("goals_scored"), stats.get("assists"),
                        stats.get("clean_sheets"), stats.get("goals_conceded"),
                        stats.get("own_goals"), stats.get("penalties_saved"),
                        stats.get("penalties_missed"), stats.get("yellow_cards"),
                        stats.get("red_cards"), stats.get("saves"), stats.get("bonus"),
                        stats.get("bps"),
                        stats.get("defensive_contribution"), stats.get("total_points"),
                        stats.get("starts"), utcnow(),
                    ),
                )
                n += 1
            return n

        b.defer(_live)
        return b

    def _parse_entry(self, doc: RawDoc) -> ParsedBatch:
        return ParsedBatch()  # consumed by fplsync, which needs squad context

    def _parse_entry_picks(self, doc: RawDoc) -> ParsedBatch:
        return ParsedBatch()

    def _parse_league(self, doc: RawDoc) -> ParsedBatch:
        return ParsedBatch()

    def _parse_set_pieces(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        teams = doc.payload.get("teams", [])

        def _notes(conn) -> int:
            n = 0
            for t in teams:
                for note in t.get("notes", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO raw_documents(source_id,doc_type,external_id,"
                        "content_hash,payload_inline,content_bytes,fetched_at) "
                        "VALUES('fpl_official','set_piece_note',?,?,?,?,?)",
                        (
                            f"{t['id']}:{note.get('external_text', '')[:40]}",
                            str(abs(hash(note.get("info_message", "")))),
                            jdump(note), len(jdump(note)), utcnow(),
                        ),
                    )
                    n += 1
            return n

        b.defer(_notes)
        return b


# --- shared helpers -------------------------------------------------------------


def _upsert_history_rows(conn, element_id: int, history: list[dict]) -> int:
    """`element-summary` history rows -> player_fixture_stats."""
    from ..config import get_settings

    season_id = get_settings().current_season
    pid = by_external_id("fpl_element", f"{season_id}:{element_id}")
    if pid is None:
        return 0
    n = 0
    for h in history:
        fx = query_one(
            "SELECT id FROM fixtures WHERE season_id=? AND fpl_fixture_id=?",
            (season_id, h.get("fixture")),
        )
        if not fx:
            continue
        ps = query_one(
            "SELECT team_id FROM player_seasons WHERE player_id=? AND season_id=?",
            (pid, season_id),
        )
        conn.execute(
            "INSERT INTO player_fixture_stats(player_id,fixture_id,team_id,was_home,minutes,"
            "goals_scored,assists,clean_sheets,goals_conceded,own_goals,penalties_saved,"
            "penalties_missed,yellow_cards,red_cards,saves,bonus,bps,defensive_contribution,"
            "total_points,starts,xg,xa,source_ids_json,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'[\"fpl_official\"]',?) "
            "ON CONFLICT(player_id,fixture_id) DO UPDATE SET minutes=excluded.minutes,"
            "goals_scored=excluded.goals_scored,assists=excluded.assists,"
            "clean_sheets=excluded.clean_sheets,goals_conceded=excluded.goals_conceded,"
            "yellow_cards=excluded.yellow_cards,red_cards=excluded.red_cards,saves=excluded.saves,"
            "bonus=excluded.bonus,bps=excluded.bps,"
            "defensive_contribution=excluded.defensive_contribution,"
            "total_points=excluded.total_points,starts=excluded.starts,updated_at=excluded.updated_at",
            (
                pid, fx["id"], ps["team_id"] if ps else None, int(bool(h.get("was_home"))),
                h.get("minutes"), h.get("goals_scored"), h.get("assists"), h.get("clean_sheets"),
                h.get("goals_conceded"), h.get("own_goals"), h.get("penalties_saved"),
                h.get("penalties_missed"), h.get("yellow_cards"), h.get("red_cards"),
                h.get("saves"), h.get("bonus"), h.get("bps"), h.get("defensive_contribution"),
                h.get("total_points"), h.get("starts"),
                _f(h.get("expected_goals")), _f(h.get("expected_assists")), utcnow(),
            ),
        )
        n += 1
    return n


def _f(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _all_element_ids(season_id: str) -> list[int]:
    return [
        r["fpl_element_id"]
        for r in query(
            "SELECT fpl_element_id FROM player_seasons WHERE season_id=? AND "
            "fpl_element_id IS NOT NULL",
            (season_id,),
        )
    ]


def _season_from_events(events: list[dict]) -> str:
    for e in events:
        if e.get("deadline_time"):
            y = datetime.fromisoformat(_iso(e["deadline_time"])).astimezone(timezone.utc).year
            return f"{y}-{str(y + 1)[2:]}"
    from ..config import get_settings

    return get_settings().current_season


def deadline_proximity(season_id: str) -> str:
    """far|near|imminent — jobs consult this to tighten cadence (docs/04)."""
    from ..config import get_settings

    row = query_one(
        "SELECT deadline_utc FROM gameweeks WHERE season_id=? AND deadline_utc > ? "
        "ORDER BY deadline_utc LIMIT 1",
        (season_id, utcnow()),
    )
    if not row:
        return "far"
    hours = (
        datetime.fromisoformat(row["deadline_utc"]) - datetime.now(timezone.utc)
    ).total_seconds() / 3600
    turbo = get_settings().deadline_turbo_hours
    if hours <= 2:
        return "imminent"
    return "near" if hours <= turbo else "far"


def next_deadline(season_id: str) -> dict | None:
    row = query_one(
        "SELECT gameweek, deadline_utc FROM gameweeks WHERE season_id=? AND deadline_utc > ? "
        "ORDER BY deadline_utc LIMIT 1",
        (season_id, utcnow()),
    )
    if not row:
        return None
    secs = (
        datetime.fromisoformat(row["deadline_utc"]) - datetime.now(timezone.utc)
    ).total_seconds()
    return {
        "gameweek": row["gameweek"],
        "deadline_utc": row["deadline_utc"],
        "seconds_remaining": int(secs),
    }


def current_gameweek(season_id: str) -> int:
    row = query_one(
        "SELECT gameweek FROM gameweeks WHERE season_id=? AND is_current=1", (season_id,)
    )
    if row:
        return row["gameweek"]
    nxt = next_deadline(season_id)
    return nxt["gameweek"] if nxt else 1


def next_gameweek(season_id: str) -> int:
    row = query_one("SELECT gameweek FROM gameweeks WHERE season_id=? AND is_next=1", (season_id,))
    if row:
        return row["gameweek"]
    nxt = next_deadline(season_id)
    return nxt["gameweek"] if nxt else current_gameweek(season_id)
