"""vaastav/Fantasy-Premier-League — the training-set backbone. docs/02 tier 1.

Takes *all* available seasons for numeric training data (it costs nothing and the
minutes/goals models want every row). Season weighting happens at training time.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import AsyncIterator

from ..db.engine import query_one
from ..resolve.entities import by_external_id, link_external_id, upsert_player
from ..resolve.normalise import norm_name
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, load_payload, utcnow

log = logging.getLogger(__name__)

RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]


def _i(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class VaastavConnector(Connector):
    id = "vaastav_history"
    category = "fpl"
    default_cadence = "0 3 * * 2"
    rate_limit_per_min = 60
    parser_version = 3

    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        seasons = ctx.params.get("seasons") or SEASONS
        for season in seasons:
            for path, doc_type in (
                (f"{season}/players_raw.csv", "vaastav_players"),
                (f"{season}/gws/merged_gw.csv", "vaastav_gws"),
                (f"{season}/teams.csv", "vaastav_teams"),
            ):
                try:
                    r = await fetch_url(f"{RAW}/{path}", self.id, per_min=self.rate_limit_per_min)
                    yield RawDoc(doc_type, r.text, external_id=season, url=f"{RAW}/{path}")
                except Exception:  # noqa: BLE001 - early seasons lack some files
                    log.info("vaastav %s not available", path)

    def parse(self, doc: RawDoc) -> ParsedBatch:
        if doc.doc_type != "vaastav_gws":
            return ParsedBatch()
        season = doc.external_id
        rows = list(csv.DictReader(io.StringIO(doc.payload)))
        b = ParsedBatch()
        b.add("seasons", [{"id": season, "is_current": 0}], ["id"])

        def _load(conn) -> int:
            n = 0
            team_ids = _ensure_teams(conn, season, rows)
            fixture_ids = _build_fixtures(conn, season, rows, team_ids)
            codes = season_codes(season)
            for r in rows:
                name = (r.get("name") or "").replace("_", " ").strip()
                if not name:
                    continue
                element = r.get("element")
                code = codes.get(str(element))
                pid = by_external_id("fpl_element", f"{season}:{element}") if element else None
                # FPL's `code` is the one id that survives seasons. Matching on the bare name
                # instead split ~870 players into 2-4 rows each ("Petr Cech 1", mis-encoded
                # accents, renames) and fused two different Wilsons into one.
                if pid is None and code:
                    pid = by_external_id("fpl", code)
                if pid is None:
                    pid = (_create(conn, name) if code else _find_or_create(conn, name))
                if element:
                    link_external_id(conn, pid, "fpl_element", f"{season}:{element}",
                                     "deterministic", 0.9)
                if code and by_external_id("fpl", code) is None:
                    link_external_id(conn, pid, "fpl", code, "exact")
                team_id = _row_team(r, team_ids)
                fixture_id = fixture_ids.get(_i(r.get("fixture")))
                if fixture_id is None:
                    continue
                if team_id is None:
                    # Old-schema rows: infer the club from the fixture we just built.
                    fx = query_one(
                        "SELECT home_team_id, away_team_id FROM fixtures WHERE id=?", (fixture_id,)
                    )
                    if fx:
                        team_id = (fx["home_team_id"] if _is_true(r.get("was_home"))
                                   else fx["away_team_id"])
                # Historic prices matter: without them every backtest values the whole
                # league at £4.0m and the budget constraint stops meaning anything.
                if r.get("value") and r.get("kickoff_time"):
                    conn.execute(
                        "INSERT OR IGNORE INTO player_prices(player_id,season_id,observed_at,"
                        "price,selected_by_percent,transfers_in_event,transfers_out_event,"
                        "net_transfers) VALUES(?,?,?,?,NULL,?,?,?)",
                        (pid, season, r["kickoff_time"], _i(r.get("value")),
                         _i(r.get("transfers_in")), _i(r.get("transfers_out")),
                         _i(r.get("transfers_balance"))),
                    )
                conn.execute(
                    "INSERT INTO player_seasons(player_id,season_id,team_id,fpl_element_id,position)"
                    " VALUES(?,?,?,?,?) ON CONFLICT(player_id,season_id) DO UPDATE SET "
                    "team_id=COALESCE(excluded.team_id,team_id)",
                    (pid, season, team_id, _i(element), (r.get("position") or "MID").upper()[:3]),
                )
                conn.execute(
                    "INSERT INTO player_fixture_stats(player_id,fixture_id,team_id,was_home,minutes,"
                    "goals_scored,assists,clean_sheets,goals_conceded,own_goals,penalties_saved,"
                    "penalties_missed,yellow_cards,red_cards,saves,bonus,bps,total_points,starts,"
                    "xg,xa,npxg,defensive_contribution,source_ids_json,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                    "'[\"vaastav_history\"]',?) "
                    "ON CONFLICT(player_id,fixture_id) DO NOTHING",
                    (
                        pid, fixture_id, team_id, int(_is_true(r.get("was_home"))),
                        _i(r.get("minutes"), 0), _i(r.get("goals_scored"), 0),
                        _i(r.get("assists"), 0), _i(r.get("clean_sheets"), 0),
                        _i(r.get("goals_conceded"), 0), _i(r.get("own_goals"), 0),
                        _i(r.get("penalties_saved"), 0), _i(r.get("penalties_missed"), 0),
                        _i(r.get("yellow_cards"), 0), _i(r.get("red_cards"), 0),
                        _i(r.get("saves"), 0), _i(r.get("bonus"), 0), _i(r.get("bps"), 0),
                        _i(r.get("total_points"), 0), _i(r.get("starts")),
                        _f(r.get("expected_goals")), _f(r.get("expected_assists")),
                        _f(r.get("npxG")), _i(r.get("defensive_contribution")), utcnow(),
                    ),
                )
                n += 1
            return n

        b.defer(_load)
        return b


def _find_or_create(conn, full_name: str) -> int:
    row = query_one(
        "SELECT player_id FROM player_aliases WHERE alias_norm=? LIMIT 1", (norm_name(full_name),)
    )
    if row:
        return row["player_id"]
    return _create(conn, full_name)


def _create(conn, full_name: str) -> int:
    parts = full_name.split(" ")
    return upsert_player(conn, full_name, parts[0], " ".join(parts[1:]) or parts[0], parts[-1])


def season_codes(season: str) -> dict[str, str]:
    """element id -> FPL `code` for one season, from the archived players_raw.csv."""
    row = query_one(
        "SELECT * FROM raw_documents WHERE source_id='vaastav_history' "
        "AND doc_type='vaastav_players' AND external_id=? ORDER BY id DESC LIMIT 1", (season,))
    if row is None:
        return {}
    text = load_payload(row)
    if not isinstance(text, str):
        return {}
    return {r["id"]: r["code"] for r in csv.DictReader(io.StringIO(text))
            if r.get("id") and r.get("code")}


def _ensure_teams(conn, season: str, rows: list[dict]) -> dict[str, int]:
    """Team name -> internal id. Pre-2020/21 merged_gw.csv has no `team` column, so the
    names come from that season's teams.csv instead (same source, already archived)."""
    names = {r.get("team") for r in rows if r.get("team")}
    if not names:
        return _teams_from_teams_csv(conn, season)
    out: dict[str, int] = {}
    for name in names:
        row = query_one("SELECT id FROM teams WHERE season_id=? AND name=?", (season, name))
        if row is None:
            cur = conn.execute(
                "INSERT INTO teams(season_id,fpl_team_id,name,short_name) VALUES(?,NULL,?,?)",
                (season, name, name[:3].upper()),
            )
            out[norm_name(name)] = cur.lastrowid
        else:
            out[norm_name(name)] = row["id"]
    return out


def _teams_from_teams_csv(conn, season: str) -> dict[str, int]:
    """Returns a map keyed by *FPL team id as a string*, for the old-schema path."""
    from .base import load_payload

    doc = query_one(
        "SELECT * FROM raw_documents WHERE source_id='vaastav_history' "
        "AND doc_type='vaastav_teams' AND external_id=? ORDER BY id DESC LIMIT 1",
        (season,),
    )
    if doc is None:
        return {}
    out: dict[str, int] = {}
    for t in csv.DictReader(io.StringIO(load_payload(doc))):
        fpl_id, name = _i(t.get("id")), (t.get("name") or "").strip()
        if fpl_id is None or not name:
            continue
        conn.execute(
            "INSERT INTO teams(season_id,fpl_team_id,name,short_name) VALUES(?,?,?,?) "
            "ON CONFLICT(season_id,fpl_team_id) DO UPDATE SET name=excluded.name",
            (season, fpl_id, name, (t.get("short_name") or name[:3]).upper()),
        )
        row = query_one(
            "SELECT id FROM teams WHERE season_id=? AND fpl_team_id=?", (season, fpl_id)
        )
        if row:
            out[str(fpl_id)] = row["id"]
    return out


def _row_team(r: dict, team_ids: dict) -> int | None:
    """The player's own club. Newer files name it; older ones only imply it."""
    name = r.get("team")
    if name:
        return team_ids.get(norm_name(name))
    return None


def _row_opponent(r: dict, team_ids: dict) -> int | None:
    """`opponent_team` is a numeric FPL team id, usable only with the teams.csv map."""
    opp = r.get("opponent_team")
    return team_ids.get(str(_i(opp))) if opp is not None else None


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _build_fixtures(conn, season: str, rows: list[dict], team_ids: dict[str, int]) -> dict[int, int]:
    """fpl_fixture_id -> internal fixture id, reconstructed from the player rows.

    vaastav's `opponent_team` is a numeric FPL team id that is not stable across seasons
    and is not in this file, so the home/away pair is derived from the rows themselves:
    a `was_home` row names the home side, a `was_home=False` row names the away side.
    """
    sides: dict[int, dict] = {}
    for r in rows:
        fx = _i(r.get("fixture"))
        if fx is None:
            continue
        team = _row_team(r, team_ids)
        opponent = _row_opponent(r, team_ids)
        if team is None and opponent is None:
            continue
        entry = sides.setdefault(
            fx, {"home": None, "away": None, "gw": _i(r.get("GW")),
                 "kickoff": r.get("kickoff_time"),
                 "hs": _i(r.get("team_h_score")), "as": _i(r.get("team_a_score"))}
        )
        if _is_true(r.get("was_home")):
            entry["home"] = entry["home"] or team
            entry["away"] = entry["away"] or opponent
        else:
            entry["away"] = entry["away"] or team
            entry["home"] = entry["home"] or opponent

    out: dict[int, int] = {}
    for fx, e in sides.items():
        if e["home"] is None or e["away"] is None:
            continue  # a fixture only one side appeared in is unusable
        conn.execute(
            "INSERT INTO fixtures(season_id,fpl_fixture_id,gameweek,kickoff_utc,home_team_id,"
            "away_team_id,finished,home_score,away_score,competition) "
            "VALUES(?,?,?,?,?,?,1,?,?,'PL') "
            "ON CONFLICT(season_id,fpl_fixture_id) DO UPDATE SET gameweek=excluded.gameweek,"
            "kickoff_utc=COALESCE(excluded.kickoff_utc,kickoff_utc),"
            "home_score=COALESCE(excluded.home_score,home_score),"
            "away_score=COALESCE(excluded.away_score,away_score)",
            (season, fx, e["gw"], e["kickoff"], e["home"], e["away"], e["hs"], e["as"]),
        )
        row = query_one(
            "SELECT id FROM fixtures WHERE season_id=? AND fpl_fixture_id=?", (season, fx)
        )
        if row:
            out[fx] = row["id"]
    return out


def duplicate_players_by_code() -> dict[int, int]:
    """{duplicate player id: canonical id} for every footballer split across rows.

    Grouped by FPL `code` via each season's element links. A row reachable from two codes
    is two people already fused by the old name matching; it is left out rather than
    allowed to chain both groups into one.
    """
    import re

    from ..db.engine import query

    by_code: dict[str, set[int]] = {}
    for s in [r["id"] for r in query("SELECT id FROM seasons")]:
        for element, code in season_codes(s).items():
            pid = by_external_id("fpl_element", f"{s}:{element}")
            if pid is not None:
                by_code.setdefault(code, set()).add(pid)
    live = {r["external_id"]: r["player_id"]
            for r in query("SELECT external_id, player_id FROM player_external_ids WHERE system='fpl'")}
    for code, pid in live.items():
        by_code.setdefault(code, set()).add(pid)

    seen: dict[int, int] = {}
    for pids in by_code.values():
        for p in pids:
            seen[p] = seen.get(p, 0) + 1
    last_season = {r["player_id"]: r["s"] for r in
                   query("SELECT player_id, MAX(season_id) s FROM player_seasons GROUP BY player_id")}
    names = {r["id"]: r["canonical_name"] for r in query("SELECT id, canonical_name FROM players")}

    mapping: dict[int, int] = {}
    for code, pids in by_code.items():
        pids = {p for p in pids if seen[p] == 1}
        if len(pids) < 2:
            continue
        if live.get(code) in pids:
            canon = live[code]
        else:
            canon = min(pids, key=lambda p: (bool(re.search(r"\d|�", names.get(p, ""))),
                                             -int((last_season.get(p) or "0")[:4]), p))
        mapping.update({p: canon for p in pids if p != canon})
    return mapping
