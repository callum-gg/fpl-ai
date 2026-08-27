"""Players, predictions, features, claims, fixtures and teams. docs/09."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...config import get_settings
from ...connectors.fpl_official import current_gameweek, next_deadline, next_gameweek
from ...db.engine import query, query_one
from ...features.build import feature_explanations
from ...models import team_goals
from ...models.predict import horizon_points, latest, prediction_for
from ...optimiser.recommend import SET_SOURCES

router = APIRouter(prefix="/api", tags=["players"])


def _season() -> str:
    return get_settings().current_season


@router.get("/players")
def list_players(
    position: str | None = None,
    team: int | None = None,
    max_price: int | None = None,
    min_minutes: int | None = None,
    sort: str = "exp_points_gw",
    owned_by_squad: int | None = None,
    limit: int = Query(500, le=1000),
    offset: int = 0,
) -> list[dict]:
    season = _season()
    gw = next_gameweek(season)
    sql = [
        "SELECT p.id, p.web_name, p.canonical_name, ps.position, ps.team_id, t.name team_name,",
        "  t.short_name team_short,",
        "  (SELECT price FROM player_prices WHERE player_id=p.id AND season_id=ps.season_id",
        "   ORDER BY observed_at DESC LIMIT 1) price,",
        "  (SELECT selected_by_percent FROM player_prices WHERE player_id=p.id",
        "   AND season_id=ps.season_id ORDER BY observed_at DESC LIMIT 1) owned_pct,",
        "  (SELECT SUM(COALESCE(s.minutes,0)) FROM player_fixture_stats s",
        "   JOIN fixtures f ON f.id=s.fixture_id WHERE s.player_id=p.id AND f.season_id=?) minutes,",
        "  (SELECT SUM(COALESCE(s.total_points,0)) FROM player_fixture_stats s",
        "   JOIN fixtures f ON f.id=s.fixture_id WHERE s.player_id=p.id AND f.season_id=?) points",
        "FROM players p JOIN player_seasons ps ON ps.player_id=p.id AND ps.season_id=?",
        "LEFT JOIN teams t ON t.id=ps.team_id WHERE 1=1",
    ]
    params: list = [season, season, season]
    if position:
        sql.append("AND ps.position=?")
        params.append(position)
    if team:
        sql.append("AND ps.team_id=?")
        params.append(team)
    if owned_by_squad:
        sql.append(
            # The squad you own, not merely the newest state row: drafts and accepted
            # plans are states too, and neither means you own the player.
            "AND p.id IN (SELECT sp.player_id FROM squad_picks sp JOIN squad_states ss "
            "ON ss.id=sp.squad_state_id WHERE ss.squad_id=? AND ss.source IN (?,?) "
            "AND ss.captured_at=(SELECT MAX(captured_at) FROM squad_states "
            "                    WHERE squad_id=? AND source IN (?,?)))"
        )
        params += [owned_by_squad, *SET_SOURCES, owned_by_squad, *SET_SOURCES]

    rows = [dict(r) for r in query(" ".join(sql), tuple(params))]

    preds = {p["player_id"]: p for p in latest(season, gw)}
    horizon = horizon_points(season, gw, 5)
    for r in rows:
        pred = preds.get(r["id"])
        r["exp_points_gw"] = pred["exp_points"] if pred else None
        r["p_start"] = pred["p_start"] if pred else None
        r["p_haul"] = pred["p_haul_10"] if pred else None
        r["sd_points"] = pred["sd_points"] if pred else None
        r["exp_points_horizon"] = round(sum(horizon.get(r["id"], [])), 2) if r["id"] in horizon \
            else None
        r["value"] = (
            round(r["exp_points_horizon"] / (r["price"] / 10), 2)
            if r.get("price") and r.get("exp_points_horizon") else None
        )
        r["form_sparkline"] = _sparkline(r["id"], season)

    if max_price:
        rows = [r for r in rows if (r.get("price") or 0) <= max_price]
    if min_minutes:
        rows = [r for r in rows if (r.get("minutes") or 0) >= min_minutes]

    reverse = sort not in ("price", "web_name")
    rows.sort(key=lambda r: (r.get(sort) is None, r.get(sort) or 0), reverse=reverse)
    return rows[offset: offset + limit]


def _sparkline(player_id: int, season: str, n: int = 6) -> list[int]:
    rows = query(
        "SELECT COALESCE(s.total_points,0) p FROM player_fixture_stats s "
        "JOIN fixtures f ON f.id=s.fixture_id WHERE s.player_id=? AND f.season_id=? "
        "AND f.finished=1 ORDER BY f.kickoff_utc DESC LIMIT ?",
        (player_id, season, n),
    )
    return [r["p"] for r in reversed(rows)]


@router.get("/players/compare")
def compare_players(ids: str, gws: str = "1-6") -> dict:
    season = _season()
    player_ids = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    start, end = _gw_range(gws)
    out = []
    for pid in player_ids:
        base = query_one(
            "SELECT p.id, p.web_name, ps.position, t.name team FROM players p "
            "JOIN player_seasons ps ON ps.player_id=p.id AND ps.season_id=? "
            "LEFT JOIN teams t ON t.id=ps.team_id WHERE p.id=?",
            (season, pid),
        )
        if base is None:
            continue
        entry = dict(base)
        entry["predictions"] = [
            prediction_for(pid, season, gw) for gw in range(start, end + 1)
        ]
        out.append(entry)
    return {"players": out, "gameweeks": list(range(start, end + 1))}


@router.get("/players/{player_id}")
def get_player(player_id: int) -> dict:
    season = _season()
    row = query_one(
        "SELECT p.*, ps.position, ps.team_id, ps.fpl_element_id, t.name team_name "
        "FROM players p LEFT JOIN player_seasons ps ON ps.player_id=p.id AND ps.season_id=? "
        "LEFT JOIN teams t ON t.id=ps.team_id WHERE p.id=?",
        (season, player_id),
    )
    if row is None:
        raise HTTPException(404, {"error": {"code": "not_found", "message": "player not found"}})
    d = dict(row)
    d["price"] = query_one(
        "SELECT price, selected_by_percent FROM player_prices WHERE player_id=? AND season_id=? "
        "ORDER BY observed_at DESC LIMIT 1",
        (player_id, season),
    )
    d["availability"] = [
        dict(r) for r in query(
            "SELECT * FROM availability WHERE player_id=? ORDER BY observed_at DESC LIMIT 8",
            (player_id,),
        )
    ]
    d["set_pieces"] = [
        dict(r) for r in query(
            "SELECT role, MIN(rank) rank FROM set_piece_roles WHERE player_id=? AND season_id=? "
            "GROUP BY role",
            (player_id, season),
        )
    ]
    gw = next_gameweek(season)
    d["prediction"] = prediction_for(player_id, season, gw)
    d["upcoming"] = _upcoming_fixtures(d.get("team_id"), season)
    return d


@router.get("/players/{player_id}/history")
def player_history(player_id: int, season: str | None = None) -> list[dict]:
    season = season or _season()
    return [
        dict(r)
        for r in query(
            "SELECT s.*, f.gameweek, f.kickoff_utc, f.competition, "
            "CASE WHEN f.home_team_id=s.team_id THEN ta.name ELSE th.name END opponent "
            "FROM player_fixture_stats s JOIN fixtures f ON f.id=s.fixture_id "
            "JOIN teams th ON th.id=f.home_team_id JOIN teams ta ON ta.id=f.away_team_id "
            "WHERE s.player_id=? AND f.season_id=? ORDER BY f.kickoff_utc DESC",
            (player_id, season),
        )
    ]


@router.get("/players/{player_id}/predictions")
def player_predictions(player_id: int, gws: str = "1-6") -> list[dict]:
    season = _season()
    start, end = _gw_range(gws)
    return [
        p for p in (prediction_for(player_id, season, gw) for gw in range(start, end + 1))
        if p is not None
    ]


@router.get("/players/{player_id}/features")
def player_features(player_id: int, gw: int | None = None) -> list[dict]:
    season = _season()
    return feature_explanations(player_id, season, gw or next_gameweek(season))


@router.get("/players/{player_id}/claims")
def player_claims(player_id: int, days: int = 14) -> list[dict]:
    rows = query(
        "SELECT c.*, rd.source_id, rd.url, rd.published_at, v.youtube_id, v.title video_title, "
        "a.title article_title, a.outlet, ch.trust_weight "
        "FROM claims c JOIN raw_documents rd ON rd.id=c.raw_doc_id "
        "LEFT JOIN videos v ON v.raw_doc_id=c.raw_doc_id "
        "LEFT JOIN channels ch ON ch.channel_id=v.channel_id "
        "LEFT JOIN articles a ON a.raw_doc_id=c.raw_doc_id "
        "WHERE c.player_id=? AND c.extracted_at > datetime('now', ?) "
        "ORDER BY c.extracted_at DESC",
        (player_id, f"-{days} day"),
    )
    out = []
    seen_groups: set[int] = set()
    for r in rows:
        d = dict(r)
        group = d.get("semantic_group")
        if group and group in seen_groups:
            # Near-duplicates collapse into a "seen on N sites" badge, not N entries.
            for existing in out:
                if existing.get("semantic_group") == group:
                    existing["duplicate_count"] = existing.get("duplicate_count", 1) + 1
                    break
            continue
        if group:
            seen_groups.add(group)
        d["duplicate_count"] = 1
        d["title"] = d.get("video_title") or d.get("article_title")
        if d.get("youtube_id") and d.get("start_s") is not None:
            d["deep_link"] = f"https://youtube.com/watch?v={d['youtube_id']}&t={int(d['start_s'])}s"
        else:
            d["deep_link"] = d.get("url")
        out.append(d)
    return out


# --- fixtures, gameweeks, teams -------------------------------------------------


@router.get("/gameweeks/current")
def gameweek_current() -> dict:
    season = _season()
    nxt = next_deadline(season) or {}
    state = query_one(
        "SELECT chips_used_json FROM squad_states ORDER BY captured_at DESC LIMIT 1"
    )
    import json

    from ...rules import chips_available

    used = json.loads(state["chips_used_json"]) if state and state["chips_used_json"] else []
    used = [{"name": c.get("name"), "gameweek": c.get("gameweek") or c.get("event")}
            for c in used if isinstance(c, dict)]
    return {
        "season_id": season,
        "gameweek": nxt.get("gameweek", current_gameweek(season)),
        "deadline_utc": nxt.get("deadline_utc"),
        "seconds_remaining": nxt.get("seconds_remaining"),
        "chips_available": chips_available(nxt.get("gameweek", 1), used),
    }


@router.get("/gameweeks")
def gameweeks() -> list[dict]:
    return [
        dict(r) for r in query(
            "SELECT * FROM gameweeks WHERE season_id=? ORDER BY gameweek", (_season(),)
        )
    ]


@router.get("/fixtures")
def fixtures(gw: int | None = None, team: int | None = None) -> list[dict]:
    sql = (
        "SELECT f.*, th.name home_name, th.short_name home_short, "
        "ta.name away_name, ta.short_name away_short FROM fixtures f "
        "JOIN teams th ON th.id=f.home_team_id JOIN teams ta ON ta.id=f.away_team_id "
        "WHERE f.season_id=?"
    )
    params: list = [_season()]
    if gw is not None:
        sql += " AND f.gameweek=?"
        params.append(gw)
    if team is not None:
        sql += " AND (f.home_team_id=? OR f.away_team_id=?)"
        params += [team, team]
    return [dict(r) for r in query(sql + " ORDER BY f.kickoff_utc", tuple(params))]


@router.get("/fixture-ticker")
def fixture_ticker(gws: str = "1-8") -> dict:
    """team x gameweek matrix with model difficulty and DGW/BGW flags."""
    season = _season()
    start, end = _gw_range(gws)
    teams = [dict(r) for r in query(
        "SELECT id, name, short_name FROM teams WHERE season_id=? ORDER BY name", (season,)
    )]
    rows = query(
        "SELECT gameweek, home_team_id h, away_team_id a, fdr_home, fdr_away FROM fixtures "
        "WHERE season_id=? AND competition='PL' AND gameweek BETWEEN ? AND ?",
        (season, start, end),
    )
    tm = team_goals.load_model_or_fit(season)
    matrix: dict[int, dict[int, list]] = {t["id"]: {gw: [] for gw in range(start, end + 1)}
                                          for t in teams}
    for r in rows:
        for team_id, opp_id, fdr, home in (
            (r["h"], r["a"], r["fdr_home"], True),
            (r["a"], r["h"], r["fdr_away"], False),
        ):
            if team_id in matrix and r["gameweek"] in matrix[team_id]:
                lh, la = tm.lambdas(r["h"], r["a"])
                matrix[team_id][r["gameweek"]].append(
                    {
                        "opponent_id": opp_id,
                        "is_home": home,
                        "fdr_official": fdr,
                        "model_difficulty": round((la if home else lh), 2),
                        "expected_goals": round((lh if home else la), 2),
                    }
                )
    return {
        "gameweeks": list(range(start, end + 1)),
        "teams": [
            {
                **t,
                "fixtures": {
                    str(gw): {
                        "matches": matrix[t["id"]][gw],
                        "dgw": len(matrix[t["id"]][gw]) > 1,
                        "bgw": len(matrix[t["id"]][gw]) == 0,
                    }
                    for gw in range(start, end + 1)
                },
            }
            for t in teams
        ],
    }


@router.get("/teams")
def teams() -> list[dict]:
    return [dict(r) for r in query(
        "SELECT * FROM teams WHERE season_id=? ORDER BY name", (_season(),)
    )]


@router.get("/teams/{team_id}/strength")
def team_strength(team_id: int) -> list[dict]:
    return team_goals.team_strengths_over_time(team_id, _season())


def _upcoming_fixtures(team_id: int | None, season: str, n: int = 8) -> list[dict]:
    if team_id is None:
        return []
    return [
        dict(r)
        for r in query(
            "SELECT f.gameweek, f.kickoff_utc, f.competition, "
            "CASE WHEN f.home_team_id=? THEN 1 ELSE 0 END is_home, "
            "CASE WHEN f.home_team_id=? THEN ta.name ELSE th.name END opponent, "
            "CASE WHEN f.home_team_id=? THEN f.fdr_home ELSE f.fdr_away END difficulty "
            "FROM fixtures f JOIN teams th ON th.id=f.home_team_id "
            "JOIN teams ta ON ta.id=f.away_team_id "
            "WHERE f.season_id=? AND (f.home_team_id=? OR f.away_team_id=?) "
            "AND f.kickoff_utc >= datetime('now') ORDER BY f.kickoff_utc LIMIT ?",
            (team_id, team_id, team_id, season, team_id, team_id, n),
        )
    ]


def _gw_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    return int(spec), int(spec)
