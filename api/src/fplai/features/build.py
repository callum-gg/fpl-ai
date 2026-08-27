"""Assemble FeatureCtx objects and persist feature_values. docs/05.

Every slice loaded here is filtered to `< as_of`. That single choice is what makes the
leakage test in docs/12 layer 4 pass: rebuilding GW k from scratch reproduces exactly
what was computed live, because no query can see past the deadline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import lru_cache

from ..db.engine import query, query_one, upsert_many, utcnow, writer
from ..db.settings_store import global_settings
from . import builders  # noqa: F401 - importing registers every feature
from .registry import FEATURE_VERSION, FeatureCtx, compute_all

log = logging.getLogger(__name__)

PROMOTED_2026_27 = {"Coventry City", "Ipswich Town", "Hull City"}


def deadline_of(season_id: str, gameweek: int) -> str:
    row = query_one(
        "SELECT deadline_utc FROM gameweeks WHERE season_id=? AND gameweek=?", (season_id, gameweek)
    )
    if row:
        return row["deadline_utc"]
    # Historic seasons backfilled without gameweek rows: fall back to first kickoff.
    row = query_one(
        "SELECT MIN(kickoff_utc) k FROM fixtures WHERE season_id=? AND gameweek=?",
        (season_id, gameweek),
    )
    return (row["k"] if row and row["k"] else utcnow())


def _player_history(player_id: int, as_of: str, limit: int = 40) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT s.*, f.kickoff_utc, f.gameweek, f.season_id, f.competition "
            "FROM player_fixture_stats s JOIN fixtures f ON f.id = s.fixture_id "
            "WHERE s.player_id=? AND f.kickoff_utc < ? "
            "ORDER BY f.kickoff_utc DESC LIMIT ?",
            (player_id, as_of, limit),
        )
    ]


def _team_history(team_id: int | None, as_of: str, limit: int = 12) -> list[dict]:
    if team_id is None:
        return []
    return [
        dict(r)
        for r in query(
            "SELECT f.*, "
            "CASE WHEN f.home_team_id=? THEN f.away_score ELSE f.home_score END goals_conceded, "
            "CASE WHEN f.home_team_id=? THEN f.home_score ELSE f.away_score END goals_scored "
            "FROM fixtures f WHERE (f.home_team_id=? OR f.away_team_id=?) "
            "AND f.finished=1 AND f.kickoff_utc < ? ORDER BY f.kickoff_utc DESC LIMIT ?",
            (team_id, team_id, team_id, team_id, as_of, limit),
        )
    ]


def _upcoming(team_id: int | None, as_of: str, limit: int = 10) -> list[dict]:
    if team_id is None:
        return []
    rows = query(
        "SELECT f.*, CASE WHEN f.home_team_id=? THEN f.fdr_home ELSE f.fdr_away END difficulty "
        "FROM fixtures f WHERE (f.home_team_id=? OR f.away_team_id=?) AND f.kickoff_utc >= ? "
        "ORDER BY f.kickoff_utc LIMIT ?",
        (team_id, team_id, team_id, as_of, limit),
    )
    return [dict(r) for r in rows]


def _availability(player_id: int, as_of: str) -> list[dict]:
    """Latest observation per source, strictly before the deadline."""
    rows = query(
        "SELECT a.* FROM availability a JOIN ("
        "  SELECT source_id, MAX(observed_at) m FROM availability "
        "  WHERE player_id=? AND observed_at < ? GROUP BY source_id"
        ") x ON x.source_id=a.source_id AND x.m=a.observed_at WHERE a.player_id=?",
        (player_id, as_of, player_id),
    )
    return [dict(r) for r in rows]


def _claims(player_id: int, as_of: str, days: int = 14) -> list[dict]:
    cutoff = (datetime.fromisoformat(as_of.replace("Z", "+00:00")) - timedelta(days=days)).isoformat()
    rows = query(
        "SELECT c.*, v.channel_id, ch.trust_weight, "
        "CASE WHEN v.id IS NOT NULL THEN 'video' "
        "     WHEN sp.platform IS NOT NULL THEN sp.platform ELSE 'article' END platform "
        "FROM claims c "
        "LEFT JOIN videos v ON v.raw_doc_id = c.raw_doc_id "
        "LEFT JOIN channels ch ON ch.channel_id = v.channel_id "
        "LEFT JOIN social_posts sp ON sp.raw_doc_id = c.raw_doc_id "
        "WHERE c.player_id=? AND c.extracted_at < ? AND c.extracted_at >= ?",
        (player_id, as_of, cutoff),
    )
    weights = global_settings().get("text.trust_weights", {})
    out = []
    for r in rows:
        d = dict(r)
        d["trust_weight"] = d.get("trust_weight") or weights.get(
            {"video": "youtube_default", "reddit": "reddit", "x": "tier1_journalist"}.get(
                d.get("platform"), "rss_default"
            ),
            1.0,
        )
        out.append(d)
    return out


def _ownership(player_id: int, season_id: str, gameweek: int, as_of: str) -> dict:
    rows = query(
        "SELECT scope, owned_pct, effective_ownership FROM ownership_snapshots "
        "WHERE player_id=? AND season_id=? AND gameweek<=? AND observed_at < ? "
        "ORDER BY observed_at DESC LIMIT 6",
        (player_id, season_id, gameweek, as_of),
    )
    out: dict = {}
    for r in rows:
        if r["scope"] == "overall" and "overall" not in out:
            out["overall"] = r["owned_pct"]
        if r["scope"] == "top10k" and "top10k" not in out:
            out["top10k"] = r["owned_pct"]
            out["effective"] = r["effective_ownership"]
    return out


def _price(player_id: int, season_id: str, as_of: str) -> dict:
    row = query_one(
        "SELECT price, selected_by_percent, net_transfers FROM player_prices "
        "WHERE player_id=? AND season_id=? AND observed_at < ? ORDER BY observed_at DESC LIMIT 1",
        (player_id, season_id, as_of),
    )
    return dict(row) if row else {}


def _set_pieces(player_id: int, season_id: str, as_of: str) -> dict:
    rows = query(
        "SELECT role, MIN(rank) rank FROM set_piece_roles WHERE player_id=? AND season_id=? "
        "AND observed_at < ? GROUP BY role",
        (player_id, season_id, as_of),
    )
    return {r["role"]: r["rank"] for r in rows}


def _odds(fixture_id: int | None, team_id: int | None, player_id: int) -> dict:
    if fixture_id is None:
        return {}
    from ..connectors.odds_api import odds_movement, team_lambdas

    lams = team_lambdas(fixture_id)
    if lams is None:
        return {}
    fx = query_one("SELECT home_team_id FROM fixtures WHERE id=?", (fixture_id,))
    is_home = bool(fx and fx["home_team_id"] == team_id)
    out = {
        "team_lambda": lams[0] if is_home else lams[1],
        "opp_lambda": lams[1] if is_home else lams[0],
        "movement_48h": odds_movement(fixture_id),
    }
    scorer = query_one(
        "SELECT devig_prob FROM odds_snapshots WHERE fixture_id=? AND market='anytime_scorer' "
        "AND player_id=? ORDER BY observed_at DESC LIMIT 1",
        (fixture_id, player_id),
    )
    if scorer:
        out["p_anytime_scorer"] = scorer["devig_prob"]
    return out


@lru_cache(maxsize=4)
def _team_model_as_of(season_id: str, as_of: str):
    """Team strengths as they stood at this deadline, and only as they stood then.

    Refitting per deadline instead of loading the shipped artifact is the difference
    between a feature and a leak: the active artifact has seen the whole season, so using
    it to build a GW7 row would quietly tell the model how GW7 turned out.

    `build_gameweek` resolves `as_of` once and hands the same value to every player, so
    this fits once per gameweek, not once per player — the cache is what makes that true.
    """
    from ..models.team_goals import fit

    seasons = [r["id"] for r in query("SELECT id FROM seasons WHERE id<=? ORDER BY id",
                                      (season_id,))]
    return fit(seasons or [season_id], as_of=as_of)


def build_ctx(
    player_id: int, season_id: str, gameweek: int, fixture_id: int | None = None,
    as_of: str | None = None,
) -> FeatureCtx | None:
    as_of = as_of or deadline_of(season_id, gameweek)
    ps = query_one(
        "SELECT ps.position, ps.team_id, p.birth_date, p.nationality, t.name team_name "
        "FROM player_seasons ps JOIN players p ON p.id=ps.player_id "
        "LEFT JOIN teams t ON t.id=ps.team_id WHERE ps.player_id=? AND ps.season_id=?",
        (player_id, season_id),
    )
    if ps is None:
        return None

    team_id = ps["team_id"]
    gw_fixtures = query(
        "SELECT * FROM fixtures WHERE season_id=? AND gameweek=? AND competition='PL' "
        "AND (home_team_id=? OR away_team_id=?)",
        (season_id, gameweek, team_id, team_id),
    )
    fixture = None
    if fixture_id is not None:
        fixture = query_one("SELECT * FROM fixtures WHERE id=?", (fixture_id,))
    elif gw_fixtures:
        fixture = gw_fixtures[0]

    is_home = bool(fixture and fixture["home_team_id"] == team_id)
    opponent = None
    if fixture:
        opponent = fixture["away_team_id"] if is_home else fixture["home_team_id"]

    upcoming = _upcoming(team_id, as_of)
    team_hist = _team_history(team_id, as_of)
    extras = {
        "birth_date": ps["birth_date"],
        "promoted": ps["team_name"] in PROMOTED_2026_27,
        "n_fixtures_this_gw": len(gw_fixtures),
        "kickoff_utc": fixture["kickoff_utc"] if fixture else None,
        "fdr": (fixture["fdr_home"] if is_home else fixture["fdr_away"]) if fixture else None,
        "team_days_rest": _days_rest(team_hist, as_of),
        "opp_days_rest": _days_rest(_team_history(opponent, as_of, 3), as_of),
        "federation": _federation(ps["nationality"]),
        "price_percentile_in_position": _price_percentile(player_id, season_id, ps["position"],
                                                          as_of),
    }
    if fixture:
        from ..connectors.lineups import predicted_lineup_prob

        extras["predicted_lineup_prob"] = predicted_lineup_prob(fixture["id"], player_id)

    # The team model's own view of this fixture. Without it `team_expected_goals` and
    # `opp_expected_goals` fall back to odds alone — and with no odds source configured
    # that left every rate model with no fixture-difficulty signal at all, which is
    # exactly what their zero gain importances were saying.
    if fixture and opponent is not None:
        tm = _team_model_as_of(season_id, as_of)
        lh, la = tm.lambdas(fixture["home_team_id"], fixture["away_team_id"])
        extras["model_team_lambda"], extras["model_opp_lambda"] = (
            (lh, la) if is_home else (la, lh)
        )
        extras["opp_attack_rating"] = tm.attack_of(opponent)
        extras["opp_defence_rating"] = tm.defence_of(opponent)

    return FeatureCtx(
        player_id=player_id,
        season_id=season_id,
        gameweek=gameweek,
        fixture_id=fixture["id"] if fixture else None,
        as_of=as_of,
        position=ps["position"],
        team_id=team_id,
        opponent_team_id=opponent,
        is_home=is_home,
        history=_player_history(player_id, as_of),
        team_history=team_hist,
        opp_history=_team_history(opponent, as_of),
        upcoming=upcoming,
        availability=_availability(player_id, as_of),
        claims=_claims(player_id, as_of),
        odds=_odds(fixture["id"] if fixture else None, team_id, player_id),
        ownership=_ownership(player_id, season_id, gameweek, as_of),
        set_pieces=_set_pieces(player_id, season_id, as_of),
        price=_price(player_id, season_id, as_of),
        extras=extras,
    )


def _price_percentile(player_id: int, season_id: str, position: str, as_of: str) -> float | None:
    """Where this player sits in his position's price distribution, 0..1."""
    row = query_one(
        "WITH latest AS ("
        "  SELECT pp.player_id, pp.price FROM player_prices pp JOIN player_seasons ps "
        "  ON ps.player_id=pp.player_id AND ps.season_id=pp.season_id "
        "  WHERE pp.season_id=? AND ps.position=? AND pp.observed_at < ? "
        "  AND pp.observed_at=(SELECT MAX(observed_at) FROM player_prices "
        "    WHERE player_id=pp.player_id AND season_id=pp.season_id AND observed_at < ?)"
        ") SELECT AVG(CASE WHEN price <= (SELECT price FROM latest WHERE player_id=?) "
        "THEN 1.0 ELSE 0.0 END) pct FROM latest",
        (season_id, position, as_of, as_of, player_id),
    )
    return row["pct"] if row and row["pct"] is not None else None


def _days_rest(history: list[dict], as_of: str) -> float | None:
    if not history:
        return None
    try:
        last = datetime.fromisoformat((history[0].get("kickoff_utc") or "").replace("Z", "+00:00"))
        return (datetime.fromisoformat(as_of.replace("Z", "+00:00")) - last).days
    except (ValueError, AttributeError):
        return None


_FEDERATIONS = {
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Chile": "CONMEBOL",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Iran": "AFC",
    "Nigeria": "CAF", "Ghana": "CAF", "Senegal": "CAF", "Egypt": "CAF",
    "Ivory Coast": "CAF", "Morocco": "CAF", "Algeria": "CAF", "Mali": "CAF",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Jamaica": "CONCACAF", "New Zealand": "OFC",
}


def _federation(nationality: str | None) -> str:
    return _FEDERATIONS.get(nationality or "", "UEFA")


def build_gameweek(
    season_id: str, gameweek: int, player_ids: list[int] | None = None, persist: bool = True
) -> int:
    """Compute and store every feature for every player in one gameweek."""
    as_of = deadline_of(season_id, gameweek)
    if player_ids is None:
        player_ids = [
            r["player_id"]
            for r in query("SELECT player_id FROM player_seasons WHERE season_id=?", (season_id,))
        ]

    now = utcnow()
    rows: list[dict] = []
    for pid in player_ids:
        ctx = build_ctx(pid, season_id, gameweek, as_of=as_of)
        if ctx is None:
            continue
        values = compute_all(ctx)
        fixture_key = ctx.fixture_id or 0
        for name, value in values.items():
            if value is None:
                continue
            rows.append(
                {
                    "season_id": season_id,
                    "gameweek": gameweek,
                    "player_id": pid,
                    "fixture_key": fixture_key,
                    "name": name,
                    "value": float(value),
                    "computed_at": now,
                    "feature_version": FEATURE_VERSION,
                }
            )

    if persist and rows:
        with writer() as conn:
            for i in range(0, len(rows), 5000):
                upsert_many(
                    conn, "feature_values", rows[i:i + 5000],
                    ["season_id", "gameweek", "player_id", "fixture_key", "name"],
                )
    log.info("built %d feature values for %s GW%s", len(rows), season_id, gameweek)
    return len(rows)


def load_matrix(season_id: str, gameweeks: list[int] | None = None):
    """Wide feature matrix for training. The tall table stays the source of truth for the
    UI's 'why'; this is the pivot the trainer actually consumes (docs/03 note)."""
    import pandas as pd

    sql = "SELECT season_id,gameweek,player_id,fixture_key,name,value FROM feature_values WHERE season_id=?"
    params: list = [season_id]
    if gameweeks:
        sql += f" AND gameweek IN ({','.join('?' * len(gameweeks))})"
        params += gameweeks
    df = pd.DataFrame([dict(r) for r in query(sql, tuple(params))])
    if df.empty:
        return df
    return df.pivot_table(
        index=["season_id", "gameweek", "player_id", "fixture_key"],
        columns="name", values="value", aggfunc="last",
    ).reset_index()


def feature_explanations(player_id: int, season_id: str, gameweek: int) -> list[dict]:
    """Powers GET /api/players/{id}/features — value plus percentile within position."""
    rows = query(
        "SELECT name, value FROM feature_values WHERE season_id=? AND gameweek=? AND player_id=?",
        (season_id, gameweek, player_id),
    )
    out = []
    for r in rows:
        pct = query_one(
            "SELECT AVG(CASE WHEN value <= ? THEN 1.0 ELSE 0.0 END) p FROM feature_values "
            "WHERE season_id=? AND gameweek=? AND name=?",
            (r["value"], season_id, gameweek, r["name"]),
        )
        from .registry import REGISTRY

        meta = REGISTRY.get(r["name"])
        out.append(
            {
                "name": r["name"],
                "value": r["value"],
                "percentile": pct["p"] if pct else None,
                "group": meta.group if meta else "misc",
                "description": meta.description if meta else "",
            }
        )
    return sorted(out, key=lambda x: (x["group"], x["name"]))
