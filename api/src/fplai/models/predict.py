"""Assemble models + features into `predictions` rows. docs/06.

This is the seam where everything meets: the feature store feeds the component models,
the components feed the correlated simulator, and the simulator's marginals are what
the UI and optimiser read.
"""

from __future__ import annotations

import logging

from ..db.engine import jdump, query, query_one, utcnow, writer
from ..db.settings_store import global_settings
from ..defaults import normalise_position
from ..features.build import build_ctx, deadline_of
from ..features.registry import compute_all
from . import bonus as bonus_mod
from . import minutes as minutes_mod
from . import rates as rates_mod
from . import team_goals as team_mod
from .base import load_active
from .simulate import FixtureInput, PlayerInput, save_draws, simulate_gameweek

log = logging.getLogger(__name__)


def _now() -> str:
    return utcnow()


def load_models() -> dict:
    """Every active artefact. Missing ones fall back to heuristics so the app works
    before any training has happened."""
    team = load_active("team_goals")
    return {
        "minutes": load_active("minutes"),
        "team_goals": team_mod.TeamModel.from_dict(team) if isinstance(team, dict) else team,
        "rates": rates_mod.RateModels(
            goals=load_active("goals90"),
            assists=load_active("assists90"),
            defcon=load_active("defcon"),
            saves=load_active("saves90"),
            cards=load_active("cards90"),
        ),
        "bonus": load_active("bonus") or bonus_mod.BonusModel(),
    }


def _team_model(models: dict, season_id: str, as_of: str) -> team_mod.TeamModel:
    tm = models.get("team_goals")
    if isinstance(tm, team_mod.TeamModel):
        return tm
    seasons = [r["id"] for r in query("SELECT id FROM seasons ORDER BY id DESC LIMIT 4")]
    fitted = team_mod.fit(seasons or [season_id], as_of)
    models["team_goals"] = fitted
    return fitted


def build_inputs(
    season_id: str, gameweek: int, models: dict, player_ids: list[int] | None = None
) -> tuple[list[FixtureInput], dict[int, dict]]:
    """One FixtureInput per PL fixture in the gameweek, populated with PlayerInputs."""
    as_of = deadline_of(season_id, gameweek)
    tm = _team_model(models, season_id, as_of)

    fixtures = query(
        "SELECT * FROM fixtures WHERE season_id=? AND gameweek=? AND competition='PL'",
        (season_id, gameweek),
    )
    if player_ids is None:
        player_ids = [
            r["player_id"]
            for r in query(
                "SELECT player_id FROM player_seasons WHERE season_id=? AND team_id IS NOT NULL",
                (season_id,),
            )
        ]

    by_team: dict[int, list[int]] = {}
    positions: dict[int, str] = {}
    for r in query(
        "SELECT player_id, team_id, position FROM player_seasons WHERE season_id=?", (season_id,)
    ):
        if r["player_id"] in set(player_ids) and r["team_id"]:
            by_team.setdefault(r["team_id"], []).append(r["player_id"])
            positions[r["player_id"]] = r["position"]

    bonus_model = models["bonus"]
    rate_models = models["rates"]
    minutes_artifact = models["minutes"]

    out: list[FixtureInput] = []
    feature_cache: dict[int, dict] = {}

    for fx in fixtures:
        lh, la = team_mod.blended_lambdas(tm, fx["id"])
        fi = FixtureInput(
            fixture_id=fx["id"],
            home_team_id=fx["home_team_id"],
            away_team_id=fx["away_team_id"],
            lambda_home=lh,
            lambda_away=la,
            rho=tm.rho,
        )
        for team_id in (fx["home_team_id"], fx["away_team_id"]):
            for pid in by_team.get(team_id, []):
                ctx = build_ctx(pid, season_id, gameweek, fixture_id=fx["id"], as_of=as_of)
                if ctx is None:
                    continue
                f = compute_all(ctx)
                feature_cache[pid] = f
                position = normalise_position(positions.get(pid, ctx.position))
                team_lambda = lh if team_id == fx["home_team_id"] else la

                mp = _minutes_for(minutes_artifact, ctx, f, fx["id"], pid)
                defcon = rate_models.defcon or rates_mod.DefconModel(None)

                fi.players.append(
                    PlayerInput(
                        player_id=pid,
                        position=position,
                        team_id=team_id,
                        fixture_id=fx["id"],
                        p_start=mp.p_start,
                        p_cameo=mp.p_cameo,
                        exp_minutes=mp.exp_minutes,
                        goals90=_conditioned(rate_models.goals90(f, position), team_lambda),
                        assists90=_conditioned(rate_models.assists90(f, position), team_lambda),
                        defcon_rate90=defcon.expected_actions(f, 90.0) or (
                            f.get("defcon_actions90") or 0.0
                        ),
                        defcon_dispersion=defcon.dispersion,
                        saves90=rate_models.saves90(f, position),
                        cards90=rate_models.cards90(f, position),
                        exp_bps=bonus_model.expected_bps(f, position, 90.0),
                        # ponytail: every squad shares one rotation shock at the league-average
                        # weight. Per-manager churn needs team XI history in FeatureCtx; the
                        # feature that claimed to supply it never produced a single value.
                    )
                )
        out.append(fi)
    return out, feature_cache


def _conditioned(rate90: float, team_lambda: float | None) -> float:
    scale = (team_lambda / rates_mod.LEAGUE_AVG_TEAM_XG) if team_lambda else 1.0
    return max(0.0, rate90 * scale)


def _minutes_for(artifact, ctx, features: dict, fixture_id: int, player_id: int):
    from ..connectors.lineups import confirmed_start

    suspended = any(a["status"] == "suspended" for a in ctx.availability)
    return minutes_mod.predict(
        artifact,
        features,
        suspended=suspended,
        confirmed_start=confirmed_start(fixture_id, player_id),
        fpl_chance=features.get("fpl_chance_of_playing"),
        disagreement=features.get("source_disagreement_score") or 0.0,
    )


def run(
    season_id: str, gameweek: int, n_sims: int | None = None, player_ids: list[int] | None = None,
    persist: bool = True,
) -> dict:
    """Full predict pass for one gameweek. Returns a summary dict for the CLI/API."""
    from ..config import get_settings

    n_sims = n_sims or get_settings().sim_iterations
    models = load_models()
    fixtures, _features = build_inputs(season_id, gameweek, models, player_ids)
    if not fixtures:
        log.warning("no PL fixtures for %s GW%s", season_id, gameweek)
        return {"players": 0, "fixtures": 0}

    sim = simulate_gameweek(fixtures, n_sims=n_sims)
    generated_at = _now()

    run_id = None
    if persist:
        with writer() as conn:
            cur = conn.execute(
                "INSERT INTO model_runs(started_at,finished_at,season_id,gameweek,models_json,"
                "n_sims) VALUES(?,?,?,?,?,?)",
                (generated_at, _now(), season_id, gameweek,
                 jdump({k: type(v).__name__ for k, v in models.items()}), n_sims),
            )
            run_id = cur.lastrowid

    rows = []
    for fx in fixtures:
        for p in fx.players:
            s = sim.summary(p.player_id)
            if not s:
                continue
            comp = sim.components.get(p.player_id, {})
            base = s["exp_points"]
            rows.append(
                {
                    "player_id": p.player_id,
                    "season_id": season_id,
                    "gameweek": gameweek,
                    "fixture_id": p.fixture_id,
                    "fixture_key": p.fixture_id,
                    "generated_at": generated_at,
                    "p_start": p.p_start,
                    "p_appear": p.p_start + p.p_cameo,
                    "exp_minutes": p.exp_minutes,
                    "exp_goals": float(comp["goals"].mean()) if "goals" in comp else None,
                    "exp_assists": float(comp["assists"].mean()) if "assists" in comp else None,
                    "p_clean_sheet": float(comp["clean_sheet"].mean())
                    if "clean_sheet" in comp else None,
                    "exp_saves": float(comp["saves"].mean()) if "saves" in comp else None,
                    "exp_defcon_points": float(comp["defcon_points"].mean())
                    if "defcon_points" in comp else None,
                    "exp_bonus": float(comp["bonus"].mean()) if "bonus" in comp else None,
                    "exp_cards_penalty": float(comp["cards_penalty"].mean())
                    if "cards_penalty" in comp else None,
                    "exp_conceded_penalty": float(comp["conceded_penalty"].mean())
                    if "conceded_penalty" in comp else None,
                    "exp_points": base,
                    "sd_points": s["sd_points"],
                    "p10": s["p10"],
                    "p50": s["p50"],
                    "p90": s["p90"],
                    "p_haul_10": s["p_haul_10"],
                    "p_blank_2": s["p_blank_2"],
                    "base_exp_points": base,
                    "adjustment": 0.0,
                    "adjustment_reason": None,
                    "model_run_id": run_id,
                }
            )

    if persist and rows:
        from ..db.engine import upsert_many

        with writer() as conn:
            upsert_many(
                conn, "predictions", rows,
                ["season_id", "gameweek", "player_id", "fixture_key", "generated_at"],
            )
        save_draws(sim, season_id, gameweek)

    # The bounded post-hoc text adjustment runs last, and only if enabled.
    if persist and global_settings().get("adjustment.enabled", True):
        from ..llm.adjust import apply_adjustments

        try:
            apply_adjustments(season_id, gameweek, generated_at)
        except Exception:
            log.exception("adjustment layer failed; predictions kept unadjusted")

    log.info("predicted %d player-fixtures for %s GW%s", len(rows), season_id, gameweek)
    return {
        "players": len(rows),
        "fixtures": len(fixtures),
        "generated_at": generated_at,
        "n_sims": n_sims,
        "bonus_regime_warning": models["bonus"].regime_warning
        if hasattr(models["bonus"], "regime_warning") else None,
    }


def latest(season_id: str, gameweek: int) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT p.* FROM predictions p JOIN ("
            "  SELECT player_id, fixture_key, MAX(generated_at) g FROM predictions "
            "  WHERE season_id=? AND gameweek=? GROUP BY player_id, fixture_key"
            ") x ON x.player_id=p.player_id AND x.fixture_key=p.fixture_key AND x.g=p.generated_at "
            "WHERE p.season_id=? AND p.gameweek=?",
            (season_id, gameweek, season_id, gameweek),
        )
    ]


def horizon_points(season_id: str, start_gw: int, n_gws: int) -> dict[int, list[float]]:
    """player_id -> expected points per gameweek over the horizon. DGW legs are summed."""
    out: dict[int, list[float]] = {}
    for i in range(n_gws):
        gw = start_gw + i
        for row in latest(season_id, gw):
            arr = out.setdefault(row["player_id"], [0.0] * n_gws)
            arr[i] += row["exp_points"] or 0.0
    return out


def prediction_for(player_id: int, season_id: str, gameweek: int) -> dict | None:
    row = query_one(
        "SELECT * FROM predictions WHERE player_id=? AND season_id=? AND gameweek=? "
        "ORDER BY generated_at DESC LIMIT 1",
        (player_id, season_id, gameweek),
    )
    return dict(row) if row else None
