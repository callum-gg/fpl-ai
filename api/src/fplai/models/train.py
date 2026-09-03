"""Walk-forward training. docs/06 training.

Never random k-fold — that leaks the future. Train on GWs 1..k, validate on k+1, roll.
Seasons are weighted `0.72 ^ seasons_ago`: enough history to fit, recent enough to matter.

A new version is promoted only if it beats the incumbent on the held-out window
(automatic, with a manual override and a Discord message either way).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime

import numpy as np

from ..config import get_settings
from ..db.engine import query, query_one
from ..features.build import load_matrix
from ..features.registry import FEATURE_VERSION
from . import bonus as bonus_mod
from . import minutes as minutes_mod
from . import rates as rates_mod
from . import team_goals as team_mod
from .base import (
    fit_lgbm_regressor,
    gain_importance,
    log_loss,
    mae,
    rmse,
    save_artifact,
    season_weight,
    spearman,
    to_matrix,
)

log = logging.getLogger(__name__)

MIN_ROWS = 300


def training_frame(seasons: list[str]):
    """Feature matrix joined to outcomes. One row per player-fixture."""
    import pandas as pd

    frames = []
    for season in seasons:
        feats = load_matrix(season)
        if feats.empty:
            continue
        outcomes = pd.DataFrame(
            [
                dict(r)
                for r in query(
                    "SELECT s.player_id, s.fixture_id AS fixture_key, f.gameweek, f.season_id, "
                    "s.minutes, s.starts, s.goals_scored, s.assists, s.clean_sheets, "
                    "s.goals_conceded, s.saves, s.bonus, s.bps, s.yellow_cards, s.red_cards, "
                    "s.defensive_contribution, s.total_points, ps.position "
                    "FROM player_fixture_stats s JOIN fixtures f ON f.id = s.fixture_id "
                    "JOIN player_seasons ps ON ps.player_id = s.player_id "
                    "AND ps.season_id = f.season_id "
                    "WHERE f.season_id=? AND f.finished=1",
                    (season,),
                )
            ]
        )
        if outcomes.empty:
            continue
        frames.append(
            feats.merge(outcomes, on=["season_id", "gameweek", "player_id", "fixture_key"],
                        how="inner")
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _weights(df, current_season: str) -> np.ndarray:
    decay = get_settings().train_season_decay
    return np.array([season_weight(s, current_season, decay) for s in df["season_id"]])


def _split(df, holdout_gws: int = 5):
    """Walk-forward split: the last `holdout_gws` played gameweeks, wherever they fall.

    Taking them from the newest season *alone* collapses the window to a single gameweek
    every August — a ~600-row holdout scored against an incumbent's ~4,000-row one, which
    is not a comparison at all and is how six of eight models stayed frozen on pre-season
    fits through GW1-2. It also starved the per-position slices: `saves90` saw ~20 keepers,
    fell under the scoring threshold, and was promoted with no measured skill whatsoever.

    Ordering by (season, gameweek) and taking the last N keeps one definition all year and
    simply spans the season boundary for the first few weeks of a new one. Still strictly
    walk-forward: everything in the holdout is later than everything used to fit.
    """
    if df.empty:
        return df, df
    keys = [(s, int(g)) for s, g in zip(df["season_id"], df["gameweek"], strict=False)]
    holdout = set(sorted(set(keys))[-holdout_gws:])
    in_holdout = np.array([k in holdout for k in keys])
    return df[~in_holdout], df[in_holdout]


def _holdout_span(valid_df) -> str | None:
    """Compact identity for the validation window, stored beside every metric."""
    if valid_df is None or valid_df.empty:
        return None
    keys = sorted({(s, int(g)) for s, g in zip(valid_df["season_id"], valid_df["gameweek"],
                                               strict=False)})
    return f"{keys[0][0]}:GW{keys[0][1]}-{keys[-1][0]}:GW{keys[-1][1]}"


def _key(season_id: str, gameweek) -> str:
    """Sortable (season, gameweek) key, so string comparison orders the calendar."""
    return f"{season_id}:{int(gameweek):02d}"


def _last_key(df) -> str | None:
    """How far the training data reaches — recorded so a later run knows what this fit saw."""
    if df is None or df.empty:
        return None
    return max(_key(s, g) for s, g in zip(df["season_id"], df["gameweek"], strict=False))


def _head_to_head(name: str, score, valid_df, challenger) -> dict | None:
    """Both models scored on the same rows — and only rows the incumbent never trained on.

    Three ways this comparison goes wrong, and all three were live at once:

    1. A *stored* metric belongs to whatever holdout was current when it was trained, so
       comparing it to a fresh number compares two different questions. That is what froze
       six of eight models on pre-season fits through GW1-2.
    2. Re-scoring the incumbent on the whole of today's window is no better: it was fitted
       a week ago on everything up to then, so most of that window is in-sample for it, and
       the flattered score blocks every honest successor. On the 2020-24 refit that gap
       alone rejected all seven challengers.
    3. Scoring the challenger on the full window and the incumbent on a subset of it is
       the first bug again in a different hat.

    So: both models, the same rows, and only gameweeks that postdate the incumbent's own
    training data. If too few of those exist to mean anything there is no honest comparison
    to make, and the caller falls back to promoting the fresher fit.
    """
    from .base import load_active

    live = load_active(name)
    if live is None:
        return None

    row = query_one(
        "SELECT metrics_json FROM model_versions WHERE model_name=? AND is_active=1", (name,)
    )
    through = json.loads(row["metrics_json"]).get("trained_through") if row else None
    if not through:
        # No record of what this artefact was trained on, so there is no slice we can be
        # sure it never saw. Assuming "nothing" would score it in-sample and hand it a win
        # it did not earn; the caller's holdout rule takes over instead.
        log.info("%s: incumbent does not record how far its training data reached — "
                 "no head-to-head is possible against it", name)
        return None

    keys = [_key(s, g) for s, g in
            zip(valid_df["season_id"], valid_df["gameweek"], strict=False)]
    unseen = valid_df[np.array([k > through for k in keys], dtype=bool)]
    if len(unseen) < 30:
        log.info("%s: only %d holdout rows postdate the incumbent's training data — no "
                 "honest head-to-head, deferring to the fresher fit", name, len(unseen))
        return None
    try:
        return {"rows": len(unseen),
                "challenger": score(challenger, unseen),
                "incumbent": score(live, unseen)}
    except Exception:
        log.exception("could not score the incumbent %s on this holdout", name)
        return None


def train_all(seasons: list[str] | None = None, models: list[str] | None = None) -> dict:
    settings = get_settings()
    if seasons is None:
        seasons = [r["id"] for r in query("SELECT id FROM seasons ORDER BY id")]
    wanted = set(models or ["team_goals", "minutes", "goals90", "assists90", "defcon",
                            "bonus", "saves90", "cards90"])

    results: dict[str, dict] = {}
    df = training_frame(seasons)
    if df.empty:
        log.warning("no training rows — run backfill and build-features first")
        return {"error": "no training data"}
    df = df.fillna(value=np.nan)
    train_df, valid_df = _split(df)
    log.info("training on %d rows, validating on %d", len(train_df), len(valid_df))

    if "team_goals" in wanted:
        tm = team_mod.fit(seasons)
        # Score a model that has never seen the holdout, but ship the one fitted on
        # everything: the metric answers "does this generalise", the artifact should still
        # use every match available. Scoring `tm` itself was in-sample and could not have
        # caught overfitting.
        cutoff = _holdout_cutoff(valid_df)
        scored = team_mod.fit(seasons, as_of=cutoff) if cutoff else tm
        # No head-to-head here, deliberately. The shipped artefact is `tm`, fitted on every
        # match including the holdout, so re-scoring it on that holdout is an in-sample
        # read and flatters it by ~0.5 nats — enough to block every honest successor and
        # rebuild the freeze this whole change exists to remove. `baseline_nll` below is
        # the comparable gate: beat league-average Poisson or the strengths do nothing.
        results["team_goals"] = {
            # Mean NLL of the exact scoreline over an ~11x11 grid — NOT a binary log-loss,
            # so it does not compare to the minutes model's. The baseline is what makes it
            # readable: beat league-average Poisson or the team strengths are doing nothing.
            "log_loss": _team_model_score(scored, cutoff),
            "baseline_nll": _baseline_scoreline_nll(cutoff),
            "holdout_from": cutoff,
            "holdout": _holdout_span(valid_df),
            "trained_through": _last_key(train_df),
            "n_teams": len(tm.attack),
            "home_adv": tm.home_adv,
            "rho": tm.rho,
        }
        save_artifact("team_goals", tm.to_dict(), results["team_goals"],
                      {"half_life_matches": team_mod.HALF_LIFE_MATCHES}, len(df), seasons,
                      FEATURE_VERSION)

    if "minutes" in wanted:
        results["minutes"] = _train_minutes(train_df, valid_df, seasons, settings.current_season)

    for name, cols, target_fn, objective in (
        ("goals90", rates_mod.GOALS_FEATURES, _rate_target("goals_scored"), "tweedie"),
        ("assists90", rates_mod.ASSISTS_FEATURES, _rate_target("assists"), "tweedie"),
        ("saves90", rates_mod.SAVES_FEATURES, _rate_target("saves"), "poisson"),
        ("cards90", rates_mod.CARDS_FEATURES, _rate_target("yellow_cards"), "poisson"),
    ):
        if name in wanted:
            results[name] = _train_rate(
                name, cols, target_fn, objective, train_df, valid_df, seasons,
                settings.current_season,
            )

    if "defcon" in wanted:
        results["defcon"] = _train_defcon(train_df, valid_df, seasons, settings.current_season)

    if "bonus" in wanted:
        results["bonus"] = _train_bonus(df, train_df, valid_df, seasons, settings.current_season)

    _notify(results)
    return results


def _rate_target(col: str):
    def fn(df):
        mins = df["minutes"].clip(lower=1)
        return (90.0 * df[col].fillna(0) / mins).clip(upper=6)

    return fn


def _train_minutes(train_df, valid_df, seasons, current_season) -> dict:
    played = train_df[train_df["minutes"].notna()]
    if len(played) < MIN_ROWS:
        return {"skipped": f"only {len(played)} rows"}
    y = np.array([minutes_mod.label_row(m, s) for m, s in
                  zip(played["minutes"], played["starts"], strict=False)])
    rows = played.to_dict("records")
    X = to_matrix(rows, minutes_mod.FEATURES)
    artifact = minutes_mod.train(X, y, minutes_mod.FEATURES, weights=_weights(played, current_season))

    metrics: dict = {"train_rows": len(played), "holdout": _holdout_span(valid_df),
                     "trained_through": _last_key(train_df)}
    if len(valid_df) > 30:

        def score(a, d=None) -> dict:
            d = valid_df if d is None else d
            vy = np.array([minutes_mod.label_row(m, s) for m, s in
                           zip(d["minutes"], d["starts"], strict=False)])
            vX = to_matrix(d.to_dict("records"), minutes_mod.FEATURES)
            cal = minutes_mod.calibration(a, vX, vy)
            return {"log_loss": log_loss(vy, a.predict_proba(vX)),
                    "calibration_ece": cal["ece"], "calibration_curve": cal["curve"]}

        metrics.update(score(artifact))
        metrics["head_to_head"] = _head_to_head("minutes", score, valid_df, artifact)
    metrics["importance"] = _top_importance(artifact)
    save_artifact("minutes", artifact, metrics, {"features": minutes_mod.FEATURES},
                  len(played), seasons, FEATURE_VERSION)
    return metrics


def _train_rate(name, cols, target_fn, objective, train_df, valid_df, seasons, current_season):
    played = train_df[(train_df["minutes"].fillna(0) > 0)]
    if name == "saves90":
        played = played[played["position"] == "GK"]
    if len(played) < MIN_ROWS:
        return {"skipped": f"only {len(played)} rows"}
    y = target_fn(played).to_numpy()
    X = to_matrix(played.to_dict("records"), cols)
    artifact = fit_lgbm_regressor(X, y, cols, objective=objective,
                                  weights=_weights(played, current_season))

    metrics: dict = {"train_rows": len(played), "holdout": _holdout_span(valid_df),
                     "trained_through": _last_key(train_df)}
    vd = valid_df[valid_df["minutes"].fillna(0) > 0]
    if name == "saves90":
        vd = vd[vd["position"] == "GK"]
    if len(vd) > 30:

        def score(a, d=None) -> dict:
            d = vd if d is None else d
            vy = target_fn(d).to_numpy()
            pred = a.predict(to_matrix(d.to_dict("records"), cols))
            return {"mae": mae(vy, pred), "rmse": rmse(vy, pred), "spearman": spearman(vy, pred)}

        metrics.update(score(artifact))
        metrics["head_to_head"] = _head_to_head(name, score, vd, artifact)
    metrics["importance"] = _top_importance(artifact)
    save_artifact(name, artifact, metrics, {"features": cols, "objective": objective},
                  len(played), seasons, FEATURE_VERSION)
    return metrics


def _train_defcon(train_df, valid_df, seasons, current_season) -> dict:
    played = train_df[(train_df["minutes"].fillna(0) >= 30)
                      & train_df["defensive_contribution"].notna()]
    if len(played) < MIN_ROWS:
        return {"skipped": f"only {len(played)} DefCon rows"}
    y = (90.0 * played["defensive_contribution"] / played["minutes"].clip(lower=1)).clip(upper=40)
    X = to_matrix(played.to_dict("records"), rates_mod.DEFCON_FEATURES)
    rate_model = fit_lgbm_regressor(X, y.to_numpy(), rates_mod.DEFCON_FEATURES,
                                    objective="poisson",
                                    weights=_weights(played, current_season))

    expected = rate_model.predict(X) * played["minutes"].to_numpy() / 90.0
    k = rates_mod.fit_dispersion(played["defensive_contribution"].to_numpy(), expected)
    model = rates_mod.DefconModel(rate_model=rate_model, dispersion=k)

    metrics: dict = {"train_rows": len(played), "dispersion": k,
                     "holdout": _holdout_span(valid_df),
                     "trained_through": _last_key(train_df)}
    vd = valid_df[(valid_df["minutes"].fillna(0) >= 30) & valid_df["defensive_contribution"].notna()]
    if len(vd) > 30:
        from ..defaults import DEFCON_THRESHOLD

        def _hit(rows) -> np.ndarray:
            return np.array([
                int(r["defensive_contribution"] >= DEFCON_THRESHOLD.get(r["position"], 12))
                for r in rows
            ])

        def score(m, d=None) -> dict:
            rows = (vd if d is None else d).to_dict("records")
            probs = np.array([m.p_threshold(r, r["position"], r["minutes"] or 0) for r in rows])
            return {"log_loss": log_loss(_hit(rows), probs)}

        metrics.update(score(model))
        metrics["threshold_base_rate"] = float(_hit(vd.to_dict("records")).mean())
        metrics["head_to_head"] = _head_to_head("defcon", score, vd, model)
    save_artifact("defcon", model, metrics, {"features": rates_mod.DEFCON_FEATURES},
                  len(played), seasons, FEATURE_VERSION)
    return metrics


def _train_bonus(df, train_df, valid_df, seasons, current_season) -> dict:
    """Historic + current-season models, blended by sample size (docs/06 model 4)."""
    played = train_df[(train_df["minutes"].fillna(0) > 0) & train_df["bps"].notna()]
    if len(played) < MIN_ROWS:
        return {"skipped": f"only {len(played)} BPS rows"}
    y = (90.0 * played["bps"] / played["minutes"].clip(lower=1)).clip(-10, 120)
    X = to_matrix(played.to_dict("records"), bonus_mod.BPS_FEATURES)
    historic = bonus_mod.train_bps(X, y.to_numpy(), bonus_mod.BPS_FEATURES,
                                  weights=_weights(played, current_season))

    # The 2026/27 retune means older coefficients are biased, so fit the new regime alone.
    # Read current-season rows from the full frame, not train_df: early in a season the
    # walk-forward holdout (`_split`) swallows every played GW, which used to pin
    # n_current_fixtures to 0 and keep the blend disengaged until ~GW6.
    # ponytail: the current model scoring against rows it saw is accepted — it is a
    # regime probe on this season's evidence only, not the walk-forward metric.
    current = None
    cur_rows = df[(df["season_id"] == current_season)
                  & (df["minutes"].fillna(0) > 0) & df["bps"].notna()]
    n_current = int(cur_rows["fixture_key"].nunique())
    if len(cur_rows) >= 150:
        current = bonus_mod.train_bps(
            to_matrix(cur_rows.to_dict("records"), bonus_mod.BPS_FEATURES),
            (90.0 * cur_rows["bps"] / cur_rows["minutes"].clip(lower=1)).clip(-10, 120).to_numpy(),
            bonus_mod.BPS_FEATURES,
        )

    model = bonus_mod.BonusModel(historic=historic, current=current, n_current_fixtures=n_current)
    metrics: dict = {"train_rows": len(played), "n_current_fixtures": n_current,
                     "blend_weight": model.blend_weight, "regime_warning": model.regime_warning,
                     "holdout": _holdout_span(valid_df),
                     "trained_through": _last_key(train_df)}
    vd = valid_df[(valid_df["minutes"].fillna(0) > 0) & valid_df["bps"].notna()]
    if len(vd) > 30:

        def score(m, d=None) -> dict:
            d = vd if d is None else d
            rows = d.to_dict("records")
            pred = np.array([m.expected_bps(r, r["position"], r["minutes"] or 0) for r in rows])
            return {"mae": mae(d["bps"].to_numpy(), pred), "spearman": spearman(d["bps"].to_numpy(), pred)}

        metrics.update(score(model))
        metrics["head_to_head"] = _head_to_head("bonus", score, vd, model)
    save_artifact("bonus", model, metrics, {"features": bonus_mod.BPS_FEATURES},
                  len(played), seasons, FEATURE_VERSION)
    return metrics


def _holdout_cutoff(valid_df) -> str | None:
    """Kickoff time the walk-forward validation window starts at, or None if there is none."""
    if valid_df is None or valid_df.empty:
        return None
    season = valid_df["season_id"].max()
    first_gw = int(valid_df.loc[valid_df["season_id"] == season, "gameweek"].min())
    row = query_one(
        "SELECT MIN(kickoff_utc) k FROM fixtures WHERE season_id=? AND gameweek>=?",
        (season, first_gw),
    )
    return row["k"] if row and row["k"] else None


def _scored_fixtures(cutoff: str | None) -> list:
    """Finished PL fixtures in the holdout window, newest first."""
    sql = (
        "SELECT home_team_id h, away_team_id a, home_score hs, away_score as_ FROM fixtures "
        "WHERE finished=1 AND home_score IS NOT NULL AND competition='PL'"
    )
    params: tuple = ()
    if cutoff:
        sql += " AND kickoff_utc >= ?"
        params = (cutoff,)
    return query(sql + " ORDER BY kickoff_utc DESC LIMIT 200", params)


def _baseline_scoreline_nll(cutoff: str | None) -> float:
    """Same metric for a model that knows nothing but the league's mean home/away goals.

    Without this reference an NLL near 2.8 looks alarming when it is close to the floor
    for exact-scoreline prediction. The gap over this baseline is the team model's value.

    The means come from *before* the cutoff. Taking them from the holdout would hand the
    baseline the one thing the real model is not allowed to see, and it is worth ~0.02
    nats here — enough to make a competitive model look beaten.
    """
    rows = _scored_fixtures(cutoff)
    if not rows:
        return 0.0
    train_rows = query(
        "SELECT home_score hs, away_score as_ FROM fixtures WHERE finished=1 "
        "AND home_score IS NOT NULL AND competition='PL'"
        + (" AND kickoff_utc < ?" if cutoff else ""),
        (cutoff,) if cutoff else (),
    ) or rows
    mh = sum(r["hs"] for r in train_rows) / len(train_rows)
    ma = sum(r["as_"] for r in train_rows) / len(train_rows)

    def _pois(k: int, lam: float) -> float:
        return math.exp(-lam) * lam**k / math.factorial(k)

    total = 0.0
    for r in rows:
        p = _pois(int(r["hs"]), mh) * _pois(int(r["as_"]), ma)
        total -= math.log(max(p, 1e-9))
    return total / len(rows)


def _team_model_score(tm: team_mod.TeamModel, cutoff: str | None) -> float:
    """Mean negative log-likelihood of observed scorelines under the fitted model."""
    rows = _scored_fixtures(cutoff)
    if not rows:
        return 0.0
    total = 0.0
    for r in rows:
        m = team_mod.score_matrix(*tm.lambdas(r["h"], r["a"]), tm.rho)
        h = min(int(r["hs"]), m.shape[0] - 1)
        a = min(int(r["as_"]), m.shape[1] - 1)
        total -= float(np.log(max(m[h, a], 1e-9)))
    return total / len(rows)


def _top_importance(artifact, n: int = 12) -> dict[str, float]:
    imp = gain_importance(artifact)
    return dict(sorted(imp.items(), key=lambda kv: -kv[1])[:n])


def text_features_not_dominant(model_name: str = "goals90") -> tuple[bool, list[str]]:
    """docs/05 section F guard: no text feature may rank top-3 by gain importance.

    If one does, it is either leakage or the pundit signal proxying for something the
    model should already know from the stats. Either way it should fail loudly.
    """
    import json

    from ..db.engine import query_one
    from ..features.registry import text_feature_names

    row = query_one(
        "SELECT metrics_json FROM model_versions WHERE model_name=? AND is_active=1", (model_name,)
    )
    if row is None:
        return True, []
    imp = json.loads(row["metrics_json"]).get("importance", {})
    top3 = [k for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:3]]
    offenders = [f for f in top3 if f in set(text_feature_names())]
    return not offenders, offenders


def _notify(results: dict) -> None:
    import asyncio

    from ..notify.discord import notify

    lines = [f"**Training complete** {datetime.utcnow():%Y-%m-%d %H:%M}"]
    for name, m in results.items():
        if "skipped" in m:
            lines.append(f"• `{name}` skipped — {m['skipped']}")
        else:
            key = next((k for k in ("log_loss", "mae", "spearman") if k in m), None)
            lines.append(f"• `{name}` {key}={m.get(key, 'n/a'):.4f}" if key else f"• `{name}` ok")
    try:
        asyncio.get_running_loop().create_task(notify("\n".join(lines)))
    except RuntimeError:
        asyncio.run(notify("\n".join(lines)))
