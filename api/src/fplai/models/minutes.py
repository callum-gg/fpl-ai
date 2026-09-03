"""Model 2 — minutes. The one that decides whether the app is any good.

Three-class LightGBM (start / bench_with_cameo / no_appearance) plus conditional
minutes regressors, isotonic-calibrated on held-out gameweeks.

Hard overrides are applied *after* the model, never learned:
  suspended                 -> P(start) = 0
  confirmed lineup exists   -> collapse to the truth
  FPL flag 0%               -> P(appear) capped at 0.02

An uncalibrated minutes model quietly destroys everything downstream, so ECE is
reported alongside log-loss and the calibration curve is served to the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import ClassifierArtifact, ece, expected_calibration_curve, fit_lgbm_multiclass

log = logging.getLogger(__name__)

CLASSES = ("no_appearance", "cameo", "start")

# Monotonic constraints where the direction is obvious and we want it enforced, not hoped for.
MONOTONE = {
    "start_streak": 1,
    "starts_last5": 1,
    "mins_last5_weighted": 1,
    "injury_status_consensus": 1,
    "predicted_lineup_prob": 1,
    "fpl_chance_of_playing": 1,
    "sub_appearance_rate": -1,
    "team_matches_next_14_days": -1,
}

FEATURES = [
    "start_streak", "starts_last5", "mins_last1", "mins_last3", "mins_last5",
    "mins_last5_weighted", "mins_last10", "sub_appearance_rate", "avg_sub_on_minute",
    "injury_status_consensus", "fpl_chance_of_playing", "source_disagreement_score",
    "days_since_injury_report", "news_signal_gap", "predicted_lineup_prob",
    "days_since_last_match", "player_minutes_last_7_days", "player_minutes_last_14_days",
    "team_matches_next_14_days", "midweek_european_flag", "european_competition_tier",
    "international_break_flag", "new_manager_flag",
    "owned_pct", "is_template_flag", "kickoff_slot", "age",
    # Price is the only quality signal that exists before a ball is kicked, and
    # FPL prices it as a nailedness proxy too (docs/06 model 2).
    "price", "price_rank_in_position",
    "x_breaking_news_flag", "yt_mention_count", "block_text_present", "block_odds_present",
]


@dataclass
class MinutesPrediction:
    p_no_appearance: float
    p_cameo: float
    p_start: float
    exp_minutes: float

    @property
    def p_appear(self) -> float:
        return self.p_cameo + self.p_start


def label_row(minutes: int | None, started: int | None) -> int:
    """0 no appearance, 1 cameo, 2 start."""
    m = minutes or 0
    if m == 0:
        return 0
    if started or m >= 60:
        return 2
    return 1


def train(X, y, feature_names: list[str], weights=None) -> ClassifierArtifact:
    constraints = [MONOTONE.get(f, 0) for f in feature_names]
    return fit_lgbm_multiclass(
        X, y, feature_names, n_classes=3, monotone=constraints, weights=weights,
        params={"num_leaves": 48, "learning_rate": 0.05, "min_data_in_leaf": 60,
                "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1},
    )


EXP_MINUTES_IF_START = 78.0   # starters get subbed; this is the empirical PL mean
EXP_MINUTES_IF_CAMEO = 19.0


def predict(
    artifact: ClassifierArtifact | None,
    features: dict[str, float | None],
    *,
    suspended: bool = False,
    confirmed_start: bool | None = None,
    fpl_chance: float | None = None,
    disagreement: float = 0.0,
) -> MinutesPrediction:
    """Model probabilities, then the hard overrides. Order matters: truth wins."""
    if _between_seasons(features):
        probs = _spread(_preseason_p_start(features))
    elif artifact is None:
        probs = _heuristic(features)
    else:
        probs = artifact.predict_one(features)

    p_no, p_cameo, p_start = float(probs[0]), float(probs[1]), float(probs[2])

    # 1. Suspension is absolute.
    if suspended:
        return MinutesPrediction(1.0, 0.0, 0.0, 0.0)

    # 2. A confirmed lineup is truth, not evidence — collapse to it.
    if confirmed_start is True:
        return MinutesPrediction(0.0, 0.05, 0.95, 0.95 * EXP_MINUTES_IF_START + 0.05 * 25)
    if confirmed_start is False:
        return MinutesPrediction(0.55, 0.42, 0.03, 0.42 * EXP_MINUTES_IF_CAMEO + 0.03 * 45)

    # 3. FPL's own flag at 0% caps appearance probability.
    if fpl_chance is not None and fpl_chance <= 0:
        p_appear = min(p_cameo + p_start, 0.02)
        scale = p_appear / max(p_cameo + p_start, 1e-9)
        p_cameo, p_start = p_cameo * scale, p_start * scale
        p_no = 1 - p_appear

    # 4. Source disagreement widens the distribution rather than shifting it: probability
    #    mass moves from the confident tails toward the uncertain middle (docs/05 section D).
    if disagreement > 0:
        blend = min(0.4, disagreement)
        p_start = p_start * (1 - blend) + (p_start + p_cameo) / 2 * blend
        p_cameo = p_cameo * (1 - blend) + (p_start + p_cameo) / 2 * blend
        total = p_no + p_cameo + p_start
        p_no, p_cameo, p_start = p_no / total, p_cameo / total, p_start / total

    exp_min = p_start * EXP_MINUTES_IF_START + p_cameo * EXP_MINUTES_IF_CAMEO
    return MinutesPrediction(p_no, p_cameo, p_start, exp_min)


def _between_seasons(f: dict[str, float | None]) -> bool:
    """True when no match has been played recently enough for the trained model to read.

    `player_minutes_last_7_days` and `..._14_days` are the model's two strongest splits,
    and before a ball is kicked they are zero for every player alike — so the booster
    reads the whole league as doubtful and ranks a nailed-on £4.5m defender above a
    £15.5m striker. Last season's form plus price is a worse model but an honest one.

    The question is whether *the league* has played, not whether *he* has. Asked with his
    own minutes it answers yes for every unused substitute, because an unplayed GW1 leaves
    him with zero minutes in 14 days and a last appearance back in May — so the one player
    the evidence has just spoken loudest about is the one routed past the model, onto a
    price-led prior that ranks him as nailed. `days_since_team_match` is the discriminator:
    his club played on Saturday, so this is a team-sheet decision, not the close season.

    ponytail: a 30-day gap, not a fixture-calendar lookup. Nothing short of the summer
    break leaves a whole club that idle, and a genuine long absentee is caught by
    `injury_status_consensus` inside the heuristic anyway.
    """
    team_idle = f.get("days_since_team_match")
    if team_idle is not None and team_idle <= 30:
        return False
    return (f.get("player_minutes_last_14_days") or 0) == 0 and (
        f.get("days_since_last_match") or 0
    ) > 30


def _heuristic(f: dict[str, float | None]) -> np.ndarray:
    """Cold-start fallback before the model is trained, so the app works on day one.

    With match history it leans on start streak and recent minutes. With none — GW1 of a
    new season — those are all zero, so it falls back to price percentile within position.
    FPL's own pricing is the only quality signal that exists before a ball is kicked, and
    without it a £4.5m bench forward ranks alongside a £15.5m striker.
    """
    streak = f.get("start_streak") or 0
    mins5 = f.get("mins_last5") or 0
    avail = f.get("injury_status_consensus")
    avail = 1.0 if avail is None else avail

    form_signal = 0.12 * min(streak, 5) + (mins5 / 450) * 0.55
    price_pct = f.get("price_rank_in_position")
    if form_signal <= 0 and price_pct is not None:
        # No history at all: price percentile spread over a plausible starter range.
        p_start = (0.10 + 0.80 * float(price_pct)) * avail
    else:
        p_start = min(0.95, 0.12 + form_signal) * avail

    return _spread(p_start)


def _spread(p_start: float) -> np.ndarray:
    """One start probability -> the three-class distribution the caller expects."""
    p_start = max(0.01, min(0.95, p_start))
    p_cameo = min(0.5, (1 - p_start) * 0.45)
    return np.array([max(0.01, 1 - p_start - p_cameo), p_cameo, p_start])


# Weight on last season's record after a normal summer, versus FPL's own pricing. Below
# 100 days idle is just the close season; a year out and the record means almost nothing.
_HISTORY_WEIGHT_FRESH = 0.6
_HISTORY_WEIGHT_STALE = 0.15
_STALE_AFTER_DAYS = 365.0


def _preseason_p_start(f: dict[str, float | None]) -> float:
    """Start probability before a ball is kicked, when no recent match exists to read.

    Three things a human weighs and the trained model cannot see in July:

    1. *How long since he played.* Ninety days is just the summer and last season still
       counts. A year means a lost season, a foreign league or no top-flight record at
       all, and the only honest signal left is what FPL priced him at.
    2. *Is he being rested, or dropped?* `start_streak` is the strongest predictor in
       March and the worst one in July, because it is zero for everyone rested in a dead
       rubber — Haaland and Saka both end last season on nil. So the pre-season read
       ignores the streak and asks the rate question instead: of his last five, how many
       did he start, and how much of the last ten games' minutes did he actually play.
    3. *Has the manager changed?* Nine did this summer. Last season's teamsheet is a
       weaker guide under a new manager, so lean further on price.

    Rate over streak is the whole point: three starts in five with a full pre-season ahead
    is a nailed striker who was wrapped in cotton wool in May, not a rotation risk.
    """
    avail = f.get("injury_status_consensus")
    avail = 1.0 if avail is None else float(avail)

    price_pct = f.get("price_rank_in_position")
    price_prior = 0.10 + 0.80 * float(price_pct) if price_pct is not None else 0.5

    days = f.get("days_since_last_match")
    starts5, mins10 = f.get("starts_last5"), f.get("mins_last10")
    if days is None or (starts5 is None and mins10 is None):
        # Never played a top-flight match we hold: a new signing or a promoted side's
        # player. Price is the only thing anyone knows about him either.
        return max(0.01, min(0.95, price_prior * avail))

    # Two views of the same question, because each misses something on its own: a starter
    # hooked on 60 minutes every week looks part-time by minutes, and a regular substitute
    # who plays 45 every week looks like a starter by minutes but starts nothing.
    start_rate = (float(starts5) / 5.0) if starts5 is not None else None
    mins_share = (float(mins10) / 900.0) if mins10 is not None else None
    if start_rate is None:
        history = mins_share
    elif mins_share is None:
        history = start_rate
    else:
        history = 0.55 * start_rate + 0.45 * mins_share

    stale = (float(days) - 100.0) / (_STALE_AFTER_DAYS - 100.0)
    weight = _HISTORY_WEIGHT_FRESH - (_HISTORY_WEIGHT_FRESH - _HISTORY_WEIGHT_STALE) * stale
    weight = max(_HISTORY_WEIGHT_STALE, min(_HISTORY_WEIGHT_FRESH, weight))
    if f.get("new_manager_flag"):
        weight *= 0.75

    p_start = (weight * min(1.0, history) + (1 - weight) * price_prior) * avail
    return max(0.01, min(0.95, p_start))


def calibration(artifact: ClassifierArtifact, X, y) -> dict:
    probs = artifact.predict_proba(X)
    p_start = probs[:, 2]
    started = (np.asarray(y) == 2).astype(int)
    return {
        "ece": ece(started, p_start),
        "curve": expected_calibration_curve(started, p_start),
    }
