"""Model 3 — per-90 rate models, and the DefCon threshold model. docs/06.

Rates are predicted per 90, then scaled by expected minutes and conditioned on the
fixture via `team_expected_goals / team_avg_xg`.

DefCon gets a **threshold classifier**, not a mean. The payoff is a step function, so
modelling the mean and thresholding it is measurably worse; a negative binomial count
model lets us integrate over the threshold exactly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ..defaults import DEFCON_THRESHOLD
from .base import RegressorArtifact, fit_lgbm_regressor

log = logging.getLogger(__name__)

LEAGUE_AVG_TEAM_XG = 1.42

GOALS_FEATURES = [
    "xg90_last3", "xg90_last5", "xg90_last10", "npxg90_last5", "npxg90_last10",
    "shots90", "sot90", "touches_box90", "big_chances90", "conversion_ratio",
    "xg_overperformance_last10", "is_first_pen_taker", "pens_taken_share",
    "team_expected_goals", "opponent_defence_rating", "is_home",
    "p_anytime_scorer_odds", "opp_goals_conceded_ewma", "age",
    "transfermarkt_value_pct_of_squad", "promoted_club_flag", "block_odds_present",
    "price_rank_in_position",
]

ASSISTS_FEATURES = [
    "xa90_last3", "xa90_last5", "xa90_last10", "key_passes90", "xgi90_last5",
    "corner_duty", "direct_fk_duty", "team_expected_goals", "opponent_defence_rating",
    "is_home", "opp_goals_conceded_ewma", "block_odds_present",
    "price_rank_in_position",
]

DEFCON_FEATURES = [
    "defcon_actions90", "defcon_hit_rate_last5", "defcon_hit_rate_last10", "defcon_margin",
    "is_home", "opponent_attack_rating", "opp_expected_goals", "mins_last5_weighted",
    "start_streak", "block_advanced_present",
]

SAVES_FEATURES = [
    "saves90", "saves_per_shot_faced", "opp_expected_goals", "opponent_attack_rating",
    "is_home", "opp_goals_conceded_ewma",
]

CARDS_FEATURES = ["card_rate90", "is_home", "derby_flag", "opponent_attack_rating", "age"]


def train_rate(X, y, feature_names: list[str], weights=None, objective: str = "tweedie"):
    return fit_lgbm_regressor(X, y, feature_names, objective=objective, weights=weights)


def scale_to_fixture(rate90: float, exp_minutes: float, team_lambda: float | None) -> float:
    """Rate per 90 -> expected count for this fixture, conditioned on the team's goal
    expectation so a good matchup lifts the whole attack, not just the model's guess."""
    scale = (team_lambda / LEAGUE_AVG_TEAM_XG) if team_lambda else 1.0
    return max(0.0, rate90 * (exp_minutes / 90.0) * scale)


def blend_with_scorer_market(
    model_p_scores: float, market_p_scores: float | None, w: float = 0.5
) -> float:
    """The market's P(anytime scorer) is a strong calibration target for goals90 x minutes."""
    if market_p_scores is None:
        return model_p_scores
    a = max(1e-6, min(1 - 1e-6, model_p_scores))
    b = max(1e-6, min(1 - 1e-6, market_p_scores))
    return float(a ** (1 - w) * b**w)


# --- DefCon threshold model -----------------------------------------------------


@dataclass
class DefconModel:
    """Negative binomial over DefCon action counts, so P(>= threshold) is exact.

    `dispersion` is the NB shape parameter k; larger means closer to Poisson. Fitted by
    method of moments on residual variance, which is plenty for a count this noisy.
    """

    rate_model: RegressorArtifact | None
    dispersion: float = 6.0

    def expected_actions(self, features: dict, exp_minutes: float) -> float:
        rate90 = (
            self.rate_model.predict_one(features)
            if self.rate_model is not None
            else (features.get("defcon_actions90") or 0.0)
        )
        return max(0.0, rate90 * (exp_minutes / 90.0))

    def p_threshold(self, features: dict, position: str, exp_minutes: float) -> float:
        mu = self.expected_actions(features, exp_minutes)
        thr = DEFCON_THRESHOLD.get(position, 12)
        return nb_survival(mu, self.dispersion, thr)


def nb_survival(mu: float, k: float, threshold: int) -> float:
    """P(X >= threshold) for a negative binomial with mean mu and dispersion k."""
    if mu <= 0:
        return 0.0
    if threshold <= 0:
        return 1.0
    p = k / (k + mu)
    cdf = 0.0
    log_p_k = k * math.log(p)
    term = math.exp(log_p_k)  # P(X=0)
    for x in range(threshold):
        cdf += term
        term *= (k + x) / (x + 1) * (1 - p)
    return max(0.0, min(1.0, 1.0 - cdf))


def nb_sample(rng: np.random.Generator, mu, k: float, size: int) -> np.ndarray:
    """Gamma-Poisson mixture, which is what a negative binomial is.

    `mu` may be a scalar or one rate per draw. Per draw matters: a player whose minutes
    vary across simulations has a different rate in each, and P(X >= threshold) is convex
    in mu over the DefCon range, so collapsing the vector to its mean under-states the
    threshold probability — measured 1.07x for a nailed starter and 2.55x for a rotation
    risk, which is most of why expected DefCon came in at half the observed rate.
    """
    mu = np.asarray(mu, dtype=float)
    if mu.ndim == 0:
        mu = np.full(size, float(mu))
    out = np.zeros(size, dtype=int)
    live = mu > 0
    if live.any():
        out[live] = rng.poisson(rng.gamma(shape=k, scale=mu[live] / k))
    return out


def fit_dispersion(counts: np.ndarray, means: np.ndarray) -> float:
    """Method-of-moments k from Var = mu + mu^2/k."""
    counts, means = np.asarray(counts, dtype=float), np.asarray(means, dtype=float)
    mask = means > 0.5
    if mask.sum() < 30:
        return 6.0
    var = float(np.var(counts[mask]))
    mu = float(np.mean(means[mask]))
    excess = var - mu
    if excess <= 0:
        return 50.0  # essentially Poisson
    return float(max(1.0, min(50.0, mu**2 / excess)))


# --- rate model bundle ----------------------------------------------------------


@dataclass
class RateModels:
    goals: RegressorArtifact | None = None
    assists: RegressorArtifact | None = None
    defcon: DefconModel | None = None
    saves: RegressorArtifact | None = None
    cards: RegressorArtifact | None = None

    def goals90(self, f: dict, position: str) -> float:
        if self.goals is not None:
            return max(0.0, self.goals.predict_one(f))
        observed = f.get("xg90_last5") or f.get("xg90_last10")
        return float(observed) if observed else _cold_start_prior(f, position, "goals")

    def assists90(self, f: dict, position: str) -> float:
        if self.assists is not None:
            return max(0.0, self.assists.predict_one(f))
        observed = f.get("xa90_last5") or f.get("xa90_last10")
        return float(observed) if observed else _cold_start_prior(f, position, "assists")

    def saves90(self, f: dict, position: str) -> float:
        if position != "GK":
            return 0.0
        if self.saves is not None:
            return max(0.0, self.saves.predict_one(f))
        return float(f.get("saves90") or 2.8)

    def cards90(self, f: dict, position: str) -> float:
        if self.cards is not None:
            return max(0.0, self.cards.predict_one(f))
        return float(f.get("card_rate90") or 0.15)


_PRIORS = {
    "goals": {"GK": 0.002, "DEF": 0.05, "MID": 0.14, "FWD": 0.33},
    "assists": {"GK": 0.005, "DEF": 0.06, "MID": 0.14, "FWD": 0.11},
}


def _prior(position: str, kind: str) -> float:
    return _PRIORS[kind].get(position, 0.1)


def _cold_start_prior(f: dict, position: str, kind: str) -> float:
    """Positional prior scaled by price percentile when there is no history to read.

    Spread is 0.4x to 2.2x the positional mean, which roughly matches the observed gap
    between the cheapest and most expensive players in a position.
    """
    base = _prior(position, kind)
    pct = f.get("price_rank_in_position")
    if pct is None:
        return base
    return base * (0.4 + 1.8 * float(pct))


# Rare events stay as fixed empirical rates rather than fitted models — the data is too
# thin to fit them and pretending otherwise adds noise, not signal (docs/06).
EMPIRICAL_RATES = {
    "own_goal_per_90": 0.006,
    "pen_missed_per_pen": 0.22,
    "pen_saved_per_90_gk": 0.012,
    "red_card_per_90": 0.010,
    "penalty_won_per_90_fwd": 0.03,
}
