"""Model 4 — bonus via simulated BPS ranking. docs/06.

Bonus is *competitive*, not absolute: modelling E[bonus] directly ignores who else is
on the pitch. So we predict the BPS distribution per player, then simulate the
within-fixture ranking and allocate 3/2/1 (with FPL's tie rules).

2026/27 regime handling: BPS was retuned this summer to reduce DefCon overlap and to
improve prospects for goalkeepers, full-backs and attackers. Coefficients learned from
2025/26 are therefore biased, so a separate lightweight model is fitted on this
season's rows only and blended with weight `w_new = n_new / (n_new + 40)`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import RegressorArtifact, fit_lgbm_regressor

log = logging.getLogger(__name__)

BPS_FEATURES = [
    "bps90_last5", "bonus_rate_last10", "mins_last5_weighted", "xgi90_last5",
    "defcon_actions90", "is_home", "team_expected_goals", "opponent_defence_rating",
    "saves90", "season_bps_regime", "is_first_pen_taker",
]

# Sensible fallbacks before anything is trained, by position.
PRIOR_BPS90 = {"GK": 18.0, "DEF": 19.0, "MID": 18.0, "FWD": 17.0}
BPS_SD = 11.0


@dataclass
class BonusModel:
    """Historic model blended with a current-season-only model as the sample grows."""

    historic: RegressorArtifact | None = None
    current: RegressorArtifact | None = None
    n_current_fixtures: int = 0
    prior_fixtures: int = 40
    sd: float = BPS_SD

    @property
    def blend_weight(self) -> float:
        """w_new = n_new / (n_new + 40). Shifts toward the new regime as evidence arrives."""
        n = self.n_current_fixtures
        return n / (n + self.prior_fixtures) if (n + self.prior_fixtures) else 0.0

    @property
    def regime_warning(self) -> str | None:
        """Surfaced as a banner in the model performance screen for the first ~8 GWs."""
        if self.n_current_fixtures >= 8 * 10:
            return None
        return (
            f"BPS model is on limited 2026/27 data (n = {self.n_current_fixtures} fixtures); "
            "bonus predictions are wider than usual."
        )

    def expected_bps(self, features: dict, position: str, exp_minutes: float) -> float:
        prior = PRIOR_BPS90.get(position, 18.0)
        hist = self.historic.predict_one(features) if self.historic is not None else prior
        if self.current is not None:
            w = self.blend_weight
            cur = self.current.predict_one(features)
            rate = (1 - w) * hist + w * cur
        else:
            rate = hist
        return max(0.0, rate * (exp_minutes / 90.0))


def train_bps(X, y, feature_names: list[str], weights=None) -> RegressorArtifact:
    return fit_lgbm_regressor(X, y, feature_names, objective="regression", weights=weights)


def allocate_bonus(bps_by_player: dict[int, float]) -> dict[int, int]:
    """FPL's rules: 3/2/1 to the top three BPS scores, with ties sharing the higher award.

    Tie handling matters and is commonly botched: two players tied on top both get 3 and
    the next gets 1; three tied on top all get 3 and nobody gets 2 or 1.
    """
    if not bps_by_player:
        return {}
    ranked = sorted(bps_by_player.items(), key=lambda kv: -kv[1])
    out: dict[int, int] = {p: 0 for p in bps_by_player}
    scores = sorted({v for v in bps_by_player.values()}, reverse=True)

    top1 = [p for p, v in ranked if v == scores[0]]
    for p in top1:
        out[p] = 3
    if len(top1) >= 3 or len(scores) < 2:
        return out

    top2 = [p for p, v in ranked if v == scores[1]]
    award2 = 2 if len(top1) == 1 else 1
    for p in top2:
        out[p] = award2
    if len(top1) + len(top2) >= 3 or len(scores) < 3 or len(top1) > 1:
        return out

    if len(top1) == 1 and len(top2) == 1:
        for p in (p for p, v in ranked if v == scores[2]):
            out[p] = 1
    return out


def simulate_fixture_bonus(
    rng: np.random.Generator,
    players: list[dict],
    n_sims: int,
    sd: float = BPS_SD,
) -> dict[int, np.ndarray]:
    """Per-sim bonus for every player in one fixture, from their sampled BPS ranking.

    `players` entries need `player_id` and `exp_bps`. Returns player_id -> array of
    bonus points, one per simulation.
    """
    if not players:
        return {}
    ids = [p["player_id"] for p in players]
    mu = np.array([p["exp_bps"] for p in players], dtype=float)
    # BPS is right-skewed and non-negative; a truncated normal is close enough and fast.
    draws = np.maximum(0.0, rng.normal(mu[None, :], sd, size=(n_sims, len(ids))))

    out = {pid: np.zeros(n_sims, dtype=np.int8) for pid in ids}
    order = np.argsort(-draws, axis=1)
    for s in range(n_sims):
        row = draws[s]
        idx = order[s]
        # Ties are rare in continuous draws, so rank directly and apply the 3/2/1 ladder.
        for award, rank in ((3, 0), (2, 1), (1, 2)):
            if rank < len(idx) and row[idx[rank]] > 0:
                out[ids[idx[rank]]][s] = award
    return out
