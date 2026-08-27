"""Utility functions and rank-aware objectives. docs/07 — where risk actually lives.

The optimiser never maximises raw expected points unless you ask it to. `U(p,g)` is
chosen by the squad's risk setting, exposed in the UI as a single slider from -1 to +1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SAFE_LAMBDA = 0.35
AGGRESSIVE_LAMBDA = 0.25
HAUL_BONUS = 1.8  # points of utility per unit of P(haul >= 10), aggressive mode only


@dataclass
class RiskProfile:
    """Derived from the squad's `risk` slider plus its differential/rank toggles."""

    risk: float = 0.0                    # -1 safe .. 0 balanced .. +1 aggressive
    prefer_differentials: bool = False
    rank_mode: str = "maximise_points"   # maximise_points | climb_rank | protect_rank

    @property
    def label(self) -> str:
        if self.risk <= -0.34:
            return "safe"
        return "aggressive" if self.risk >= 0.34 else "balanced"

    @property
    def lam(self) -> float:
        """Signed variance coefficient: negative penalises SD, positive rewards it."""
        if self.risk < 0:
            return self.risk * SAFE_LAMBDA        # -1 -> -0.35
        return self.risk * AGGRESSIVE_LAMBDA      # +1 -> +0.25

    def utility(self, exp_points: float, sd_points: float | None = None,
                p_haul: float | None = None, effective_ownership: float | None = None) -> float:
        u = exp_points + self.lam * (sd_points or 0.0)
        if self.risk > 0 and p_haul is not None:
            u += self.risk * HAUL_BONUS * p_haul
        if self.prefer_differentials and effective_ownership is not None:
            # Reward low ownership mildly; the honest version of "differential" is the
            # rank-aware pass below, this is just a nudge for the linear proxy.
            u += 0.02 * self.risk_sign * max(0.0, 40.0 - effective_ownership) / 10.0
        return u

    @property
    def risk_sign(self) -> float:
        return 1.0 if self.risk >= 0 else -1.0


def from_settings(squad_settings: dict) -> RiskProfile:
    return RiskProfile(
        risk=float(squad_settings.get("risk", 0.0)),
        prefer_differentials=bool(squad_settings.get("prefer_differentials", False)),
        rank_mode=squad_settings.get("rank_mode", "maximise_points"),
    )


def variant_profile(variant: str) -> RiskProfile:
    return {
        "safe": RiskProfile(risk=-1.0),
        "balanced": RiskProfile(risk=0.0),
        "aggressive": RiskProfile(risk=1.0, prefer_differentials=True),
    }.get(variant, RiskProfile())


# --- rank-aware evaluation ------------------------------------------------------


def evaluate_against_rivals(
    my_draws: np.ndarray, rival_draws: dict[int, np.ndarray]
) -> dict:
    """Monte Carlo rank outcomes against rivals' *actual* squads.

    MILP cannot optimise a rank objective directly, so candidates are generated under a
    linear proxy and then scored here against the joint simulation draws.
    """
    if not rival_draws:
        return {}
    rivals = np.stack(list(rival_draws.values()))          # (rivals, sims)
    beats = (my_draws[None, :] > rivals).mean(axis=1)      # P(beat each rival)
    rank = 1 + (rivals > my_draws[None, :]).sum(axis=0)    # my rank per sim
    return {
        "p_beat_each": {rid: float(p) for rid, p in zip(rival_draws, beats, strict=False)},
        "p_win_league": float((rank == 1).mean()),
        "e_rank": float(rank.mean()),
        "p_gain_rank": float((rank < rank.mean()).mean()),
        "p_hold_position": float((rank <= np.median(rank)).mean()),
    }


def rank_objective(evaluation: dict, mode: str) -> float:
    """Which rank statistic to maximise. The right differential depends entirely on
    whether you are chasing or defending, and against whom."""
    if not evaluation:
        return 0.0
    if mode == "climb_rank":
        return evaluation.get("p_win_league", 0.0)
    if mode == "protect_rank":
        return evaluation.get("p_hold_position", 0.0)
    return -evaluation.get("e_rank", 0.0)


def eo_covariance_penalty(
    player_ids: list[int], eo_by_player: dict[int, float], draws: dict[int, np.ndarray]
) -> float:
    """Linear proxy for rank risk: owning what everyone owns has low rank variance.

    Used inside the MILP objective, where the true rank utility cannot be expressed.
    """
    if not draws:
        return 0.0
    total = 0.0
    for pid in player_ids:
        arr = draws.get(pid)
        if arr is None:
            continue
        eo = eo_by_player.get(pid, 0.0) / 100.0
        total += float(arr.std()) * (1.0 - eo)
    return total


def blend_horizon(values: list[float], decay: float) -> float:
    """decay(g) = horizon_decay ^ (g - g0). Default 0.84: GW+1 counts 1.0, GW+5 ~ 0.50."""
    return sum(v * (decay**i) for i, v in enumerate(values))
