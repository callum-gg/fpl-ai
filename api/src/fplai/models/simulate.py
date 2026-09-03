"""Correlated Monte Carlo over a gameweek. docs/06 simulation.

Sampling teammates independently is the single most common error in public FPL
projections and it makes every variance number wrong. Here:

1. sample the scoreline jointly from the team model,
2. sample who starts, correlated within a team by a shared rotation shock — managers
   rotate in blocks, not independently,
3. allocate the team's *actual* sampled goals multinomially over its players, which
   preserves the negative correlation between teammates that independent sampling
   destroys,
4. sample DefCon, saves and cards,
5. compute BPS, rank within the fixture, allocate bonus,
6. sum FPL points.

The raw draws are kept so the optimiser can evaluate squad-level variance and
mini-league rank probabilities against the *joint* distribution, not marginals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..defaults import DEFCON_POINTS, DEFCON_THRESHOLD
from .bonus import simulate_fixture_bonus
from .rates import EMPIRICAL_RATES, nb_sample
from .team_goals import score_matrix

log = logging.getLogger(__name__)

# .get(..., MID) at the call sites: an unrecognised position from a historical feed
# must not raise out of the middle of a simulation and lose the whole gameweek.
GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_RATE = 0.90  # share of goals with an FPL assist (DB-wide 2020/21-2025/26)


@dataclass
class PlayerInput:
    """Everything the simulator needs about one player in one fixture."""

    player_id: int
    position: str
    team_id: int
    fixture_id: int
    p_start: float
    p_cameo: float
    exp_minutes: float
    goals90: float
    assists90: float
    defcon_rate90: float
    defcon_dispersion: float
    saves90: float
    cards90: float
    exp_bps: float
    rotation_sensitivity: float = 1.0  # scaled by manager_rotation_index


@dataclass
class FixtureInput:
    fixture_id: int
    home_team_id: int
    away_team_id: int
    lambda_home: float
    lambda_away: float
    rho: float = -0.05
    players: list[PlayerInput] = field(default_factory=list)


@dataclass
class SimResult:
    points: dict[int, np.ndarray]           # player_id -> per-sim points
    minutes: dict[int, np.ndarray]
    n_sims: int
    components: dict[int, dict] = field(default_factory=dict)

    def summary(self, player_id: int) -> dict:
        p = self.points.get(player_id)
        if p is None or len(p) == 0:
            return {}
        return {
            "exp_points": float(p.mean()),
            "sd_points": float(p.std()),
            "p10": float(np.percentile(p, 10)),
            "p50": float(np.percentile(p, 50)),
            "p90": float(np.percentile(p, 90)),
            "p_haul_10": float((p >= 10).mean()),
            "p_blank_2": float((p <= 2).mean()),
        }


def _sample_scorelines(rng, fx: FixtureInput, n: int) -> tuple[np.ndarray, np.ndarray]:
    m = score_matrix(fx.lambda_home, fx.lambda_away, fx.rho)
    flat = m.ravel()
    draws = rng.choice(len(flat), size=n, p=flat / flat.sum())
    return draws // m.shape[1], draws % m.shape[1]


def simulate_gameweek(
    fixtures: list[FixtureInput], n_sims: int = 10000, seed: int | None = 42
) -> SimResult:
    rng = np.random.default_rng(seed)
    points: dict[int, np.ndarray] = {}
    minutes: dict[int, np.ndarray] = {}
    components: dict[int, dict] = {}

    for fx in fixtures:
        home_goals, away_goals = _sample_scorelines(rng, fx, n_sims)

        for team_id, own_goals, conceded in (
            (fx.home_team_id, home_goals, away_goals),
            (fx.away_team_id, away_goals, home_goals),
        ):
            squad = [p for p in fx.players if p.team_id == team_id]
            if not squad:
                continue

            # --- 2. correlated starts: one rotation shock per team per sim -------------
            shock = rng.normal(0, 1, size=n_sims)
            mins = {}
            played = {}
            for p in squad:
                z = rng.normal(0, 1, size=n_sims)
                # Blend the shared shock with idiosyncratic noise: rho ~ 0.35 within a team.
                latent = 0.59 * shock * p.rotation_sensitivity + 0.81 * z
                start_cut = _z_for(p.p_start)
                cameo_cut = _z_for(p.p_start + p.p_cameo)
                starts = latent >= start_cut
                cameos = (~starts) & (latent >= cameo_cut)
                m = np.zeros(n_sims)
                # E[minutes | start] is 78 (EXP_MINUTES_IF_START), not the unconditional
                # exp_minutes = p_start*78 + p_cameo*19 — feeding that in deflated starter
                # minutes for every 0.4 < p_start < 0.6 player (CS/attack shares with it).
                m[starts] = np.clip(rng.normal(78.0, 14, starts.sum()), 1, 90)
                m[cameos] = np.clip(rng.normal(19, 9, cameos.sum()), 1, 45)
                mins[p.player_id] = m
                played[p.player_id] = m > 0

            # --- 3. multinomial allocation of the team's actual sampled goals ----------
            share_g = np.array([max(1e-6, p.goals90) for p in squad])
            share_a = np.array([max(1e-6, p.assists90) for p in squad])
            minute_share = np.stack([mins[p.player_id] / 90.0 for p in squad])  # (players, sims)

            wg = share_g[:, None] * minute_share
            wa = share_a[:, None] * minute_share
            goals_alloc = _multinomial_by_column(rng, own_goals, wg)
            # Assists cannot exceed goals; ~90% of goals carry an FPL assist (measured
            # 0.89-0.94 across 2020/21-2025/26 in this DB — Opta-style 0.70 is the wrong
            # definition for FPL scoring).
            assisted = rng.binomial(own_goals, ASSIST_RATE)
            assists_alloc = _multinomial_by_column(rng, assisted, wa)

            clean_sheet = conceded == 0

            for i, p in enumerate(squad):
                m = mins[p.player_id]
                on = played[p.player_id]
                pts = np.zeros(n_sims)
                pts += np.where(m >= 60, 2, np.where(m > 0, 1, 0))

                g = goals_alloc[i]
                a = assists_alloc[i]
                pts += GOAL_POINTS.get(p.position, GOAL_POINTS['MID']) * g
                pts += 3 * a

                if p.position in ("GK", "DEF", "MID"):
                    pts += CS_POINTS.get(p.position, CS_POINTS['MID']) * (clean_sheet & (m >= 60))
                conceded_penalty = np.zeros(n_sims)
                if p.position in ("GK", "DEF"):
                    conceded_penalty = conceded // 2 * on
                    pts -= conceded_penalty

                # --- 4. DefCon, saves, cards -----------------------------------------
                # One rate per draw, not the mean of them: he racks up DefCon actions in
                # the sims where he plays 90 and none in the sims where he doesn't, and
                # averaging first throws away exactly the tail that clears the threshold.
                mu_defcon = p.defcon_rate90 * (m / 90.0)
                actions = nb_sample(rng, mu_defcon, p.defcon_dispersion, n_sims)
                hit = actions >= DEFCON_THRESHOLD.get(p.position, 12)
                defcon_pts = DEFCON_POINTS * hit * on
                pts += defcon_pts

                if p.position == "GK":
                    saves = rng.poisson(np.maximum(0, p.saves90 * m / 90.0))
                    pts += saves // 3
                    pts += 5 * rng.binomial(1, EMPIRICAL_RATES["pen_saved_per_90_gk"], n_sims) * on
                else:
                    saves = np.zeros(n_sims, dtype=int)

                yellows = rng.poisson(np.maximum(0, p.cards90 * m / 90.0))
                reds = rng.binomial(1, EMPIRICAL_RATES["red_card_per_90"], n_sims) * on
                cards_penalty = np.minimum(yellows, 1) + 3 * reds
                pts -= cards_penalty
                pts -= 2 * rng.binomial(1, EMPIRICAL_RATES["own_goal_per_90"], n_sims) * on

                components[p.player_id] = {
                    "goals": g, "assists": a, "clean_sheet": clean_sheet & (m >= 60),
                    "defcon_points": defcon_pts, "saves": saves,
                    # Both of these reached `predictions` as NULL on every row ever written,
                    # so the UI could show what a player was expected to earn but never what
                    # the model expected him to lose.
                    "cards_penalty": cards_penalty, "conceded_penalty": conceded_penalty,
                }
                points[p.player_id] = pts
                minutes[p.player_id] = m

        # --- 5. BPS ranking within the fixture, then bonus -----------------------------
        bonus = simulate_fixture_bonus(
            rng,
            [
                {"player_id": p.player_id,
                 "exp_bps": p.exp_bps * (minutes.get(p.player_id, np.zeros(1)).mean() / 90.0 or 0)}
                for p in fx.players
                if p.player_id in points
            ],
            n_sims,
        )
        for pid, arr in bonus.items():
            points[pid] = points[pid] + arr
            components.setdefault(pid, {})["bonus"] = arr

    return SimResult(points=points, minutes=minutes, n_sims=n_sims, components=components)


def _z_for(p: float) -> float:
    """Threshold on a standard normal such that P(Z >= cut) = p."""
    from scipy.stats import norm

    return float(norm.ppf(1 - min(max(p, 1e-6), 1 - 1e-6)))


def _multinomial_by_column(rng, totals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Allocate `totals[s]` events across players using `weights[:, s]`, per simulation.

    This is the step that preserves teammate anti-correlation: the goals are shared out
    of a fixed pot, so one player scoring means another did not.
    """
    n_players, n_sims = weights.shape
    out = np.zeros((n_players, n_sims), dtype=int)
    max_total = int(totals.max()) if len(totals) else 0
    if max_total == 0 or n_players == 0:
        return out

    # A rate model can emit NaN, and a simulation where nobody played leaves a column of
    # zeros. Either would make rng.multinomial reject the probabilities, so both collapse
    # to a uniform share — the goals were scored by someone.
    w = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    col_sums = w.sum(axis=0)
    dead = col_sums <= 0
    if dead.any():
        w[:, dead] = 1.0
        col_sums[dead] = float(n_players)
    probs = (w / col_sums).T  # (n_sims, n_players)

    for k in range(1, max_total + 1):
        for idx in np.flatnonzero(totals == k):
            out[:, idx] = rng.multinomial(k, probs[idx])
    return out


def squad_distribution(sim: SimResult, player_ids: list[int], captain_id: int | None = None,
                       multiplier: int = 2) -> np.ndarray:
    """Per-sim total for a set of players. The joint draws are the whole point: summing
    marginal means and adding variances would overstate the spread badly."""
    total = np.zeros(sim.n_sims)
    for pid in player_ids:
        arr = sim.points.get(pid)
        if arr is None:
            continue
        total += arr * (multiplier if pid == captain_id else 1)
    return total


def save_draws(sim: SimResult, season_id: str, gameweek: int) -> str | None:
    """Persist the current GW's draws so the optimiser and rank mode can reuse them."""
    from ..config import get_settings

    path = get_settings().models_dir / "sims" / f"{season_id}-gw{gameweek}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(path, **{str(k): v for k, v in sim.points.items()})
        return str(path)
    except OSError as e:
        log.warning("could not save sim draws: %s", e)
        return None


def load_draws(season_id: str, gameweek: int) -> dict[int, np.ndarray] | None:
    from ..config import get_settings

    path = get_settings().models_dir / "sims" / f"{season_id}-gw{gameweek}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {int(k): data[k] for k in data.files}
