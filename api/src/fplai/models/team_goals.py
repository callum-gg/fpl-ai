"""Model 1 — team goals. Bivariate Poisson with Dixon-Coles, blended with the market.

Attack/defence strengths are time-decayed latent parameters (half-life ~60 matches),
home advantage is fitted, promoted sides shrink toward the league mean.

Then the market gets the final say where it exists:
    lambda_final = lambda_model^(1-w) * lambda_odds^w,   w ~ 0.65
The market is better than this model at forecasting goals, and pretending otherwise is
vanity. `w` is learned by backtest and drops to 0 when no odds are available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from ..db.engine import query, query_one
from ..db.settings_store import global_settings

log = logging.getLogger(__name__)

HALF_LIFE_MATCHES = 60.0
MAX_GOALS = 9


@lru_cache(maxsize=512)
def club_key(team_id: int | None) -> str | None:
    """Stable club identity across seasons, because `teams.id` is not one.

    `teams` carries one row per club *per season*, so Arsenal is seven different ids in
    this database. Keying strengths on the raw id fits each club seven separate times off
    38 matches each, and the 60-match half-life can never decay across a season boundary
    — which is how a Dixon-Coles model ends up losing to league-average Poisson. The
    alias table already holds the canonical key, so use it.
    """
    if team_id is None:
        return None
    row = query_one("SELECT name FROM teams WHERE id=?", (team_id,))
    if row is None:
        return None
    from ..resolve.entities import resolve_team

    return resolve_team(row["name"]) or row["name"].strip().lower()


@dataclass
class TeamModel:
    attack: dict[int, float] = field(default_factory=dict)
    defence: dict[int, float] = field(default_factory=dict)
    home_adv: float = 0.26
    rho: float = -0.05  # Dixon-Coles low-score correction
    base: float = 0.15

    def attack_of(self, team_id: int | None) -> float | None:
        return self.attack.get(club_key(team_id))

    def defence_of(self, team_id: int | None) -> float | None:
        return self.defence.get(club_key(team_id))

    def lambdas(self, home_team: int, away_team: int) -> tuple[float, float]:
        lh = math.exp(
            self.base + self.home_adv + (self.attack_of(home_team) or 0.0)
            - (self.defence_of(away_team) or 0.0)
        )
        la = math.exp(
            self.base + (self.attack_of(away_team) or 0.0)
            - (self.defence_of(home_team) or 0.0)
        )
        return max(0.15, min(5.0, lh)), max(0.15, min(5.0, la))

    def to_dict(self) -> dict:
        return {
            "attack": dict(self.attack),
            "defence": dict(self.defence),
            "home_adv": self.home_adv,
            "rho": self.rho,
            "base": self.base,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TeamModel:
        return cls(
            attack=dict(d.get("attack", {})),
            defence=dict(d.get("defence", {})),
            home_adv=d.get("home_adv", 0.26),
            rho=d.get("rho", -0.05),
            base=d.get("base", 0.15),
        )


def dixon_coles_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Low-score dependence correction. Independent Poisson underrates 0-0 and 1-1."""
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    if h == 0 and a == 1:
        return 1 + lh * rho
    if h == 1 and a == 0:
        return 1 + la * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def fit(season_ids: list[str], as_of: str | None = None) -> TeamModel:
    sql = (
        "SELECT home_team_id h, away_team_id a, home_score hs, away_score as_, kickoff_utc "
        "FROM fixtures WHERE finished=1 AND home_score IS NOT NULL AND competition='PL' "
        f"AND season_id IN ({','.join('?' * len(season_ids))})"
    )
    params: list = list(season_ids)
    if as_of:
        sql += " AND kickoff_utc < ?"
        params.append(as_of)
    rows = [dict(r) for r in query(sql + " ORDER BY kickoff_utc", tuple(params))]
    if len(rows) < 40:
        log.warning("team model: only %d matches, returning league-average priors", len(rows))
        return TeamModel()

    for r in rows:
        r["hk"], r["ak"] = club_key(r["h"]), club_key(r["a"])
    rows = [r for r in rows if r["hk"] and r["ak"]]
    if len(rows) < 40:
        log.warning("team model: only %d resolvable matches", len(rows))
        return TeamModel()
    teams = sorted({r["hk"] for r in rows} | {r["ak"] for r in rows})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Exponential time decay: recent matches matter more.
    decay = np.array([0.5 ** ((len(rows) - 1 - i) / HALF_LIFE_MATCHES) for i in range(len(rows))])
    hs = np.array([r["hs"] for r in rows], dtype=float)
    as_ = np.array([r["as_"] for r in rows], dtype=float)
    hi = np.array([idx[r["hk"]] for r in rows])
    ai = np.array([idx[r["ak"]] for r in rows])

    def nll(params: np.ndarray) -> float:
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv, rho, base = params[2 * n], params[2 * n + 1], params[2 * n + 2]
        lh = np.exp(base + home_adv + attack[hi] - defence[ai]).clip(0.05, 8)
        la = np.exp(base + attack[ai] - defence[hi]).clip(0.05, 8)
        ll = hs * np.log(lh) - lh + as_ * np.log(la) - la
        tau = np.ones_like(ll)
        m00 = (hs == 0) & (as_ == 0)
        m01 = (hs == 0) & (as_ == 1)
        m10 = (hs == 1) & (as_ == 0)
        m11 = (hs == 1) & (as_ == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-6, None)
        # Ridge on the strengths shrinks promoted sides toward the league mean.
        penalty = 0.02 * (np.sum(attack**2) + np.sum(defence**2))
        return -float(np.sum(decay * (ll + np.log(tau)))) + penalty

    x0 = np.concatenate([np.zeros(2 * n), [0.26, -0.05, 0.15]])
    bounds = [(-2, 2)] * (2 * n) + [(0, 0.8), (-0.25, 0.25), (-1, 1.5)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400, "maxfun": 8000})
    p = res.x
    # Identify by centring: attack and defence are only defined up to a constant.
    attack = p[:n] - p[:n].mean()
    defence = p[n:2 * n] - p[n:2 * n].mean()
    return TeamModel(
        attack={teams[i]: float(attack[i]) for i in range(n)},
        defence={teams[i]: float(defence[i]) for i in range(n)},
        home_adv=float(p[2 * n]),
        rho=float(p[2 * n + 1]),
        base=float(p[2 * n + 2]),
    )


def blended_lambdas(model: TeamModel, fixture_id: int, w: float | None = None) -> tuple[float, float]:
    """Geometric blend of model and market. w=0 when the market is silent."""
    fx = query_one("SELECT home_team_id, away_team_id FROM fixtures WHERE id=?", (fixture_id,))
    if fx is None:
        return 1.4, 1.2
    lm_h, lm_a = model.lambdas(fx["home_team_id"], fx["away_team_id"])

    from ..connectors.odds_api import team_lambdas

    market = team_lambdas(fixture_id)
    if market is None:
        return lm_h, lm_a
    if w is None:
        w = float(global_settings().get("model.odds_blend_weight", 0.65))
    lo_h, lo_a = market
    return (lm_h ** (1 - w)) * (lo_h**w), (lm_a ** (1 - w)) * (lo_a**w)


def score_matrix(lh: float, la: float, rho: float = -0.05) -> np.ndarray:
    """Joint distribution over scorelines, Dixon-Coles corrected and renormalised."""
    h = np.array([_pois(k, lh) for k in range(MAX_GOALS + 1)])
    a = np.array([_pois(k, la) for k in range(MAX_GOALS + 1)])
    m = np.outer(h, a)
    for i in range(2):
        for j in range(2):
            m[i, j] *= dixon_coles_tau(i, j, lh, la, rho)
    return m / m.sum()


def clean_sheet_prob(lam_conceded: float) -> float:
    return math.exp(-lam_conceded)


def concede_dist(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    d = np.array([_pois(k, lam) for k in range(max_goals + 1)])
    return d / d.sum()


def _pois(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def team_strengths_over_time(team_id: int, season_id: str) -> list[dict]:
    """Powers GET /api/teams/{id}/strength — refit at each gameweek boundary."""
    out = []
    gws = [r["gameweek"] for r in query(
        "SELECT DISTINCT gameweek FROM fixtures WHERE season_id=? AND finished=1 "
        "AND gameweek IS NOT NULL ORDER BY gameweek", (season_id,)
    )]
    for gw in gws[::3]:  # every third GW keeps this cheap enough to serve live
        as_of = query_one(
            "SELECT MAX(kickoff_utc) k FROM fixtures WHERE season_id=? AND gameweek<=?",
            (season_id, gw),
        )
        m = fit([season_id], as_of["k"] if as_of else None)
        out.append(
            {
                "gameweek": gw,
                "attack": m.attack.get(team_id, 0.0),
                "defence": m.defence.get(team_id, 0.0),
            }
        )
    return out


_MODEL_CACHE: dict[str, TeamModel] = {}


def load_model_or_fit(season_id: str) -> TeamModel:
    """Active artefact if one exists, else fit on the fly and memoise.

    The fixture ticker and team-strength endpoints need this synchronously, and a cold
    fit over one season is fast enough to serve.
    """
    if season_id in _MODEL_CACHE:
        return _MODEL_CACHE[season_id]
    from .base import load_active

    stored = load_active("team_goals")
    model = TeamModel.from_dict(stored) if isinstance(stored, dict) else (
        stored if isinstance(stored, TeamModel) else fit([season_id])
    )
    _MODEL_CACHE[season_id] = model
    return model
