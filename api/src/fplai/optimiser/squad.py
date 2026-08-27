"""Single-gameweek squad + XI MILP. docs/07.

Exact optimisation, not heuristics — the problem is small enough to solve properly in
seconds. Used for the initial GW1 build, for wildcards, and as the inner solve of the
multi-gameweek planner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pulp

from ..config import get_settings
from ..defaults import MAX_PER_CLUB, POSITION_QUOTA, SQUAD_SIZE, START_BUDGET, XI_MAX, XI_MIN
from .risk import RiskProfile

log = logging.getLogger(__name__)

PREFILTER_TOP_N = 250  # docs/07 performance: keep the model small, always retain owned players


@dataclass
class Candidate:
    player_id: int
    position: str
    team_id: int
    price: int              # tenths; buy price
    selling_price: int      # tenths; what we get if we already own him
    utility: float          # U(p,g) from the risk profile
    exp_points: float
    sd_points: float = 0.0
    p_haul: float = 0.0
    owned: bool = False
    effective_ownership: float = 0.0
    name: str = ""
    team_short: str = ""


@dataclass
class SquadSolution:
    squad: list[int] = field(default_factory=list)
    xi: list[int] = field(default_factory=list)
    bench_order: list[int] = field(default_factory=list)
    captain: int | None = None
    vice: int | None = None
    objective: float = 0.0
    exp_points: float = 0.0
    sd_points: float = 0.0
    formation: str = ""
    status: str = "optimal"
    spend: int = 0
    bank: int = 0


def _solver():
    s = get_settings()
    limit = s.optimiser_time_limit_s
    if s.optimiser_solver.upper() == "HIGHS":
        try:
            return pulp.HiGHS_CMD(msg=False, timeLimit=limit)
        except Exception:  # noqa: BLE001 - HiGHS is optional; CBC ships with PuLP
            log.info("HiGHS unavailable, falling back to CBC")
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=limit)


CHEAP_KEEP_PER_POSITION = 8


def prefilter(candidates: list[Candidate], keep: list[int] | None = None,
              top_n: int = PREFILTER_TOP_N) -> list[Candidate]:
    """Top N by utility, plus the players a legal squad structurally needs.

    Ranking by utility alone prunes away every £4.0m enabler, because cheap bench fodder
    has low expected points *by definition*. That leaves the solver unable to build a
    legal bench inside budget, so it either overspends on positions 12-15 or reports the
    problem infeasible. Retaining the cheapest few per position costs nothing and is what
    makes the budget constraint behave like real FPL.
    """
    keep_set = set(keep or []) | {c.player_id for c in candidates if c.owned}
    ranked = sorted(candidates, key=lambda c: -c.utility)
    out = list(ranked[:top_n])
    have = {c.player_id for c in out}

    by_position: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_position.setdefault(c.position, []).append(c)
    for pos_candidates in by_position.values():
        cheapest = sorted(pos_candidates, key=lambda c: (c.price, -c.utility))
        for c in cheapest[:CHEAP_KEEP_PER_POSITION]:
            if c.player_id not in have:
                out.append(c)
                have.add(c.player_id)

    out += [c for c in candidates if c.player_id in keep_set and c.player_id not in have]
    return out


def solve_squad(
    candidates: list[Candidate],
    budget: int = START_BUDGET,
    profile: RiskProfile | None = None,
    bench_weight: float = 0.12,
    banned_clubs: list[int] | None = None,
    must_own: list[int] | None = None,
    locked: list[int] | None = None,
    forbidden_sets: list[frozenset[int]] | None = None,
    captain_multiplier: int = 2,
    bench_boost: bool = False,
    max_bench_value: int | None = None,
    min_bank: int = 0,
) -> SquadSolution:
    """15-man squad + starting XI + captain, in one MILP."""
    profile = profile or RiskProfile()
    banned = set(banned_clubs or [])
    cands = [c for c in candidates if c.team_id not in banned]
    if not cands:
        return SquadSolution(status="infeasible")

    by_id = {c.player_id: c for c in cands}
    ids = list(by_id)

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ids, cat="Binary")   # in squad
    s = pulp.LpVariable.dicts("s", ids, cat="Binary")   # in XI
    c = pulp.LpVariable.dicts("c", ids, cat="Binary")   # captain
    v = pulp.LpVariable.dicts("v", ids, cat="Binary")   # vice

    # --- objective -----------------------------------------------------------------
    bench_term = 0 if bench_boost else bench_weight
    prob += pulp.lpSum(
        s[i] * by_id[i].utility
        + c[i] * by_id[i].utility * (captain_multiplier - 1)
        + (x[i] - s[i]) * by_id[i].utility * bench_term
        for i in ids
    )

    # --- squad composition ---------------------------------------------------------
    prob += pulp.lpSum(x[i] for i in ids) == SQUAD_SIZE
    for pos, want in POSITION_QUOTA.items():
        prob += pulp.lpSum(x[i] for i in ids if by_id[i].position == pos) == want

    teams = {by_id[i].team_id for i in ids}
    for t in teams:
        prob += pulp.lpSum(x[i] for i in ids if by_id[i].team_id == t) <= MAX_PER_CLUB

    prob += pulp.lpSum(x[i] * by_id[i].price for i in ids) <= budget - min_bank

    # --- starting XI ---------------------------------------------------------------
    prob += pulp.lpSum(s[i] for i in ids) == 11
    for pos in POSITION_QUOTA:
        prob += pulp.lpSum(s[i] for i in ids if by_id[i].position == pos) >= XI_MIN[pos]
        prob += pulp.lpSum(s[i] for i in ids if by_id[i].position == pos) <= XI_MAX[pos]
    for i in ids:
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]
        prob += v[i] <= s[i]
        prob += c[i] + v[i] <= 1  # captain and vice must be different players
    prob += pulp.lpSum(c[i] for i in ids) == 1
    prob += pulp.lpSum(v[i] for i in ids) == 1

    # --- personal constraints ------------------------------------------------------
    for pid in (must_own or []) + (locked or []):
        if pid in x:
            prob += x[pid] == 1
    if max_bench_value is not None:
        prob += pulp.lpSum((x[i] - s[i]) * by_id[i].price for i in ids) <= max_bench_value

    # --- diversification cuts (solution pool) --------------------------------------
    for banned_set in forbidden_sets or []:
        present = [i for i in banned_set if i in x]
        if present:
            prob += pulp.lpSum(x[i] for i in present) <= len(present) - 1

    prob.solve(_solver())
    status = pulp.LpStatus[prob.status].lower()
    if status not in ("optimal", "not solved"):
        return SquadSolution(status=status)

    squad = [i for i in ids if x[i].value() and x[i].value() > 0.5]
    xi = [i for i in ids if s[i].value() and s[i].value() > 0.5]
    captain = next((i for i in ids if c[i].value() and c[i].value() > 0.5), None)
    vice = next((i for i in ids if v[i].value() and v[i].value() > 0.5), None)
    bench = [i for i in squad if i not in xi]

    return SquadSolution(
        squad=squad,
        xi=xi,
        bench_order=order_bench(bench, by_id),
        captain=captain,
        vice=vice,
        objective=float(pulp.value(prob.objective) or 0),
        exp_points=sum(by_id[i].exp_points for i in xi)
        + (by_id[captain].exp_points if captain else 0),
        sd_points=_squad_sd([by_id[i] for i in xi]),
        formation=_formation(xi, by_id),
        spend=sum(by_id[i].price for i in squad),
        bank=budget - sum(by_id[i].price for i in squad),
        status="optimal",
    )


def order_bench(bench: list[int], by_id: dict[int, Candidate]) -> list[int]:
    """GK always bench slot 1 (position 12); outfielders by descending utility, because
    autosubs fire in bench order and the most likely replacement should come first."""
    gk = [i for i in bench if by_id[i].position == "GK"]
    out = sorted((i for i in bench if by_id[i].position != "GK"),
                 key=lambda i: -by_id[i].utility)
    return gk + out


def _formation(xi: list[int], by_id: dict[int, Candidate]) -> str:
    counts = {pos: sum(1 for i in xi if by_id[i].position == pos) for pos in POSITION_QUOTA}
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def _squad_sd(members: list[Candidate]) -> float:
    """Independent-variance approximation for the linear objective. The honest joint SD
    comes from the simulation draws in `planner.evaluate_with_draws`."""
    return float(sum(c.sd_points**2 for c in members) ** 0.5)


def best_xi(squad_ids: list[int], by_id: dict[int, Candidate],
            captain_multiplier: int = 2, bench_boost: bool = False) -> SquadSolution:
    """Pick the XI, captain and bench order out of a squad we already own."""
    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    ids = [i for i in squad_ids if i in by_id]
    s = pulp.LpVariable.dicts("s", ids, cat="Binary")
    c = pulp.LpVariable.dicts("c", ids, cat="Binary")
    v = pulp.LpVariable.dicts("v", ids, cat="Binary")

    prob += pulp.lpSum(
        s[i] * by_id[i].utility + c[i] * by_id[i].utility * (captain_multiplier - 1)
        for i in ids
    )
    prob += pulp.lpSum(s[i] for i in ids) == (len(ids) if bench_boost else 11)
    if not bench_boost:
        for pos in POSITION_QUOTA:
            prob += pulp.lpSum(s[i] for i in ids if by_id[i].position == pos) >= XI_MIN[pos]
            prob += pulp.lpSum(s[i] for i in ids if by_id[i].position == pos) <= XI_MAX[pos]
    for i in ids:
        prob += c[i] <= s[i]
        prob += v[i] <= s[i]
        prob += c[i] + v[i] <= 1
    prob += pulp.lpSum(c[i] for i in ids) == 1
    prob += pulp.lpSum(v[i] for i in ids) == 1

    prob.solve(_solver())
    xi = [i for i in ids if s[i].value() and s[i].value() > 0.5]
    captain = next((i for i in ids if c[i].value() and c[i].value() > 0.5), None)
    vice = next((i for i in ids if v[i].value() and v[i].value() > 0.5), None)
    bench = [i for i in ids if i not in xi]
    return SquadSolution(
        squad=ids,
        xi=xi,
        bench_order=order_bench(bench, by_id),
        captain=captain,
        vice=vice,
        objective=float(pulp.value(prob.objective) or 0),
        exp_points=sum(by_id[i].exp_points for i in xi)
        + (by_id[captain].exp_points if captain else 0),
        sd_points=_squad_sd([by_id[i] for i in xi]),
        formation=_formation(xi, by_id),
    )


def diverse_squads(
    candidates: list[Candidate], n: int = 3, **kwargs
) -> list[SquadSolution]:
    """Structurally different squads via no-good cuts: solve, forbid that exact 15, repeat.

    Gives the initial-build screen a 5-4-1 premium-heavy option next to a balanced one
    rather than fifteen near-identical squads.
    """
    out: list[SquadSolution] = []
    forbidden: list[frozenset[int]] = list(kwargs.pop("forbidden_sets", []) or [])
    for _ in range(n):
        sol = solve_squad(candidates, forbidden_sets=forbidden, **kwargs)
        if sol.status != "optimal" or not sol.squad:
            break
        out.append(sol)
        forbidden.append(frozenset(sol.squad))
    return out


def bench_fodder_budget(candidates: list[Candidate]) -> int:
    """Cheapest legal bench that still leaves a playing GK and bodies for autosubs."""
    cheap_gk = sorted((c for c in candidates if c.position == "GK"), key=lambda c: c.price)
    cheap_out = sorted((c for c in candidates if c.position != "GK"), key=lambda c: c.price)
    if len(cheap_gk) < 1 or len(cheap_out) < 3:
        return 170
    return cheap_gk[0].price + sum(c.price for c in cheap_out[:3])
