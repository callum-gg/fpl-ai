"""Multi-gameweek transfer planner MILP. docs/07.

Decision variables per player p and gameweek g in the horizon:
  x[p,g] in squad · s[p,g] in XI · c[p,g] captain · in/out[p,g] transfers
  ft[g] free transfers carried in (0..5) · hits[g] paid transfers · chip[t,g]

Free Hit gets a parallel shadow squad so the reversion at g+1 is modelled exactly rather
than approximated, and selling price uses the real 50%-of-profit rule — being 0.1 out
quietly makes every plan infeasible in practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pulp

from ..defaults import (
    HIT_COST,
    MAX_FREE_TRANSFERS,
    MAX_PER_CLUB,
    POSITION_QUOTA,
    SQUAD_SIZE,
    XI_MAX,
    XI_MIN,
)
from ..rules import chip_set
from .risk import RiskProfile
from .squad import Candidate, SquadSolution, _formation, _solver, _squad_sd, order_bench

log = logging.getLogger(__name__)

BIG_M = 20


@dataclass
class PlanContext:
    """The squad's real state at the start of the horizon."""

    start_gw: int
    horizon: int
    current_squad: list[int]
    bank: int
    free_transfers: int
    chips_used: list[dict] = field(default_factory=list)
    chip_active: str | None = None
    selling_prices: dict[int, int] = field(default_factory=dict)
    decay: float = 0.84
    bench_weight: float = 0.12
    max_hits_per_gw: int = 1
    max_transfers_per_gw: int = 3
    banned_clubs: list[int] = field(default_factory=list)
    locked_players: list[int] = field(default_factory=list)
    must_own: list[int] = field(default_factory=list)
    min_bank: int = 0
    max_bench_value: int | None = None
    allow_chips: bool = True
    chips_allowed: set[str] | None = None  # None = unrestricted; else only these chips
    wildcard_earliest_gw: int = 0
    save_second_set: bool = False
    force_chip: tuple[str, int] | None = None   # (chip, gameweek)
    force_in: list[int] = field(default_factory=list)
    force_out: list[int] = field(default_factory=list)


@dataclass
class GameweekPlan:
    gameweek: int
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: int = 0
    chip: str | None = None
    xi: list[int] = field(default_factory=list)
    bench_order: list[int] = field(default_factory=list)
    captain: int | None = None
    vice: int | None = None
    squad: list[int] = field(default_factory=list)
    exp_points: float = 0.0
    sd_points: float = 0.0
    formation: str = ""
    free_transfers_in: int = 0
    bank_after: int = 0


@dataclass
class Plan:
    gameweeks: list[GameweekPlan] = field(default_factory=list)
    objective: float = 0.0
    exp_points_gw: float = 0.0
    exp_points_horizon: float = 0.0
    sd_points_gw: float = 0.0
    total_hits: int = 0
    status: str = "optimal"

    @property
    def first(self) -> GameweekPlan | None:
        return self.gameweeks[0] if self.gameweeks else None


def solve_plan(
    utility: dict[int, dict[int, Candidate]],   # gameweek -> player_id -> candidate
    ctx: PlanContext,
    profile: RiskProfile | None = None,
) -> Plan:
    profile = profile or RiskProfile()
    gws = [ctx.start_gw + i for i in range(ctx.horizon)]
    ids = sorted({pid for gw in gws for pid in utility.get(gw, {})} | set(ctx.current_squad))
    if not ids:
        return Plan(status="infeasible")

    meta = _player_meta(utility, ids, ctx)
    banned = set(ctx.banned_clubs)
    ids = [i for i in ids if meta[i]["team_id"] not in banned or i in ctx.current_squad]

    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    x = {(p, g): pulp.LpVariable(f"x_{p}_{g}", cat="Binary") for p in ids for g in gws}
    s = {(p, g): pulp.LpVariable(f"s_{p}_{g}", cat="Binary") for p in ids for g in gws}
    c = {(p, g): pulp.LpVariable(f"c_{p}_{g}", cat="Binary") for p in ids for g in gws}
    tin = {(p, g): pulp.LpVariable(f"in_{p}_{g}", cat="Binary") for p in ids for g in gws}
    tout = {(p, g): pulp.LpVariable(f"out_{p}_{g}", cat="Binary") for p in ids for g in gws}
    ft = {g: pulp.LpVariable(f"ft_{g}", lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer")
          for g in gws}
    hits = {g: pulp.LpVariable(f"hits_{g}", lowBound=0, cat="Integer") for g in gws}

    chips = _chip_vars(prob, gws, ctx)

    # --- objective -----------------------------------------------------------------
    terms = []
    for i, g in enumerate(gws):
        d = ctx.decay**i
        bb = chips.get(("bboost", g))
        for p in ids:
            u = _u(utility, g, p, profile)
            tc = chips.get(("3xc", g))
            terms.append(d * u * s[(p, g)])
            # Captain gets +1x normally; Triple Captain adds another multiple. Modelled
            # as a separate linear term because c * chip would be quadratic.
            terms.append(d * u * c[(p, g)])
            if tc is not None:
                cap_tc = pulp.LpVariable(f"captc_{p}_{g}", cat="Binary")
                prob += cap_tc <= c[(p, g)]
                prob += cap_tc <= tc
                prob += cap_tc >= c[(p, g)] + tc - 1
                terms.append(d * u * cap_tc)
            # Bench contributes at bench_weight, or fully under Bench Boost.
            bench_var = x[(p, g)] - s[(p, g)]
            terms.append(d * u * ctx.bench_weight * bench_var)
            if bb is not None:
                bench_bb = pulp.LpVariable(f"bb_{p}_{g}", cat="Binary")
                prob += bench_bb <= bb
                prob += bench_bb <= x[(p, g)] - s[(p, g)] + 1
                prob += bench_bb >= bb + (x[(p, g)] - s[(p, g)]) - 1
                terms.append(d * u * (1 - ctx.bench_weight) * bench_bb)
        terms.append(-HIT_COST * d * hits[g])
    prob += pulp.lpSum(terms)

    # --- per-gameweek squad and XI legality -----------------------------------------
    for g in gws:
        prob += pulp.lpSum(x[(p, g)] for p in ids) == SQUAD_SIZE
        for pos, want in POSITION_QUOTA.items():
            prob += pulp.lpSum(x[(p, g)] for p in ids if meta[p]["position"] == pos) == want
        for t in {meta[p]["team_id"] for p in ids}:
            prob += pulp.lpSum(x[(p, g)] for p in ids if meta[p]["team_id"] == t) <= MAX_PER_CLUB

        prob += pulp.lpSum(s[(p, g)] for p in ids) == 11
        for pos in POSITION_QUOTA:
            in_pos = [p for p in ids if meta[p]["position"] == pos]
            prob += pulp.lpSum(s[(p, g)] for p in in_pos) >= XI_MIN[pos]
            prob += pulp.lpSum(s[(p, g)] for p in in_pos) <= XI_MAX[pos]
        for p in ids:
            prob += s[(p, g)] <= x[(p, g)]
            prob += c[(p, g)] <= s[(p, g)]
        prob += pulp.lpSum(c[(p, g)] for p in ids) == 1


    # --- money ----------------------------------------------------------------------
    # bank[g] is what remains after gameweek g's transfers. Selling uses the real
    # 50%-of-profit price, so a plan that looks affordable here is affordable in FPL.
    bank = {g: pulp.LpVariable(f"bank_{g}", lowBound=ctx.min_bank) for g in gws}
    for i, g in enumerate(gws):
        prev_bank = bank[gws[i - 1]] if i else ctx.bank
        prob += bank[g] == (
            prev_bank
            + pulp.lpSum(tout[(p, g)] * meta[p]["selling_price"] for p in ids)
            - pulp.lpSum(tin[(p, g)] * meta[p]["price"] for p in ids)
        )

    # --- transfer continuity --------------------------------------------------------
    owned = set(ctx.current_squad)
    for i, g in enumerate(gws):
        prev = gws[i - 1] if i else None
        for p in ids:
            before = x[(p, prev)] if prev else (1 if p in owned else 0)
            prob += x[(p, g)] - before == tin[(p, g)] - tout[(p, g)]
            prob += tin[(p, g)] + tout[(p, g)] <= 1
        prob += pulp.lpSum(tin[(p, g)] for p in ids) == pulp.lpSum(tout[(p, g)] for p in ids)
        prob += pulp.lpSum(tin[(p, g)] for p in ids) <= ctx.max_transfers_per_gw + BIG_M * (
            pulp.lpSum(chips.get((ch, g), 0) for ch in ("wildcard", "freehit"))
        )

    # --- free transfers and hits -----------------------------------------------------
    for i, g in enumerate(gws):
        free_chip = pulp.lpSum(chips.get((ch, g), 0) for ch in ("wildcard", "freehit"))
        prob += hits[g] >= pulp.lpSum(tin[(p, g)] for p in ids) - ft[g] - BIG_M * free_chip
        prob += hits[g] <= ctx.max_hits_per_gw + BIG_M * free_chip
        if i == 0:
            prob += ft[g] == ctx.free_transfers
        else:
            prev = gws[i - 1]
            wc_prev = chips.get(("wildcard", prev))
            fh_prev = chips.get(("freehit", prev))
            # ponytail: never `var or 0` on LpVariables — pulp defines __bool__ as
            # False, which would silently collapse the relaxation to zero.
            free_prev = pulp.lpSum(v for v in (wc_prev, fh_prev) if v is not None)
            # ft[g] = min(5, ft[g-1] - used + 1), linearised: <= all arms, and the
            # objective's hit penalty pushes it up to whichever binds.
            used = pulp.lpSum(tin[(p, prev)] for p in ids) - hits[prev]
            # Chip transfers come out of the chip, not the bank — relax the drain
            # under a wildcard/free hit.
            prob += ft[g] <= ft[prev] - used + 1 + BIG_M * free_prev
            # Always valid, and the only cap on a Free Hit GW (its transfers are
            # separate, so the bank just accrues +1 as if you'd made no transfers).
            prob += ft[g] <= ft[prev] + 1
            # A wildcard forfeits banked FTs: you enter the next gameweek with exactly
            # one. Lower + upper arms pin it — ft only floats where nothing cares.
            if wc_prev is not None:
                prob += ft[g] <= 1 + BIG_M * (1 - wc_prev)
                prob += ft[g] >= wc_prev
            # A Free Hit leaves the bank untouched. Lower arm preserves (not accrues):
            # demanding ft[prev]+1 here would demand 6 at a full bank against the cap
            # of 5 — infeasible. The upper arms still give the +1 (capped) whenever
            # hit pressure pushes ft up, and the value is exact at the cap.
            if fh_prev is not None:
                prob += ft[g] >= ft[prev] - BIG_M * (1 - fh_prev)
            prob += ft[g] <= MAX_FREE_TRANSFERS

    # --- personal constraints --------------------------------------------------------
    for g in gws:
        for pid in ctx.locked_players:
            if (pid, g) in x:
                prob += x[(pid, g)] == 1
        for pid in ctx.must_own:
            if (pid, g) in x:
                prob += x[(pid, g)] == 1
    for pid in ctx.force_in:
        if (pid, ctx.start_gw) in x:
            prob += x[(pid, ctx.start_gw)] == 1
    for pid in ctx.force_out:
        if (pid, ctx.start_gw) in x:
            prob += x[(pid, ctx.start_gw)] == 0

    # --- Free Hit reversion ----------------------------------------------------------
    # The squad at g+1 must return to the squad at g-1, so a Free Hit is genuinely
    # one-week-only rather than a free wildcard.
    for i, g in enumerate(gws):
        fh = chips.get(("freehit", g))
        if fh is None or i + 1 >= len(gws):
            continue
        nxt = gws[i + 1]
        for p in ids:
            before = x[(p, gws[i - 1])] if i else (1 if p in owned else 0)
            prob += x[(p, nxt)] - before <= BIG_M * (1 - fh)
            prob += before - x[(p, nxt)] <= BIG_M * (1 - fh)

    prob.solve(_solver())
    status = pulp.LpStatus[prob.status].lower()
    if status not in ("optimal", "not solved"):
        log.warning("planner returned %s", status)
        return Plan(status=status)

    return _extract(prob, x, s, c, tin, tout, hits, ft, chips, gws, ids, meta, utility, profile, ctx)


def _chip_vars(prob, gws: list[int], ctx: PlanContext) -> dict:
    """One binary per (chip, gameweek), with set-1 expiry and one-chip-per-GW enforced.

    Policy gates (chips_allowed / wildcard_earliest_gw / save_second_set) remove the
    variable entirely: inside a short horizon saving a chip is worth nothing, so a
    schedulable chip gets burned in GW2 even when the chip calendar says there is no
    fixture-driven case for it. An explicit force or chip_active overrides policy —
    if you activated it, it plays.
    """
    chips: dict[tuple[str, int], pulp.LpVariable] = {}
    if not ctx.allow_chips:
        return chips
    used_by_set = {(u["name"], chip_set(u["gameweek"])) for u in ctx.chips_used}
    forced = {ctx.force_chip} if ctx.force_chip else set()
    if ctx.chip_active:
        forced.add((ctx.chip_active, ctx.start_gw))
    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        for g in gws:
            if (chip, chip_set(g)) in used_by_set:
                continue  # already spent in this half
            if (chip, g) not in forced:
                if ctx.chips_allowed is not None and chip not in ctx.chips_allowed:
                    continue  # calendar: no fixture-driven case for this chip yet
                if chip == "wildcard" and g < ctx.wildcard_earliest_gw:
                    continue  # squad policy: no wildcard this early
                if ctx.save_second_set and chip_set(g) == 2:
                    continue  # squad policy: keep the second set in reserve
            chips[(chip, g)] = pulp.LpVariable(f"chip_{chip}_{g}", cat="Binary")

    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        for cs in (1, 2):
            in_set = [v for (ch, g), v in chips.items() if ch == chip and chip_set(g) == cs]
            if in_set:
                prob += pulp.lpSum(in_set) <= 1
    for g in gws:
        same_gw = [v for (_, gg), v in chips.items() if gg == g]
        if same_gw:
            prob += pulp.lpSum(same_gw) <= 1   # one chip per gameweek, total

    if ctx.force_chip:
        chip, g = ctx.force_chip
        if (chip, g) in chips:
            prob += chips[(chip, g)] == 1
    if ctx.chip_active and (ctx.chip_active, ctx.start_gw) in chips:
        prob += chips[(ctx.chip_active, ctx.start_gw)] == 1
    return chips


def _player_meta(utility: dict, ids: list[int], ctx: PlanContext) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    for pid in ids:
        cand = None
        for gw in sorted(utility):
            if pid in utility[gw]:
                cand = utility[gw][pid]
                break
        if cand is None:
            meta[pid] = {"position": "MID", "team_id": -1, "price": 40,
                         "selling_price": ctx.selling_prices.get(pid, 40), "name": ""}
            continue
        meta[pid] = {
            "position": cand.position,
            "team_id": cand.team_id,
            "price": cand.price,
            "selling_price": ctx.selling_prices.get(pid, cand.selling_price or cand.price),
            "name": cand.name,
        }
    return meta


def _u(utility: dict, gw: int, pid: int, profile: RiskProfile) -> float:
    cand = utility.get(gw, {}).get(pid)
    if cand is None:
        return 0.0
    return cand.utility


def _extract(prob, x, s, c, tin, tout, hits, ft, chips, gws, ids, meta, utility, profile, ctx) -> Plan:
    plan = Plan(objective=float(pulp.value(prob.objective) or 0))
    bank = ctx.bank
    for g in gws:
        squad = [p for p in ids if _on(x.get((p, g)))]
        xi = [p for p in ids if _on(s.get((p, g)))]
        captain = next((p for p in ids if _on(c.get((p, g)))), None)
        ins = [p for p in ids if _on(tin.get((p, g)))]
        outs = [p for p in ids if _on(tout.get((p, g)))]
        chip = next((ch for (ch, gg), var in chips.items() if gg == g and _on(var)), None)
        bench = [p for p in squad if p not in xi]
        cands = utility.get(g, {})
        bank += sum(meta[p]["selling_price"] for p in outs) - sum(meta[p]["price"] for p in ins)

        xi_cands = [cands[p] for p in xi if p in cands]
        vice = _pick_vice(xi, captain, cands)
        plan.gameweeks.append(
            GameweekPlan(
                gameweek=g,
                transfers_in=ins,
                transfers_out=outs,
                hits=int(hits[g].value() or 0),
                chip=chip,
                xi=xi,
                bench_order=order_bench(bench, cands) if cands else bench,
                captain=captain,
                vice=vice,
                squad=squad,
                exp_points=sum(cd.exp_points for cd in xi_cands)
                + (cands[captain].exp_points if captain in cands else 0.0),
                sd_points=_squad_sd(xi_cands),
                formation=_formation(xi, cands) if cands else "",
                free_transfers_in=int(ft[g].value() or 0),
                bank_after=bank,
            )
        )
    plan.total_hits = sum(g.hits for g in plan.gameweeks)
    if plan.gameweeks:
        plan.exp_points_gw = plan.gameweeks[0].exp_points
        plan.sd_points_gw = plan.gameweeks[0].sd_points
        plan.exp_points_horizon = sum(
            g.exp_points * (ctx.decay**i) for i, g in enumerate(plan.gameweeks)
        )
    return plan


def _pick_vice(xi: list[int], captain: int | None, cands: dict[int, Candidate]) -> int | None:
    others = [p for p in xi if p != captain and p in cands]
    return max(others, key=lambda p: cands[p].exp_points) if others else None


def _on(var) -> bool:
    return var is not None and var.value() is not None and var.value() > 0.5


def no_transfer_baseline(
    utility: dict[int, dict[int, Candidate]], ctx: PlanContext, profile: RiskProfile | None = None
) -> Plan:
    """The 'do nothing' plan. An FPL tool that cannot recommend inaction is a bad one."""
    frozen = PlanContext(**{**ctx.__dict__, "max_transfers_per_gw": 0, "max_hits_per_gw": 0,
                            "allow_chips": False})
    return solve_plan(utility, frozen, profile)


def roll_transfer_baseline(
    utility: dict[int, dict[int, Candidate]], ctx: PlanContext, profile: RiskProfile | None = None
) -> Plan:
    """Make no transfer this week, then optimise freely from next week with the extra FT."""
    rolled = PlanContext(**ctx.__dict__)
    rolled.max_transfers_per_gw = ctx.max_transfers_per_gw
    plan = solve_plan(utility, rolled, profile)
    # Force GW1 of the horizon to be transfer-free by re-solving with a tightened start.
    frozen_first = PlanContext(**{**ctx.__dict__, "force_in": [], "force_out": []})
    frozen_first.max_transfers_per_gw = 0
    first = solve_plan({ctx.start_gw: utility.get(ctx.start_gw, {})}, frozen_first, profile)
    if first.gameweeks and plan.gameweeks:
        plan.gameweeks[0] = first.gameweeks[0]
    return plan


def to_solution(gw_plan: GameweekPlan, cands: dict[int, Candidate]) -> SquadSolution:
    return SquadSolution(
        squad=gw_plan.squad,
        xi=gw_plan.xi,
        bench_order=gw_plan.bench_order,
        captain=gw_plan.captain,
        vice=gw_plan.vice,
        exp_points=gw_plan.exp_points,
        sd_points=gw_plan.sd_points,
        formation=gw_plan.formation,
    )
