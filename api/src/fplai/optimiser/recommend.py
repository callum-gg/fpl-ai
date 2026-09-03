"""Turn predictions + squad state into stored recommendations. docs/07 + docs/09.

Always produces three named variants (safe / balanced / aggressive) plus the "no
transfer" and "roll your transfer" baselines, each with its delta versus doing nothing.
If the best plan beats inaction by less than `min_expected_gain_to_act`, the app says so
plainly — an FPL tool that cannot recommend inaction is a bad FPL tool.
"""

from __future__ import annotations

import logging

from ..config import get_settings
from ..connectors.fpl_official import next_gameweek
from ..db.engine import jdump, query, query_one, utcnow, writer
from ..db.settings_store import squad_settings
from ..models.predict import horizon_points, latest
from ..models.simulate import load_draws
from ..rules import selling_price
from . import chips as chips_mod
from .planner import GameweekPlan, Plan, PlanContext, solve_plan
from .risk import RiskProfile, evaluate_against_rivals, from_settings, rank_objective, variant_profile
from .squad import Candidate, prefilter

log = logging.getLogger(__name__)

VARIANTS = ("safe", "balanced", "aggressive")


def _now() -> str:
    return utcnow()


# What a squad_state row means. Only SET_SOURCES are the squad you actually own: a
# 'planned' state is an accepted recommendation and 'draft' is the scratch copy, and
# neither may masquerade as reality just by being the most recent row written.
SET_SOURCES = ("fpl_sync", "manual")
DRAFT = "draft"


def _state(squad_id: int, sources: tuple[str, ...], gameweek: int | None = None) -> dict | None:
    placeholders = ",".join("?" * len(sources))
    row = query_one(
        f"SELECT * FROM squad_states WHERE squad_id=? AND source IN ({placeholders}) " +
        ("AND gameweek=? " if gameweek else "") +
        "ORDER BY gameweek DESC, captured_at DESC LIMIT 1",
        (squad_id, *sources, gameweek) if gameweek else (squad_id, *sources),
    )
    if row is None:
        return None
    state = dict(row)
    # Annotated here rather than per-caller: every screen that lists a squad wants the
    # name and club, and `position` on a pick is its 1-15 slot, so the GK/DEF/MID/FWD
    # name has to arrive under a separate key.
    state["picks"] = [
        dict(r) for r in query(
            "SELECT sp.*, p.web_name, ps.position position_name, ps.team_id, "
            "  t.short_name team_short "
            "FROM squad_picks sp "
            "LEFT JOIN players p ON p.id=sp.player_id "
            "LEFT JOIN player_seasons ps ON ps.player_id=sp.player_id AND ps.season_id=? "
            "LEFT JOIN teams t ON t.id=ps.team_id "
            "WHERE sp.squad_state_id=? ORDER BY sp.position",
            (get_settings().current_season, row["id"]),
        )
    ]
    return state


def save_state(conn, squad_id: int, gameweek: int, source: str, picks: list[dict], *,
               bank: int = 0, squad_value: int = 0, free_transfers: int = 1,
               chips_used_json: str = "[]", chip_active: str | None = None) -> int:
    """Write one squad_state and its picks. Every state in the app goes through here.

    captured_at is second-resolution on purpose (utcnow is the single timestamp format the
    whole app compares as strings), so two writes in the same second would otherwise trip
    the UNIQUE(squad_id, gameweek, source, captured_at) constraint. A second write in the
    same second is a supersede, not a duplicate, so it replaces the first.
    """
    now = utcnow()
    conn.execute(
        "DELETE FROM squad_states WHERE squad_id=? AND gameweek=? AND source=? AND captured_at=?",
        (squad_id, gameweek, source, now),
    )
    state_id = conn.execute(
        "INSERT INTO squad_states(squad_id,gameweek,source,bank,squad_value,free_transfers,"
        "chips_used_json,chip_active,captured_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (squad_id, gameweek, source, bank, squad_value, free_transfers, chips_used_json,
         chip_active, now),
    ).lastrowid
    for i, pick in enumerate(picks, start=1):
        conn.execute(
            "INSERT OR REPLACE INTO squad_picks(squad_state_id,player_id,position,is_captain,"
            "is_vice,purchase_price,selling_price) VALUES(?,?,?,?,?,?,?)",
            (state_id, pick["player_id"], pick.get("position", i),
             int(pick.get("is_captain") or 0), int(pick.get("is_vice") or 0),
             pick.get("purchase_price"), pick.get("selling_price")),
        )
    return state_id


def current_state(squad_id: int, gameweek: int | None = None) -> dict | None:
    """The squad you actually own. Drafts and accepted plans are deliberately invisible here."""
    return _state(squad_id, SET_SOURCES, gameweek)


def draft_state(squad_id: int, gameweek: int | None = None) -> dict | None:
    """The editable scratch copy, if one has been started."""
    return _state(squad_id, (DRAFT,), gameweek)


def working_state(squad_id: int, use_draft: bool, gameweek: int | None = None) -> dict | None:
    """What a recommendation should optimise from."""
    if use_draft:
        return draft_state(squad_id, gameweek) or current_state(squad_id, gameweek)
    return current_state(squad_id, gameweek)


def build_candidates(
    season_id: str, gameweek: int, profile: RiskProfile, state: dict | None
) -> dict[int, Candidate]:
    """player_id -> Candidate with utility computed under the given risk profile."""
    preds = latest(season_id, gameweek)
    owned = {p["player_id"]: p for p in (state or {}).get("picks", [])}

    meta = {
        r["player_id"]: r
        for r in query(
            "SELECT ps.player_id, ps.position, ps.team_id, p.web_name, p.canonical_name, "
            "  t.short_name team_short "
            "FROM player_seasons ps JOIN players p ON p.id=ps.player_id "
            "LEFT JOIN teams t ON t.id=ps.team_id WHERE ps.season_id=?",
            (season_id,),
        )
    }
    prices = {
        r["player_id"]: r["price"]
        for r in query(
            "SELECT player_id, price FROM player_prices pp WHERE season_id=? AND observed_at=("
            "  SELECT MAX(observed_at) FROM player_prices WHERE player_id=pp.player_id "
            "  AND season_id=pp.season_id)",
            (season_id,),
        )
    }
    eo = {
        r["player_id"]: r["effective_ownership"] or r["owned_pct"] or 0.0
        for r in query(
            "SELECT player_id, effective_ownership, owned_pct FROM ownership_snapshots "
            "WHERE season_id=? AND gameweek<=? GROUP BY player_id",
            (season_id, gameweek),
        )
    }

    # Double gameweeks produce two prediction rows per player; sum them.
    agg: dict[int, dict] = {}
    for p in preds:
        a = agg.setdefault(
            p["player_id"], {"exp": 0.0, "sd2": 0.0, "haul": 0.0}
        )
        a["exp"] += p["exp_points"] or 0.0
        a["sd2"] += (p["sd_points"] or 0.0) ** 2
        a["haul"] = max(a["haul"], p["p_haul_10"] or 0.0)

    out: dict[int, Candidate] = {}
    for pid, a in agg.items():
        m = meta.get(pid)
        if m is None or m["team_id"] is None:
            continue
        price = prices.get(pid, 40)
        pick = owned.get(pid)
        sell = (
            selling_price(pick["purchase_price"], price)
            if pick and pick.get("purchase_price")
            else price
        )
        sd = a["sd2"] ** 0.5
        out[pid] = Candidate(
            player_id=pid,
            position=m["position"],
            team_id=m["team_id"],
            price=price,
            selling_price=sell,
            exp_points=a["exp"],
            sd_points=sd,
            p_haul=a["haul"],
            utility=profile.utility(a["exp"], sd, a["haul"], eo.get(pid)),
            owned=pid in owned,
            effective_ownership=eo.get(pid, 0.0),
            name=m["web_name"] or m["canonical_name"],
            team_short=m["team_short"] or "",
        )
    return out


def build_horizon_candidates(
    season_id: str, start_gw: int, horizon: int, profile: RiskProfile, state: dict | None
) -> dict[int, dict[int, Candidate]]:
    """gameweek -> player_id -> Candidate. Later gameweeks reuse the earliest known
    price, because the price model's drift over 5 weeks is smaller than its error."""
    base = build_candidates(season_id, start_gw, profile, state)
    ep_horizon = horizon_points(season_id, start_gw, horizon)
    out: dict[int, dict[int, Candidate]] = {}
    for i in range(horizon):
        gw = start_gw + i
        gw_cands: dict[int, Candidate] = {}
        for pid, c in base.items():
            ep = ep_horizon.get(pid, [0.0] * horizon)[i] if pid in ep_horizon else 0.0
            if i == 0:
                gw_cands[pid] = c
                continue
            gw_cands[pid] = Candidate(
                player_id=pid, position=c.position, team_id=c.team_id, price=c.price,
                selling_price=c.selling_price, exp_points=ep,
                sd_points=c.sd_points * (ep / c.exp_points if c.exp_points else 1.0),
                p_haul=c.p_haul, owned=c.owned, effective_ownership=c.effective_ownership,
                utility=profile.utility(ep, c.sd_points, c.p_haul, c.effective_ownership),
                name=c.name, team_short=c.team_short,
            )
        out[gw] = gw_cands
    return out


FREE_TRANSFER_CHIPS = ("wildcard", "freehit")


def transfers_already_made(squad_id: int, gameweek: int, current_ids: list[int]) -> int:
    """How many of this gameweek's transfers have already been spent.

    A `manual` or `draft` state records the squad *after* a transfer but carries the free
    transfer count from before it. On 28 Aug that told the planner a third transfer was
    free when two were already gone, and the "0 hits" plan it produced actually cost -4.
    Last gameweek's settled squad is the honest baseline, so diff against that.
    """
    if gameweek <= 1 or not current_ids:
        return 0
    prev = query_one(
        "SELECT id FROM squad_states WHERE squad_id=? AND gameweek=? "
        "ORDER BY (source='fpl_sync') DESC, captured_at DESC LIMIT 1",
        (squad_id, gameweek - 1),
    )
    if prev is None:
        return 0
    previous = {
        r["player_id"]
        for r in query("SELECT player_id FROM squad_picks WHERE squad_state_id=?", (prev["id"],))
    }
    if not previous:
        return 0
    return len(set(current_ids) - previous)


def free_transfers_remaining(squad_id: int, gameweek: int, state: dict | None) -> int:
    stored = int((state or {}).get("free_transfers", 1) or 0)
    picks = (state or {}).get("picks", [])
    if (state or {}).get("chip_active") in FREE_TRANSFER_CHIPS:
        return stored  # transfers are free under a wildcard or free hit; nothing is spent
    used = transfers_already_made(squad_id, gameweek, [p["player_id"] for p in picks])
    if used:
        log.info("squad %s GW%s: %d transfer(s) already made, %d free transfer(s) left",
                 squad_id, gameweek, used, max(0, stored - used))
    return max(0, stored - used)


def _plan_context(
    squad_id: int, gameweek: int, settings: dict, state: dict | None,
    chips_allowed: set[str] | None = None,
) -> PlanContext:
    picks = (state or {}).get("picks", [])
    chip_strategy = settings.get("chip_strategy", {})
    return PlanContext(
        start_gw=gameweek,
        horizon=int(settings.get("horizon_gws", 5)),
        current_squad=[p["player_id"] for p in picks],
        bank=(state or {}).get("bank", 0),
        free_transfers=free_transfers_remaining(squad_id, gameweek, state),
        chips_used=_chips_used(state),
        chip_active=(state or {}).get("chip_active"),
        selling_prices={p["player_id"]: p.get("selling_price") or 0 for p in picks},
        decay=float(settings.get("horizon_decay", 0.84)),
        bench_weight=float(settings.get("bench_weight", 0.12)),
        max_hits_per_gw=int(settings.get("max_hits_per_gw", 1)),
        max_transfers_per_gw=int(settings.get("max_transfers_per_gw", 3)),
        banned_clubs=settings.get("banned_clubs", []),
        locked_players=settings.get("locked_players", []),
        must_own=settings.get("must_own", []),
        max_bench_value=settings.get("max_bench_value"),
        chips_allowed=chips_allowed,
        wildcard_earliest_gw=int(chip_strategy.get("wildcard_earliest_gw", 0)),
        save_second_set=bool(chip_strategy.get("save_second_set", False)),
    )


def _chips_used(state: dict | None) -> list[dict]:
    import json

    if not state or not state.get("chips_used_json"):
        return []
    try:
        raw = json.loads(state["chips_used_json"])
    except json.JSONDecodeError:
        return []
    return [
        {"name": c.get("name") or c.get("chip"), "gameweek": c.get("gameweek") or c.get("event")}
        for c in raw
        if isinstance(c, dict)
    ]


def recommend(
    squad_id: int,
    season_id: str,
    gameweek: int | None = None,
    variants: list[str] | None = None,
    constraints: dict | None = None,
    persist: bool = True,
    use_draft: bool = False,
) -> list[dict]:
    """The main entry point behind POST /api/squads/{id}/recommend."""
    gameweek = gameweek or next_gameweek(season_id)
    settings = squad_settings(squad_id)
    state = working_state(squad_id, use_draft)
    variants = variants or list(VARIANTS)

    # The chip calendar is computed once, here, and does two jobs: payload display and
    # the planner's gate. A chip the calendar can't justify (no doubles/blanks known —
    # actionable=False) gets no MILP variable at all; otherwise the solver burns it
    # inside the horizon because saving it is worth nothing there.
    state_teams = sorted({
        p["team_id"] for p in (state or {}).get("picks", []) if p.get("team_id")
    })
    chip_recos = chips_mod.plan_chips(
        season_id, gameweek, state_teams, chips_used=_chips_used(state)
    )
    chips_allowed = {r.chip for r in chip_recos if r.actionable}

    results: list[dict] = []
    baseline_points: float | None = None

    for variant in variants:
        profile = variant_profile(variant) if variant in VARIANTS else from_settings(settings)
        cands = build_horizon_candidates(
            season_id, gameweek, int(settings.get("horizon_gws", 5)), profile, state
        )
        first_gw = cands.get(gameweek, {})
        keep = list((state or {}).get("picks", []))
        kept_ids = [p["player_id"] for p in keep]
        pruned = {
            gw: {c.player_id: c for c in prefilter(list(v.values()), keep=kept_ids)}
            for gw, v in cands.items()
        }

        ctx = _plan_context(squad_id, gameweek, settings, state, chips_allowed)
        _apply_constraints(ctx, constraints)

        if not ctx.current_squad:
            plan = _initial_squad_plan(pruned, ctx, profile, gameweek)
            kind = "initial_squad"
        else:
            plan = solve_plan(pruned, ctx, profile)
            kind = "transfer_plan"

        if plan.status != "optimal" or not plan.gameweeks:
            log.warning("variant %s returned %s", variant, plan.status)
            continue

        if baseline_points is None and ctx.current_squad:
            baseline_points = _do_nothing_points(pruned, ctx, profile)

        payload = _payload(plan, pruned, first_gw, ctx, settings, season_id, gameweek,
                           baseline_points, variant, state, kind, chip_recos)

        # Three cards that hold the same team are one choice wearing three hats. Once the
        # projections compress, the risk terms (0.25*sd, 1.8*p_haul, the differential
        # nudge) stop separating anything and every variant lands on the same squad — as
        # `balanced` and `aggressive` did for the whole of GW2. Say so rather than letting
        # the UI imply a decision that was never offered.
        payload["same_as"] = _matching_variant(payload, results)
        if payload["same_as"]:
            log.info("variant %s is identical to %s", variant, payload["same_as"])

        rec = {
            "squad_id": squad_id,
            "gameweek": gameweek,
            "generated_at": _now(),
            "variant": variant,
            "kind": kind,
            "horizon_gws": ctx.horizon,
            "objective_value": plan.objective,
            "exp_points_gw": plan.exp_points_gw,
            "exp_points_horizon": plan.exp_points_horizon,
            "sd_points_gw": plan.sd_points_gw,
            "hits_taken": plan.total_hits,
            "chip_suggested": plan.gameweeks[0].chip,
            "payload_json": jdump(payload),
            "llm_rationale": None,
            "llm_critique": None,
            "model_run_id": None,
        }
        if persist:
            with writer() as conn:
                cur = conn.execute(
                    "INSERT INTO recommendations(squad_id,gameweek,generated_at,variant,kind,"
                    "horizon_gws,objective_value,exp_points_gw,exp_points_horizon,sd_points_gw,"
                    "hits_taken,chip_suggested,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        squad_id, gameweek, rec["generated_at"], variant, kind, ctx.horizon,
                        plan.objective, plan.exp_points_gw, plan.exp_points_horizon,
                        plan.sd_points_gw, plan.total_hits, rec["chip_suggested"],
                        rec["payload_json"],
                    ),
                )
                rec["id"] = cur.lastrowid
            _write_evidence(rec["id"], payload, season_id, gameweek)
        rec["payload"] = payload
        results.append(rec)
    return results


def _plan_fingerprint(payload: dict) -> tuple:
    """What makes two recommendations the same decision: the fifteen, who starts, and
    who wears the armband. Objective values differ by risk weighting even when the team
    is byte-identical, so they are deliberately not part of this."""
    lineup = payload.get("lineup") or {}
    return (
        tuple(sorted(p["player_id"] for p in payload.get("squad", []))),
        tuple(sorted(p["player_id"] for p in lineup.get("xi", []))),
        lineup.get("captain"),
        payload.get("chip"),
    )


def _matching_variant(payload: dict, earlier: list[dict]) -> str | None:
    """The name of an already-emitted variant holding this exact team, if any."""
    mine = _plan_fingerprint(payload)
    for rec in earlier:
        if _plan_fingerprint(rec.get("payload") or {}) == mine:
            return rec["variant"]
    return None


def _apply_constraints(ctx: PlanContext, constraints: dict | None) -> None:
    if not constraints:
        return
    ctx.force_in = constraints.get("force_in", []) or []
    ctx.force_out = constraints.get("force_out", []) or []
    ctx.locked_players = list(set(ctx.locked_players) | set(constraints.get("lock", []) or []))
    if constraints.get("budget_override") is not None:
        ctx.bank = int(constraints["budget_override"])
    if constraints.get("chip"):
        ctx.force_chip = (constraints["chip"], ctx.start_gw)
    if constraints.get("max_hits") is not None:
        ctx.max_hits_per_gw = int(constraints["max_hits"])


def _initial_squad_plan(pruned, ctx: PlanContext, profile: RiskProfile, gameweek: int) -> Plan:
    """No incumbent squad: a plain budget build with no transfer cost.

    Caps bench spend explicitly (docs/07 initial build). Bench points only count at
    `bench_weight`, so without a cap the solver happily parks £20m on players who mostly
    do not play — money that belongs in the starting XI.
    """
    from ..defaults import START_BUDGET
    from .squad import bench_fodder_budget, solve_squad

    cands = list(pruned.get(gameweek, {}).values())
    # force_out on an initial build means "never pick him", which the squad solver
    # expresses by dropping the candidate entirely. Without this the what-if endpoint
    # silently returns the unconstrained squad, which looks like the constraint had no cost.
    excluded = set(ctx.force_out)
    if excluded:
        cands = [c for c in cands if c.player_id not in excluded]
    must_own = list({*ctx.must_own, *ctx.force_in})

    bench_cap = ctx.max_bench_value or int(bench_fodder_budget(cands) * 1.25)
    sol = solve_squad(
        cands, budget=START_BUDGET, profile=profile, bench_weight=ctx.bench_weight,
        banned_clubs=ctx.banned_clubs, must_own=must_own, locked=ctx.locked_players,
        max_bench_value=bench_cap,
    )
    if sol.status != "optimal":
        # An over-tight bench cap is the likeliest cause; retry without it rather than
        # returning "no squad" for a solvable problem.
        log.info("initial build infeasible with bench cap %s, retrying uncapped", bench_cap)
        sol = solve_squad(
            cands, budget=START_BUDGET, profile=profile, bench_weight=ctx.bench_weight,
            banned_clubs=ctx.banned_clubs, must_own=must_own, locked=ctx.locked_players,
        )
    if sol.status != "optimal":
        return Plan(status=sol.status)
    return Plan(
        gameweeks=[
            GameweekPlan(
                gameweek=gameweek, squad=sol.squad, xi=sol.xi, bench_order=sol.bench_order,
                captain=sol.captain, vice=sol.vice, exp_points=sol.exp_points,
                sd_points=sol.sd_points, formation=sol.formation, bank_after=sol.bank,
                free_transfers_in=ctx.free_transfers,
            )
        ],
        objective=sol.objective,
        exp_points_gw=sol.exp_points,
        exp_points_horizon=sol.exp_points,
        sd_points_gw=sol.sd_points,
    )


def _do_nothing_points(pruned, ctx: PlanContext, profile: RiskProfile) -> float:
    frozen = PlanContext(**ctx.__dict__)
    frozen.max_transfers_per_gw = 0
    frozen.max_hits_per_gw = 0
    frozen.allow_chips = False
    frozen.force_in = []
    frozen.force_out = []
    plan = solve_plan(pruned, frozen, profile)
    return plan.exp_points_horizon if plan.status == "optimal" else 0.0


def _payload(plan, pruned, first_gw, ctx, settings, season_id, gameweek, baseline, variant,
             state, kind="transfer_plan", chip_recos=None):
    gw0 = plan.gameweeks[0]
    cands = pruned.get(gameweek, {})

    def _name(pid: int) -> str:
        """A pick can sit outside the pruned candidate set (departed, unpriced). Name him
        anyway — an id rendered where a name belongs is a bug the whole way down."""
        row = query_one(
            "SELECT COALESCE(web_name, canonical_name) name FROM players WHERE id=?", (pid,)
        )
        return (row["name"] if row else None) or str(pid)

    def _p(pid: int) -> dict:
        c = cands.get(pid)
        return {
            "player_id": pid,
            "name": c.name if c else _name(pid),
            "team_short": c.team_short if c else None,
            "position": c.position if c else None,
            "price": c.price if c else None,
            "selling_price": c.selling_price if c else None,
            "exp_points": round(c.exp_points, 2) if c else None,
        }

    delta_vs_nothing = (
        plan.exp_points_horizon - baseline if baseline is not None else None
    )
    threshold = float(settings.get("min_expected_gain_to_act", 0.8))
    act = delta_vs_nothing is None or delta_vs_nothing >= threshold

    transfers = []
    for out_id, in_id in zip(gw0.transfers_out, gw0.transfers_in, strict=False):
        transfers.append(
            {
                "out": _p(out_id),
                "in": _p(in_id),
                "delta_exp_points_gw": round(
                    (cands[in_id].exp_points if in_id in cands else 0)
                    - (cands[out_id].exp_points if out_id in cands else 0), 2
                ),
            }
        )

    # Calendar comes from recommend() (computed once against the real squad's teams)
    # so the display and the planner's gate can never disagree.
    chip_calendar = [r.to_dict() for r in (chip_recos or [])]

    return {
        "variant": variant,
        "gameweek": gameweek,
        "horizon": {"gws": ctx.horizon, "decay": ctx.decay},
        "transfers": transfers,
        "hits": gw0.hits,
        "chip": gw0.chip,
        "lineup": {
            "xi": [_p(p) for p in gw0.xi],
            "bench_order": [_p(p) for p in gw0.bench_order],
            "captain": gw0.captain,
            "vice": gw0.vice,
            "formation": gw0.formation,
        },
        "squad": [_p(p) for p in gw0.squad],
        "totals": {
            "exp_points_gw": round(plan.exp_points_gw, 1),
            "sd_points_gw": round(plan.sd_points_gw, 1),
            "exp_points_horizon": round(plan.exp_points_horizon, 1),
            "p_haul_captain": round(cands[gw0.captain].p_haul, 3)
            if gw0.captain in cands else None,
            "bank_after": gw0.bank_after,
            "free_transfers": gw0.free_transfers_in,
        },
        "alternatives": _alternatives(plan, baseline, threshold),
        "recommendation": (
            "act" if act else "do_nothing"
        ),
        "headline": _headline(transfers, gw0, delta_vs_nothing, act, threshold, kind),
        "future_gameweeks": [
            {
                "gameweek": g.gameweek,
                "transfers_in": [_p(p) for p in g.transfers_in],
                "transfers_out": [_p(p) for p in g.transfers_out],
                "hits": g.hits,
                "chip": g.chip,
                "exp_points": round(g.exp_points, 1),
            }
            for g in plan.gameweeks[1:]
        ],
        "chip_calendar": chip_calendar,
        "chip_warnings": chips_mod.expiry_warnings(gameweek, ctx.chips_used),
        "data_warnings": _data_warnings(season_id, gameweek, list(gw0.xi)),
        "delta_vs_do_nothing": round(delta_vs_nothing, 2) if delta_vs_nothing is not None else None,
    }


# How old the availability feed may be before the plan stops trusting itself. Team news
# lands in Friday press conferences, so anything older than half a day can be a full
# news cycle behind FPL's own flags.
STALE_AVAILABILITY_HOURS = 12.0

# Odds drift steadily and then sharply once team news lands, so a day-old price is a
# different market. Wider than the availability window because a stale price still beats
# no price — the team model alone is the worse of the two.
STALE_ODDS_HOURS = 24.0


def _hours_since(ts: str | None) -> float | None:
    if not ts:
        return None
    from datetime import datetime, timezone

    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 3600


def _data_warnings(season_id: str, gameweek: int, xi: list[int]) -> list[str]:
    """What the plan could not know, said out loud.

    Every number above is only as fresh as the feed under it, and the two feeds that move
    fastest — FPL's own injury flags and the prediction pass that reads them — are exactly
    the two that go stale silently. A plan that cannot say "my team news is a day old" will
    recommend a flagged player with total confidence, which is worse than not answering.
    """
    out: list[str] = []

    avail = query_one("SELECT MAX(observed_at) t FROM availability WHERE source_id='fpl_official'")
    avail_at = avail["t"] if avail else None
    age = _hours_since(avail_at)
    if age is None:
        out.append("No FPL availability data at all — injury flags are unknown.")
    elif age > STALE_AVAILABILITY_HOURS:
        out.append(
            f"FPL injury flags are {age:.0f}h old (last synced {avail_at}). "
            "Re-run the bootstrap sync before trusting this XI."
        )

    # The market is a better forecaster of goals than the team model, and the blend simply
    # disengages when there are no odds — quietly, with every number still rendering. That
    # is how GW1-2 were planned on the team model alone while a working odds key sat in the
    # .env and the poll job had never once been run.
    odds = query_one(
        "SELECT MAX(observed_at) t FROM odds_snapshots o JOIN fixtures f ON f.id=o.fixture_id "
        "WHERE f.season_id=? AND f.gameweek=?",
        (season_id, gameweek),
    )
    odds_at = odds["t"] if odds else None
    odds_age = _hours_since(odds_at)
    if odds_age is None:
        out.append(
            "No match odds for this gameweek — expected goals come from the team model "
            "alone and the market blend is disengaged. Run the odds_poll job."
        )
    elif odds_age > STALE_ODDS_HOURS:
        out.append(f"Match odds are {odds_age:.0f}h old (last polled {odds_at}) — prices "
                   "move most in the 24h before a deadline.")

    pred = query_one(
        "SELECT MAX(generated_at) t FROM predictions WHERE season_id=? AND gameweek=?",
        (season_id, gameweek),
    )
    pred_at = pred["t"] if pred else None
    if pred_at and avail_at and pred_at < avail_at:
        out.append(
            f"Predictions ({pred_at}) predate the latest team news ({avail_at}) — re-predict."
        )

    # Last line of defence: whatever the model priced him at, a flagged player in the XI
    # gets named. This is the check that survives a stale minutes model.
    if xi:
        placeholders = ",".join("?" * len(xi))
        flagged = query(
            f"SELECT p.web_name name, a.status, a.chance_pct FROM availability a "
            f"JOIN players p ON p.id=a.player_id JOIN ("
            f"  SELECT player_id, MAX(observed_at) m FROM availability "
            f"  WHERE source_id='fpl_official' AND player_id IN ({placeholders}) "
            f"  GROUP BY player_id"
            f") x ON x.player_id=a.player_id AND x.m=a.observed_at "
            f"WHERE a.source_id='fpl_official' AND a.status != 'available'",
            tuple(xi),
        )
        for r in flagged:
            chance = f" ({r['chance_pct']}%)" if r["chance_pct"] is not None else ""
            out.append(f"{r['name']} is flagged {r['status']}{chance} but is in the XI.")
    return out


def _alternatives(plan: Plan, baseline: float | None, threshold: float) -> list[dict]:
    out = []
    if baseline is not None:
        out.append({"label": "no change", "delta": round(baseline - plan.exp_points_horizon, 2)})
        out.append(
            {"label": "roll transfer",
             "delta": round(baseline + 0.4 - plan.exp_points_horizon, 2)}
        )
    return out


def _headline(transfers, gw0, delta, act, threshold, kind: str = "transfer_plan") -> str:
    if kind == "initial_squad":
        chip = f", playing {gw0.chip}" if gw0.chip else ""
        return (
            f"Initial squad built{chip}: {gw0.formation}, projected "
            f"{gw0.exp_points:.1f} points in GW{gw0.gameweek}."
        )
    if not act:
        return (
            f"The recommendation is to do nothing this week — the best plan gains only "
            f"{delta:.1f} points over the horizon, below your {threshold:.1f} threshold."
        )
    if not transfers:
        chip = f" and play {gw0.chip}" if gw0.chip else ""
        return f"Keep the squad{chip}; the lineup change alone is worth it."
    bits = ", ".join(f"{t['out']['name']} → {t['in']['name']}" for t in transfers)
    hit = f" (taking a -{gw0.hits * 4} hit)" if gw0.hits else ""
    chip = f", playing {gw0.chip}" if gw0.chip else ""
    gain = f" for +{delta:.1f} pts over the horizon" if delta is not None else ""
    return f"Transfer {bits}{hit}{chip}{gain}."


def _write_evidence(rec_id: int, payload: dict, season_id: str, gameweek: int) -> None:
    """Link the recommendation to the predictions and features behind it, so the UI's
    evidence panel is a view over the real chain rather than a re-summary."""
    rows = []
    for entry in payload.get("squad", []):
        pid = entry["player_id"]
        pred = query_one(
            "SELECT id FROM predictions WHERE player_id=? AND season_id=? AND gameweek=? "
            "ORDER BY generated_at DESC LIMIT 1",
            (pid, season_id, gameweek),
        )
        rows.append(
            {
                "subject_type": "recommendation",
                "subject_id": rec_id,
                "player_id": pid,
                "evidence_type": "prediction",
                "raw_doc_id": None,
                "claim_id": None,
                "feature_name": None,
                "weight": entry.get("exp_points"),
                "note": f"prediction {pred['id']}" if pred else None,
            }
        )
    if rows:
        with writer() as conn:
            conn.executemany(
                "INSERT INTO evidence_links(subject_type,subject_id,player_id,evidence_type,"
                "raw_doc_id,claim_id,feature_name,weight,note) VALUES(?,?,?,?,?,?,?,?,?)",
                [tuple(r.values()) for r in rows],
            )


def rank_aware_evaluation(squad_id: int, season_id: str, gameweek: int,
                          candidate_squads: list[list[int]]) -> list[dict]:
    """Second pass for mini-league mode: score MILP candidates against rivals' real squads
    using the joint simulation draws. MILP cannot optimise rank directly, so this is where
    'the right differential' is actually decided."""
    draws = load_draws(season_id, gameweek)
    if not draws:
        return []
    settings = squad_settings(squad_id)
    rivals = _rival_draws(squad_id, season_id, gameweek, draws)
    out = []
    for squad in candidate_squads:
        mine = sum((draws[p] for p in squad if p in draws), start=None)
        if mine is None:
            continue
        evaluation = evaluate_against_rivals(mine, rivals)
        out.append(
            {
                "squad": squad,
                "evaluation": evaluation,
                "score": rank_objective(evaluation, settings.get("rank_mode", "maximise_points")),
            }
        )
    return sorted(out, key=lambda r: -r["score"])


def _rival_draws(squad_id: int, season_id: str, gameweek: int, draws: dict) -> dict[int, object]:
    """Rivals' squads come from the public picks endpoint, ingested by the league job."""
    rows = query(
        "SELECT rival_entry_ids_json FROM squad_leagues WHERE squad_id=?", (squad_id,)
    )
    import json

    entry_ids: list[int] = []
    for r in rows:
        entry_ids += json.loads(r["rival_entry_ids_json"] or "[]")

    out = {}
    for entry_id in entry_ids[:50]:
        picks = query(
            "SELECT sp.player_id FROM squad_picks sp JOIN squad_states ss "
            "ON ss.id=sp.squad_state_id JOIN squads s ON s.id=ss.squad_id "
            "WHERE s.fpl_entry_id=? AND ss.gameweek=?",
            (entry_id, gameweek),
        )
        ids = [r["player_id"] for r in picks if r["player_id"] in draws]
        if len(ids) >= 11:
            total = None
            for pid in ids:
                total = draws[pid] if total is None else total + draws[pid]
            out[entry_id] = total
    return out
