"""Chip gating and chip FT semantics in the planner MILP (optimiser/planner.py).

The gating rules are pure constraint-building and test without a solve. The wildcard
FT-reset needs a real solve — one small end-to-end scenario, because this exact code
once shipped a silent `var or 0` pulp bug that only a solve would expose.
"""

from __future__ import annotations

import pulp
import pytest
from fplai.optimiser.planner import PlanContext, _chip_vars, solve_plan
from fplai.optimiser.squad import Candidate
from fplai.rules import chip_set


def _ctx(**kw) -> PlanContext:
    base = dict(start_gw=2, horizon=5, current_squad=[1], bank=0, free_transfers=1)
    base.update(kw)
    return PlanContext(**base)


def _vars(ctx: PlanContext) -> set[tuple[str, int]]:
    prob = pulp.LpProblem("chip_gate", pulp.LpMaximize)
    gws = list(range(ctx.start_gw, ctx.start_gw + ctx.horizon))
    return set(_chip_vars(prob, gws, ctx))


def test_no_allowed_chips_means_no_chip_variables():
    # Early season, no doubles/blanks known: the calendar marks everything
    # actionable=False, so the solver must have nothing to burn.
    assert _vars(_ctx(chips_allowed=set())) == set()


def test_allowed_chip_gets_variables_across_the_horizon():
    v = _vars(_ctx(chips_allowed={"wildcard"}))
    assert v == {("wildcard", g) for g in range(2, 7)}


def test_wildcard_earliest_gw_blocks_early_wildcards_only():
    ctx = _ctx(chips_allowed={"wildcard"}, wildcard_earliest_gw=4)
    assert _vars(ctx) == {("wildcard", g) for g in (4, 5, 6)}


def test_save_second_set_blocks_set_two_only():
    ctx = _ctx(start_gw=19, horizon=3, save_second_set=True)
    v = _vars(ctx)
    assert v and all(chip_set(g) == 1 for _, g in v)


def test_chip_active_overrides_the_gate():
    # You activated it in FPL — the plan must show it, calendar or no calendar.
    assert ("bboost", 2) in _vars(_ctx(chips_allowed=set(), chip_active="bboost"))


def test_spent_chip_stays_blocked_even_when_active():
    ctx = _ctx(chip_active="bboost", chips_used=[{"name": "bboost", "gameweek": 1}])
    assert ("bboost", 2) not in _vars(ctx)


# --- end-to-end: the wildcard's effect on the free-transfer bank ---------------------

QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _squad_pool(prefix: str, quality: float, id_start: int) -> tuple[list[Candidate], list[int]]:
    """Spares included (so the solver has choice); second return value is a
    quota-exact 15-man squad to own at the start."""
    pool: list[Candidate] = []
    exact: list[int] = []
    pid = id_start
    for pos, want in QUOTA.items():
        for i in range(want + 2):
            pid += 1
            pool.append(
                Candidate(
                    player_id=pid, position=pos, team_id=100 + (pid % 8), price=80,
                    selling_price=80, utility=quality, exp_points=quality,
                    name=f"{prefix}{pid}",
                )
            )
            if i < want:
                exact.append(pid)
    return pool, exact


def _wildcard_ctx(current: list[int], **kw) -> PlanContext:
    base = dict(
        start_gw=1, horizon=2, current_squad=current, bank=800, free_transfers=2,
        force_chip=("wildcard", 1), max_transfers_per_gw=3,
    )
    base.update(kw)
    return PlanContext(**base)


def test_wildcard_transfers_are_free_and_reset_the_bank():
    """A wildcard rebuild must be hit-free regardless of banked FTs, and must forfeit
    the bank: you enter the next gameweek with one free transfer — not zero
    (infeasible), not the drained balance, and not the preserved bank."""
    bad, current = _squad_pool("bad", quality=2.0, id_start=0)
    good, _ = _squad_pool("good", quality=5.0, id_start=1000)   # distinct ids: a real
    # Every GW must offer every candidate — the planner may field anyone in any GW.
    pool = {c.player_id: c for c in [*bad, *good]}
    utility = {1: pool, 2: dict(pool)}
    plan = solve_plan(utility, _wildcard_ctx(current))
    assert plan.status == "optimal"
    gw1, gw2 = plan.gameweeks
    assert gw1.chip == "wildcard"
    assert gw1.hits == 0                       # WC transfers cost no points
    assert len(gw1.transfers_in) > 2           # rebuild well past the 2 banked FTs
    assert gw2.free_transfers_in == 1          # reset to one, forfeiting the bank


def test_free_hit_at_a_full_bank_stays_feasible():
    """Regression: the FH lower arm used to demand ft[prev]+1 — 6 against the cap of 5
    at a full bank — making every full-bank Free Hit plan infeasible. The bank is
    untouched: it carries in whole, and the cap is the only ceiling."""
    bad, current = _squad_pool("bad", quality=2.0, id_start=0)
    pool = {c.player_id: c for c in bad}
    utility = {1: dict(pool), 2: dict(pool)}
    plan = solve_plan(utility, _wildcard_ctx(current, free_transfers=5,
                                             force_chip=("freehit", 1)))
    assert plan.status == "optimal"
    gw1, gw2 = plan.gameweeks
    assert gw1.chip == "freehit"
    assert gw1.hits == 0
    assert gw2.free_transfers_in == 5          # untouched, still at the cap


def test_free_hit_preserves_a_partial_bank():
    bad, current = _squad_pool("bad", quality=2.0, id_start=0)
    pool = {c.player_id: c for c in bad}
    utility = {1: dict(pool), 2: dict(pool)}
    plan = solve_plan(utility, _wildcard_ctx(current, free_transfers=2,
                                             force_chip=("freehit", 1)))
    assert plan.status == "optimal"
    gw1, gw2 = plan.gameweeks
    assert gw1.chip == "freehit"
    # The squad reverts after a FH, so GW2 has no transfer pressure and ft may float
    # conservatively below its true +1 accrual; it must never exceed the true value
    # nor drop below the preserved bank.
    assert 2 <= gw2.free_transfers_in <= 3


# ══ free-transfer accounting ═════════════════════════════════════════════════


@pytest.fixture
def player_pool(seeded_season):
    """Five real player rows, because squad_picks has a foreign key to players.id."""
    from fplai.db.engine import writer
    from fplai.resolve.entities import upsert_player

    with writer() as conn:
        return [
            upsert_player(conn, f"FT Pool {i}", "FT", f"Pool{i}", f"Pool{i}")
            for i in range(5)
        ]


def _seed_two_gameweeks(squad_id: int, gw1_ids, gw2_ids, *, stored_ft: int,
                        chip: str | None = None, gw2_source: str = "manual"):
    """GW1 settled by an FPL sync, GW2 a working state whose picks already moved."""
    from fplai.db.engine import writer

    with writer() as conn:
        conn.execute("INSERT OR IGNORE INTO squads(id,name,season_id) VALUES(?,?, '2026-27')",
                     (squad_id, f"S{squad_id}"))
        for gw, ids, source, ft in ((1, gw1_ids, "fpl_sync", 1),
                                    (2, gw2_ids, gw2_source, stored_ft)):
            sid = conn.execute(
                "INSERT INTO squad_states(squad_id,gameweek,source,bank,squad_value,"
                "free_transfers,chips_used_json,chip_active,captured_at) "
                "VALUES(?,?,?,0,1000,?,'[]',?,?)",
                (squad_id, gw, source, ft, chip if gw == 2 else None,
                 f"2026-08-2{gw}T12:00:00+00:00"),
            ).lastrowid
            for pos, pid in enumerate(ids, start=1):
                conn.execute(
                    "INSERT INTO squad_picks(squad_state_id,player_id,position) VALUES(?,?,?)",
                    (sid, pid, pos),
                )


def test_transfers_already_made_this_week_are_not_still_free(seeded_season, player_pool):
    """The 28 Aug bug: a manual state carried FT=1 after two transfers had been spent, so
    the planner priced a third move as free when it actually cost -4."""
    from fplai.optimiser.recommend import current_state, free_transfers_remaining

    a, b, c, d = player_pool[:4]
    _seed_two_gameweeks(701, [a, b, c], [a, d, c], stored_ft=1)
    state = current_state(701, 2)
    assert free_transfers_remaining(701, 2, state) == 0     # one made, one FT, none left

    _seed_two_gameweeks(702, [a, b, c], [a, b, c], stored_ft=1)
    assert free_transfers_remaining(702, 2, current_state(702, 2)) == 1   # nothing moved yet


def test_two_transfers_on_one_free_transfer_leaves_none(seeded_season, player_pool):
    from fplai.optimiser.recommend import current_state, free_transfers_remaining

    a, b, c, d, e = player_pool[:5]
    _seed_two_gameweeks(703, [a, b, c], [a, d, e], stored_ft=1)
    assert free_transfers_remaining(703, 2, current_state(703, 2)) == 0


def test_a_wildcard_makes_transfers_free_so_none_are_deducted(seeded_season, player_pool):
    from fplai.optimiser.recommend import current_state, free_transfers_remaining

    a, b, c, d, e = player_pool[:5]
    _seed_two_gameweeks(704, [a, b, c], [a, d, e], stored_ft=2, chip="wildcard")
    assert free_transfers_remaining(704, 2, current_state(704, 2)) == 2


def test_gameweek_one_has_no_previous_squad_to_diff_against(seeded_season, player_pool):
    from fplai.optimiser.recommend import transfers_already_made

    assert transfers_already_made(705, 1, list(player_pool[:3])) == 0
    assert transfers_already_made(705, 2, []) == 0


# ══ variant de-duplication ═══════════════════════════════════════════════════


def _payload(squad, xi, captain, chip=None):
    return {"squad": [{"player_id": p} for p in squad],
            "lineup": {"xi": [{"player_id": p} for p in xi], "captain": captain},
            "chip": chip}


def test_identical_plans_are_marked_rather_than_shown_as_three_choices():
    """`balanced` and `aggressive` returned byte-identical GW2 plans — same fifteen, same
    XI, same captain, same objective — while the UI presented them as two decisions."""
    from fplai.optimiser.recommend import _matching_variant

    fifteen, xi = list(range(1, 16)), list(range(1, 12))
    safe = {"variant": "safe", "payload": _payload(fifteen, xi, captain=5)}
    balanced = _payload(fifteen, xi, captain=5)
    assert _matching_variant(balanced, [safe]) == "safe"

    # A different captain is a different decision, even with the same fifteen.
    assert _matching_variant(_payload(fifteen, xi, captain=7), [safe]) is None
    # So is a different XI out of the same squad.
    other_xi = [*xi[:-1], 12]
    assert _matching_variant(_payload(fifteen, other_xi, captain=5), [safe]) is None
    # And so is playing a chip.
    assert _matching_variant(_payload(fifteen, xi, 5, chip="wildcard"), [safe]) is None
    # Nothing to compare against yet.
    assert _matching_variant(balanced, []) is None
