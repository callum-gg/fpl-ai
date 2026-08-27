"""Layer 3 — the FPL rules engine. docs/12.

Pure functions, exhaustively tested. If these pass, the rules engine is right; if they
fail, every plan the optimiser produces is quietly wrong by a fraction of a million.
"""

from __future__ import annotations

import pytest
from fplai.rules import (
    apply_autosubs,
    chip_legal,
    chips_available,
    effective_captain,
    expiring_chips,
    formation_string,
    free_transfers_next,
    gameweek_points,
    hit_cost,
    legal_formations,
    player_points,
    selling_price,
    valid_formation,
    validate_squad,
)
from hypothesis import given
from hypothesis import strategies as st

# ── selling price ─────────────────────────────────────────────────────────────


def test_selling_price_worked_example():
    """docs/12: purchase 70, now 75 -> sells at 72 (profit 5, half rounded down = 2)."""
    assert selling_price(70, 75) == 72


@pytest.mark.parametrize(
    ("bought", "now", "expected"),
    [
        (70, 70, 70),   # no change
        (70, 71, 70),   # +0.1: half of 1 rounds down to 0
        (70, 72, 71),   # +0.2 -> +0.1
        (70, 73, 71),   # +0.3 -> +0.1
        (70, 74, 72),   # +0.4 -> +0.2
        (70, 75, 72),   # +0.5 -> +0.2
        (70, 65, 65),   # losses are absorbed in full
        (45, 60, 52),   # +1.5 -> +0.7
        (125, 140, 132),
    ],
)
def test_selling_price_rounding_edges(bought, now, expected):
    assert selling_price(bought, now) == expected


@given(bought=st.integers(38, 150), now=st.integers(38, 150))
def test_selling_price_never_exceeds_current_and_never_below_loss(bought, now):
    sell = selling_price(bought, now)
    assert sell <= max(bought, now)
    if now <= bought:
        assert sell == now          # a fall is passed on in full
    else:
        assert bought <= sell <= now  # a rise is shared


# ── free transfers and hits ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ft", "made", "expected"),
    [(1, 0, 2), (1, 1, 1), (2, 1, 2), (5, 0, 5), (5, 5, 1), (2, 3, 1), (0, 0, 1)],
)
def test_free_transfer_accrual_caps_at_five(ft, made, expected):
    assert free_transfers_next(ft, made) == expected


@given(ft=st.integers(0, 5), made=st.integers(0, 15))
def test_free_transfers_stay_in_range(ft, made):
    assert 0 <= free_transfers_next(ft, made) <= 5


def test_hit_cost_uses_three_when_you_had_two():
    """docs/12: 'used 3 when you had 2' costs one hit."""
    assert hit_cost(2, 3) == 4
    assert hit_cost(2, 2) == 0
    assert hit_cost(1, 4) == 12


def test_wildcard_and_free_hit_make_transfers_free():
    assert hit_cost(1, 12, chip="wildcard") == 0
    assert hit_cost(1, 12, chip="freehit") == 0
    assert free_transfers_next(2, 12, chip="wildcard") == 3


# ── chips ─────────────────────────────────────────────────────────────────────


def test_chip_set_one_expires_at_gameweek_nineteen():
    used = [{"name": "wildcard", "gameweek": 5}]
    assert chip_legal("wildcard", 10, used)[0] is False   # already spent in set 1
    assert chip_legal("wildcard", 20, used)[0] is True    # set 2 is a fresh chip


def test_one_chip_per_gameweek():
    used = [{"name": "bboost", "gameweek": 12}]
    ok, reason = chip_legal("3xc", 12, used)
    assert ok is False
    assert "GW12" in reason


def test_chips_available_shrinks_as_they_are_used():
    assert set(chips_available(5, [])) == {"wildcard", "freehit", "bboost", "3xc"}
    used = [{"name": "wildcard", "gameweek": 4}, {"name": "3xc", "gameweek": 6}]
    assert set(chips_available(8, used)) == {"freehit", "bboost"}


def test_expiring_chips_are_only_the_unplayed_set_one():
    used = [{"name": "wildcard", "gameweek": 4}]
    assert set(expiring_chips(17, used)) == {"freehit", "bboost", "3xc"}
    assert expiring_chips(25, used) == []   # set 1 is gone; nothing left to warn about


# ── squad legality ────────────────────────────────────────────────────────────


def _squad(n_gk=2, n_def=5, n_mid=5, n_fwd=3, price=60, teams=None):
    picks = []
    counts = {"GK": n_gk, "DEF": n_def, "MID": n_mid, "FWD": n_fwd}
    i = 0
    for pos, n in counts.items():
        for _ in range(n):
            picks.append(
                {"player_id": i, "position": pos, "price": price,
                 "team_id": (teams[i] if teams else i % 8)}
            )
            i += 1
    return picks


def test_valid_squad_passes():
    """15 players at £6.0m is £90.0m — a legal squad inside the £100.0m budget."""
    result = validate_squad(_squad(price=60), budget=1000, bank=0)
    assert result.ok, result.errors
    assert result.errors == []


def test_squad_size_and_quota_violations_are_reported():
    result = validate_squad(_squad(n_fwd=2), budget=1000)
    assert not result.ok
    assert any("FWD" in e for e in result.errors)


def test_max_three_players_per_club():
    result = validate_squad(_squad(teams=[1] * 15), budget=2000)
    assert not result.ok
    assert any("max 3" in e for e in result.errors)


def test_budget_violation_is_reported():
    result = validate_squad(_squad(price=100), budget=1000)
    assert not result.ok
    assert any("exceeds budget" in e for e in result.errors)


# ── formations ────────────────────────────────────────────────────────────────


def test_every_legal_formation_validates():
    for d, m, f in legal_formations():
        xi = (
            [{"position": "GK"}]
            + [{"position": "DEF"}] * d
            + [{"position": "MID"}] * m
            + [{"position": "FWD"}] * f
        )
        assert valid_formation(xi), f"{d}-{m}-{f} should be legal"


def test_illegal_formations_rejected():
    assert not valid_formation([{"position": "DEF"}] * 11)          # no keeper
    assert not valid_formation(
        [{"position": "GK"}] + [{"position": "DEF"}] * 2 + [{"position": "MID"}] * 8
    )                                                                # only 2 defenders
    assert not valid_formation([{"position": "GK"}] * 2 + [{"position": "MID"}] * 9)


def test_formation_string():
    xi = (
        [{"position": "GK"}]
        + [{"position": "DEF"}] * 4
        + [{"position": "MID"}] * 4
        + [{"position": "FWD"}] * 2
    )
    assert formation_string(xi) == "4-4-2"


# ── autosubs ──────────────────────────────────────────────────────────────────


def _p(pid, pos, minutes, points=0):
    return {"player_id": pid, "position": pos, "minutes": minutes, "points": points}


def test_goalkeeper_is_only_replaced_by_the_bench_goalkeeper():
    xi = [_p(1, "GK", 0)] + [_p(i, "DEF", 90) for i in range(2, 5)] + \
         [_p(i, "MID", 90) for i in range(5, 9)] + [_p(i, "FWD", 90) for i in range(9, 12)]
    bench = [_p(12, "GK", 90), _p(13, "MID", 90), _p(14, "DEF", 90), _p(15, "FWD", 90)]
    final, subs = apply_autosubs(xi, bench)
    assert [s["in"]["player_id"] for s in subs] == [12]
    assert sum(1 for p in final if p["position"] == "GK") == 1


def test_outfield_autosub_respects_bench_order():
    xi = [_p(1, "GK", 90)] + [_p(i, "DEF", 90) for i in range(2, 6)] + \
         [_p(i, "MID", 90) for i in range(6, 10)] + [_p(10, "FWD", 90), _p(11, "FWD", 0)]
    bench = [_p(12, "GK", 0), _p(13, "FWD", 90), _p(14, "MID", 90)]
    final, subs = apply_autosubs(xi, bench)
    assert [s["in"]["player_id"] for s in subs] == [13]   # first eligible in bench order
    assert valid_formation(final)


def test_autosub_is_skipped_when_it_would_break_the_formation():
    # Exactly 3 defenders: losing one cannot be replaced by a midfielder.
    xi = [_p(1, "GK", 90), _p(2, "DEF", 90), _p(3, "DEF", 90), _p(4, "DEF", 0)] + \
         [_p(i, "MID", 90) for i in range(5, 10)] + [_p(10, "FWD", 90), _p(11, "FWD", 90)]
    bench = [_p(12, "GK", 0), _p(13, "MID", 90)]
    final, subs = apply_autosubs(xi, bench)
    assert subs == []
    assert 4 in [p["player_id"] for p in final]


def test_vice_captain_takes_over_only_when_the_captain_blanks():
    xi = [_p(1, "GK", 90), _p(2, "DEF", 0)]
    assert effective_captain(xi, captain_id=1, vice_id=2) == 1
    assert effective_captain(xi, captain_id=2, vice_id=1) == 1   # captain played 0
    xi2 = [_p(1, "GK", 0), _p(2, "DEF", 0)]
    assert effective_captain(xi2, captain_id=1, vice_id=2) == 1  # neither played: stays


# ── scoring ───────────────────────────────────────────────────────────────────


def test_player_points_matches_fpl_scoring():
    # A defender: 90 mins (2) + goal (6) + clean sheet (4) + 2 bonus + DefCon (2) = 16
    stats = {"minutes": 90, "goals_scored": 1, "clean_sheets": 1, "bonus": 2,
             "defensive_contribution": 12, "goals_conceded": 0}
    assert player_points(stats, "DEF") == 16

    # A midfielder scoring twice with a yellow: 2 + 10 + 1(CS) - 1 = 12
    mid = {"minutes": 90, "goals_scored": 2, "clean_sheets": 1, "yellow_cards": 1}
    assert player_points(mid, "MID") == 12

    # A keeper: 2 + 4(CS) + 2 saves-points(6 saves) + 5 pen save = 13
    gk = {"minutes": 90, "clean_sheets": 1, "saves": 6, "penalties_saved": 1,
          "goals_conceded": 0}
    assert player_points(gk, "GK") == 13


def test_goals_conceded_penalty_is_per_two():
    stats = {"minutes": 90, "goals_conceded": 3}
    # 2 appearance - 1 (3 // 2) = 1
    assert player_points(stats, "DEF") == 1


def test_defcon_threshold_differs_by_position():
    defender = {"minutes": 90, "defensive_contribution": 10}
    midfielder = {"minutes": 90, "defensive_contribution": 10}
    assert player_points(defender, "DEF") == 4    # 2 + 2 DefCon (threshold 10)
    assert player_points(midfielder, "MID") == 2  # threshold is 12, so no DefCon


def test_gameweek_points_applies_captain_autosub_and_hits():
    xi = [_p(1, "GK", 90, 6)] + [_p(i, "DEF", 90, 5) for i in range(2, 6)] + \
         [_p(i, "MID", 90, 4) for i in range(6, 10)] + [_p(10, "FWD", 90, 8), _p(11, "FWD", 0, 0)]
    bench = [_p(12, "GK", 0, 0), _p(13, "FWD", 90, 7)]
    result = gameweek_points(xi, bench, captain_id=10, vice_id=1, hits=1)
    # 6 + 20 + 16 + 8 + 7(autosub) + 8(captain doubles) - 4(hit)
    assert result["captain_id"] == 10
    assert result["hits_cost"] == 4
    assert result["total"] == 6 + 20 + 16 + 8 + 7 + 8 - 4


def test_triple_captain_and_bench_boost():
    xi = [_p(1, "GK", 90, 2)] + [_p(i, "DEF", 90, 2) for i in range(2, 6)] + \
         [_p(i, "MID", 90, 2) for i in range(6, 10)] + [_p(10, "FWD", 90, 10), _p(11, "FWD", 90, 2)]
    bench = [_p(12, "GK", 90, 3), _p(13, "FWD", 90, 4)]

    tc = gameweek_points(xi, bench, captain_id=10, vice_id=1, chip="3xc")
    normal = gameweek_points(xi, bench, captain_id=10, vice_id=1)
    assert tc["total"] - normal["total"] == 10      # one extra captain multiple

    bb = gameweek_points(xi, bench, captain_id=10, vice_id=1, chip="bboost")
    assert bb["total"] - normal["total"] == 3 + 4   # the whole bench now counts
