"""Layers 4, 5 and 6 — features, models, optimiser. docs/12."""

from __future__ import annotations

import numpy as np
import pytest
from fplai.db.engine import query_one, writer

# ══ Layer 4 — features ═══════════════════════════════════════════════════════


@pytest.fixture
def player_with_history(seeded_season):
    """One player with three finished fixtures before GW1's deadline and one after.

    The late fixture is the leakage trap: any builder that reads it is peeking.
    """
    from fplai.resolve.entities import upsert_player

    existing = query_one("SELECT id FROM players WHERE canonical_name='Leak Test'")
    if existing:
        return existing["id"]

    with writer() as conn:
        team = query_one("SELECT id FROM teams WHERE short_name='ALP'")["id"]
        opp = query_one("SELECT id FROM teams WHERE short_name='BET'")["id"]
        pid = upsert_player(conn, "Leak Test", "Leak", "Test", "Test")
        conn.execute(
            "INSERT OR REPLACE INTO player_seasons(player_id,season_id,team_id,position) "
            "VALUES(?,?,?,'MID')",
            (pid, seeded_season, team),
        )
        fixtures = []
        for i, (kickoff, minutes, goals) in enumerate(
            [
                ("2026-08-01T15:00:00+00:00", 90, 1),
                ("2026-08-08T15:00:00+00:00", 90, 0),
                ("2026-08-15T15:00:00+00:00", 45, 2),
                # AFTER the GW1 deadline of 2026-08-22T10:00 — must never be read.
                ("2026-08-25T15:00:00+00:00", 90, 5),
            ]
        ):
            cur = conn.execute(
                "INSERT INTO fixtures(season_id,fpl_fixture_id,gameweek,kickoff_utc,"
                "home_team_id,away_team_id,finished,home_score,away_score) "
                "VALUES(?,?,?,?,?,?,1,2,1)",
                (seeded_season, 9000 + i, i + 1, kickoff, team, opp),
            )
            fx = cur.lastrowid
            fixtures.append(fx)
            conn.execute(
                "INSERT INTO player_fixture_stats(player_id,fixture_id,team_id,was_home,minutes,"
                "goals_scored,assists,bonus,bps,total_points,starts,xg,xa) "
                "VALUES(?,?,?,1,?,?,0,0,20,?,1,?,0.2)",
                (pid, fx, team, minutes, goals, goals * 5 + 2, goals * 0.6),
            )
    return pid


def test_features_never_read_past_the_deadline(player_with_history, seeded_season):
    """Layer 4, the critical one. The 5-goal fixture sits after the GW1 deadline."""
    from fplai.features.build import build_ctx

    ctx = build_ctx(player_with_history, seeded_season, 1)
    assert ctx is not None
    kickoffs = [h["kickoff_utc"] for h in ctx.history]
    assert all(k < ctx.as_of for k in kickoffs), kickoffs
    assert len(ctx.history) == 3
    # Total goals in-window is 3; the leaked fixture would push it to 8.
    assert sum(h["goals_scored"] for h in ctx.history) == 3


def test_rebuilding_features_is_deterministic(player_with_history, seeded_season):
    from fplai.features.build import build_ctx
    from fplai.features.registry import compute_all

    first = compute_all(build_ctx(player_with_history, seeded_season, 1))
    second = compute_all(build_ctx(player_with_history, seeded_season, 1))
    assert first == second


def test_missing_block_indicators_are_set(player_with_history, seeded_season):
    """With no odds source configured, every feature still computes and the block
    indicator says so — the model learns 'lean on the team model', not 'zero'."""
    from fplai.features.build import build_ctx
    from fplai.features.registry import compute_all

    values = compute_all(build_ctx(player_with_history, seeded_season, 1))
    assert values["block_odds_present"] == 0.0
    assert values["block_text_present"] == 0.0
    assert "mins_last3" in values


def test_start_streak_counts_consecutive_starts(player_with_history, seeded_season):
    from fplai.features.build import build_ctx
    from fplai.features.registry import REGISTRY

    ctx = build_ctx(player_with_history, seeded_season, 1)
    # All three in-window fixtures are seeded with starts=1, and the post-deadline one
    # is invisible, so the streak is exactly 3.
    assert REGISTRY["start_streak"].fn(ctx) == 3.0


def test_every_registered_feature_has_metadata():
    from fplai.features.registry import REGISTRY

    for name, f in REGISTRY.items():
        assert f.version >= 1, name
        assert f.missing_strategy in ("zero", "positional_mean", "shrink_to_prior",
                                      "indicator", "nan"), name
        assert f.group, name


def test_a_failing_builder_does_not_break_the_row(seeded_season):
    from fplai.features.registry import REGISTRY, FeatureCtx, compute_all, feature

    @feature("deliberately_broken", group="misc")
    def _broken(ctx):
        raise ValueError("boom")

    ctx = FeatureCtx(
        player_id=1, season_id=seeded_season, gameweek=1, fixture_id=None,
        as_of="2026-08-22T10:00:00+00:00", position="MID", team_id=1,
        opponent_team_id=2, is_home=True,
    )
    values = compute_all(ctx)
    assert values["deliberately_broken"] is None
    assert len(values) > 10          # everything else still computed
    REGISTRY.pop("deliberately_broken")


# ══ Layer 5 — models ═════════════════════════════════════════════════════════


def test_minutes_hard_overrides_beat_the_model():
    from fplai.models.minutes import predict

    features = {"start_streak": 5, "mins_last5": 450, "price_rank_in_position": 0.99}

    suspended = predict(None, features, suspended=True)
    assert suspended.p_start == 0.0 and suspended.exp_minutes == 0.0

    confirmed = predict(None, features, confirmed_start=True)
    assert confirmed.p_start > 0.9

    benched = predict(None, features, confirmed_start=False)
    assert benched.p_start < 0.1

    flagged = predict(None, features, fpl_chance=0)
    assert flagged.p_appear <= 0.02


def test_minutes_probabilities_form_a_distribution():
    from fplai.models.minutes import predict

    for streak in (0, 1, 3, 5):
        p = predict(None, {"start_streak": streak, "mins_last5": streak * 90})
        total = p.p_no_appearance + p.p_cameo + p.p_start
        assert 0.99 <= total <= 1.01
        assert all(0 <= v <= 1 for v in (p.p_no_appearance, p.p_cameo, p.p_start))


def test_between_seasons_bypasses_the_trained_minutes_model():
    """Before a ball is kicked, `player_minutes_last_7/14_days` are zero for everyone.

    They are the booster's two strongest splits, so it reads the whole league as doubtful.
    The guard must route pre-season players to the price/record blend instead.
    """
    from fplai.models.minutes import _between_seasons, predict

    preseason = {"starts_last5": 5, "mins_last10": 900, "days_since_last_match": 89,
                 "price_rank_in_position": 0.9, "player_minutes_last_14_days": 0}
    midseason = dict(preseason, player_minutes_last_14_days=180, days_since_last_match=6)
    assert _between_seasons(preseason)
    assert not _between_seasons(midseason)

    class _Doubtful:
        """Stands in for a booster reading every player as a doubt out of distribution."""

        def predict_one(self, _features):
            return np.array([0.7, 0.2, 0.1])

    assert predict(_Doubtful(), midseason).p_start == pytest.approx(0.1)
    assert predict(_Doubtful(), preseason).p_start > 0.8


def test_date_only_timestamps_stay_comparable_with_aware_ones():
    """`players.birth_date` is a plain date, everything else carries an offset.

    Parsing it naive made `asof - birth` raise TypeError, which the feature registry
    swallows as "no value" — so `age` was null for every player in every season while
    578 of 595 had a birth date on file. Silent, and invisible in the output.
    """
    from fplai.features.builders import _dt

    birth, asof = _dt("1995-09-15"), _dt("2026-08-21T17:30:00+00:00")
    assert birth.tzinfo is not None
    assert 30 < (asof - birth).days / 365.25 < 31   # the subtraction is the whole point
    assert _dt("2026-08-21T17:30:00Z").tzinfo is not None
    assert _dt("not-a-date") is None and _dt(None) is None


def test_preseason_forgives_an_end_of_season_rest():
    """A nailed striker wrapped in cotton wool in May ends the season on a nil streak.

    Mid-season that is the strongest bearish signal there is; in July it means nothing,
    so the pre-season read must use the start *rate*, not the streak.
    """
    from fplai.models.minutes import _preseason_p_start

    rested_star = {"starts_last5": 3, "mins_last10": 720, "start_streak": 0,
                   "days_since_last_match": 94, "price_rank_in_position": 1.0}
    regular_sub = {"starts_last5": 0, "mins_last10": 450, "start_streak": 0,
                   "days_since_last_match": 90, "price_rank_in_position": 0.3}

    assert _preseason_p_start(rested_star) > 0.7
    # Same zero streak, same "played recently" — but he starts nothing, and it must show.
    assert _preseason_p_start(regular_sub) < 0.4


def test_preseason_leans_on_price_as_the_record_goes_stale():
    """A year idle is a lost season or a foreign league; the teamsheet stops meaning much."""
    from fplai.models.minutes import _preseason_p_start

    record = {"starts_last5": 5, "mins_last10": 900, "price_rank_in_position": 0.2}
    fresh = _preseason_p_start(dict(record, days_since_last_match=90))
    stale = _preseason_p_start(dict(record, days_since_last_match=400))
    assert fresh > stale  # a strong record counts for less once it is a year old

    # No record at all — a new signing or a promoted side's player — is priced, not guessed.
    assert _preseason_p_start({"price_rank_in_position": 0.92}) > 0.8
    assert _preseason_p_start({"price_rank_in_position": 0.02}) < 0.2


def test_source_disagreement_widens_rather_than_shifts():
    """docs/05: high disagreement should widen the minutes distribution, not move it."""
    from fplai.models.minutes import predict

    f = {"start_streak": 4, "mins_last5": 360}
    calm = predict(None, f, disagreement=0.0)
    noisy = predict(None, f, disagreement=0.35)
    assert noisy.p_start < calm.p_start          # confidence bleeds out of the tail
    assert noisy.p_no_appearance + noisy.p_cameo + noisy.p_start == pytest.approx(1.0, abs=0.01)


def test_defcon_threshold_probability_is_monotonic_in_rate():
    """A step-function payoff modelled as a count, not a mean (docs/06 model 3)."""
    from fplai.models.rates import DefconModel, nb_survival

    model = DefconModel(rate_model=None, dispersion=5.0)
    probs = [
        model.p_threshold({"defcon_actions90": rate}, "DEF", 90.0)
        for rate in (2, 5, 8, 11, 15)
    ]
    assert probs == sorted(probs)
    assert 0 <= probs[0] < probs[-1] <= 1
    assert nb_survival(0, 5, 10) == 0.0
    assert nb_survival(10, 5, 0) == 1.0


def test_defcon_threshold_is_position_dependent():
    from fplai.models.rates import DefconModel

    model = DefconModel(rate_model=None, dispersion=5.0)
    features = {"defcon_actions90": 11.0}
    # Defenders need 10, midfielders 12 — the same rate must not give the same probability.
    assert model.p_threshold(features, "DEF", 90) > model.p_threshold(features, "MID", 90)


def test_bonus_allocation_follows_fpl_tie_rules():
    from fplai.models.bonus import allocate_bonus

    assert allocate_bonus({1: 40, 2: 30, 3: 20, 4: 10}) == {1: 3, 2: 2, 3: 1, 4: 0}
    # Two tied on top both get 3, next gets 1 — nobody gets 2.
    assert allocate_bonus({1: 40, 2: 40, 3: 20}) == {1: 3, 2: 3, 3: 1}
    # Three tied on top all get 3.
    assert allocate_bonus({1: 40, 2: 40, 3: 40, 4: 10}) == {1: 3, 2: 3, 3: 3, 4: 0}
    assert allocate_bonus({}) == {}


def test_bps_regime_blend_shifts_toward_the_new_season():
    """2026/27 retuned BPS, so the new-regime model must gain weight as data arrives."""
    from fplai.models.bonus import BonusModel

    assert BonusModel(n_current_fixtures=0).blend_weight == 0.0
    assert BonusModel(n_current_fixtures=40).blend_weight == pytest.approx(0.5)
    assert BonusModel(n_current_fixtures=360).blend_weight > 0.89
    assert BonusModel(n_current_fixtures=0).regime_warning is not None
    assert BonusModel(n_current_fixtures=500).regime_warning is None


def test_team_model_lambdas_are_bounded_and_home_advantaged():
    from fplai.models.team_goals import TeamModel, score_matrix

    m = TeamModel(attack={1: 0.5, 2: 0.0}, defence={1: 0.2, 2: 0.0}, home_adv=0.3)
    lh, la = m.lambdas(1, 2)
    assert 0.15 <= lh <= 5.0 and 0.15 <= la <= 5.0
    # Same two teams, reversed venue: the home side should expect more.
    lh2, la2 = m.lambdas(2, 1)
    assert lh > la2

    matrix = score_matrix(1.5, 1.2)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_devigging_removes_the_overround():
    from fplai.connectors.odds_api import devig_multiplicative, devig_shin

    implied = [1 / 2.0, 1 / 3.5, 1 / 4.5]     # sums to well over 1
    assert sum(implied) > 1.0
    for fair in (devig_multiplicative(implied), devig_shin(implied)):
        assert sum(fair) == pytest.approx(1.0, abs=1e-6)
        assert all(0 < p < 1 for p in fair)
        assert fair[0] > fair[1] > fair[2]     # ordering preserved


def test_poisson_fit_recovers_sensible_lambdas():
    from fplai.connectors.odds_api import poisson_lambdas_from_market

    lh, la = poisson_lambdas_from_market(0.55, 0.25, 0.20)
    assert lh > la                             # home favourite
    assert 0.5 < lh < 4.0 and 0.2 < la < 3.0


def test_simulation_preserves_teammate_anti_correlation():
    """docs/06: independent sampling destroys the negative correlation between teammates
    and makes every variance number wrong."""
    from fplai.models.simulate import FixtureInput, PlayerInput, simulate_gameweek

    def striker(pid):
        return PlayerInput(
            player_id=pid, position="FWD", team_id=1, fixture_id=1,
            p_start=0.99, p_cameo=0.0, exp_minutes=90, goals90=0.8, assists90=0.2,
            defcon_rate90=2, defcon_dispersion=5, saves90=0, cards90=0.1, exp_bps=20,
        )

    fx = FixtureInput(1, 1, 2, 1.6, 1.1, players=[striker(1), striker(2)])
    sim = simulate_gameweek([fx], n_sims=3000, seed=7)
    a, b = sim.components[1]["goals"], sim.components[2]["goals"]
    # They share a fixed pot of team goals, so their goal counts must not be independent.
    assert np.corrcoef(a, b)[0, 1] < 0.0


def test_simulation_is_deterministic_given_a_seed():
    from fplai.models.simulate import FixtureInput, PlayerInput, simulate_gameweek

    def build():
        p = PlayerInput(1, "MID", 1, 1, 0.9, 0.05, 80, 0.3, 0.3, 6, 5, 0, 0.1, 22)
        return [FixtureInput(1, 1, 2, 1.5, 1.2, players=[p])]

    a = simulate_gameweek(build(), n_sims=500, seed=42)
    b = simulate_gameweek(build(), n_sims=500, seed=42)
    assert np.array_equal(a.points[1], b.points[1])


def test_simulated_points_are_never_absurd():
    """A realistic XI, so goals are shared out rather than all landing on one player."""
    from fplai.models.simulate import FixtureInput, PlayerInput, simulate_gameweek

    squad = [
        PlayerInput(i, pos, 1, 1, 0.9, 0.05, 85, goals, assists, defcon, 5, 0, 0.2, 22)
        for i, (pos, goals, assists, defcon) in enumerate(
            [("GK", 0.0, 0.0, 1.0)]
            + [("DEF", 0.05, 0.05, 9.0)] * 4
            + [("MID", 0.25, 0.25, 7.0)] * 4
            + [("FWD", 0.55, 0.20, 3.0)] * 2,
            start=1,
        )
    ]
    sim = simulate_gameweek([FixtureInput(1, 1, 2, 1.4, 1.1, players=squad)], n_sims=1000, seed=3)
    for pid in range(1, 12):
        pts = sim.points[pid]
        assert pts.min() >= -6        # red card plus own goal is about as bad as it gets
        # A defender hat-trick with a clean sheet and bonus genuinely reaches the low 30s;
        # this guards against runaway values, not against a legitimate fat tail.
        assert pts.max() <= 45
        summary = sim.summary(pid)
        assert summary["p10"] <= summary["p50"] <= summary["p90"]
        assert 0 <= summary["p_haul_10"] <= 1


def test_season_weighting_decays_with_age():
    from fplai.models.base import season_weight

    assert season_weight("2026-27", "2026-27") == 1.0
    assert season_weight("2025-26", "2026-27") == pytest.approx(0.72)
    assert season_weight("2024-25", "2026-27") == pytest.approx(0.5184)
    assert season_weight("2016-17", "2026-27") < 0.05


def test_calibration_metrics_behave():
    from fplai.models.base import ece, log_loss, spearman

    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece(perfect, perfect) == pytest.approx(0.0, abs=1e-9)
    # Predicting the base rate is *calibrated* even though it is not sharp, so ECE is 0.
    assert ece(perfect, np.full(4, 0.5)) == pytest.approx(0.0, abs=1e-9)
    # Confidently wrong is what ECE is meant to catch: 0.9 predicted against a 50% rate.
    assert ece(perfect, np.full(4, 0.9)) == pytest.approx(0.4, abs=1e-9)
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert log_loss(np.array([1, 0]), np.array([0.9, 0.1])) < 0.2


# ══ Layer 6 — optimiser ══════════════════════════════════════════════════════


def _candidates(n_per_pos=8, base_price=45):
    from fplai.optimiser.squad import Candidate

    out, pid = [], 1
    for pos in ("GK", "DEF", "MID", "FWD"):
        for i in range(n_per_pos):
            ep = 2.0 + i * 0.4
            out.append(
                Candidate(
                    player_id=pid, position=pos, team_id=pid % 10,
                    price=base_price + i * 5, selling_price=base_price + i * 5,
                    utility=ep, exp_points=ep, sd_points=2.0 + i * 0.1,
                    p_haul=0.05 * i, name=f"{pos}{i}",
                )
            )
            pid += 1
    return out


def test_solved_squad_satisfies_every_constraint():
    from fplai.defaults import MAX_PER_CLUB, POSITION_QUOTA, SQUAD_SIZE
    from fplai.optimiser.squad import solve_squad

    cands = _candidates()
    by_id = {c.player_id: c for c in cands}
    sol = solve_squad(cands, budget=1000)

    assert sol.status == "optimal"
    assert len(sol.squad) == SQUAD_SIZE
    for pos, want in POSITION_QUOTA.items():
        assert sum(1 for p in sol.squad if by_id[p].position == pos) == want
    clubs: dict[int, int] = {}
    for p in sol.squad:
        clubs[by_id[p].team_id] = clubs.get(by_id[p].team_id, 0) + 1
    assert max(clubs.values()) <= MAX_PER_CLUB
    assert sum(by_id[p].price for p in sol.squad) <= 1000
    assert len(sol.xi) == 11
    assert sol.captain in sol.xi
    assert sol.vice in sol.xi
    assert sol.captain != sol.vice
    assert set(sol.bench_order) == set(sol.squad) - set(sol.xi)


def test_starting_eleven_is_a_legal_formation():
    from fplai.optimiser.squad import solve_squad
    from fplai.rules import valid_formation

    cands = _candidates()
    by_id = {c.player_id: c for c in cands}
    sol = solve_squad(cands, budget=1000)
    xi = [{"position": by_id[p].position} for p in sol.xi]
    assert valid_formation(xi)


def test_bench_goalkeeper_is_first_in_bench_order():
    from fplai.optimiser.squad import solve_squad

    cands = _candidates()
    by_id = {c.player_id: c for c in cands}
    sol = solve_squad(cands, budget=1000)
    assert by_id[sol.bench_order[0]].position == "GK"


def test_raising_a_players_score_never_removes_him_from_the_xi():
    """Monotonicity (docs/12 layer 6)."""
    from fplai.optimiser.squad import solve_squad

    cands = _candidates()
    sol = solve_squad(cands, budget=1000)
    picked = sol.xi[0]

    boosted = []
    for c in cands:
        if c.player_id == picked:
            c = type(c)(**{**c.__dict__, "utility": c.utility + 5, "exp_points": c.exp_points + 5})
        boosted.append(c)
    sol2 = solve_squad(boosted, budget=1000)
    assert picked in sol2.xi


def test_budget_is_respected_when_tight():
    from fplai.optimiser.squad import solve_squad

    sol = solve_squad(_candidates(base_price=40), budget=700)
    if sol.status == "optimal":
        by_id = {c.player_id: c for c in _candidates(base_price=40)}
        assert sum(by_id[p].price for p in sol.squad) <= 700


def test_banned_club_players_are_excluded():
    from fplai.optimiser.squad import solve_squad

    cands = _candidates()
    by_id = {c.player_id: c for c in cands}
    sol = solve_squad(cands, budget=1000, banned_clubs=[3])
    assert all(by_id[p].team_id != 3 for p in sol.squad)


def test_must_own_players_are_forced_in():
    from fplai.optimiser.squad import solve_squad

    cands = _candidates()
    cheap_gk = min((c for c in cands if c.position == "GK"), key=lambda c: c.utility)
    sol = solve_squad(cands, budget=1000, must_own=[cheap_gk.player_id])
    assert cheap_gk.player_id in sol.squad


def test_prefilter_keeps_cheap_enablers_and_owned_players():
    """Ranking by utility alone prunes every bench enabler and makes budgets unsolvable."""
    from fplai.optimiser.squad import prefilter

    cands = _candidates(n_per_pos=40)
    kept = prefilter(cands, keep=[cands[-1].player_id], top_n=20)
    kept_ids = {c.player_id for c in kept}
    assert cands[-1].player_id in kept_ids
    for pos in ("GK", "DEF", "MID", "FWD"):
        cheapest = min((c for c in cands if c.position == pos), key=lambda c: c.price)
        assert cheapest.player_id in kept_ids, f"cheapest {pos} was pruned"


def test_diverse_squads_are_structurally_different():
    from fplai.optimiser.squad import diverse_squads

    squads = diverse_squads(_candidates(), n=3, budget=1000)
    assert len(squads) >= 2
    signatures = {frozenset(s.squad) for s in squads}
    assert len(signatures) == len(squads)


def test_risk_profile_changes_the_utility_ordering():
    from fplai.optimiser.risk import RiskProfile

    safe, balanced, aggressive = RiskProfile(-1.0), RiskProfile(0.0), RiskProfile(1.0)
    steady, volatile = (6.0, 1.0), (6.0, 6.0)     # (exp_points, sd)

    assert safe.utility(*steady) > safe.utility(*volatile)
    assert balanced.utility(*steady) == pytest.approx(balanced.utility(*volatile))
    assert aggressive.utility(*volatile) > aggressive.utility(*steady)
    assert safe.label == "safe" and aggressive.label == "aggressive"


def test_horizon_decay_discounts_later_gameweeks():
    from fplai.optimiser.risk import blend_horizon

    assert blend_horizon([10, 10, 10], 1.0) == pytest.approx(30)
    assert blend_horizon([10, 10, 10], 0.84) == pytest.approx(10 + 8.4 + 7.056)


def test_chip_expiry_warnings_escalate_toward_gameweek_nineteen():
    from fplai.optimiser.chips import expiry_warnings

    assert expiry_warnings(10, []) == []                    # too early to nag
    mid = expiry_warnings(16, [])
    assert mid and all(w["severity"] in ("medium", "high") for w in mid)
    late = expiry_warnings(19, [])
    assert late and any(w["severity"] == "critical" for w in late)
    assert expiry_warnings(25, []) == []                    # set 1 is gone


def test_chip_calendar_refuses_to_guess_before_fixtures_vary():
    """Every club plays once a week early on, so naming a gameweek is false precision."""
    from fplai.optimiser.chips import _has_fixture_variation

    flat = {gw: 11 for gw in range(1, 39)}
    assert _has_fixture_variation(flat) is False
    varied = {**flat, 25: 13, 29: 8}          # a double and a blank
    assert _has_fixture_variation(varied) is True


def test_chip_recommendation_marks_itself_unactionable_when_confidence_is_none():
    from fplai.optimiser.chips import ChipRecommendation

    guess = ChipRecommendation(chip="bboost", best_gw=1, gain=12.0, confidence="none")
    assert guess.actionable is False
    assert guess.to_dict()["actionable"] is False

    real = ChipRecommendation(chip="bboost", best_gw=34, gain=18.0, confidence="high")
    assert real.actionable is True


def test_forcing_a_player_out_of_an_initial_build_actually_excludes_him():
    """A what-if that silently ignores its constraint reports a cost of zero, which is
    worse than refusing — it looks like the idea was free."""
    from fplai.optimiser.planner import PlanContext
    from fplai.optimiser.recommend import _initial_squad_plan
    from fplai.optimiser.risk import RiskProfile

    cands = {c.player_id: c for c in _candidates(n_per_pos=10)}
    ctx = PlanContext(start_gw=1, horizon=1, current_squad=[], bank=0, free_transfers=1)

    unconstrained = _initial_squad_plan({1: cands}, ctx, RiskProfile(), 1)
    # Exclude someone the solver actually wanted. The globally highest-utility player is
    # not necessarily picked — a premium goalkeeper is deliberately poor value.
    picked = max(unconstrained.gameweeks[0].xi, key=lambda pid: cands[pid].utility)

    ctx.force_out = [picked]
    constrained = _initial_squad_plan({1: cands}, ctx, RiskProfile(), 1)
    assert picked not in constrained.gameweeks[0].squad
    # Removing an option can never help. The synthetic pool has interchangeable players,
    # so an equal-value replacement is expected; the tolerance is for float noise only.
    assert constrained.exp_points_gw <= unconstrained.exp_points_gw + 1e-6


def test_forcing_a_player_into_an_initial_build_includes_him():
    from fplai.optimiser.planner import PlanContext
    from fplai.optimiser.recommend import _initial_squad_plan
    from fplai.optimiser.risk import RiskProfile

    cands = {c.player_id: c for c in _candidates(n_per_pos=10)}
    worst = min(cands.values(), key=lambda c: c.utility)
    ctx = PlanContext(start_gw=1, horizon=1, current_squad=[], bank=0, free_transfers=1,
                      force_in=[worst.player_id])
    plan = _initial_squad_plan({1: cands}, ctx, RiskProfile(), 1)
    assert worst.player_id in plan.gameweeks[0].squad
