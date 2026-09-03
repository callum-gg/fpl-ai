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


def test_an_unused_substitute_is_not_the_close_season():
    """GW1 bench-warmers were being handed the pre-season prior, which is price-led.

    A striker left out of the opening weekend has zero minutes in 14 days and a last
    appearance back in May, which is indistinguishable from July if you only read his own
    row. The guard then skipped the model for the one player it had just learned the most
    about, and a 25%-start forward came back priced as nailed. `days_since_team_match` is
    what tells the two apart: his club played on Saturday.
    """
    from fplai.models.minutes import _between_seasons, predict

    idle = {"starts_last5": 3, "mins_last10": 689, "days_since_last_match": 96,
            "price_rank_in_position": 0.96, "player_minutes_last_14_days": 0}
    assert _between_seasons(idle)                                   # no team signal: unchanged
    assert _between_seasons(dict(idle, days_since_team_match=96))    # genuine close season
    assert not _between_seasons(dict(idle, days_since_team_match=6))  # his club played

    class _Benched:
        def predict_one(self, _features):
            return np.array([0.69, 0.15, 0.16])

    benched = dict(idle, days_since_team_match=6)
    assert predict(_Benched(), benched).p_start == pytest.approx(0.16)
    assert predict(_Benched(), idle).p_start > 0.6


def test_a_source_with_no_verdict_does_not_vote():
    """Scrapers list a player before they know anything, as `unknown` with no percentage.

    Scored as 0.7 that was a vote for mild doubt on a fit player, and it also made him
    disagree with the sources that did know — and disagreement *widens* the minutes
    distribution. Two fabricated signals from one empty row.
    """
    from fplai.features.builders import _avail_score

    assert _avail_score({"status": "unknown", "chance_pct": None}) is None
    assert _avail_score({"status": "available", "chance_pct": None}) == 1.0
    assert _avail_score({"status": "doubt", "chance_pct": 25}) == 0.25


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


def test_nb_sample_uses_one_rate_per_draw_not_their_mean():
    """The rotation-risk case: he plays 90 or he plays nothing, and only the 90s score.

    Sampling at mean(mu) instead of per-draw mu under-stated expected DefCon by roughly
    half across GW1-2. P(X >= threshold) is convex in mu here, so the two are not
    interchangeable and the gap widens as start probability falls.
    """
    from fplai.models.rates import nb_sample

    rng = np.random.default_rng(0)
    n, k, threshold = 40000, 5.0, 10
    # 60% of draws at a full 90 minutes' rate, 40% at zero — mean 5.1, but the mass that
    # clears 10 actions lives entirely in the 8.5 half.
    mu = np.where(rng.random(n) < 0.6, 8.5, 0.0)

    per_draw = nb_sample(rng, mu, k, n)
    collapsed = nb_sample(rng, float(mu.mean()), k, n) * (mu > 0)

    assert (per_draw >= threshold).mean() > 1.5 * (collapsed >= threshold).mean()
    assert per_draw[mu == 0].max() == 0          # no actions without minutes
    # A scalar rate must still behave, since that is the whole API for a fixed-minutes call.
    assert nb_sample(rng, 0.0, k, 50).sum() == 0
    assert nb_sample(rng, 6.0, k, 500).mean() == pytest.approx(6.0, rel=0.25)


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


# ══ promotion gate ═══════════════════════════════════════════════════════════


@pytest.fixture
def _no_incumbent(monkeypatch):
    """`_should_promote` with a controllable incumbent row and auto-promote forced on."""
    from fplai.models import base

    state: dict = {"incumbent": None}

    def fake_query_one(sql, params=()):
        return state["incumbent"]

    monkeypatch.setattr(base, "query_one", fake_query_one)

    class _S:
        model_auto_promote = True

    monkeypatch.setattr(base, "get_settings", lambda: _S())
    return state


def _incumbent(**metrics):
    import json as _json

    return {"metrics_json": _json.dumps(metrics)}


def test_cold_start_promotes_but_an_unmeasured_challenger_never_does(_no_incumbent):
    """The 27 Aug saves90 bug: no MAE on the run, promoted over a measured incumbent."""
    from fplai.models.base import _should_promote

    assert _should_promote("saves90", {"train_rows": 100})           # nothing to beat yet

    _no_incumbent["incumbent"] = _incumbent(mae=1.32, holdout="2025-26:GW34-2025-26:GW38")
    assert not _should_promote("saves90", {"train_rows": 100})       # unmeasured: refuse
    assert not _should_promote("saves90", {"importance": {}})


def test_promotion_prefers_a_head_to_head_on_the_same_holdout(_no_incumbent):
    """Both models answering the same question is the only comparison worth acting on."""
    from fplai.models.base import _should_promote

    # The incumbent's *stored* score is flattering — it came from an easier window.
    _no_incumbent["incumbent"] = _incumbent(log_loss=0.43, holdout="2025-26:GW34-2025-26:GW38")

    # Scored side by side on rows the incumbent never trained on, the challenger wins, so
    # promote — even though its raw number looks worse than the incumbent's stored one.
    assert _should_promote("minutes", {
        "log_loss": 0.97, "holdout": "2026-27:GW1-2026-27:GW1",
        "head_to_head": {"rows": 600, "challenger": {"log_loss": 0.97},
                         "incumbent": {"log_loss": 1.15}}})
    # ...and refuse when the head-to-head goes the other way.
    assert not _should_promote("minutes", {
        "log_loss": 0.97, "holdout": "2026-27:GW1-2026-27:GW1",
        "head_to_head": {"rows": 600, "challenger": {"log_loss": 0.97},
                         "incumbent": {"log_loss": 0.80}}})


def test_higher_is_better_metrics_compare_the_right_way(_no_incumbent):
    from fplai.models.base import _should_promote

    _no_incumbent["incumbent"] = _incumbent(spearman=0.26, holdout="h1")
    assert _should_promote("goals90", {
        "spearman": 0.31, "holdout": "h1",
        "head_to_head": {"challenger": {"spearman": 0.31}, "incumbent": {"spearman": 0.26}}})
    assert not _should_promote("goals90", {
        "spearman": 0.21, "holdout": "h1",
        "head_to_head": {"challenger": {"spearman": 0.21}, "incumbent": {"spearman": 0.26}}})


def test_scores_from_different_holdouts_are_not_treated_as_comparable(_no_incumbent):
    """Without a head-to-head, a mismatched window voids the comparison rather than
    letting an unfalsifiable incumbent sit there forever."""
    from fplai.models.base import _should_promote

    _no_incumbent["incumbent"] = _incumbent(log_loss=0.43, holdout="2025-26:GW34-2025-26:GW38")
    assert _should_promote("minutes", {"log_loss": 0.97, "holdout": "2026-27:GW1-2026-27:GW1"})
    # Same window, no head-to-head: the raw numbers *are* comparable, so honour them.
    same = "2025-26:GW34-2025-26:GW38"
    assert not _should_promote("minutes", {"log_loss": 0.97, "holdout": same})
    assert _should_promote("minutes", {"log_loss": 0.31, "holdout": same})


def test_walk_forward_split_keeps_one_window_across_the_season_boundary():
    """August used to collapse the holdout to a single gameweek of a brand new season."""
    import pandas as pd
    from fplai.models.train import _holdout_span, _split

    rows = [{"season_id": "2025-26", "gameweek": gw, "v": 1} for gw in range(30, 39)]
    rows += [{"season_id": "2026-27", "gameweek": 1, "v": 1}]
    train, valid = _split(pd.DataFrame(rows), holdout_gws=5)

    keys = sorted({(r.season_id, r.gameweek) for r in valid.itertuples()})
    assert len(keys) == 5                       # not 1, even one week into a new season
    assert keys[-1] == ("2026-27", 1)
    assert keys[0] == ("2025-26", 35)          # four from last season, one from this one
    # Strictly walk-forward: nothing in train is later than the earliest holdout key.
    assert max((r.season_id, r.gameweek) for r in train.itertuples()) < keys[0]
    assert _holdout_span(valid) == "2025-26:GW35-2026-27:GW1"


# ══ feature coverage ═════════════════════════════════════════════════════════


def test_no_model_declares_a_feature_nothing_can_build():
    """The invariant that has no exceptions: if a model asks for it, something must
    produce it. `manager_rotation_index` failed this for the whole of GW1-2."""
    import fplai.features.builders  # noqa: F401 - importing registers every builder
    from fplai.features.build import declared_features
    from fplai.features.registry import BLOCK_INDICATORS, REGISTRY

    buildable = set(REGISTRY) | set(BLOCK_INDICATORS)
    for model, names in declared_features().items():
        unwired = sorted(set(names) - buildable)
        assert not unwired, f"{model} declares features nothing builds: {unwired}"


def test_the_minutes_guard_feature_is_actually_computed(player_with_history, seeded_season):
    """`_between_seasons` reads `days_since_team_match` off the computed feature dict.

    It was written, unit-tested against a hand-built dict, and never once produced for a
    real player — so the guard silently no-opped and every unused substitute kept the
    price-led pre-season prior that ranks him as nailed.
    """
    from fplai.features.build import build_ctx
    from fplai.features.registry import compute_all

    values = compute_all(build_ctx(player_with_history, seeded_season, 1))
    assert values["days_since_team_match"] is not None
    assert values["days_since_last_match"] is not None


def test_audit_coverage_names_features_the_store_never_received(
    player_with_history, seeded_season
):
    from fplai.db.engine import writer
    from fplai.features.build import audit_coverage

    def put(fixture_key, name, value):
        with writer() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feature_values(season_id,gameweek,player_id,"
                "fixture_key,name,value,computed_at,feature_version) VALUES(?,?,?,?,?,?,?,?)",
                (seeded_season, 7, player_with_history, fixture_key, name, value,
                 "2026-09-01T00:00:00+00:00", 1),
            )

    put(0, "price", 55.0)
    put(1, "price", 61.0)          # two distinct values -> not constant
    put(0, "is_home", 1.0)
    put(1, "is_home", 1.0)         # one value everywhere -> constant

    report = audit_coverage(seeded_season, 7)
    assert report["healthy"] and report["unwired"] == []
    # Declared, has a builder, but no source has ever written it — a data gap, not a bug.
    assert "predicted_lineup_prob" in report["never_written"]
    # Written for this gameweek but the same value throughout: nothing can split on it.
    assert "is_home" in report["constant"]
    assert "price" not in report["constant"]


# ══ weekly scorecard ═════════════════════════════════════════════════════════


def test_spearman_handles_ties_and_degenerate_input():
    from fplai.models.backtest import _spearman_rho

    assert _spearman_rho([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert _spearman_rho([1, 1, 1], [1, 2, 3]) is None      # no variance, no correlation
    assert _spearman_rho([1, 2], [2, 1]) is None            # too few points to mean anything
    # Monotone but non-linear is still a perfect *rank* correlation.
    assert _spearman_rho([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)


def test_the_scorecard_only_grades_predictions_made_before_the_deadline(
    player_with_history, seeded_season
):
    """Scoring the newest vintage would grade the model on rows written after the matches
    were played — a scorecard that flatters itself is worse than no scorecard."""
    from fplai.db.engine import writer
    from fplai.models.backtest import _predictions_before_deadline

    def put(generated_at, exp_points):
        with writer() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions(player_id,season_id,gameweek,fixture_key,"
                "generated_at,p_start,exp_points,p_haul_10) VALUES(?,?,1,0,?,0.9,?,0.05)",
                (player_with_history, seeded_season, generated_at, exp_points),
            )

    put("2026-08-20T09:00:00+00:00", 4.0)          # before the 2026-08-22T10:00 deadline
    put("2026-08-23T09:00:00+00:00", 99.0)         # after the matches: must be ignored

    rows = _predictions_before_deadline(seeded_season, 1)
    assert [r["exp_points"] for r in rows] == [4.0]


def test_an_artefact_written_by_another_filesystem_still_loads(tmp_path, monkeypatch):
    """Paths were stored absolute, so a row written in the container said
    `/app/data/models/minutes-*.pkl` and nothing on the host could open it. `load_active`
    warned and returned None, and every caller quietly served heuristic numbers instead."""
    import pickle

    from fplai.models import base

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    artefact = models_dir / "minutes-20260101000000.pkl"
    artefact.write_bytes(pickle.dumps({"i am": "the model"}))

    class _S:
        models_dir = str(tmp_path / "models")

    monkeypatch.setattr(base, "get_settings", lambda: _S())

    # The container's path does not exist here, but the filename does.
    assert base._resolve_artefact("/app/data/models/minutes-20260101000000.pkl") == artefact
    assert base._resolve_artefact(str(artefact)) == artefact          # already correct
    assert base._resolve_artefact("/app/data/models/never-trained.pkl") is None


def test_the_head_to_head_only_uses_rows_the_incumbent_never_trained_on(monkeypatch):
    """The subtle half of the promotion bug. The incumbent was fitted a week ago on
    everything up to then, so most of today's walk-forward window is in-sample for it;
    scoring it there flatters it and blocks every honest successor. Measured on a 2020-24
    refit, that gap alone rejected all seven challengers."""
    import json as _json

    import pandas as pd
    from fplai.models import train

    valid = pd.DataFrame(
        [{"season_id": "2026-27", "gameweek": gw, "row": i}
         for gw in (1, 2, 3) for i in range(40)]
    )
    # `_incumbent_score` imports load_active from .base at call time, so patch it there.
    monkeypatch.setattr("fplai.models.base.load_active", lambda name: "the-incumbent")
    monkeypatch.setattr(
        train, "query_one",
        lambda sql, params=(): {"metrics_json": _json.dumps({"trained_through": "2026-27:02"})},
    )

    seen = {}

    def score(model, d=None):
        seen["rows"] = len(d)
        seen["gameweeks"] = sorted(set(d["gameweek"]))
        return {"mae": 1.0}

    result = train._head_to_head("goals90", score, valid, "the-challenger")
    assert result["challenger"] == {"mae": 1.0} and result["incumbent"] == {"mae": 1.0}
    assert result["rows"] == 40
    assert seen["gameweeks"] == [3]        # GW1-2 were in the incumbent's training data
    assert seen["rows"] == 40

    # Nothing postdates it -> no honest comparison, so the caller falls back.
    monkeypatch.setattr(
        train, "query_one",
        lambda sql, params=(): {"metrics_json": _json.dumps({"trained_through": "2026-27:09"})},
    )
    assert train._head_to_head("goals90", score, valid, "the-challenger") is None


def test_training_records_how_far_its_data_reached():
    import pandas as pd
    from fplai.models.train import _last_key

    df = pd.DataFrame([
        {"season_id": "2025-26", "gameweek": 38},
        {"season_id": "2026-27", "gameweek": 2},
        {"season_id": "2026-27", "gameweek": 10},
    ])
    assert _last_key(df) == "2026-27:10"
    assert _last_key(pd.DataFrame()) is None
    # Zero-padded so plain string comparison still orders GW9 before GW10.
    assert _last_key(df) > "2026-27:09"


def test_an_incumbent_of_unknown_provenance_gets_no_head_to_head(monkeypatch):
    """Without `trained_through` there is no slice we can be sure it never saw, and
    assuming 'nothing' would score it in-sample and hand it a win it did not earn."""
    import pandas as pd
    from fplai.models import train

    valid = pd.DataFrame([{"season_id": "2026-27", "gameweek": 3} for _ in range(100)])
    monkeypatch.setattr("fplai.models.base.load_active", lambda name: "the-incumbent")
    monkeypatch.setattr(train, "query_one", lambda sql, params=(): {"metrics_json": "{}"})

    def score(model, d=None):
        raise AssertionError("must not score an incumbent of unknown provenance")

    assert train._head_to_head("goals90", score, valid, "the-challenger") is None


def test_an_unknown_position_code_does_not_abort_a_gameweek():
    """The 2024-25 archive labels 20 players `AM`, which raised KeyError out of the
    simulator's scoring dicts and killed all 38 backtest gameweeks before they started."""
    from fplai.defaults import normalise_position
    from fplai.models.simulate import CS_POINTS, GOAL_POINTS

    assert normalise_position("AM") == "MID"
    assert normalise_position("GKP") == "GK"
    assert normalise_position("gk") == "GK"
    assert normalise_position(None) == "MID"
    assert normalise_position("something new") == "MID"
    for real in ("GK", "DEF", "MID", "FWD"):
        assert normalise_position(real) == real
    # And the scoring tables must not raise even if something slips through unnormalised.
    assert GOAL_POINTS.get("AM", GOAL_POINTS["MID"]) == 5
    assert CS_POINTS.get("AM", CS_POINTS["MID"]) == 1


def test_the_simulator_reports_what_a_player_is_expected_to_lose():
    """`exp_cards_penalty` and `exp_conceded_penalty` were written as NULL on every
    prediction row ever produced, so the UI could show expected returns but never the
    expected deductions behind a number."""
    from fplai.models.simulate import FixtureInput, PlayerInput, simulate_gameweek

    def player(pid, position, cards90=0.4):
        return PlayerInput(
            player_id=pid, position=position, team_id=1 if pid < 10 else 2, fixture_id=1,
            p_start=0.95, p_cameo=0.03, exp_minutes=78.0, goals90=0.1, assists90=0.1,
            defcon_rate90=6.0, defcon_dispersion=5.0, saves90=0.0, cards90=cards90,
            exp_bps=20.0,
        )

    fx = FixtureInput(fixture_id=1, home_team_id=1, away_team_id=2,
                      lambda_home=1.6, lambda_away=1.4, rho=-0.05)
    fx.players = [player(1, "DEF"), player(2, "MID"), player(11, "GK", cards90=0.0)]
    sim = simulate_gameweek([fx], n_sims=3000)

    keeper = sim.components[11]
    defender = sim.components[1]
    midfielder = sim.components[2]

    for comp in (keeper, defender, midfielder):
        assert "cards_penalty" in comp and "conceded_penalty" in comp
        assert comp["cards_penalty"].mean() >= 0
    # A booked defender loses real points. The keeper has no yellow-card rate but reds are
    # modelled independently of it, so his penalty is small rather than zero.
    assert defender["cards_penalty"].mean() > 0.2
    assert 0 <= keeper["cards_penalty"].mean() < 0.1
    # Goals conceded only dock keepers and defenders.
    assert keeper["conceded_penalty"].mean() > 0
    assert defender["conceded_penalty"].mean() > 0
    assert midfielder["conceded_penalty"].mean() == 0
