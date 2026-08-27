"""Every feature in docs/05, sections A-F. Importing this module registers them.

Each builder is a pure function of the FeatureCtx, which is already truncated at
`as_of`, so leakage is structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from ..defaults import DEFCON_THRESHOLD
from .registry import FeatureCtx, ewma, feature, mean, rate_per90, safe_div, shrink

# Positional priors for shrinking low-minute players toward something sane.
POS_XG90_PRIOR = {"GK": 0.005, "DEF": 0.04, "MID": 0.12, "FWD": 0.30}
POS_XA90_PRIOR = {"GK": 0.005, "DEF": 0.05, "MID": 0.13, "FWD": 0.10}
POS_DEFCON90_PRIOR = {"GK": 1.0, "DEF": 8.5, "MID": 7.0, "FWD": 3.0}


def _last(ctx: FeatureCtx, n: int) -> list[dict]:
    return ctx.history[:n]


def _mins(rows: list[dict]) -> list[float]:
    return [r.get("minutes") or 0 for r in rows]


def _played(rows: list[dict]) -> list[dict]:
    return [r for r in rows if (r.get("minutes") or 0) > 0]


def _dt(s: str | None):
    """Parse a stored timestamp, always offset-aware.

    Everything written through `utcnow()` carries an offset, but a few columns are plain
    dates — `players.birth_date` is `1995-09-15`. Those parse naive, and subtracting a
    naive datetime from an aware `as_of` raises TypeError, which the feature registry
    swallows as "no value". That is why `age` was null for every player in every season
    despite 578 of 595 having a birth date on file. Assume UTC rather than hand back a
    value that cannot be compared with anything else here.
    """
    if not s:
        return None
    try:
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ══ A. Player form and underlying performance ═══════════════════════════════════

for _window in (1, 3, 5, 10):

    def _make_mins(n: int):
        @feature(f"mins_last{n}", deps=["player_fixture_stats"], group="form", missing="zero",
                 description=f"Total minutes over the last {n} fixtures")
        def _f(ctx: FeatureCtx, _n=n) -> float | None:
            rows = _last(ctx, _n)
            return float(sum(_mins(rows))) if rows else None

        return _f

    _make_mins(_window)


@feature("mins_last5_weighted", deps=["player_fixture_stats"], version=2, group="form",
         missing="zero")
def mins_last5_weighted(ctx: FeatureCtx) -> float | None:
    """Exponentially-weighted minutes, half-life 3 games."""
    return ewma(_mins(_last(ctx, 5)), half_life=3.0)


@feature("starts_last5", deps=["player_fixture_stats"], group="form", missing="zero")
def starts_last5(ctx: FeatureCtx) -> float | None:
    rows = _last(ctx, 5)
    if not rows:
        return None
    return float(sum(1 for r in rows if (r.get("starts") or 0) or (r.get("minutes") or 0) >= 60))


@feature("start_streak", deps=["player_fixture_stats"], group="form", missing="zero")
def start_streak(ctx: FeatureCtx) -> float:
    """Consecutive starts — the strongest single minutes predictor."""
    streak = 0
    for r in ctx.history:
        if (r.get("starts") or 0) or (r.get("minutes") or 0) >= 60:
            streak += 1
        else:
            break
    return float(streak)


@feature("sub_appearance_rate", deps=["player_fixture_stats"], group="form", missing="zero")
def sub_appearance_rate(ctx: FeatureCtx) -> float | None:
    """Separates rotation risk from cameo risk."""
    rows = _last(ctx, 10)
    if not rows:
        return None
    subs = sum(1 for r in rows if 0 < (r.get("minutes") or 0) < 60)
    return subs / len(rows)


@feature("avg_sub_on_minute", deps=["player_fixture_stats"], group="form")
def avg_sub_on_minute(ctx: FeatureCtx) -> float | None:
    cameos = [90 - (r.get("minutes") or 0) for r in _last(ctx, 10)
              if 0 < (r.get("minutes") or 0) < 60]
    return mean(cameos)


def _per90_feature(name: str, col: str, prior_map: dict, n: int, group: str = "form"):
    @feature(f"{name}_last{n}", deps=["player_fixture_stats"], group=group,
             missing="shrink_to_prior", description=f"{col} per 90 over last {n}, EWMA-weighted")
    def _f(ctx: FeatureCtx, _col=col, _n=n, _prior=prior_map) -> float | None:
        rows = _played(_last(ctx, _n))
        minutes = sum(_mins(rows))
        if minutes <= 0:
            return None
        total = sum(r.get(_col) or 0 for r in rows)
        raw = rate_per90(total, minutes)
        return shrink(raw, _prior.get(ctx.position, 0.1), minutes / 90.0)

    return _f


for _window in (3, 5, 10):
    _per90_feature("xg90", "xg", POS_XG90_PRIOR, _window)
    _per90_feature("xa90", "xa", POS_XA90_PRIOR, _window)
    _per90_feature("npxg90", "npxg", POS_XG90_PRIOR, _window)


@feature("xgi90_last5", deps=["player_fixture_stats"], group="form", missing="shrink_to_prior")
def xgi90_last5(ctx: FeatureCtx) -> float | None:
    """Expected goal involvement per 90."""
    rows = _played(_last(ctx, 5))
    minutes = sum(_mins(rows))
    if minutes <= 0:
        return None
    total = sum((r.get("xg") or 0) + (r.get("xa") or 0) for r in rows)
    return rate_per90(total, minutes)


for _stat_col, _stat_name in (
    ("shots", "shots90"),
    ("shots_on_target", "sot90"),
    ("touches_in_box", "touches_box90"),
    ("big_chances", "big_chances90"),
    ("key_passes", "key_passes90"),
):

    def _make_vol(col: str, fname: str):
        @feature(fname, deps=["player_fixture_stats"], group="form", missing="nan",
                 description=f"{col} per 90 over the last 5 — volume is more stable than conversion")
        def _f(ctx: FeatureCtx, _col=col) -> float | None:
            rows = _played(_last(ctx, 5))
            minutes = sum(_mins(rows))
            return rate_per90(sum(r.get(_col) or 0 for r in rows), minutes)

        return _f

    _make_vol(_stat_col, _stat_name)


@feature("conversion_ratio", deps=["player_fixture_stats"], group="form")
def conversion_ratio(ctx: FeatureCtx) -> float | None:
    """Goals / xG. Used to *regress* hot streaks, not to chase them."""
    rows = _played(_last(ctx, 10))
    xg = sum(r.get("xg") or 0 for r in rows)
    goals = sum(r.get("goals_scored") or 0 for r in rows)
    return safe_div(goals, xg)


@feature("xg_overperformance_last10", deps=["player_fixture_stats"], group="form", missing="zero")
def xg_overperformance_last10(ctx: FeatureCtx) -> float | None:
    """Explicit mean-reversion signal: goals minus xG."""
    rows = _played(_last(ctx, 10))
    if not rows:
        return None
    return sum((r.get("goals_scored") or 0) - (r.get("xg") or 0) for r in rows)


@feature("defcon_actions90", deps=["player_fixture_stats"], group="defcon",
         missing="shrink_to_prior")
def defcon_actions90(ctx: FeatureCtx) -> float | None:
    """CBIT (DEF) or CBIRT (MID/FWD) count per 90."""
    rows = _played(_last(ctx, 10))
    minutes = sum(_mins(rows))
    if minutes <= 0:
        return None
    total = sum(_defcon_count(r, ctx.position) for r in rows)
    return shrink(rate_per90(total, minutes), POS_DEFCON90_PRIOR.get(ctx.position, 5.0),
                  minutes / 90.0)


def _defcon_count(row: dict, position: str) -> float:
    """Prefer the official count; reconstruct from FBref components when it is absent."""
    if row.get("defensive_contribution") is not None:
        return float(row["defensive_contribution"])
    cbi = sum(row.get(c) or 0 for c in ("clearances", "blocks", "interceptions", "tackles"))
    if position != "DEF":
        cbi += row.get("recoveries") or 0
    return float(cbi)


for _window in (5, 10):

    def _make_hit_rate(n: int):
        @feature(f"defcon_hit_rate_last{n}", deps=["player_fixture_stats"], group="defcon",
                 missing="zero",
                 description="Share of games clearing the DefCon threshold — this matters more "
                             "than the mean, because the payoff is a step function")
        def _f(ctx: FeatureCtx, _n=n) -> float | None:
            rows = [r for r in _last(ctx, _n) if (r.get("minutes") or 0) >= 45]
            if not rows:
                return None
            thr = DEFCON_THRESHOLD.get(ctx.position, 12)
            return sum(1 for r in rows if _defcon_count(r, ctx.position) >= thr) / len(rows)

        return _f

    _make_hit_rate(_window)


@feature("defcon_margin", deps=["player_fixture_stats"], group="defcon")
def defcon_margin(ctx: FeatureCtx) -> float | None:
    """Mean distance above/below threshold — how safe a DefCon asset really is."""
    rows = [r for r in _last(ctx, 10) if (r.get("minutes") or 0) >= 45]
    if not rows:
        return None
    thr = DEFCON_THRESHOLD.get(ctx.position, 12)
    return mean([_defcon_count(r, ctx.position) - thr for r in rows])


@feature("bps90_last5", deps=["player_fixture_stats"], group="bonus", missing="nan")
def bps90_last5(ctx: FeatureCtx) -> float | None:
    rows = _played(_last(ctx, 5))
    return rate_per90(sum(r.get("bps") or 0 for r in rows), sum(_mins(rows)))


@feature("bonus_rate_last10", deps=["player_fixture_stats"], group="bonus", missing="zero")
def bonus_rate_last10(ctx: FeatureCtx) -> float | None:
    rows = _played(_last(ctx, 10))
    if not rows:
        return None
    return sum(1 for r in rows if (r.get("bonus") or 0) > 0) / len(rows)


@feature("season_bps_regime", deps=[], group="bonus", missing="zero")
def season_bps_regime(ctx: FeatureCtx) -> float:
    """BPS was retuned for 2026/27 to cut DefCon overlap and favour GKs, full-backs and
    attackers. Coefficients learned from 2025/26 are therefore biased; this categorical
    lets the model separate the regimes (see docs/06, model 4)."""
    try:
        return float(int(ctx.season_id.split("-")[0]) >= 2026)
    except (ValueError, IndexError):
        return 0.0


@feature("saves90", deps=["player_fixture_stats"], group="form", missing="zero")
def saves90(ctx: FeatureCtx) -> float | None:
    if ctx.position != "GK":
        return 0.0
    rows = _played(_last(ctx, 10))
    return rate_per90(sum(r.get("saves") or 0 for r in rows), sum(_mins(rows)))


@feature("saves_per_shot_faced", deps=["player_fixture_stats"], group="form")
def saves_per_shot_faced(ctx: FeatureCtx) -> float | None:
    if ctx.position != "GK":
        return None
    rows = _played(_last(ctx, 10))
    saves = sum(r.get("saves") or 0 for r in rows)
    conceded = sum(r.get("goals_conceded") or 0 for r in rows)
    return safe_div(saves, saves + conceded)


@feature("is_first_pen_taker", deps=["set_piece_roles"], group="setpiece", missing="zero")
def is_first_pen_taker(ctx: FeatureCtx) -> float:
    return float(ctx.set_pieces.get("penalties") == 1)


@feature("pens_taken_share", deps=["set_piece_roles"], group="setpiece", missing="zero")
def pens_taken_share(ctx: FeatureCtx) -> float:
    rank = ctx.set_pieces.get("penalties")
    return {1: 0.9, 2: 0.08, 3: 0.02}.get(rank, 0.0)


@feature("corner_duty", deps=["set_piece_roles"], group="setpiece", missing="zero")
def corner_duty(ctx: FeatureCtx) -> float:
    ranks = [ctx.set_pieces.get("corners_left"), ctx.set_pieces.get("corners_right")]
    return float(any(r == 1 for r in ranks if r))


@feature("direct_fk_duty", deps=["set_piece_roles"], group="setpiece", missing="zero")
def direct_fk_duty(ctx: FeatureCtx) -> float:
    return float(ctx.set_pieces.get("direct_fk") == 1)


@feature("card_rate90", deps=["player_fixture_stats"], group="form", missing="zero")
def card_rate90(ctx: FeatureCtx) -> float | None:
    rows = _played(_last(ctx, 10))
    cards = sum((r.get("yellow_cards") or 0) + (r.get("red_cards") or 0) for r in rows)
    return rate_per90(cards, sum(_mins(rows)))


@feature("age", deps=["players"], group="bio")
def age(ctx: FeatureCtx) -> float | None:
    birth = _dt(ctx.extras.get("birth_date"))
    asof = _dt(ctx.as_of)
    if not birth or not asof:
        return None
    return (asof - birth).days / 365.25


@feature("days_since_debut", deps=["player_fixture_stats"], group="bio")
def days_since_debut(ctx: FeatureCtx) -> float | None:
    if not ctx.history:
        return None
    first = _dt(ctx.history[-1].get("kickoff_utc"))
    asof = _dt(ctx.as_of)
    return (asof - first).days if first and asof else None


@feature("promoted_club_flag", deps=["teams"], group="bio", missing="zero")
def promoted_club_flag(ctx: FeatureCtx) -> float:
    """Coventry, Ipswich and Hull came up for 2026/27 and have no PL history to learn from."""
    return float(bool(ctx.extras.get("promoted")))


@feature("transfermarkt_value_pct_of_squad", deps=["feature_values"], group="bio")
def transfermarkt_value_pct_of_squad(ctx: FeatureCtx) -> float | None:
    """The only useful prior for a promoted club's players."""
    val = ctx.extras.get("transfermarkt_value_eur")
    squad = ctx.extras.get("squad_value_eur")
    return safe_div(val, squad)


# ══ B. Fixture and opponent ═════════════════════════════════════════════════════


@feature("is_home", deps=["fixtures"], group="fixture", missing="zero")
def is_home(ctx: FeatureCtx) -> float:
    return float(ctx.is_home)


@feature("fdr_official", deps=["fixtures"], group="fixture")
def fdr_official(ctx: FeatureCtx) -> float | None:
    """Kept only as a comparison baseline against the model's own difficulty."""
    return ctx.extras.get("fdr")


@feature("opponent_defence_rating", deps=["teams"], group="fixture")
def opponent_defence_rating(ctx: FeatureCtx) -> float | None:
    return ctx.extras.get("opp_defence_rating")


@feature("opponent_attack_rating", deps=["teams"], group="fixture")
def opponent_attack_rating(ctx: FeatureCtx) -> float | None:
    return ctx.extras.get("opp_attack_rating")


@feature("opp_xga_per_game_last6", deps=["player_fixture_stats"], group="fixture")
def opp_xga_per_game_last6(ctx: FeatureCtx) -> float | None:
    rows = ctx.opp_history[:6]
    return mean([r.get("goals_conceded") for r in rows]) if rows else None


@feature("opp_clean_sheet_rate", deps=["player_fixture_stats"], group="fixture", missing="zero")
def opp_clean_sheet_rate(ctx: FeatureCtx) -> float | None:
    rows = ctx.opp_history[:10]
    if not rows:
        return None
    return sum(1 for r in rows if (r.get("goals_conceded") or 0) == 0) / len(rows)


@feature("opp_goals_conceded_ewma", deps=["player_fixture_stats"], group="fixture")
def opp_goals_conceded_ewma(ctx: FeatureCtx) -> float | None:
    return ewma([r.get("goals_conceded") for r in ctx.opp_history[:10]], half_life=5)


@feature("team_expected_goals", deps=["odds_snapshots", "team_model"], group="fixture")
def team_expected_goals(ctx: FeatureCtx) -> float | None:
    """From the market where odds exist, the team model otherwise."""
    return ctx.odds.get("team_lambda") or ctx.extras.get("model_team_lambda")


@feature("opp_expected_goals", deps=["odds_snapshots", "team_model"], group="fixture")
def opp_expected_goals(ctx: FeatureCtx) -> float | None:
    return ctx.odds.get("opp_lambda") or ctx.extras.get("model_opp_lambda")


@feature("p_clean_sheet_odds", deps=["odds_snapshots"], group="fixture")
def p_clean_sheet_odds(ctx: FeatureCtx) -> float | None:
    """Devigged market probability where the market exists."""
    lam = ctx.odds.get("opp_lambda")
    return math.exp(-lam) if lam is not None else None


@feature("p_anytime_scorer_odds", deps=["odds_snapshots"], group="fixture")
def p_anytime_scorer_odds(ctx: FeatureCtx) -> float | None:
    """A *feature*, not the answer — but close to a free market-calibrated goal model."""
    return ctx.odds.get("p_anytime_scorer")


@feature("odds_movement_48h", deps=["odds_snapshots"], group="fixture", missing="zero")
def odds_movement_48h(ctx: FeatureCtx) -> float | None:
    """Drift in team goal expectation. Steam is information."""
    return ctx.odds.get("movement_48h")


for _window in (3, 5, 8):

    def _make_run(n: int):
        @feature(f"fixture_run_score_next{n}", deps=["fixtures"], group="fixture",
                 description="Decayed sum of upcoming difficulty — the planner's headline number")
        def _f(ctx: FeatureCtx, _n=n) -> float | None:
            rows = ctx.upcoming[:_n]
            if not rows:
                return None
            total = 0.0
            for i, fx in enumerate(rows):
                difficulty = fx.get("difficulty")
                if difficulty is None:
                    continue
                total += (0.84**i) * (6 - difficulty)
            return total

        return _f

    _make_run(_window)


@feature("n_fixtures_this_gw", deps=["fixtures"], group="fixture", missing="zero")
def n_fixtures_this_gw(ctx: FeatureCtx) -> float:
    return float(ctx.extras.get("n_fixtures_this_gw", 1))


@feature("dgw_flag", deps=["fixtures"], group="fixture", missing="zero")
def dgw_flag(ctx: FeatureCtx) -> float:
    return float(ctx.extras.get("n_fixtures_this_gw", 1) > 1)


@feature("bgw_flag", deps=["fixtures"], group="fixture", missing="zero")
def bgw_flag(ctx: FeatureCtx) -> float:
    return float(ctx.extras.get("n_fixtures_this_gw", 1) == 0)


@feature("kickoff_slot", deps=["fixtures"], group="fixture", missing="zero")
def kickoff_slot(ctx: FeatureCtx) -> float | None:
    """0 early Sat, 1 Sat afternoon, 2 Sat evening, 3 Sunday, 4 midweek. Affects rotation."""
    ko = _dt(ctx.extras.get("kickoff_utc"))
    if not ko:
        return None
    if ko.weekday() in (1, 2, 3):
        return 4.0
    if ko.weekday() == 6:
        return 3.0
    return float(min(2, max(0, (ko.hour - 11) // 3)))


@feature("derby_flag", deps=["fixtures"], group="fixture", missing="zero")
def derby_flag(ctx: FeatureCtx) -> float:
    return float(bool(ctx.extras.get("is_derby")))


@feature("dead_rubber_flag", deps=["fixtures"], group="fixture", missing="zero")
def dead_rubber_flag(ctx: FeatureCtx) -> float:
    """End-of-season motivation. Crude, but real."""
    return float(ctx.gameweek >= 36 and bool(ctx.extras.get("mid_table")))


# ══ C. Congestion, rest and travel ══════════════════════════════════════════════


@feature("days_since_last_match", deps=["player_fixture_stats"], group="congestion")
def days_since_last_match(ctx: FeatureCtx) -> float | None:
    """Player-level, not team-level — a benched player is fresh."""
    played = _played(ctx.history)
    if not played:
        return None
    last = _dt(played[0].get("kickoff_utc"))
    asof = _dt(ctx.as_of)
    return (asof - last).total_seconds() / 86400 if last and asof else None


for _days in (7, 14, 21):

    def _make_load(days: int):
        @feature(f"player_minutes_last_{days}_days", deps=["player_fixture_stats"],
                 group="congestion", missing="zero",
                 description=f"Minutes in the last {days} days — the real fatigue proxy")
        def _f(ctx: FeatureCtx, _window_days=days) -> float | None:
            asof = _dt(ctx.as_of)
            if not asof:
                return None
            cutoff = asof - timedelta(days=_window_days)
            return float(
                sum(
                    r.get("minutes") or 0
                    for r in ctx.history
                    if (_dt(r.get("kickoff_utc")) or asof) >= cutoff
                )
            )

        return _f

    _make_load(_days)


@feature("team_matches_next_14_days", deps=["fixtures"], group="congestion", missing="zero")
def team_matches_next_14_days(ctx: FeatureCtx) -> float:
    """Includes UCL/UEL/FA Cup/EFL — this is the rotation driver."""
    asof = _dt(ctx.as_of)
    if not asof:
        return 0.0
    horizon = asof + timedelta(days=14)
    return float(
        sum(1 for f in ctx.upcoming if (_dt(f.get("kickoff_utc")) or horizon) <= horizon)
    )


@feature("midweek_european_flag", deps=["fixtures"], group="congestion", missing="zero")
def midweek_european_flag(ctx: FeatureCtx) -> float:
    asof = _dt(ctx.as_of)
    if not asof:
        return 0.0
    window = asof + timedelta(days=5)
    return float(
        any(
            f.get("competition") in ("UCL", "UEL", "UECL")
            and (_dt(f.get("kickoff_utc")) or window) <= window
            for f in ctx.upcoming
        )
    )


@feature("european_competition_tier", deps=["fixtures"], group="congestion", missing="zero")
def european_competition_tier(ctx: FeatureCtx) -> float:
    """Thursday Europa is worse than Tuesday UCL for a Saturday lunchtime kickoff."""
    tiers = {"UCL": 1.0, "UEL": 2.0, "UECL": 3.0}
    for f in ctx.upcoming[:4]:
        if f.get("competition") in tiers:
            return tiers[f["competition"]]
    return 0.0


@feature("days_rest_diff_vs_opponent", deps=["fixtures"], group="congestion", missing="zero")
def days_rest_diff_vs_opponent(ctx: FeatureCtx) -> float | None:
    """One team on 3 days' rest against one on 7 is a genuine edge."""
    mine = ctx.extras.get("team_days_rest")
    theirs = ctx.extras.get("opp_days_rest")
    if mine is None or theirs is None:
        return None
    return float(mine - theirs)


@feature("international_break_flag", deps=["fixtures"], group="congestion", missing="zero")
def international_break_flag(ctx: FeatureCtx) -> float:
    return float(bool(ctx.extras.get("after_intl_break")))


@feature("intl_travel_km", deps=["players"], group="congestion", missing="zero")
def intl_travel_km(ctx: FeatureCtx) -> float:
    """Long-haul returnees get benched. Approximated from nationality -> federation."""
    if not ctx.extras.get("after_intl_break"):
        return 0.0
    federation = ctx.extras.get("federation", "UEFA")
    return {"UEFA": 800.0, "CAF": 5000.0, "CONMEBOL": 10000.0, "CONCACAF": 7000.0,
            "AFC": 9000.0, "OFC": 16000.0}.get(federation, 800.0)


@feature("manager_rotation_index", deps=["player_fixture_stats"], group="congestion")
def manager_rotation_index(ctx: FeatureCtx) -> float | None:
    """How much this manager's XI churns in congested weeks versus normal ones."""
    return ctx.extras.get("manager_rotation_index")


@feature("manager_tenure_days", deps=["teams"], group="congestion")
def manager_tenure_days(ctx: FeatureCtx) -> float | None:
    return ctx.extras.get("manager_tenure_days")


@feature("new_manager_flag", deps=["teams"], group="congestion", missing="zero")
def new_manager_flag(ctx: FeatureCtx) -> float:
    """Nine new managers this season, so historic team priors are shakier than usual.
    Giving the model this lets it widen its own uncertainty rather than being confidently wrong."""
    tenure = ctx.extras.get("manager_tenure_days")
    return float(tenure is not None and tenure < 120)


# ══ D. Availability and news-derived ════════════════════════════════════════════


@feature("fpl_chance_of_playing", deps=["availability"], group="availability", missing="indicator")
def fpl_chance_of_playing(ctx: FeatureCtx) -> float | None:
    for a in ctx.availability:
        if a["source_id"] == "fpl_official":
            return float(a["chance_pct"]) if a["chance_pct"] is not None else (
                100.0 if a["status"] == "available" else 0.0
            )
    return None


STATUS_SCORE = {"available": 1.0, "doubt": 0.5, "injured": 0.0, "suspended": 0.0, "unknown": 0.7}
SOURCE_WEIGHT = {"fpl_official": 2.0, "premier_injuries": 1.5, "physioroom": 1.0,
                 "transfermarkt": 0.6, "claims": 0.8}


@feature("injury_status_consensus", deps=["availability", "claims"], group="availability",
         missing="indicator")
def injury_status_consensus(ctx: FeatureCtx) -> float | None:
    """Weighted vote across premierinjuries, physioroom, transfermarkt and claims."""
    num = den = 0.0
    for a in ctx.availability:
        w = SOURCE_WEIGHT.get(a["source_id"], 0.5)
        score = (a["chance_pct"] / 100.0) if a["chance_pct"] is not None else STATUS_SCORE.get(
            a["status"], 0.7
        )
        num += w * score
        den += w
    return num / den if den else None


@feature("source_disagreement_score", deps=["availability"], group="availability", missing="zero")
def source_disagreement_score(ctx: FeatureCtx) -> float | None:
    """Variance across sources. High disagreement should *widen* the minutes distribution,
    not shift it — the simulator reads this directly."""
    scores = [
        (a["chance_pct"] / 100.0) if a["chance_pct"] is not None
        else STATUS_SCORE.get(a["status"], 0.7)
        for a in ctx.availability
    ]
    if len(scores) < 2:
        return 0.0
    m = sum(scores) / len(scores)
    return math.sqrt(sum((s - m) ** 2 for s in scores) / len(scores))


@feature("days_since_injury_report", deps=["availability"], group="availability")
def days_since_injury_report(ctx: FeatureCtx) -> float | None:
    asof = _dt(ctx.as_of)
    reports = [a for a in ctx.availability if a["status"] in ("injured", "doubt")]
    if not reports or not asof:
        return None
    latest = max((_dt(a["observed_at"]) for a in reports if _dt(a["observed_at"])), default=None)
    return (asof - latest).total_seconds() / 86400 if latest else None


@feature("expected_return_gw", deps=["availability"], group="availability")
def expected_return_gw(ctx: FeatureCtx) -> float | None:
    return ctx.extras.get("expected_return_gw")


@feature("news_signal_gap", deps=["availability", "claims"], group="availability", missing="zero")
def news_signal_gap(ctx: FeatureCtx) -> float | None:
    """**The edge feature.** Consensus availability minus FPL's official flag.

    Positive gap = the news says fit before FPL has updated its own number. That window
    is the most defensible edge in the whole design (docs/14, grade A).
    """
    consensus = injury_status_consensus(ctx)
    official = fpl_chance_of_playing(ctx)
    if consensus is None or official is None:
        return None
    return consensus - (official / 100.0)


@feature("press_conference_recency", deps=["claims"], group="availability")
def press_conference_recency(ctx: FeatureCtx) -> float | None:
    """Hours since the manager last spoke — the most informative *timing* feature there is."""
    asof = _dt(ctx.as_of)
    times = [
        _dt(c["extracted_at"]) for c in ctx.claims if c.get("claim_type") == "manager_quote"
    ]
    times = [t for t in times if t]
    if not times or not asof:
        return None
    return (asof - max(times)).total_seconds() / 3600


@feature("predicted_lineup_prob", deps=["lineups"], group="availability", missing="indicator")
def predicted_lineup_prob(ctx: FeatureCtx) -> float | None:
    return ctx.extras.get("predicted_lineup_prob")


# ══ E. Market and ownership ═════════════════════════════════════════════════════


@feature("owned_pct", deps=["ownership_snapshots"], group="market", missing="zero")
def owned_pct(ctx: FeatureCtx) -> float | None:
    return ctx.ownership.get("overall")


@feature("owned_pct_top10k", deps=["ownership_snapshots"], group="market")
def owned_pct_top10k(ctx: FeatureCtx) -> float | None:
    return ctx.ownership.get("top10k")


@feature("effective_ownership", deps=["ownership_snapshots"], group="market")
def effective_ownership(ctx: FeatureCtx) -> float | None:
    return ctx.ownership.get("effective")


@feature("is_template_flag", deps=["ownership_snapshots"], group="market", missing="zero")
def is_template_flag(ctx: FeatureCtx) -> float:
    eo = ctx.ownership.get("effective") or ctx.ownership.get("overall") or 0
    return float(eo > 40)


@feature("net_transfers_24h", deps=["player_prices"], group="market", missing="zero")
def net_transfers_24h(ctx: FeatureCtx) -> float | None:
    return ctx.price.get("net_transfers")


@feature("ownership_momentum_24h", deps=["ownership_snapshots"], group="market", missing="zero")
def ownership_momentum_24h(ctx: FeatureCtx) -> float | None:
    return ctx.ownership.get("momentum_24h")


@feature("price_change_progress", deps=["player_prices"], group="market", missing="zero")
def price_change_progress(ctx: FeatureCtx) -> float | None:
    """Estimated distance to the next rise/fall threshold, roughly normalised to [-1, 1]."""
    net = ctx.price.get("net_transfers")
    owners = ctx.ownership.get("overall")
    if net is None or not owners:
        return None
    return max(-1.0, min(1.0, net / (owners * 100_000 * 0.08 + 1)))


@feature("price", deps=["player_prices"], group="market", missing="nan")
def price(ctx: FeatureCtx) -> float | None:
    """Current price in tenths of a million."""
    return ctx.price.get("price")


@feature("price_rank_in_position", deps=["player_prices"], group="market", missing="zero")
def price_rank_in_position(ctx: FeatureCtx) -> float | None:
    """Price percentile within the player's position, 0..1.

    The single best cold-start quality prior: before a ball is kicked, FPL's own pricing
    is the only signal that separates a £15.5m striker from a £4.5m one. Without it the
    minutes and rate models rank a bench forward alongside Haaland in GW1.
    """
    return ctx.extras.get("price_percentile_in_position")


@feature("value_score", deps=["predictions"], group="market")
def value_score(ctx: FeatureCtx) -> float | None:
    """Expected points per £m over the horizon."""
    ep = ctx.extras.get("exp_points_horizon")
    price = ctx.price.get("price")
    return safe_div(ep, price / 10.0 if price else None)


@feature("differential_score", deps=["predictions", "ownership_snapshots"], group="market")
def differential_score(ctx: FeatureCtx) -> float | None:
    """Predicted-points percentile minus ownership percentile."""
    ep_pct = ctx.extras.get("ep_percentile")
    own_pct = ctx.extras.get("ownership_percentile")
    if ep_pct is None or own_pct is None:
        return None
    return ep_pct - own_pct


# ══ F. Text-derived aggregates ══════════════════════════════════════════════════
#
# All trust-weighted and recency-decayed with a 3-day half-life. By design these are
# never allowed to be top-3 by importance in the points model — if they are, something
# has gone wrong (docs/05 section F; the guard is tested in docs/12 layer 5).

TEXT_HALF_LIFE_DAYS = 3.0


def _decayed(ctx: FeatureCtx, predicate) -> float:
    asof = _dt(ctx.as_of)
    total = 0.0
    for c in ctx.claims:
        if not predicate(c):
            continue
        t = _dt(c.get("extracted_at"))
        days = (asof - t).total_seconds() / 86400 if (t and asof) else 0.0
        decay = 0.5 ** (max(0.0, days) / TEXT_HALF_LIFE_DAYS)
        total += decay * float(c.get("trust_weight", 1.0))
    return total


@feature("yt_mention_count", deps=["claims"], group="text", missing="zero")
def yt_mention_count(ctx: FeatureCtx) -> float:
    return _decayed(ctx, lambda c: c.get("platform") == "video")


for _call_kind, _claim_types in (
    ("buy", ("recommendation",)),
    ("sell", ("avoid",)),
    ("captain", ("captain_pick",)),
    ("avoid", ("avoid",)),
):

    def _make_calls(call: str, types: tuple):
        @feature(f"yt_{call}_calls", deps=["claims"], group="text", missing="zero")
        def _f(ctx: FeatureCtx, _types=types, _call=call) -> float:
            want_positive = _call in ("buy", "captain")
            return _decayed(
                ctx,
                lambda c: c.get("claim_type") in _types
                and (c.get("stance") == "positive") == want_positive,
            )

        return _f

    _make_calls(_call_kind, _claim_types)


@feature("yt_net_sentiment", deps=["claims"], group="text", missing="zero")
def yt_net_sentiment(ctx: FeatureCtx) -> float | None:
    """Trust-weighted mean sentiment: channel accuracy x recency decay x dupe collapse."""
    asof = _dt(ctx.as_of)
    num = den = 0.0
    seen_groups = set()
    for c in ctx.claims:
        if c.get("sentiment") is None:
            continue
        group = c.get("semantic_group")
        if group and group in seen_groups:
            continue  # a story on six sites is one fact, not six confirmations
        if group:
            seen_groups.add(group)
        t = _dt(c.get("extracted_at"))
        days = (asof - t).total_seconds() / 86400 if (t and asof) else 0.0
        w = (0.5 ** (max(0.0, days) / TEXT_HALF_LIFE_DAYS)) * float(c.get("trust_weight", 1.0))
        num += w * float(c["sentiment"])
        den += w
    return num / den if den else None


@feature("yt_consensus_strength", deps=["claims"], group="text", missing="zero")
def yt_consensus_strength(ctx: FeatureCtx) -> float:
    """Agreement across distinct *channels* — not distinct videos, which would double-count."""
    channels = {c.get("channel_id") for c in ctx.claims if c.get("channel_id")}
    if len(channels) < 2:
        return 0.0
    stances = [c.get("stance") for c in ctx.claims if c.get("channel_id") and c.get("stance")]
    if not stances:
        return 0.0
    top = max(stances.count(s) for s in set(stances))
    return top / len(stances)


@feature("pundit_disagreement", deps=["claims"], group="text", missing="zero")
def pundit_disagreement(ctx: FeatureCtx) -> float:
    """Disagreement, not consensus. A hard split is genuine uncertainty and should widen
    the distribution rather than shift the mean (docs/14, grade B)."""
    stances = [c.get("stance") for c in ctx.claims if c.get("stance") in ("positive", "negative")]
    if len(stances) < 3:
        return 0.0
    pos = stances.count("positive") / len(stances)
    return 1.0 - abs(2 * pos - 1)


@feature("reddit_mention_volume_zscore", deps=["claims"], group="text", missing="zero")
def reddit_mention_volume_zscore(ctx: FeatureCtx) -> float | None:
    vol = _decayed(ctx, lambda c: c.get("platform") == "reddit")
    baseline = ctx.extras.get("reddit_volume_mean")
    sd = ctx.extras.get("reddit_volume_sd")
    if baseline is None or not sd:
        return vol
    return (vol - baseline) / sd


@feature("reddit_sentiment", deps=["claims"], group="text", missing="zero")
def reddit_sentiment(ctx: FeatureCtx) -> float | None:
    vals = [c["sentiment"] for c in ctx.claims
            if c.get("platform") == "reddit" and c.get("sentiment") is not None]
    return mean(vals)


@feature("x_breaking_news_flag", deps=["claims"], group="text", missing="zero")
def x_breaking_news_flag(ctx: FeatureCtx) -> float:
    asof = _dt(ctx.as_of)
    for c in ctx.claims:
        if c.get("platform") != "x" or c.get("claim_type") not in ("injury", "rotation"):
            continue
        t = _dt(c.get("extracted_at"))
        if t and asof and (asof - t).total_seconds() < 6 * 3600:
            return 1.0
    return 0.0


@feature("journalist_tier1_mention", deps=["claims"], group="text", missing="zero")
def journalist_tier1_mention(ctx: FeatureCtx) -> float:
    return _decayed(ctx, lambda c: float(c.get("trust_weight", 1.0)) >= 1.5)


@feature("narrative_novelty", deps=["claims"], group="text", missing="zero")
def narrative_novelty(ctx: FeatureCtx) -> float | None:
    """How different this week's claims are from last week's, by embedding distance."""
    return ctx.extras.get("narrative_novelty")
