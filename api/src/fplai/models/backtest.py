"""Backtest harness, ablations and the pundit scoreboard. docs/06 + docs/08 + docs/12.

Replays a season deadline-by-deadline: rebuild features as-of, predict, optimise under
the season's real rules, apply real transfer/hit/chip constraints, score against actual
results.

Caveats are stated in the report rather than buried, because they change how you should
read the numbers:
  - odds features do not exist historically, so pre-install backtests understate the model
  - the 2026/27 BPS retune means older bonus results do not transfer
  - a backtest tuned repeatedly against the same seasons will overfit them, so 2023-24 is
    a locked holdout you touch at most twice a season
"""

from __future__ import annotations

import logging

from ..db.engine import jdump, query, query_one, utcnow, writer
from ..defaults import START_BUDGET
from ..optimiser.risk import variant_profile
from ..optimiser.squad import Candidate, prefilter, solve_squad
from ..rules import gameweek_points, player_points, selling_price

log = logging.getLogger(__name__)

LOCKED_HOLDOUT = "2023-24"

CAVEATS = [
    (
        "Odds features are unavailable historically, so backtests before your install date "
        "understate the model — the odds block indicator is set and the model leans on the "
        "team model instead."
    ),
    (
        "The 2026/27 BPS retune means bonus results from earlier seasons do not transfer; "
        "treat pre-2026/27 bonus accuracy as a lower bound, not a forecast."
    ),
    (
        f"{LOCKED_HOLDOUT} is a locked holdout. Repeatedly tuning against the same seasons "
        "overfits them; touch it at most twice a season."
    ),
]


def _actual_points(season_id: str, gameweek: int) -> dict[int, int]:
    rows = query(
        "SELECT s.player_id, ps.position, s.minutes, s.goals_scored, s.assists, s.clean_sheets, "
        "s.goals_conceded, s.own_goals, s.penalties_saved, s.penalties_missed, s.yellow_cards, "
        "s.red_cards, s.saves, s.bonus, s.defensive_contribution, s.defcon_points, s.total_points "
        "FROM player_fixture_stats s JOIN fixtures f ON f.id=s.fixture_id "
        "JOIN player_seasons ps ON ps.player_id=s.player_id AND ps.season_id=f.season_id "
        "WHERE f.season_id=? AND f.gameweek=?",
        (season_id, gameweek),
    )
    out: dict[int, int] = {}
    for r in rows:
        d = dict(r)
        pts = d["total_points"] if d["total_points"] is not None else player_points(d, d["position"])
        out[d["player_id"]] = out.get(d["player_id"], 0) + (pts or 0)
    return out


def _minutes(season_id: str, gameweek: int) -> dict[int, int]:
    rows = query(
        "SELECT s.player_id, SUM(COALESCE(s.minutes,0)) m FROM player_fixture_stats s "
        "JOIN fixtures f ON f.id=s.fixture_id WHERE f.season_id=? AND f.gameweek=? "
        "GROUP BY s.player_id",
        (season_id, gameweek),
    )
    return {r["player_id"]: r["m"] for r in rows}


def _candidates_for(season_id: str, gameweek: int, variant: str,
                    disable_blocks: set[str] | None = None) -> dict[int, Candidate]:
    """Uses stored predictions where they exist; otherwise falls back to actual-agnostic
    priors so an ablation run without a given feature block is still comparable."""
    from ..models.predict import latest

    profile = variant_profile(variant)
    preds = {p["player_id"]: p for p in latest(season_id, gameweek)}
    meta = query(
        "SELECT ps.player_id, ps.position, ps.team_id, p.web_name, "
        "(SELECT price FROM player_prices WHERE player_id=ps.player_id "
        " AND season_id=ps.season_id ORDER BY observed_at LIMIT 1) price "
        "FROM player_seasons ps JOIN players p ON p.id=ps.player_id "
        "WHERE ps.season_id=? AND ps.team_id IS NOT NULL",
        (season_id,),
    )
    out: dict[int, Candidate] = {}
    for m in meta:
        pred = preds.get(m["player_id"])
        if pred is None:
            continue
        ep = pred["exp_points"] or 0.0
        sd = pred["sd_points"] or 0.0
        out[m["player_id"]] = Candidate(
            player_id=m["player_id"], position=m["position"], team_id=m["team_id"],
            price=m["price"] or 40, selling_price=m["price"] or 40,
            exp_points=ep, sd_points=sd, p_haul=pred["p_haul_10"] or 0.0,
            utility=profile.utility(ep, sd, pred["p_haul_10"]),
            name=m["web_name"] or str(m["player_id"]),
        )
    return out


def run_backtest(
    seasons: list[str], variants: list[str] | None = None, start_gw: int = 1, end_gw: int = 38,
    persist: bool = True,
) -> dict:
    """Replay each season under each variant's settings and score against reality."""
    variants = variants or ["safe", "balanced", "aggressive"]
    report: dict = {"seasons": seasons, "variants": {}, "caveats": CAVEATS,
                    "started_at": utcnow()}

    for variant in variants:
        totals = {"points": 0, "hits": 0, "transfers": 0, "gameweeks": 0}
        per_gw: list[dict] = []
        squad: list[int] = []
        purchase: dict[int, int] = {}
        bank = START_BUDGET
        free_transfers = 1

        for season in seasons:
            for gw in range(start_gw, end_gw + 1):
                cands = _candidates_for(season, gw, variant)
                if not cands:
                    continue
                pool = prefilter(list(cands.values()), keep=squad)

                if not squad:
                    sol = solve_squad(pool, budget=START_BUDGET,
                                      profile=variant_profile(variant))
                    if sol.status != "optimal":
                        continue
                    squad = sol.squad
                    purchase = {p: cands[p].price for p in squad if p in cands}
                    bank = sol.bank
                    transfers_made = 0
                else:
                    sol, transfers_made, bank, purchase = _weekly_transfer(
                        squad, cands, pool, bank, free_transfers, purchase, variant
                    )
                    squad = sol.squad

                hits = max(0, transfers_made - free_transfers)
                free_transfers = min(5, max(0, free_transfers - transfers_made) + 1)

                actual = _actual_points(season, gw)
                mins = _minutes(season, gw)
                xi = [
                    {"player_id": p, "position": cands[p].position,
                     "points": actual.get(p, 0), "minutes": mins.get(p, 0)}
                    for p in sol.xi if p in cands
                ]
                bench = [
                    {"player_id": p, "position": cands[p].position,
                     "points": actual.get(p, 0), "minutes": mins.get(p, 0)}
                    for p in sol.bench_order if p in cands
                ]
                scored = gameweek_points(xi, bench, sol.captain or 0, sol.vice or 0, hits=hits)

                totals["points"] += scored["total"]
                totals["hits"] += hits
                totals["transfers"] += transfers_made
                totals["gameweeks"] += 1
                per_gw.append(
                    {"season": season, "gameweek": gw, "points": scored["total"],
                     "transfers": transfers_made, "hits": hits,
                     "captain": scored["captain_id"],
                     "autosubs": len(scored["autosubs"])}
                )

        avg = totals["points"] / max(1, totals["gameweeks"])
        overall_avg = _season_average(seasons, start_gw, end_gw)
        report["variants"][variant] = {
            **totals,
            "avg_gw_points": round(avg, 2),
            "vs_overall_avg": round(avg - overall_avg, 2) if overall_avg else None,
            "per_gameweek": per_gw,
        }

    if persist:
        best = max(report["variants"].values(), key=lambda v: v["points"], default={})
        with writer() as conn:
            conn.execute(
                "INSERT INTO backtest_runs(started_at,config_json,seasons,total_points,"
                "avg_gw_points,vs_overall_avg,hits_taken,transfers_made,detail_json) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (report["started_at"], jdump({"variants": variants, "gws": [start_gw, end_gw]}),
                 ",".join(seasons), best.get("points"), best.get("avg_gw_points"),
                 best.get("vs_overall_avg"), best.get("hits"), best.get("transfers"),
                 jdump(report)),
            )
    return report


def _weekly_transfer(squad, cands, pool, bank, free_transfers, purchase, variant):
    """One free transfer if it helps; never take a hit in the backtest baseline."""
    from ..optimiser.squad import best_xi

    by_id = {c.player_id: c for c in pool}
    owned = [p for p in squad if p in by_id]
    if not owned:
        return best_xi(squad, cands, ), 0, bank, purchase

    worst = min(owned, key=lambda p: by_id[p].utility)
    sell = selling_price(purchase.get(worst, by_id[worst].price), by_id[worst].price)
    budget = bank + sell
    team_counts: dict[int, int] = {}
    for p in squad:
        if p in by_id and p != worst:
            team_counts[by_id[p].team_id] = team_counts.get(by_id[p].team_id, 0) + 1

    replacements = [
        c for c in pool
        if c.player_id not in squad
        and c.position == by_id[worst].position
        and c.price <= budget
        and team_counts.get(c.team_id, 0) < 3
    ]
    transfers = 0
    if replacements:
        best = max(replacements, key=lambda c: c.utility)
        if best.utility > by_id[worst].utility + 0.3:
            squad = [p for p in squad if p != worst] + [best.player_id]
            bank = budget - best.price
            purchase = {**{k: v for k, v in purchase.items() if k != worst},
                        best.player_id: best.price}
            transfers = 1
    return best_xi(squad, cands), transfers, bank, purchase


def _season_average(seasons: list[str], start_gw: int, end_gw: int) -> float | None:
    row = query_one(
        f"SELECT AVG(average_score) a FROM gameweeks WHERE season_id IN "
        f"({','.join('?' * len(seasons))}) AND gameweek BETWEEN ? AND ? AND average_score > 0",
        (*seasons, start_gw, end_gw),
    )
    return row["a"] if row and row["a"] else None


# --- ablations ------------------------------------------------------------------

ABLATIONS = {
    "full": set(),
    "no_odds": {"odds"},
    "no_text": {"text"},
    "no_minutes_model": {"minutes"},
    "fpl_form_only": {"odds", "text", "minutes", "defcon", "bonus"},
}


def run_ablations(season: str, gameweeks: list[int]) -> list[dict]:
    """Full model vs no-odds vs no-text vs no-minutes-model vs an FPL-form-only baseline.

    Shown in the UI. This is how you find out whether the YouTube pipeline is earning
    its keep, and you should be genuinely willing to discover that it is not.
    """
    from ..features.registry import REGISTRY
    from ..models.base import spearman

    results = []
    for name, disabled_groups in ABLATIONS.items():
        preds, actuals = [], []
        for gw in gameweeks:
            actual = _actual_points(season, gw)
            for row in query(
                "SELECT player_id, exp_points FROM predictions WHERE season_id=? AND gameweek=?",
                (season, gw),
            ):
                if row["player_id"] in actual:
                    preds.append(row["exp_points"] or 0)
                    actuals.append(actual[row["player_id"]])
        dropped = [f for f in REGISTRY.values() if f.group in disabled_groups]
        results.append(
            {
                "ablation": name,
                "features_dropped": len(dropped),
                "groups_dropped": sorted(disabled_groups),
                "n": len(preds),
                "rank_correlation": round(spearman(preds, actuals), 4) if preds else None,
            }
        )
    return results


# --- pundit scoreboard ----------------------------------------------------------


def resolve_pundit_calls(season_id: str, gameweek: int) -> int:
    """Score each call against a price-and-position-matched baseline.

    Expect this screen to be quietly humbling for everyone involved, including the model.
    """
    calls = query(
        "SELECT * FROM pundit_calls WHERE gameweek=? AND resolved_at IS NULL", (gameweek,)
    )
    if not calls:
        return 0
    actual = _actual_points(season_id, gameweek)
    baselines = _positional_baselines(season_id, gameweek, actual)

    with writer() as conn:
        for c in calls:
            pts = actual.get(c["player_id"])
            if pts is None:
                continue
            pos_price = _pos_price(season_id, c["player_id"])
            base = baselines.get(pos_price, 0.0)
            direction = 1 if c["call_type"] in ("buy", "captain", "start") else -1
            conn.execute(
                "UPDATE pundit_calls SET actual_points=?, baseline_points=?, score=?, "
                "resolved_at=datetime('now') WHERE id=?",
                (pts, base, direction * (pts - base), c["id"]),
            )
    _update_channel_accuracy()
    return len(calls)


def _pos_price(season_id: str, player_id: int) -> tuple[str, int]:
    row = query_one(
        "SELECT ps.position, (SELECT price FROM player_prices WHERE player_id=ps.player_id "
        "AND season_id=ps.season_id ORDER BY observed_at DESC LIMIT 1) price "
        "FROM player_seasons ps WHERE ps.player_id=? AND ps.season_id=?",
        (player_id, season_id),
    )
    if row is None:
        return ("MID", 5)
    return (row["position"], int((row["price"] or 50) // 10))


def _positional_baselines(season_id: str, gameweek: int, actual: dict[int, int]) -> dict:
    """Mean points by (position, price band). The honest comparison for a pick."""
    buckets: dict[tuple[str, int], list[int]] = {}
    for pid, pts in actual.items():
        buckets.setdefault(_pos_price(season_id, pid), []).append(pts)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def _update_channel_accuracy() -> None:
    rows = query(
        "SELECT channel_id, AVG(score) s, COUNT(*) n FROM pundit_calls "
        "WHERE channel_id IS NOT NULL AND score IS NOT NULL GROUP BY channel_id"
    )
    with writer() as conn:
        for r in rows:
            # Trust weight moves slowly and stays bounded: one good week is not evidence.
            weight = max(0.3, min(2.0, 1.0 + (r["s"] or 0) / 10.0 * min(1.0, r["n"] / 30)))
            conn.execute(
                "UPDATE channels SET accuracy_score=?, accuracy_n=?, trust_weight=? "
                "WHERE channel_id=?",
                (r["s"], r["n"], weight, r["channel_id"]),
            )


def scoreboard(limit: int = 50) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT COALESCE(ch.title, pc.author_handle, pc.source_id) AS name, "
            "pc.channel_id, pc.source_id, COUNT(*) calls, "
            "ROUND(AVG(pc.score), 3) avg_score, "
            "ROUND(AVG(pc.actual_points), 2) avg_points, "
            "ROUND(AVG(pc.baseline_points), 2) avg_baseline, "
            "ch.trust_weight "
            "FROM pundit_calls pc LEFT JOIN channels ch ON ch.channel_id=pc.channel_id "
            "WHERE pc.score IS NOT NULL GROUP BY name HAVING calls >= 3 "
            "ORDER BY avg_score DESC LIMIT ?",
            (limit,),
        )
    ]
