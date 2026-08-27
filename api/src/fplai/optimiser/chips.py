"""Chip timing: a separate long-horizon planner run weekly. docs/07.

Coarse projections over the rest of the season, evaluating each chip placement against
the no-chip baseline. Half-season expiry is a hard constraint, and the app nags as the
GW19 deadline (13:30 GMT, 2 January 2027) approaches with set-1 chips unplayed —
losing a chip to the calendar is the most avoidable mistake in the game.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..db.engine import query
from ..defaults import CHIP_SET_1_LAST_GW
from ..rules import chip_set, expiring_chips

log = logging.getLogger(__name__)

CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}
NAG_FROM_GW = 15


@dataclass
class ChipRecommendation:
    chip: str
    best_gw: int
    gain: float
    second_gw: int | None = None
    second_gain: float | None = None
    confidence: str = "medium"
    reason: str = ""

    @property
    def actionable(self) -> bool:
        """False when the fixture list cannot yet distinguish one gameweek from another."""
        return self.confidence != "none"

    def to_dict(self) -> dict:
        return {
            "chip": self.chip,
            "actionable": self.actionable,
            "label": CHIP_LABELS.get(self.chip, self.chip),
            "best_gw": self.best_gw,
            "gain": round(self.gain, 1),
            "second_gw": self.second_gw,
            "second_gain": round(self.second_gain, 1) if self.second_gain is not None else None,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def fixture_counts(season_id: str, from_gw: int, to_gw: int) -> dict[int, dict[int, int]]:
    """team_id -> gameweek -> number of PL fixtures. Blanks are 0, doubles are 2."""
    rows = query(
        "SELECT gameweek, home_team_id h, away_team_id a FROM fixtures "
        "WHERE season_id=? AND competition='PL' AND gameweek BETWEEN ? AND ?",
        (season_id, from_gw, to_gw),
    )
    out: dict[int, dict[int, int]] = {}
    teams = {r["id"] for r in query("SELECT id FROM teams WHERE season_id=?", (season_id,))}
    for t in teams:
        out[t] = {gw: 0 for gw in range(from_gw, to_gw + 1)}
    for r in rows:
        for t in (r["h"], r["a"]):
            if t in out and r["gameweek"] in out[t]:
                out[t][r["gameweek"]] += 1
    return out


def plan_chips(
    season_id: str,
    start_gw: int,
    squad_team_ids: list[int],
    projections: dict[int, dict[int, float]] | None = None,
    chips_used: list[dict] | None = None,
    end_gw: int = 38,
) -> list[ChipRecommendation]:
    """Coarse chip calendar. `projections` is gameweek -> player_id -> expected points."""
    chips_used = chips_used or []
    counts = fixture_counts(season_id, start_gw, end_gw)
    gws = list(range(start_gw, end_gw + 1))
    recs: list[ChipRecommendation] = []

    squad_fixtures = {
        gw: sum(counts.get(t, {}).get(gw, 0) for t in squad_team_ids) for gw in gws
    }
    baseline = _median(list(squad_fixtures.values())) or 11

    used_names = {(u["name"], chip_set(u["gameweek"])) for u in chips_used}

    for chip, scorer in (
        ("bboost", _bench_boost_gain),
        ("3xc", _triple_captain_gain),
        ("freehit", _free_hit_gain),
        ("wildcard", _wildcard_gain),
    ):
        candidates = []
        for gw in gws:
            if (chip, chip_set(gw)) in used_names:
                continue
            gain = scorer(gw, squad_fixtures, baseline, projections)
            candidates.append((gw, gain))
        if not candidates:
            continue
        candidates.sort(key=lambda kv: -kv[1])
        best_gw, best_gain = candidates[0]
        second = candidates[1] if len(candidates) > 1 else (None, None)
        confidence = _confidence(best_gain, second[1])

        # Early in a season every club plays exactly once a week, so nothing distinguishes
        # one gameweek from another. Naming a specific gameweek then is false precision:
        # say so instead, and let doubles and blanks appear before advising a date.
        if not _has_fixture_variation(squad_fixtures):
            recs.append(
                ChipRecommendation(
                    chip=chip, best_gw=best_gw, gain=best_gain,
                    second_gw=second[0], second_gain=second[1], confidence="none",
                    reason="no doubles or blanks scheduled yet — too early to time this chip",
                )
            )
            continue

        recs.append(
            ChipRecommendation(
                chip=chip,
                best_gw=best_gw,
                gain=best_gain,
                second_gw=second[0],
                second_gain=second[1],
                confidence=confidence,
                reason=_reason(chip, best_gw, squad_fixtures.get(best_gw, 0), baseline),
            )
        )
    return recs


def _has_fixture_variation(squad_fixtures: dict[int, int]) -> bool:
    """True once some gameweek differs from the rest — a double, a blank, or a split."""
    counts = set(squad_fixtures.values())
    return len(counts) > 1


def _bench_boost_gain(gw, squad_fixtures, baseline, projections) -> float:
    """Bench Boost is worth most in a double gameweek where all four bench players play."""
    ratio = squad_fixtures.get(gw, baseline) / max(baseline, 1)
    base_bench_points = 12.0
    return base_bench_points * ratio


def _triple_captain_gain(gw, squad_fixtures, baseline, projections) -> float:
    """The gain is one extra captain haul, so it scales with the best available fixture."""
    if projections and gw in projections and projections[gw]:
        best = max(projections[gw].values())
        return float(best)
    return 7.0 * (squad_fixtures.get(gw, baseline) / max(baseline, 1))


def _free_hit_gain(gw, squad_fixtures, baseline, projections) -> float:
    """Free Hit is for blanks: the fewer of your players who have a fixture, the better."""
    played = squad_fixtures.get(gw, baseline)
    shortfall = max(0, baseline - played)
    return 4.0 * shortfall


def _wildcard_gain(gw, squad_fixtures, baseline, projections) -> float:
    """Earlier wildcards compound over more gameweeks, so gain decays with the calendar."""
    remaining = max(0, 38 - gw)
    return 14.0 * (remaining / 38.0)


def _confidence(best: float, second: float | None) -> str:
    if second is None or best <= 0:
        return "low"
    margin = (best - second) / max(best, 1e-6)
    return "high" if margin > 0.3 else ("medium" if margin > 0.1 else "low")


def _reason(chip: str, gw: int, fixtures: int, baseline: float) -> str:
    if chip == "bboost":
        return f"GW{gw} has {fixtures} squad fixtures vs a {baseline:.0f} baseline"
    if chip == "freehit":
        return f"only {fixtures} of your players have a fixture in GW{gw}"
    if chip == "3xc":
        return f"best captaincy ceiling in GW{gw}"
    return f"wildcarding at GW{gw} leaves {38 - gw} gameweeks to benefit"


def expiry_warnings(gameweek: int, chips_used: list[dict]) -> list[dict]:
    """Loud from GW15 onward. Set-1 chips die at the GW19 deadline and cannot be carried."""
    unplayed = expiring_chips(gameweek, chips_used)
    if not unplayed or gameweek < NAG_FROM_GW:
        return []
    weeks_left = CHIP_SET_1_LAST_GW - gameweek + 1
    severity = "critical" if weeks_left <= 2 else ("high" if weeks_left <= 4 else "medium")
    return [
        {
            "chip": c,
            "label": CHIP_LABELS.get(c, c),
            "gameweeks_left": weeks_left,
            "severity": severity,
            "message": (
                f"{CHIP_LABELS.get(c, c)} (first set) expires at the GW19 deadline "
                f"— {weeks_left} gameweek{'s' if weeks_left != 1 else ''} left."
            ),
        }
        for c in unplayed
    ]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
