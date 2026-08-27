"""The FPL rules engine: pure functions, no I/O. docs/07 + docs/12 layer 3.

Getting selling price wrong by 0.1 quietly makes every plan infeasible in reality, so
everything here is exhaustively tested (tests/test_rules.py) rather than assumed.

2026/27 ruleset encoded: 5 banked free transfers, two chip sets with set 1 expiring at
the GW19 deadline, one chip per gameweek, DefCon unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .defaults import (
    CHIP_SET_1_LAST_GW,
    DEFCON_POINTS,
    DEFCON_THRESHOLD,
    HIT_COST,
    MAX_FREE_TRANSFERS,
    MAX_PER_CLUB,
    POSITION_QUOTA,
    SQUAD_SIZE,
    XI_MAX,
    XI_MIN,
)

CHIPS = ("wildcard", "freehit", "bboost", "3xc")


# --- money ----------------------------------------------------------------------


def selling_price(purchase_price: int, current_price: int) -> int:
    """FPL sells at purchase + floor(profit/2). Losses are absorbed in full.

    All prices are tenths of a million: purchase 70, now 75 -> 72 (profit 5, half
    rounded down = 2).
    """
    profit = current_price - purchase_price
    if profit <= 0:
        return current_price
    return purchase_price + profit // 2


def squad_value(picks: list[dict]) -> int:
    return sum(p.get("selling_price") or p.get("price", 0) for p in picks)


# --- transfers ------------------------------------------------------------------


def free_transfers_next(current_ft: int, transfers_made: int, chip: str | None = None) -> int:
    """FT accrual with the 5 cap. Wildcard/Free Hit do not consume the banked transfer."""
    if chip in ("wildcard", "freehit"):
        return min(MAX_FREE_TRANSFERS, current_ft + 1)
    remaining = max(0, current_ft - transfers_made)
    return min(MAX_FREE_TRANSFERS, remaining + 1)


def hit_cost(current_ft: int, transfers_made: int, chip: str | None = None) -> int:
    """-4 per transfer beyond the free ones. Free under Wildcard and Free Hit."""
    if chip in ("wildcard", "freehit"):
        return 0
    return HIT_COST * max(0, transfers_made - current_ft)


# --- chips ----------------------------------------------------------------------


def chip_set(gameweek: int) -> int:
    return 1 if gameweek <= CHIP_SET_1_LAST_GW else 2


def chip_legal(
    chip: str, gameweek: int, chips_used: list[dict], chip_this_gw: str | None = None
) -> tuple[bool, str]:
    """(legal, reason). `chips_used` entries are {'name': ..., 'gameweek': ...}."""
    if chip not in CHIPS:
        return False, f"unknown chip {chip!r}"
    if chip_this_gw and chip_this_gw != chip:
        return False, f"{chip_this_gw} already played in GW{gameweek} — one chip per gameweek"
    wanted_set = chip_set(gameweek)
    for used in chips_used:
        if used["name"] == chip and chip_set(used["gameweek"]) == wanted_set:
            return False, f"{chip} already used in set {wanted_set}"
        if used["gameweek"] == gameweek:
            return False, f"{used['name']} already played in GW{gameweek}"
    return True, ""


def chips_available(gameweek: int, chips_used: list[dict]) -> list[str]:
    return [c for c in CHIPS if chip_legal(c, gameweek, chips_used)[0]]


def expiring_chips(gameweek: int, chips_used: list[dict]) -> list[str]:
    """Set-1 chips still unplayed. Losing one to the calendar is the most avoidable
    mistake in the game, so the UI nags on this from GW15."""
    if gameweek > CHIP_SET_1_LAST_GW:
        return []
    used_set1 = {u["name"] for u in chips_used if chip_set(u["gameweek"]) == 1}
    return [c for c in CHIPS if c not in used_set1]


# --- squad legality -------------------------------------------------------------


@dataclass
class SquadCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_squad(picks: list[dict], budget: int = 1000, bank: int = 0) -> SquadCheck:
    """15 players, 2/5/5/3, <=3 per club, within budget. `picks` need position/team_id/price."""
    errors: list[str] = []
    if len(picks) != SQUAD_SIZE:
        errors.append(f"squad has {len(picks)} players, needs {SQUAD_SIZE}")
    for pos, want in POSITION_QUOTA.items():
        got = sum(1 for p in picks if p["position"] == pos)
        if got != want:
            errors.append(f"{pos}: {got} selected, needs {want}")
    clubs: dict[int, int] = {}
    for p in picks:
        clubs[p["team_id"]] = clubs.get(p["team_id"], 0) + 1
    for team, n in clubs.items():
        if n > MAX_PER_CLUB:
            errors.append(f"team {team}: {n} players, max {MAX_PER_CLUB}")
    spend = sum(p["price"] for p in picks)
    if spend > budget + bank:
        errors.append(f"spend {spend} exceeds budget {budget + bank}")
    return SquadCheck(not errors, errors)


def valid_formation(xi: list[dict]) -> bool:
    if len(xi) != 11:
        return False
    counts = {pos: sum(1 for p in xi if p["position"] == pos) for pos in POSITION_QUOTA}
    return all(XI_MIN[pos] <= counts[pos] <= XI_MAX[pos] for pos in POSITION_QUOTA)


def formation_string(xi: list[dict]) -> str:
    c = {pos: sum(1 for p in xi if p["position"] == pos) for pos in POSITION_QUOTA}
    return f"{c['DEF']}-{c['MID']}-{c['FWD']}"


def legal_formations() -> list[tuple[int, int, int]]:
    out = []
    for d in range(XI_MIN["DEF"], XI_MAX["DEF"] + 1):
        for m in range(XI_MIN["MID"], XI_MAX["MID"] + 1):
            f = 10 - d - m
            if XI_MIN["FWD"] <= f <= XI_MAX["FWD"]:
                out.append((d, m, f))
    return out


# --- autosubs -------------------------------------------------------------------


def apply_autosubs(xi: list[dict], bench: list[dict]) -> tuple[list[dict], list[dict]]:
    """FPL autosub rules. `minutes` on each pick decides who played.

    Bench order matters; the goalkeeper is a special case — only a GK replaces a GK.
    A substitution is only made if the resulting XI is still a legal formation.
    """
    final = [dict(p) for p in xi]
    subs_used: list[dict] = []
    bench_gk = [p for p in bench if p["position"] == "GK"]
    bench_out = [p for p in bench if p["position"] != "GK"]

    # Goalkeeper first: only the bench GK can replace the starting GK.
    for i, p in enumerate(final):
        if p["position"] == "GK" and not p.get("minutes"):
            for gk in bench_gk:
                if gk.get("minutes"):
                    subs_used.append({"out": p, "in": gk})
                    final[i] = dict(gk)
                    break

    for i, p in enumerate(final):
        if p["position"] == "GK" or p.get("minutes"):
            continue
        for cand in bench_out:
            if not cand.get("minutes") or any(s["in"] is cand for s in subs_used):
                continue
            trial = [dict(x) for x in final]
            trial[i] = dict(cand)
            if valid_formation(trial):
                final = trial
                subs_used.append({"out": p, "in": cand})
                break
    return final, subs_used


def effective_captain(xi: list[dict], captain_id: int, vice_id: int) -> int:
    """Vice takes over only if the captain played no minutes."""
    by_id = {p["player_id"]: p for p in xi}
    cap = by_id.get(captain_id)
    if cap and cap.get("minutes"):
        return captain_id
    vice = by_id.get(vice_id)
    return vice_id if vice and vice.get("minutes") else captain_id


# --- scoring --------------------------------------------------------------------


def defcon_points(position: str, actions: int | None) -> int:
    if actions is None:
        return 0
    return DEFCON_POINTS if actions >= DEFCON_THRESHOLD.get(position, 12) else 0


def player_points(stats: dict, position: str) -> int:
    """FPL scoring for one player-fixture. Used by the backtest and the rules tests."""
    mins = stats.get("minutes") or 0
    pts = 0
    if mins > 0:
        pts += 2 if mins >= 60 else 1
    goal_points = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}[position]
    pts += goal_points * (stats.get("goals_scored") or 0)
    pts += 3 * (stats.get("assists") or 0)
    if position in ("GK", "DEF") and mins >= 60:
        pts += 4 * (stats.get("clean_sheets") or 0)
    elif position == "MID" and mins >= 60:
        pts += 1 * (stats.get("clean_sheets") or 0)
    if position in ("GK", "DEF"):
        pts -= (stats.get("goals_conceded") or 0) // 2
    pts += (stats.get("saves") or 0) // 3
    pts += 5 * (stats.get("penalties_saved") or 0)
    pts -= 2 * (stats.get("penalties_missed") or 0)
    pts -= 1 * (stats.get("yellow_cards") or 0)
    pts -= 3 * (stats.get("red_cards") or 0)
    pts -= 2 * (stats.get("own_goals") or 0)
    pts += stats.get("bonus") or 0
    if stats.get("defcon_points") is not None:
        pts += stats["defcon_points"]
    else:
        pts += defcon_points(position, stats.get("defensive_contribution"))
    return pts


def gameweek_points(
    xi: list[dict],
    bench: list[dict],
    captain_id: int,
    vice_id: int,
    chip: str | None = None,
    hits: int = 0,
) -> dict:
    """Full GW score including autosubs, captaincy fallback, chips and hits."""
    if chip == "bboost":
        final_xi, subs = [dict(p) for p in xi] + [dict(p) for p in bench], []
    else:
        final_xi, subs = apply_autosubs(xi, bench)

    cap_id = effective_captain(final_xi, captain_id, vice_id)
    multiplier = 3 if chip == "3xc" else 2

    total = 0
    breakdown = []
    for p in final_xi:
        base = p.get("points")
        if base is None:
            base = player_points(p, p["position"])
        mult = multiplier if p["player_id"] == cap_id else 1
        total += base * mult
        breakdown.append({"player_id": p["player_id"], "points": base, "multiplier": mult})

    penalty = HIT_COST * hits if chip not in ("wildcard", "freehit") else 0
    return {
        "total": total - penalty,
        "raw": total,
        "hits_cost": penalty,
        "captain_id": cap_id,
        "autosubs": [{"out": s["out"]["player_id"], "in": s["in"]["player_id"]} for s in subs],
        "breakdown": breakdown,
    }
