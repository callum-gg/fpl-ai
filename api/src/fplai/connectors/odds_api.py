"""The Odds API + devigging. docs/02 tier 3 — the highest-signal cheap data there is.

Stores raw implied probability *and* the devigged probability. Every snapshot is kept:
movement in the 48h before kickoff is itself the signal, not just the level.
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator

from functools import lru_cache

from ..db.engine import query, query_one
from ..resolve.entities import resolve_team
from .base import Connector, IngestContext, ParsedBatch, RawDoc, fetch_url, utcnow

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"


def devig_multiplicative(probs: list[float]) -> list[float]:
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else probs


def devig_shin(probs: list[float], iters: int = 60) -> list[float]:
    """Shin's method: assumes an informed-trader fraction z rather than proportional vig.

    Sharper than multiplicative on lopsided markets (short-priced favourites), which is
    exactly where FPL cares most.
    """
    n = len(probs)
    total = sum(probs)
    if n < 2 or total <= 0:
        return probs
    z = 0.0
    for _ in range(iters):
        denom = sum(math.sqrt(z * z + 4 * (1 - z) * (p * p) / total) for p in probs)
        new_z = max(0.0, min(0.2, (denom - 2) / (n - 2))) if n > 2 else z
        if abs(new_z - z) < 1e-9:
            break
        z = new_z
    out = [
        (math.sqrt(z * z + 4 * (1 - z) * (p * p) / total) - z) / (2 * (1 - z))
        for p in probs
    ]
    s = sum(out)
    return [o / s for o in out] if s > 0 else devig_multiplicative(probs)


def poisson_lambdas_from_market(
    p_home: float, p_draw: float, p_away: float, p_over25: float | None = None
) -> tuple[float, float]:
    """Fit (lambda_home, lambda_away) to the devigged 1X2 (+ over/under 2.5) surface.

    Grid search over a plausible range; small enough to be exact and instant, and it
    avoids dragging in a solver for a 2-parameter fit.
    """
    best, best_err = (1.4, 1.1), float("inf")
    for lh in [x / 20 for x in range(4, 81)]:
        for la in [x / 20 for x in range(4, 81)]:
            ph = pd = pa = 0.0
            over = 0.0
            for h in range(9):
                for a in range(9):
                    p = _pois(h, lh) * _pois(a, la)
                    if h > a:
                        ph += p
                    elif h == a:
                        pd += p
                    else:
                        pa += p
                    if h + a > 2:
                        over += p
            err = (ph - p_home) ** 2 + (pd - p_draw) ** 2 + (pa - p_away) ** 2
            if p_over25 is not None:
                err += (over - p_over25) ** 2
            if err < best_err:
                best_err, best = err, (lh, la)
    return best


def _pois(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


class OddsApiConnector(Connector):
    id = "odds_api"
    category = "odds"
    requires_keys = ["odds_api_key"]
    default_cadence = "0 */4 * * *"
    rate_limit_per_min = 5
    # 2: fixture matching canonicalises both sides, so the promoted clubs resolve. Bumping
    # this reparses the archive in place — no refetch, no extra API credits spent.
    parser_version = 2

    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        s = ctx.settings
        r = await fetch_url(
            f"{BASE}/sports/soccer_epl/odds/",
            self.id,
            params={
                "apiKey": s.odds_api_key,
                "regions": s.odds_api_regions,
                "markets": s.odds_api_markets,
                "oddsFormat": "decimal",
            },
            per_min=self.rate_limit_per_min,
        )
        yield RawDoc("odds_snapshot", r.json(), external_id=utcnow()[:13])

    def parse(self, doc: RawDoc) -> ParsedBatch:
        b = ParsedBatch()
        events = doc.payload
        observed = utcnow()

        def _odds(conn) -> int:
            n = 0
            for ev in events:
                fixture_id = _match_fixture(ev.get("home_team"), ev.get("away_team"),
                                            ev.get("commence_time"))
                if fixture_id is None:
                    continue
                for bk in ev.get("bookmakers", []):
                    for market in bk.get("markets", []):
                        outcomes = market.get("outcomes", [])
                        implied = [1 / o["price"] for o in outcomes if o.get("price")]
                        if not implied:
                            continue
                        fair = devig_shin(implied)
                        for o, ip, fp in zip(outcomes, implied, fair, strict=False):
                            selection = _selection_label(market["key"], o, ev)
                            conn.execute(
                                "INSERT OR IGNORE INTO odds_snapshots(fixture_id,source_id,"
                                "bookmaker,market,selection,price_decimal,implied_prob,devig_prob,"
                                "observed_at) VALUES(?,'odds_api',?,?,?,?,?,?,?)",
                                (fixture_id, bk.get("key"), market["key"], selection,
                                 o["price"], ip, fp, observed),
                            )
                            n += 1
            # New prices land, so the fixture-level caches below must not keep serving the
            # old ones to a long-lived API process.
            refresh()
            return n

        b.defer(_odds)
        return b


def _selection_label(market_key: str, outcome: dict, ev: dict) -> str:
    name = outcome.get("name", "")
    if market_key == "h2h":
        if name == ev.get("home_team"):
            return "home"
        if name == ev.get("away_team"):
            return "away"
        return "draw"
    if market_key == "totals":
        return f"{name.lower()}_{outcome.get('point')}"
    return name


def _same_club(name: str | None, short: str | None, key: str) -> bool:
    """Does this fixture's team row mean the same club as the odds feed's resolved key?"""
    if key in (name, short):
        return True
    return resolve_team(name) == key or (bool(short) and resolve_team(short) == key)


def _match_fixture(home: str | None, away: str | None, kickoff: str | None) -> int | None:
    """Match on canonical club identity, not on whichever string each side happens to use.

    Comparing the feed's *resolved* key straight against `teams.name`/`short_name` drops
    any fixture whose stored name and canonical alias disagree — which for 2026-27 was
    exactly the three promoted clubs ("Ipswich Town" in `teams`, "Ipswich" as the canonical
    key), so three of every ten fixtures went unpriced and their players silently fell back
    to the team model while everyone else got the market.
    """
    if not home or not away or not kickoff:
        return None
    # An unknown alias must not be fatal. Resolution failing used to return None for the
    # whole fixture, so one unrecognised club name cost both sides their market — falling
    # back to the surface form still matches against `teams.name` directly.
    hk = resolve_team(home) or home
    ak = resolve_team(away) or away
    for r in query(
        "SELECT f.id, th.name hn, th.short_name hs, ta.name an, ta.short_name a_s "
        "FROM fixtures f JOIN teams th ON th.id=f.home_team_id "
        "JOIN teams ta ON ta.id=f.away_team_id "
        "WHERE substr(f.kickoff_utc,1,10)=substr(?,1,10)",
        (kickoff,),
    ):
        if _same_club(r["hn"], r["hs"], hk) and _same_club(r["an"], r["a_s"], ak):
            return r["id"]
    return None


@lru_cache(maxsize=512)
def team_lambdas(fixture_id: int) -> tuple[float, float] | None:
    """Latest devigged market view of a fixture as (lambda_home, lambda_away).

    Cached because this is a property of the fixture but `_odds()` asks for it once per
    *player* — about 65 times per fixture, each time running two correlated subqueries
    over odds_snapshots. Uncached, that alone took a gameweek feature build from 10
    seconds to five minutes the moment odds existed. `refresh()` clears it after an
    ingest so a long-lived process never serves a stale price.
    """
    rows = query(
        "SELECT selection, AVG(devig_prob) p FROM odds_snapshots WHERE fixture_id=? AND market='h2h'"
        " AND observed_at=(SELECT MAX(observed_at) FROM odds_snapshots WHERE fixture_id=?"
        " AND market='h2h') GROUP BY selection",
        (fixture_id, fixture_id),
    )
    probs = {r["selection"]: r["p"] for r in rows}
    if not {"home", "draw", "away"} <= probs.keys():
        return None
    over = query_one(
        "SELECT AVG(devig_prob) p FROM odds_snapshots WHERE fixture_id=? AND market='totals' "
        "AND selection LIKE 'over_2.5%'",
        (fixture_id,),
    )
    return poisson_lambdas_from_market(
        probs["home"], probs["draw"], probs["away"], over["p"] if over and over["p"] else None
    )


@lru_cache(maxsize=512)
def odds_movement(fixture_id: int, hours: int = 48) -> float:
    """Drift in the home team's devigged win probability. Steam is information."""
    rows = query(
        "SELECT observed_at, AVG(devig_prob) p FROM odds_snapshots WHERE fixture_id=? "
        "AND market='h2h' AND selection='home' AND observed_at > datetime('now', ?) "
        "GROUP BY observed_at ORDER BY observed_at",
        (fixture_id, f"-{hours} hour"),
    )
    if len(rows) < 2:
        return 0.0
    return float(rows[-1]["p"] - rows[0]["p"])


def refresh() -> None:
    """Drop the fixture-level odds caches. Called after an ingest writes new prices."""
    team_lambdas.cache_clear()
    odds_movement.cache_clear()
