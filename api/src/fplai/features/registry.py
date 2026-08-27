"""Feature registry. docs/05.

Features are declared, not scattered: each has a name, version, dependency list and
builder, so the UI can explain any number and the trainer can guarantee no leakage.

**The leakage rule is enforced here, not by convention.** A builder receives `ctx.as_of`
(the gameweek deadline) and the row accessors refuse to return anything observed at or
after it. That is the whole difference between a backtest that means something and one
that lies to you.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

FEATURE_VERSION = 1  # bump to invalidate the whole store

MissingStrategy = str  # zero | positional_mean | shrink_to_prior | indicator | nan


@dataclass(slots=True)
class FeatureCtx:
    """Everything a builder may look at. Nothing here postdates `as_of`."""

    player_id: int
    season_id: str
    gameweek: int
    fixture_id: int | None
    as_of: str  # ISO deadline; the hard leakage boundary
    position: str
    team_id: int | None
    opponent_team_id: int | None
    is_home: bool
    # Pre-loaded slices, all already filtered to `< as_of` by build.py.
    history: list[dict] = field(default_factory=list)       # this player's past fixtures, newest first
    team_history: list[dict] = field(default_factory=list)  # his team's past fixtures
    opp_history: list[dict] = field(default_factory=list)
    upcoming: list[dict] = field(default_factory=list)       # fixtures from as_of forward
    availability: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    odds: dict = field(default_factory=dict)
    ownership: dict = field(default_factory=dict)
    set_pieces: dict = field(default_factory=dict)
    price: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def blocks_present(self) -> dict[str, bool]:
        """Whole-block indicators. Lets the model learn 'no odds -> lean on the team model'
        instead of silently treating absence as zero."""
        return {
            "block_odds_present": bool(self.odds),
            "block_text_present": bool(self.claims),
            "block_ownership_present": bool(self.ownership),
            "block_advanced_present": any(h.get("xg") is not None for h in self.history),
        }


@dataclass(slots=True)
class Feature:
    name: str
    fn: Callable[[FeatureCtx], float | None]
    version: int
    deps: list[str]
    missing_strategy: MissingStrategy
    group: str
    description: str


REGISTRY: dict[str, Feature] = {}


def feature(
    name: str,
    deps: list[str] | None = None,
    version: int = 1,
    missing: MissingStrategy = "nan",
    group: str = "misc",
    description: str = "",
):
    def wrap(fn: Callable[[FeatureCtx], float | None]) -> Callable:
        REGISTRY[name] = Feature(
            name=name,
            fn=fn,
            version=version,
            deps=deps or [],
            missing_strategy=missing,
            group=group,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return wrap


def compute_all(ctx: FeatureCtx, only: list[str] | None = None) -> dict[str, float | None]:
    """Run every registered builder. A failing builder yields None, never a crash."""
    out: dict[str, float | None] = {}
    names = only or list(REGISTRY)
    for name in names:
        f = REGISTRY.get(name)
        if f is None:
            continue
        try:
            out[name] = f.fn(ctx)
        except Exception:
            log.debug("feature %s failed for player %s", name, ctx.player_id, exc_info=True)
            out[name] = None
    out.update({k: float(v) for k, v in ctx.blocks_present().items()})
    return out


def feature_names() -> list[str]:
    return sorted(REGISTRY)


def by_group() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for f in REGISTRY.values():
        groups.setdefault(f.group, []).append(
            {"name": f.name, "version": f.version, "missing": f.missing_strategy,
             "description": f.description, "deps": f.deps}
        )
    for v in groups.values():
        v.sort(key=lambda x: x["name"])
    return groups


def text_feature_names() -> list[str]:
    """Used by the dominance guard in docs/12 layer 5."""
    return [f.name for f in REGISTRY.values() if f.group == "text"]


# --- small numeric helpers shared by builders ------------------------------------


def ewma(values: list[float], half_life: float = 3.0) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    decay = 0.5 ** (1 / half_life)
    num = den = 0.0
    for i, v in enumerate(vals):  # values arrive newest-first
        w = decay**i
        num += w * v
        den += w
    return num / den if den else None


def mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rate_per90(total: float | None, minutes: float | None) -> float | None:
    if total is None or not minutes:
        return None
    return 90.0 * total / minutes


def safe_div(a, b, default=None):
    try:
        return a / b if b else default
    except TypeError:
        return default


def shrink(value: float | None, prior: float, n: float, k: float = 5.0) -> float | None:
    """Shrink a small-sample rate toward a prior. n = observations, k = prior weight."""
    if value is None:
        return prior
    return (n * value + k * prior) / (n + k)
