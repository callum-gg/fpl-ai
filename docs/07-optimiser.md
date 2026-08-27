# 07 — Optimiser

MILP via `PuLP` with CBC (bundled, no licence) and an optional `HiGHS` backend for speed. Everything below is exact optimisation, not heuristics — the problem is small enough (≈700 players × 8 gameweeks) to solve properly in seconds.

## Decision variables

For each player `p` and gameweek `g` in the horizon:

- `x[p,g]` ∈ {0,1} — in the 15-man squad
- `s[p,g]` ∈ {0,1} — in the starting XI (`s ≤ x`)
- `c[p,g]` ∈ {0,1} — captain (`c ≤ s`)
- `v[p,g]` ∈ {0,1} — vice-captain
- `b[p,g,k]` ∈ {0,1} — bench order slot k ∈ {1,2,3} (outfield)
- `in[p,g]`, `out[p,g]` ∈ {0,1} — transfers
- `ft[g]` ∈ ℤ₀..₅ — free transfers carried into g
- `hits[g]` ∈ ℤ≥0 — paid transfers
- `chip[t,g]` ∈ {0,1} for t ∈ {wc, fh, bb, tc}, per half of the season

## Constraints

**Squad composition:** Σx = 15; GK = 2, DEF = 5, MID = 5, FWD = 3; ≤ 3 per club; Σ(price·x) + bank ≤ budget.

**Starting XI:** Σs = 11; exactly 1 GK; DEF ≥ 3; MID ≥ 2 (implied); FWD ≥ 1. Captain and vice distinct.

**Transfers:** `x[p,g] - x[p,g-1] = in[p,g] - out[p,g]`; `Σin[·,g] = Σout[·,g]`; `hits[g] ≥ Σin[·,g] - ft[g] - M·wc[g] - M·fh[g]`; `ft[g+1] = min(5, ft[g] - used + 1)` linearised with an indicator. Five is the cap this season.

**Money:** selling price uses the FPL 50%-of-profit-rounded-down rule, so the app tracks `purchase_price` per holding and computes `selling_price` exactly — getting this wrong by 0.1 quietly makes every plan infeasible in reality.

**Chips:**
- One chip per gameweek total.
- Chip set 1 usable only in GW ≤ 19 (deadline 2 Jan 2027); unused set-1 chips expire and must be dropped from the model at GW 20.
- Chip set 2 from GW 20.
- Wildcard: transfers free that week, squad persists.
- Free Hit: transfers free that week, and `x[p,g+1]` is forced back to `x[p,g-1]` — modelled with a parallel "shadow squad" variable set.
- Bench Boost: objective includes bench points that week.
- Triple Captain: captain multiplier 3 instead of 2.

**Personal constraints** (per squad, from settings): banned clubs, must-own players, locked players (never transfer out), max bench value, minimum bank, "no more than N players from a club playing in a blank GW", and a `max_transfers_per_gw`.

## Objective

```
maximise  Σ_g  decay(g) · [ Σ_p ( s[p,g]·U(p,g) + c[p,g]·U(p,g)·mult )
                            + bench_weight · Σ_p bench_contribution(p,g) ]
          - 4 · Σ_g decay(g) · hits[g]
          - price_penalty + price_bonus
```

- `decay(g) = horizon_decay ^ (g - g0)` — **per-squad configurable**, exactly as you asked. Default 0.84, so GW+1 counts 1.0, GW+5 ≈ 0.50.
- `horizon_gws` per-squad, default 5, range 1–8.
- `bench_weight` default 0.12 (bench points are real but conditional on autosubs firing).
- `price_bonus` — small term rewarding predicted price rises, off by default; it's a distraction unless you're chasing team value early.

### The utility function `U(p,g)` — where risk lives

Not just expected points. `U` is chosen by the squad's risk setting:

| Mode | `U(p,g)` |
|---|---|
| **Safe** | `E[pts] - λ·SD[pts]`, λ = 0.35. Prefers nailed, low-variance, template assets |
| **Balanced** | `E[pts]` (λ = 0) |
| **Aggressive** | `E[pts] + λ·SD[pts]` with λ = 0.25, plus a bonus on `P(haul ≥ 10)` |
| **Rank-chasing** | mini-league aware — see below |

Exposed in the UI as a single **risk slider (-1 … +1)** that maps to λ, plus separate toggles for "prefer differentials" and "protect rank".

### Mini-league / rank-aware mode

You asked for variance-aware optimisation with mini-leagues pulled from the API. `/api/leagues-classic/{id}/standings/` gives every rival's entry id; `/api/entry/{id}/event/{gw}/picks/` gives their squads (public, available after each deadline).

With rival squads known, the objective becomes a **rank utility over simulated outcomes** rather than a linear points sum. Since MILP can't optimise a rank objective directly, use two passes:

1. MILP generates the top-K candidate plans (K ≈ 30) under a linear proxy objective (`E[pts] - EO-weighted covariance penalty`).
2. Each candidate is evaluated by Monte Carlo against rivals' actual squads using the joint simulation draws, producing `P(finish above rival)`, `P(gain rank)`, `E[rank change]`.
3. Rank by whichever the squad's setting says: maximise `P(win league)` if you're behind, maximise `P(not losing position)` if you're ahead.

This is the honest way to do "differentials" — the right differential depends entirely on whether you're chasing or defending, and against whom.

Effective ownership from LiveFPL substitutes for rival squads at the overall-rank level.

## Candidate generation

Always produce **three named variants** per squad per gameweek (safe / balanced / aggressive), plus the "no transfer" baseline and the "roll your transfer" option, each with expected points and the delta versus doing nothing. If the best plan beats doing nothing by less than a configurable threshold (default 0.8 pts over the horizon), the app says so plainly: **"the recommendation is to do nothing this week."** An FPL tool that can't recommend inaction is a bad FPL tool.

Generate alternatives with solution-pool style diversification: solve, add a no-good cut forbidding that exact transfer set, re-solve, repeat.

## Chip timing

Separate long-horizon planner run weekly over the remaining season: coarse fixture/DGW projections, evaluates chip placement by expected gain vs. the no-chip baseline. Outputs a chip calendar with confidence, e.g. "Bench Boost best in GW 34 (+11.2), second best GW 26 (+8.4); Wildcard now sits at +14 versus GW 8". Half-season expiry is a hard constraint, and the app must nag as GW19's 2 Jan deadline approaches if set-1 chips are unused — losing a chip to the calendar is the most avoidable mistake in the game.

## Initial squad build (GW1 / Wildcard)

Same MILP with no incumbent squad and no transfer cost. Add: a diversity pass producing several structurally different squads (5-4-1 heavy premium vs balanced vs 4-4-2 mid-heavy), and a "bench fodder" sub-problem that explicitly minimises spend on positions 12–15 while keeping ≥ 1 playing GK and enough bodies for autosubs.

## Performance

Roughly 700 players × 8 gameweeks × ~6 variable families ≈ 35k binaries. CBC solves single-GW in <1s and 5-GW plans in 3–20s. Keep it fast with: pre-filtering to the top ~250 players by expected points over the horizon (always retaining currently-owned players and any manually pinned ones), warm starts from last run's solution, and a 60s time limit with the incumbent returned.
