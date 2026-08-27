# FPL AI

An AI-powered Fantasy Premier League squad builder and week-to-week manager. Runs locally, in the browser, mobile-optimised. Python backend, TypeScript frontend, SQLite storage, LLM reasoning over an OpenAI-compatible endpoint, and a real predictive model underneath.

## Read these in order

| File | What it covers |
|---|---|
| [`docs/01-architecture.md`](docs/01-architecture.md) | System shape, services, request/job flow, repo layout |
| [`docs/02-data-sources.md`](docs/02-data-sources.md) | Every source, endpoint, cost, cadence, failure mode |
| [`docs/03-database-schema.md`](docs/03-database-schema.md) | Full SQLite DDL, dedup strategy, provenance model |
| [`docs/04-ingestion.md`](docs/04-ingestion.md) | Connector contract, scheduler, entity resolution, rate limiting |
| [`docs/05-features.md`](docs/05-features.md) | Feature store — every feature, definition, and source |
| [`docs/06-ml-models.md`](docs/06-ml-models.md) | Model stack, training, simulation, backtesting |
| [`docs/07-optimiser.md`](docs/07-optimiser.md) | MILP squad + multi-GW transfer planner, chips, risk |
| [`docs/08-llm-layer.md`](docs/08-llm-layer.md) | Task registry, prompts, sentiment, chat, guardrails |
| [`docs/09-api-contracts.md`](docs/09-api-contracts.md) | REST surface, request/response shapes |
| [`docs/10-frontend.md`](docs/10-frontend.md) | Screens, components, mobile behaviour, design direction |
| [`docs/11-config.md`](docs/11-config.md) | Full `.env` template + settings precedence |
| [`docs/12-testing.md`](docs/12-testing.md) | Unit, integration, golden-dataset and model regression tests |
| [`docs/13-roadmap.md`](docs/13-roadmap.md) | Phased build order for Claude Code |
| [`docs/14-idea-appendix.md`](docs/14-idea-appendix.md) | Every other idea, honestly graded — good, dubious, bad |

## The 2026/27 ruleset the app must encode

Verified against the Premier League's own announcements for this season, so build against these, not older FPL knowledge:

- <cite index="14-1">The season runs 22 August 2026 to 30 May 2027, with 33 weekend and five midweek rounds</cite>. <cite index="14-1">Coventry City, Ipswich Town and Hull City are promoted</cite>.
- <cite index="8-1">Two sets of chips again — Wildcard, Free Hit, Triple Captain, Bench Boost, one set per half, eight in total. The first set must be played before the Gameweek 19 deadline at 13:30 GMT on Saturday 2 January and cannot be carried over. Only one chip per Gameweek.</cite>
- <cite index="2-1">Up to five free transfers can be banked, and defensive contribution (DefCon) points continue unchanged.</cite>
- <cite index="3-1">The Bonus Points System has been retuned for 2026/27 to reduce overlap with defensive contributions and improve bonus prospects for goalkeepers, full-backs and attacking players.</cite> **This matters a lot** — it means BPS coefficients learned from 2025/26 data are stale, so the bonus model needs a season-aware recalibration path (see `06-ml-models.md`).
- <cite index="3-1">Gameweek lockdown moved to 09:00 UK time the day after the Gameweek's final match, allowing post-match Opta review data to feed into BPS and DefCon.</cite> Ingestion must therefore re-fetch and reconcile scores after lockdown, not at full-time.
- <cite index="7-1">FPL now ships its own price-change predictor, and price changes still occur daily at midnight UK time.</cite>
- <cite index="1-1">Points, ranks and mini-leagues update live, with projected bonus added after 20 minutes of each match.</cite> Useful later for the deferred live-tracking feature.
- <cite index="11-1">Eleven players were reclassified by position for 2026/27</cite> — position must always be read from the live API, never cached across seasons.

Squad rules to hard-code in the optimiser: 15 players (2 GK, 5 DEF, 5 MID, 3 FWD), £100.0m starting budget, max 3 per club, starting XI needs 1 GK / ≥3 DEF / ≥1 FWD.

## Quickstart

```bash
cp .env.example .env      # fill in what you have; everything optional degrades gracefully
docker compose up -d
# http://localhost:5173  → dashboard, http://localhost:8000/docs → API

docker compose exec api python -m fplai.cli backfill          # ~3-6h, mostly rate-limit sleep
docker compose exec api python -m fplai.cli build-features --all
docker compose exec api python -m fplai.cli train
docker compose exec api python -m fplai.cli predict
docker compose exec api python -m fplai.cli create-squad "Main" --entry-id <your fpl id>
docker compose exec api python -m fplai.cli optimise 1
```

Without Docker: `pip install -e api/` then `fplai serve`, and `npm --prefix web run dev`.

### Running it locally, step by step

| Command | What it does |
|---|---|
| `fplai init` | Create the DB, apply the schema, seed the source registry |
| `fplai ingest fpl_official` | Pull players, teams, fixtures, gameweeks, prices |
| `fplai backfill` | The ordered plan from `docs/04` across every source |
| `fplai build-features --all` | Compute the feature store for a whole season |
| `fplai train` | Walk-forward training, auto-promotion if it beats the incumbent |
| `fplai predict` | Correlated Monte Carlo → `predictions` |
| `fplai optimise <squad>` | MILP squad + transfer plan, three risk variants |
| `fplai backtest --seasons 2024-25` | Replay a season and score against reality |
| `fplai sources` | Source health: enabled, keys present, last run |
| `fplai job <name>` \| `fplai job list` | Run any scheduler job by hand |
| `fplai reprocess <source> --force` | Re-parse the raw archive with zero refetches |

## What runs without any API keys

The FPL API, vaastav's history, Understat, FBref, Open-Meteo, Bluesky, Reddit and the
RSS feeds need no credentials, and that is enough to build a squad end to end. Every
key-gated source self-disables and explains itself on the Sources screen rather than
failing. `SCRAPE_ENABLED=false` stops everything ToS-adjacent in one flip.

## What it measures so far

From a real run on this machine: 7 seasons backfilled (185,890 player-fixture rows),
2.3M feature values, all eight models trained, and 2025/26 replayed gameweek by gameweek.

| | |
|---|---|
| Minutes model | log-loss 0.465, calibration ECE 0.035 |
| Rank correlation, in-sample (GW2–33) | 0.784 (n = 25,042) |
| Rank correlation, held-out (GW34–38) | 0.713 (n = 4,015) |
| Backtest, balanced variant | 2,500 pts over 38 GWs (65.8/GW), 37 transfers, 0 hits |

Read those with the caveats below. In particular the backtest replays the same season the
models were trained on, so 65.8 is optimistic; the held-out window is the honest signal,
and it holding up near the in-sample figure is what suggests the model generalises rather
than memorises. There is no stored FPL overall average for that season in this database,
so the "vs the field" comparison the plan asks for is not yet populated.

## Honest limitations

Worth knowing before you trust a number:

- **No historic odds exist**, so odds features only start from your install date. The
  model gets an explicit "odds block absent" indicator rather than silently reading zeros,
  and leans on the team model instead — but backtests before install understate it.
- **The 2026/27 BPS retune** means bonus coefficients learned from earlier seasons are
  biased. A separate current-season model blends in as `n/(n+40)` fixtures accumulate, and
  the model performance screen carries a banner until the sample is real.
- **Pre-season predictions lean on price.** With no 2026/27 matches played, the minutes and
  rate models fall back to price percentile within position — FPL's own pricing is the only
  quality signal that exists before a ball is kicked. Rankings sharpen from GW3 or so.
- **Chip timing stays silent early.** Until doubles and blanks are scheduled, every
  gameweek looks identical, so the calendar says "too early" instead of naming a date.
- **vaastav's 2016/17–2018/19 files** use an older schema that omits the club column in a
  way this parser does not reconstruct. Seven seasons (2019/20 onward, ~186k player-fixture
  rows) load, which comfortably exceeds what the models need; those older seasons would
  carry a weight of 0.03 anyway.
- **The FPL write path is fragile and unsanctioned.** Off by default, human-initiated only,
  always previewed, and it snapshots your squad before touching anything.
