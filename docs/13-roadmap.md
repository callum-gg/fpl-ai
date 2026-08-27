# 13 — Roadmap

Ordered so that every phase ends with something usable. Build quality over speed, as you said — but each phase should leave a working app, not a half-wired one.

## Phase 0 — Skeleton (½ day)
Repo, Docker Compose, FastAPI hello, Vite hello, SQLite with pragmas, Alembic baseline, `pydantic-settings` reading `.env`, the settings table + `/api/settings` endpoints, structured logging, the CLI shell. **Done when** `docker compose up` gives you both services and a settings screen that persists a value.

## Phase 1 — FPL core (1–2 days)
`fpl_official` connector, raw archive + hashing + dedup framework, `players`/`teams`/`fixtures`/`gameweeks`/`player_fixture_stats`, entry sync, squad CRUD, squad picker, basic dashboard and player table. **Done when** you can see your real team and every player's FPL stats in your own UI.

## Phase 2 — History and backfill (1 day)
`vaastav_history`, `understat`, entity resolution ladder + review screen, the backfill CLI. **Done when** you have ~10 seasons of player-gameweek rows and the review queue is empty enough to trust.

## Phase 3 — Feature store + first model (2–3 days)
Feature registry, sections A–C of `05`, minutes model, team goals model, rate models, simulation, `predictions` table, predicted points shown in the player table and on the pitch. Walk-forward training in the CLI. **Done when** the app shows its own projections and you can eyeball whether they're sane.

## Phase 4 — Optimiser (2 days)
Rules engine (selling price, FTs, chips, autosubs) with its full test suite, single-GW MILP, then multi-GW planner, three variants, "do nothing" baseline. Dashboard headline card and transfer planner screen. **Done when** it recommends a real transfer with a real number attached.

**At the end of Phase 4 the app is genuinely useful.** Everything after this is edge and explanation. Don't skip ahead to the LLM before this point — a beautiful explanation of a bad recommendation is worse than no app.

## Phase 5 — Odds and availability (1–2 days)
Odds API / Betfair, devigging, odds-blended team model, injury scrapers, availability consensus, lineup providers if you add a key. Retrain and compare in the model performance screen. **Done when** the ablation shows odds features earning their place — and if they don't, find out why before building more.

## Phase 6 — Text pipeline (3–4 days)
News RSS + extraction, YouTube (tracked + discovery + transcripts), Reddit, Bluesky, X fallback chain, chunking, embeddings, sqlite-vec, LLM claim extraction, near-dupe collapse, text features, the feed screen and evidence panels. **Done when** a player detail page shows what was said about them this week, with timestamped video links.

## Phase 7 — LLM reasoning (1–2 days)
Task registry, explanation, critique pass with constrained re-solve, chat with read-only tools, weekly digest. **Done when** you can ask "why not X?" and get an answer backed by real optimiser output.

## Phase 8 — Backtesting and honesty (2 days)
Backtest harness, ablations, calibration curves, pundit accuracy scoreboard, model performance screen. **Done when** you know whether the thing actually beats a naive baseline. Expect this phase to change your mind about at least one earlier decision.

## Phase 9 — Multi-squad and risk (1–2 days)
Risk slider wiring, rank-aware mode with mini-league ingestion, comparison view, squad cloning, per-squad settings UI. **Done when** two squads with different risk settings visibly diverge in their recommendations.

## Phase 10 — Polish (1–2 days)
Discord notifications, deadline alerts, chip-expiry nagging, mobile pass, PWA manifest, command palette, FPL push with preview and confirmation, source health dashboard.

## Future enhancements (explicitly deferred)

- **Live gameweek tracking** — you deferred this. FPL now updates ranks and projected bonus live, so the app should consume `event/{gw}/live` every couple of minutes and show live projected rank, bonus, and autosub outcomes.
- **Whisper transcription** for caption-less videos (the 3070 makes this cheap; the `transcript_source` enum is already in the schema).
- **Podcast ingestion** — the FPL podcast scene is large and identical in shape to the video pipeline once Whisper exists.
- **FPL Draft / Fantasy Champions League** support.
- **Postgres migration** and a proper task queue when it moves to a VPS.
- **Alerting on breaking news** — push a Discord message within minutes of a tier-1 source reporting an injury to a player you own.
- **A public read-only share link** for a squad's projections (needs the auth work to be real first).

## Realistic total

Roughly 15–20 focused days of Claude Code work for all ten phases, most of it in Phases 3, 6 and 8. Phases 0–4 alone — a working, useful, model-backed FPL tool — is maybe 6–8 days.
