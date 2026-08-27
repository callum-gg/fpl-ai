# 01 — Architecture

## Guiding principles

1. **Fetch once, store forever, reprocess freely.** Every byte a connector receives is archived raw before anything parses it. Reprocessing a changed parser must never require re-hitting a rate-limited source.
2. **Every number is traceable.** A recommendation points to a prediction, which points to features, which point to raw documents. The evidence panel in the UI is a UI over that chain, not a re-summary.
3. **Optional everything.** Missing an API key disables that connector and logs a warning. The app must produce a squad with nothing but the free FPL API.
4. **The model decides, the LLM explains.** The MILP optimiser working from the ML model's distributions produces the picks. The LLM annotates, argues, and proposes alternates — it never silently rewrites the squad.
5. **Settings are data, not code.** `.env` holds secrets and infrastructure. Everything tunable lives in the DB, editable in the UI, at global or squad scope.

## Services (Docker Compose)

```
┌──────────────────────────────────────────────────────────────┐
│ web            Vite + React + TS, nginx-served, port 5173     │
├──────────────────────────────────────────────────────────────┤
│ api            FastAPI + Uvicorn, port 8000                   │
│                ├── REST surface                               │
│                ├── APScheduler (in-process, jobstore=SQLite)  │
│                ├── connectors/  (ingestion)                   │
│                ├── features/    (feature store build)         │
│                ├── models/      (train + predict + simulate)  │
│                ├── optimiser/   (MILP)                        │
│                └── llm/         (task registry)               │
├──────────────────────────────────────────────────────────────┤
│ ollama         optional, GPU passthrough, port 11434          │
├──────────────────────────────────────────────────────────────┤
│ (volume) ./data/fplai.db  + ./data/raw/ + ./data/models/      │
└──────────────────────────────────────────────────────────────┘
```

One process for the API and the scheduler in v1. `APSCHEDULER_ENABLED=false` lets you run a second API container as a pure web server later without duplicate jobs. Upgrade path when jobs outgrow this: swap APScheduler for Celery + Redis, connector functions become tasks unchanged (they're already pure `run(context) -> IngestResult`).

### Windows / GPU notes

Docker Desktop on WSL2 with the NVIDIA container toolkit gives the `ollama` container the 3070. Add `deploy.resources.reservations.devices` for `driver: nvidia`. Embeddings (`bge-small-en-v1.5` via sentence-transformers) also run on GPU inside the `api` container if `EMBEDDING_DEVICE=cuda`; fall back to CPU automatically if `torch.cuda.is_available()` is false — this is what makes the eventual VPS move painless.

### Why SQLite is right here, and where it bites

Single user, single writer, tens of millions of rows at most, `WAL` mode, and one file to back up. It bites on: concurrent long-running writes during ingestion (mitigated by a single writer thread and `busy_timeout=30000`), and analytical scans over the fact tables (mitigated by attaching the same file in DuckDB read-only for training queries — `duckdb.connect().execute("ATTACH 'fplai.db' AS s (TYPE sqlite)")`). If you later move to a VPS and want a second consumer, the schema is plain enough to `pgloader` into Postgres.

## Data flow

```
sources ──> connectors ──> raw_documents (hashed, deduped)
                              │
                              ├──> parsers ──> normalised tables (players, fixtures, stats, injuries, odds…)
                              │                        │
                              │                        └──> feature_store (per player × gameweek)
                              │
                              └──> chunker + embedder ──> doc_chunks + vec index (sqlite-vec)
                                                              │
predictions ◀── models (minutes, goals, defcon, bonus…) ◀─────┘  (text-derived features join in here)
     │
     └──> monte_carlo simulation ──> per-player point distributions
                                          │
                                          ▼
                              optimiser (MILP, per squad settings)
                                          │
                                          ├──> recommendations + evidence links
                                          └──> LLM annotation / critique / chat
                                                          │
                                                          ▼
                                                     UI + Discord
```

## Repo layout

```
fpl-ai/
├── docker-compose.yml
├── .env.example
├── data/                        # gitignored volume
│   ├── fplai.db
│   ├── raw/                     # large blobs (transcripts, html) keyed by content hash
│   └── models/                  # serialised model artefacts + metadata json
├── api/
│   ├── pyproject.toml
│   ├── alembic/                 # migrations
│   └── src/fplai/
│       ├── main.py              # FastAPI app factory
│       ├── config.py            # pydantic-settings, .env → Settings
│       ├── db/
│       │   ├── engine.py        # sqlite pragmas, single-writer lock
│       │   ├── models.py        # SQLAlchemy ORM
│       │   └── repositories/
│       ├── connectors/
│       │   ├── base.py          # Connector ABC, IngestResult, retry/backoff
│       │   ├── fpl_official.py
│       │   ├── understat.py
│       │   ├── fbref.py
│       │   ├── football_data_org.py
│       │   ├── api_football.py
│       │   ├── sportmonks.py
│       │   ├── odds_api.py
│       │   ├── betfair.py
│       │   ├── livefpl.py
│       │   ├── youtube.py
│       │   ├── reddit.py
│       │   ├── bluesky.py
│       │   ├── twitter_scrape.py
│       │   ├── rss_news.py
│       │   ├── premier_injuries.py
│       │   ├── physioroom.py
│       │   ├── transfermarkt.py
│       │   ├── sofascore.py
│       │   ├── whoscored.py
│       │   ├── setpieces.py
│       │   ├── weather.py
│       │   ├── euro_fixtures.py
│       │   └── vaastav_history.py
│       ├── resolve/             # entity resolution: player/team alias matching
│       ├── features/
│       │   ├── registry.py      # @feature decorator, dependency graph
│       │   └── builders/
│       ├── models/
│       │   ├── minutes.py
│       │   ├── team_goals.py
│       │   ├── player_share.py
│       │   ├── defcon.py
│       │   ├── bonus.py
│       │   ├── saves_cards.py
│       │   ├── simulate.py
│       │   ├── train.py
│       │   └── backtest.py
│       ├── optimiser/
│       │   ├── squad.py         # initial squad MILP
│       │   ├── planner.py       # multi-GW transfer path MILP
│       │   ├── chips.py
│       │   └── risk.py          # utility functions, EO/rank-aware objectives
│       ├── llm/
│       │   ├── client.py        # OpenAI-compatible, per-task model resolution
│       │   ├── tasks/           # one module per task, each with prompt + schema
│       │   └── cache.py         # prompt-hash → response cache
│       ├── fplsync/             # read team from FPL, push transfers/lineup back
│       ├── notify/discord.py
│       ├── scheduler/jobs.py
│       ├── api/routers/
│       └── cli.py               # typer: backfill, ingest, train, backtest, predict, optimise
└── web/
    ├── package.json             # vite, react, typescript, tailwind, tanstack-query, zustand
    └── src/
        ├── main.tsx
        ├── lib/api.ts           # generated from OpenAPI schema
        ├── stores/squad.ts      # active squad id, comparison set
        ├── components/
        └── routes/
```

## Concurrency and the writer lock

All DB writes go through `db.engine.writer_session()`, guarded by a `threading.Lock`. Connectors are IO-bound and run concurrently under `asyncio`, but they buffer parsed rows and hand them to the writer in batches. This keeps `database is locked` errors at zero without giving up parallel fetching.

## Exposure later

You said this might get exposed. Build in, but leave off by default:

- `APP_AUTH_MODE=none|token|basic`. In `token` mode a middleware checks `X-App-Token` against `APP_TOKEN`, and the frontend stores it in `localStorage` after a one-field login screen.
- Bind to `127.0.0.1` by default; `BIND_HOST` env to change it.
- CORS allowlist from `ALLOWED_ORIGINS`.
- Never log secrets; the settings API redacts any key whose name matches `/(KEY|TOKEN|SECRET|PASSWORD)/`.
