# 11 — Configuration

## Precedence

```
.env  →  Settings (global, DB)  →  Settings (squad, DB)  →  request-level override
```

`.env` holds **secrets and infrastructure** — things that shouldn't be editable from a web UI. The DB holds **everything tunable**, so you can change behaviour without a restart. Where a setting exists in both, `.env` supplies the initial value on first boot and the DB wins thereafter; the settings UI marks those as "seeded from .env".

Loaded with `pydantic-settings`. Missing optional keys are fine — the corresponding connector self-disables and says so on the Sources screen.

## `.env.example`

```ini
# ─── Core ────────────────────────────────────────────────────────────────────
APP_ENV=local
TZ=Europe/London
BIND_HOST=127.0.0.1
API_PORT=8000
WEB_PORT=5173
LOG_LEVEL=INFO
DATA_DIR=./data
DATABASE_URL=sqlite:///./data/fplai.db
CURRENT_SEASON=2026-27

# Off by default; flip if you ever expose this beyond localhost
APP_AUTH_MODE=none              # none | token | basic
APP_TOKEN=
ALLOWED_ORIGINS=http://localhost:5173

# ─── LLM (OpenAI-compatible) ────────────────────────────────────────────────
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=
LLM_DEFAULT_MODEL=
LLM_TIMEOUT_S=120
LLM_MAX_RETRIES=2
LLM_CACHE_ENABLED=true

# Optional second endpoint, e.g. NVIDIA build, selectable per task as "alt:<model>"
LLM_ALT_BASE_URL=
LLM_ALT_API_KEY=

# Local models
OLLAMA_ENABLED=false
OLLAMA_BASE_URL=http://ollama:11434/v1
OLLAMA_API_KEY=ollama

# Embeddings
EMBEDDING_PROVIDER=local        # local | openai_compatible
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=auto           # auto | cuda | cpu
EMBEDDING_DIM=384

# ─── FPL ────────────────────────────────────────────────────────────────────
FPL_ENTRY_IDS=                  # comma-separated; squads can also set their own
FPL_WRITE_ENABLED=false
FPL_EMAIL=
FPL_PASSWORD=
FPL_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64)

# ─── Paid / key-gated sources (leave blank to disable) ──────────────────────
ODDS_API_KEY=
ODDS_API_REGIONS=uk
ODDS_API_MARKETS=h2h,totals

BETFAIR_APP_KEY=
BETFAIR_USERNAME=
BETFAIR_PASSWORD=
BETFAIR_CERT_PATH=
BETFAIR_KEY_PATH=

API_FOOTBALL_KEY=
API_FOOTBALL_HOST=v3.football.api-sports.io

SPORTMONKS_API_KEY=

YOUTUBE_API_KEY=
SUPADATA_API_KEY=               # transcript fallback
APIFY_TOKEN=                    # transcript fallback

REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=fpl-ai/1.0 by u/yourname

BLUESKY_HANDLE=                 # optional; reads work unauthenticated
BLUESKY_APP_PASSWORD=

FOOTBALL_DATA_ORG_KEY=

# ─── X / Twitter (unofficial, layered fallback) ─────────────────────────────
X_ENABLED=true
X_METHODS=syndication,nitter,twscrape
NITTER_INSTANCES=https://nitter.net,https://nitter.privacydev.net
TWSCRAPE_ACCOUNTS_FILE=./data/secrets/twscrape_accounts.json

# ─── Scraping ───────────────────────────────────────────────────────────────
SCRAPE_ENABLED=true             # global kill switch for all ToS-adjacent scrapers
SCRAPE_MIN_DELAY_MS=1200
SCRAPE_MAX_CONCURRENCY=4
HTTP_PROXY_URL=

# ─── Scheduler ──────────────────────────────────────────────────────────────
APSCHEDULER_ENABLED=true
INGEST_ON_STARTUP=false
DEADLINE_TURBO_HOURS=6          # tighten cadences this many hours before a deadline

# ─── Notifications ──────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL=
DISCORD_DIGEST_HOUR=8
NOTIFY_PRICE_CHANGES=true
NOTIFY_INJURY_TO_OWNED=true
NOTIFY_DEADLINE_HOURS=24,2

# ─── Modelling ──────────────────────────────────────────────────────────────
SIM_ITERATIONS=10000
MODEL_AUTO_PROMOTE=true
TRAIN_SEASON_DECAY=0.72
OPTIMISER_SOLVER=CBC            # CBC | HiGHS
OPTIMISER_TIME_LIMIT_S=60
```

## Global settings (DB, UI-editable)

```jsonc
{
  "llm.tasks": {
    "extract_claims":          {"model": "…", "temperature": 0.1},
    "resolve_entity":          {"model": "…", "temperature": 0.0},
    "classify_injury_severity":{"model": "…"},
    "summarise_video":         {"model": "…"},
    "explain_recommendation":  {"model": "…", "temperature": 0.4},
    "critique_recommendation": {"model": "…", "temperature": 0.5},
    "chat":                    {"model": "…", "temperature": 0.6},
    "weekly_digest":           {"model": "…"}
  },
  "sources.enabled":  {"understat": true, "fbref": true, "sofascore": true, "whoscored": false, "…": true},
  "sources.cadence":  {"news_rss": "*/20 * * * *", "odds_poll": "0 */4 * * *"},
  "youtube.channels": [{"channel_id": "UC…", "title": "Let's Talk FPL", "tracked": true, "trust_weight": 1.0}],
  "youtube.discovery_queries": ["FPL GW{gw} team selection", "fantasy premier league {gw} transfers"],
  "youtube.discovery_min_views": 2000,
  "rss.feeds": ["https://feeds.bbci.co.uk/sport/football/rss.xml", "…"],
  "x.handles": ["@…"],
  "reddit.subreddits": ["FantasyPL", "soccer"],
  "text.trust_weights": {"tier1_journalist": 1.5, "official_club": 2.0, "youtube_default": 0.8, "reddit": 0.4},
  "adjustment.enabled": true,
  "adjustment.max_points": 2.0,
  "ui.theme": "dark",
  "watchlist": [12, 88, 301]
}
```

## Squad settings (DB, per squad)

```jsonc
{
  "risk": -1.0,                     // -1 safe … 0 balanced … +1 aggressive
  "horizon_gws": 5,
  "horizon_decay": 0.84,
  "bench_weight": 0.12,
  "max_hits_per_gw": 1,
  "min_expected_gain_to_act": 0.8,
  "prefer_differentials": false,
  "rank_mode": "maximise_points",   // maximise_points | climb_rank | protect_rank
  "leagues": [{"league_id": 314159, "target_rank": 1, "rivals": [12345, 67890]}],
  "banned_clubs": [],
  "locked_players": [],
  "must_own": [],
  "chip_strategy": {"wildcard_earliest_gw": 4, "bench_boost_prefer_dgw": true, "save_second_set": true},
  "price_bonus_weight": 0.0,
  "auto_sync_from_fpl": true,
  "notes": "work league, currently 40 pts behind"
}
```

Cloning a squad copies settings — the intended workflow is "duplicate my main squad, crank risk to +0.8, see what it says".
