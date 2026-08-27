# 03 — Database Schema

SQLite, WAL mode, managed with Alembic. Three layers:

- **L0 raw** — immutable archive of everything fetched, content-addressed.
- **L1 normalised** — parsed entities, one row per real-world fact.
- **L2 derived** — features, predictions, recommendations, evidence.

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 30000;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;
```

---

## L0 — Raw archive and dedup

```sql
CREATE TABLE sources (
    id                  TEXT PRIMARY KEY,          -- 'fpl_official', 'youtube', ...
    display_name        TEXT NOT NULL,
    category            TEXT NOT NULL,             -- fpl|stats|odds|news|social|video|injury|meta
    requires_key        INTEGER NOT NULL DEFAULT 0,
    enabled             INTEGER NOT NULL DEFAULT 1,
    base_url            TEXT,
    rate_limit_per_min  REAL,
    trust_weight        REAL NOT NULL DEFAULT 1.0, -- learned, see pundit scoreboard
    config_json         TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE ingest_runs (
    id              INTEGER PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES sources(id),
    job_name        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL,                 -- running|ok|partial|failed
    requests_made   INTEGER NOT NULL DEFAULT 0,
    docs_new        INTEGER NOT NULL DEFAULT 0,
    docs_duplicate  INTEGER NOT NULL DEFAULT 0,
    rows_upserted   INTEGER NOT NULL DEFAULT 0,
    error_text      TEXT,
    params_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX ix_ingest_runs_source_time ON ingest_runs(source_id, started_at DESC);

-- The dedup heart of the system.
CREATE TABLE raw_documents (
    id                INTEGER PRIMARY KEY,
    source_id         TEXT NOT NULL REFERENCES sources(id),
    doc_type          TEXT NOT NULL,               -- bootstrap|fixture|element_summary|article|transcript|tweet|odds_snapshot|...
    external_id       TEXT,                        -- video id, tweet id, article url, endpoint+entity
    url               TEXT,
    content_hash      TEXT NOT NULL,               -- sha256 of NORMALISED content
    simhash           INTEGER,                     -- 64-bit, for near-dupe news
    payload_inline    TEXT,                        -- small payloads live here
    payload_path      TEXT,                        -- large blobs on disk: data/raw/{hash[:2]}/{hash}.json.zst
    content_bytes     INTEGER NOT NULL,
    published_at      TEXT,
    fetched_at        TEXT NOT NULL DEFAULT (datetime('now')),
    first_seen_run    INTEGER REFERENCES ingest_runs(id),
    last_seen_at      TEXT NOT NULL DEFAULT (datetime('now')),
    seen_count        INTEGER NOT NULL DEFAULT 1,
    parsed_at         TEXT,
    parser_version    INTEGER,
    parse_status      TEXT NOT NULL DEFAULT 'pending', -- pending|ok|failed|skipped
    parse_error       TEXT,
    supersedes_id     INTEGER REFERENCES raw_documents(id),
    meta_json         TEXT NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX ux_raw_hash        ON raw_documents(source_id, content_hash);
CREATE UNIQUE INDEX ux_raw_external    ON raw_documents(source_id, doc_type, external_id, content_hash)
    WHERE external_id IS NOT NULL;
CREATE INDEX ix_raw_type_time          ON raw_documents(doc_type, published_at DESC);
CREATE INDEX ix_raw_parse_pending      ON raw_documents(parse_status, source_id) WHERE parse_status = 'pending';
CREATE INDEX ix_raw_simhash            ON raw_documents(simhash) WHERE simhash IS NOT NULL;
```

### The four-layer dedup strategy

1. **Natural key** — `(source_id, doc_type, external_id)`. A tweet id, a video id, a fixture id. If we've seen it and the hash matches, we bump `seen_count` and `last_seen_at` and stop.
2. **Content hash** — SHA-256 over *normalised* content: for JSON, `json.dumps(obj, sort_keys=True, separators=(',',':'))` with volatile fields stripped (timestamps, request ids, `now` fields); for HTML, extracted main text with whitespace collapsed and lowercased. Identical content from the same source is never stored twice. **Crucially, changed content under the same external id IS stored** as a new row with `supersedes_id` pointing at the old one — that's how you get a free history of, say, a player's injury status changing.
3. **Near-duplicate (news only)** — 64-bit SimHash over token shingles; Hamming distance ≤ 3 against candidates from the last 7 days means "same wire story". Store it, but flag `meta_json.near_dupe_of` so the LLM and the UI count it once. A story appearing on 6 sites is one fact, not six confirmations — and if you don't do this, your "consensus" signals become garbage.
4. **Semantic (claims)** — after extraction, claims about the same player+topic+day with cosine similarity > 0.92 collapse into one claim with multiple sources. This is what stops the evidence panel showing the same rumour twelve times.

Disk layout for big blobs: zstd-compressed, content-addressed, so identical transcripts across sources share a file.

---

## L1 — Normalised football entities

```sql
CREATE TABLE seasons (
    id           TEXT PRIMARY KEY,       -- '2026-27'
    start_date   TEXT, end_date TEXT,
    is_current   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE teams (
    id            INTEGER PRIMARY KEY,   -- internal
    season_id     TEXT NOT NULL REFERENCES seasons(id),
    fpl_team_id   INTEGER,
    name          TEXT NOT NULL,
    short_name    TEXT,
    code          INTEGER,
    strength_overall_home INTEGER, strength_overall_away INTEGER,
    strength_attack_home  INTEGER, strength_attack_away  INTEGER,
    strength_defence_home INTEGER, strength_defence_away INTEGER,
    stadium_name  TEXT, stadium_lat REAL, stadium_lon REAL,
    UNIQUE(season_id, fpl_team_id)
);

CREATE TABLE players (
    id              INTEGER PRIMARY KEY,      -- stable internal id across seasons
    canonical_name  TEXT NOT NULL,
    first_name      TEXT, last_name TEXT, web_name TEXT,
    birth_date      TEXT,
    nationality     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Entity resolution: one row per (player, external system).
CREATE TABLE player_external_ids (
    player_id     INTEGER NOT NULL REFERENCES players(id),
    system        TEXT NOT NULL,            -- fpl|understat|fbref|sofascore|transfermarkt|apifootball|...
    external_id   TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    method        TEXT NOT NULL,            -- exact|fuzzy|manual|llm
    verified      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (system, external_id)
);
CREATE INDEX ix_pei_player ON player_external_ids(player_id);

CREATE TABLE player_aliases (
    id          INTEGER PRIMARY KEY,
    player_id   INTEGER NOT NULL REFERENCES players(id),
    alias       TEXT NOT NULL,              -- 'Trent', 'TAA', 'KDB', 'Bruno'
    alias_norm  TEXT NOT NULL,              -- lowercased, accent-stripped
    origin      TEXT NOT NULL,              -- manual|llm|derived
    UNIQUE(alias_norm, player_id)
);
CREATE INDEX ix_alias_norm ON player_aliases(alias_norm);
-- Same two tables exist for teams: team_external_ids, team_aliases ('Spurs','Man U','Wolves').

CREATE TABLE player_seasons (
    player_id     INTEGER NOT NULL REFERENCES players(id),
    season_id     TEXT NOT NULL REFERENCES seasons(id),
    team_id       INTEGER REFERENCES teams(id),
    fpl_element_id INTEGER,
    position      TEXT NOT NULL,            -- GK|DEF|MID|FWD  (re-read every season; 11 changed for 2026/27)
    start_price   INTEGER,                  -- tenths of a million, integer only
    PRIMARY KEY (player_id, season_id)
);

CREATE TABLE fixtures (
    id              INTEGER PRIMARY KEY,
    season_id       TEXT NOT NULL REFERENCES seasons(id),
    fpl_fixture_id  INTEGER,
    gameweek        INTEGER,                -- NULL for postponed/unscheduled
    kickoff_utc     TEXT,
    home_team_id    INTEGER NOT NULL REFERENCES teams(id),
    away_team_id    INTEGER NOT NULL REFERENCES teams(id),
    finished        INTEGER NOT NULL DEFAULT 0,
    home_score      INTEGER, away_score INTEGER,
    fdr_home        INTEGER, fdr_away INTEGER,
    competition     TEXT NOT NULL DEFAULT 'PL',   -- PL|UCL|UEL|UECL|FAC|EFL — non-PL rows drive congestion features
    UNIQUE(season_id, fpl_fixture_id)
);
CREATE INDEX ix_fixtures_gw ON fixtures(season_id, gameweek);
CREATE INDEX ix_fixtures_kick ON fixtures(kickoff_utc);

CREATE TABLE gameweeks (
    season_id     TEXT NOT NULL REFERENCES seasons(id),
    gameweek      INTEGER NOT NULL,
    deadline_utc  TEXT NOT NULL,
    is_current    INTEGER NOT NULL DEFAULT 0,
    is_next       INTEGER NOT NULL DEFAULT 0,
    finished      INTEGER NOT NULL DEFAULT 0,
    data_checked  INTEGER NOT NULL DEFAULT 0,   -- true only after the 09:00-next-day lockdown
    average_score INTEGER,
    highest_score INTEGER,
    chip_plays_json TEXT,
    PRIMARY KEY (season_id, gameweek)
);

-- The central fact table: one row per player per fixture appearance.
CREATE TABLE player_fixture_stats (
    id                  INTEGER PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES players(id),
    fixture_id          INTEGER NOT NULL REFERENCES fixtures(id),
    team_id             INTEGER NOT NULL REFERENCES teams(id),
    was_home            INTEGER NOT NULL,
    -- FPL scoring components
    minutes             INTEGER, goals_scored INTEGER, assists INTEGER,
    clean_sheets        INTEGER, goals_conceded INTEGER, own_goals INTEGER,
    penalties_saved     INTEGER, penalties_missed INTEGER,
    yellow_cards        INTEGER, red_cards INTEGER, saves INTEGER,
    bonus               INTEGER, bps INTEGER,
    defensive_contribution INTEGER,          -- the DefCon count (CBIT / CBIRT)
    defcon_points       INTEGER,
    total_points        INTEGER,
    starts              INTEGER,
    -- advanced (Understat/FBref)
    xg REAL, xa REAL, npxg REAL, xgot REAL, shots INTEGER, shots_on_target INTEGER,
    key_passes INTEGER, big_chances INTEGER, big_chances_missed INTEGER,
    touches_in_box INTEGER, progressive_carries INTEGER, sca INTEGER, gca INTEGER,
    tackles INTEGER, interceptions INTEGER, blocks INTEGER, clearances INTEGER, recoveries INTEGER,
    -- ratings
    sofascore_rating REAL, whoscored_rating REAL,
    -- provenance
    source_ids_json     TEXT NOT NULL DEFAULT '[]',
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(player_id, fixture_id)
);
CREATE INDEX ix_pfs_fixture ON player_fixture_stats(fixture_id);
CREATE INDEX ix_pfs_player  ON player_fixture_stats(player_id);

CREATE TABLE player_prices (
    player_id  INTEGER NOT NULL REFERENCES players(id),
    season_id  TEXT NOT NULL REFERENCES seasons(id),
    observed_at TEXT NOT NULL,
    price      INTEGER NOT NULL,            -- tenths
    selected_by_percent REAL,
    transfers_in_event INTEGER, transfers_out_event INTEGER,
    net_transfers      INTEGER,
    PRIMARY KEY (player_id, season_id, observed_at)
);

CREATE TABLE availability (
    id           INTEGER PRIMARY KEY,
    player_id    INTEGER NOT NULL REFERENCES players(id),
    source_id    TEXT NOT NULL REFERENCES sources(id),
    raw_doc_id   INTEGER REFERENCES raw_documents(id),
    observed_at  TEXT NOT NULL,
    status       TEXT NOT NULL,             -- available|doubt|injured|suspended|unknown
    chance_pct   INTEGER,                   -- 0..100 where known
    issue        TEXT,                      -- 'hamstring'
    expected_return TEXT,
    note         TEXT,
    UNIQUE(player_id, source_id, observed_at, status, COALESCE(note,''))
);
CREATE INDEX ix_avail_player_time ON availability(player_id, observed_at DESC);

CREATE TABLE lineups (
    id            INTEGER PRIMARY KEY,
    fixture_id    INTEGER NOT NULL REFERENCES fixtures(id),
    player_id     INTEGER NOT NULL REFERENCES players(id),
    source_id     TEXT NOT NULL REFERENCES sources(id),
    kind          TEXT NOT NULL,            -- predicted|confirmed
    is_starting   INTEGER NOT NULL,
    formation     TEXT, position_slot TEXT,
    observed_at   TEXT NOT NULL,
    UNIQUE(fixture_id, player_id, source_id, kind, observed_at)
);

CREATE TABLE odds_snapshots (
    id             INTEGER PRIMARY KEY,
    fixture_id     INTEGER NOT NULL REFERENCES fixtures(id),
    source_id      TEXT NOT NULL REFERENCES sources(id),
    bookmaker      TEXT,
    market         TEXT NOT NULL,           -- h2h|totals|clean_sheet|anytime_scorer|player_shots
    selection      TEXT NOT NULL,           -- 'home'|'over_2.5'|player external ref
    player_id      INTEGER REFERENCES players(id),
    price_decimal  REAL NOT NULL,
    implied_prob   REAL NOT NULL,
    devig_prob     REAL,
    observed_at    TEXT NOT NULL,
    UNIQUE(fixture_id, source_id, bookmaker, market, selection, observed_at)
);
CREATE INDEX ix_odds_fixture_market ON odds_snapshots(fixture_id, market, observed_at DESC);

CREATE TABLE set_piece_roles (
    player_id  INTEGER NOT NULL REFERENCES players(id),
    season_id  TEXT NOT NULL REFERENCES seasons(id),
    role       TEXT NOT NULL,               -- penalties|direct_fk|corners_left|corners_right
    rank       INTEGER NOT NULL,            -- 1 = first choice
    source_id  TEXT NOT NULL REFERENCES sources(id),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (player_id, season_id, role, source_id, observed_at)
);

CREATE TABLE weather_observations (
    fixture_id  INTEGER PRIMARY KEY REFERENCES fixtures(id),
    temp_c REAL, wind_kph REAL, precip_mm REAL, humidity REAL,
    is_forecast INTEGER NOT NULL DEFAULT 1,
    observed_at TEXT NOT NULL
);

CREATE TABLE ownership_snapshots (
    player_id     INTEGER NOT NULL REFERENCES players(id),
    gameweek      INTEGER NOT NULL,
    season_id     TEXT NOT NULL,
    scope         TEXT NOT NULL,            -- overall|top10k|top1k|league:{id}
    owned_pct     REAL, captained_pct REAL, effective_ownership REAL,
    observed_at   TEXT NOT NULL,
    PRIMARY KEY (season_id, gameweek, player_id, scope, observed_at)
);
```

## L1b — Text: articles, videos, social, and extracted claims

```sql
CREATE TABLE articles (
    id            INTEGER PRIMARY KEY,
    raw_doc_id    INTEGER NOT NULL UNIQUE REFERENCES raw_documents(id),
    title         TEXT, author TEXT, outlet TEXT,
    published_at  TEXT, url TEXT,
    body_text     TEXT NOT NULL,
    word_count    INTEGER,
    near_dupe_group INTEGER            -- shared id across syndicated copies
);

CREATE TABLE videos (
    id             INTEGER PRIMARY KEY,
    raw_doc_id     INTEGER NOT NULL UNIQUE REFERENCES raw_documents(id),
    youtube_id     TEXT NOT NULL UNIQUE,
    channel_id     TEXT NOT NULL,
    channel_title  TEXT,
    title          TEXT, description TEXT,
    published_at   TEXT, duration_s INTEGER,
    view_count     INTEGER, like_count INTEGER,
    gameweek_hint  INTEGER,
    transcript_source TEXT,             -- youtube_manual|youtube_auto|supadata|apify|whisper
    transcript_text TEXT,
    transcript_json TEXT,               -- [{start, dur, text}]
    discovered_via TEXT NOT NULL        -- tracked_channel|search_discovery
);

CREATE TABLE channels (
    channel_id      TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    tracked         INTEGER NOT NULL DEFAULT 1,
    subscriber_count INTEGER,
    accuracy_score  REAL,               -- populated by the pundit scoreboard
    accuracy_n      INTEGER NOT NULL DEFAULT 0,
    trust_weight    REAL NOT NULL DEFAULT 1.0,
    added_by        TEXT NOT NULL DEFAULT 'seed'  -- seed|discovery|manual
);

CREATE TABLE social_posts (
    id             INTEGER PRIMARY KEY,
    raw_doc_id     INTEGER NOT NULL UNIQUE REFERENCES raw_documents(id),
    platform       TEXT NOT NULL,       -- x|bluesky|reddit
    external_id    TEXT NOT NULL,
    author_handle  TEXT, author_display TEXT,
    author_is_verified INTEGER,
    body_text      TEXT NOT NULL,
    posted_at      TEXT,
    likes INTEGER, reposts INTEGER, replies INTEGER, score INTEGER,
    parent_external_id TEXT,
    retrieval_method TEXT,              -- syndication|nitter|twscrape|playwright|api
    UNIQUE(platform, external_id)
);

-- Chunked text for retrieval; vectors in a sqlite-vec virtual table.
CREATE TABLE doc_chunks (
    id           INTEGER PRIMARY KEY,
    raw_doc_id   INTEGER NOT NULL REFERENCES raw_documents(id),
    ordinal      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    start_s      REAL,                  -- video timestamp, enables deep links
    end_s        REAL,
    token_count  INTEGER,
    UNIQUE(raw_doc_id, ordinal)
);
CREATE VIRTUAL TABLE chunk_vec USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[384]);
CREATE VIRTUAL TABLE chunk_fts USING fts5(text, content='doc_chunks', content_rowid='id');

-- The structured output of LLM extraction. This is the bridge from text to features.
CREATE TABLE claims (
    id             INTEGER PRIMARY KEY,
    raw_doc_id     INTEGER NOT NULL REFERENCES raw_documents(id),
    chunk_id       INTEGER REFERENCES doc_chunks(id),
    player_id      INTEGER REFERENCES players(id),
    team_id        INTEGER REFERENCES teams(id),
    claim_type     TEXT NOT NULL,       -- injury|return|rotation|form|recommendation|captain_pick|avoid|transfer_rumour|role_change|penalty_duty|manager_quote
    stance         TEXT,                -- positive|negative|neutral
    sentiment      REAL,                -- -1..1
    confidence     REAL,                -- 0..1, model's own confidence
    horizon_gw     INTEGER,             -- which GW it refers to
    text_span      TEXT NOT NULL,       -- the supporting quote, kept short
    start_s        REAL,                -- video deep-link
    extracted_at   TEXT NOT NULL,
    extractor_model TEXT NOT NULL,
    semantic_group INTEGER,             -- collapses duplicate claims
    UNIQUE(raw_doc_id, chunk_id, player_id, claim_type, text_span)
);
CREATE INDEX ix_claims_player_time ON claims(player_id, extracted_at DESC);
CREATE INDEX ix_claims_type ON claims(claim_type, horizon_gw);
```

## L2 — Features, predictions, squads, recommendations

```sql
CREATE TABLE feature_values (
    player_id   INTEGER NOT NULL REFERENCES players(id),
    season_id   TEXT NOT NULL,
    gameweek    INTEGER NOT NULL,
    fixture_id  INTEGER REFERENCES fixtures(id),   -- non-null distinguishes DGW legs
    name        TEXT NOT NULL,
    value       REAL,
    computed_at TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    PRIMARY KEY (season_id, gameweek, player_id, COALESCE(fixture_id,0), name)
) WITHOUT ROWID;
```
> Note: a tall table is flexible but slow to train from. Build it tall, then materialise a wide `feature_matrix` parquet per season under `data/models/` for training; the tall table stays the queryable source of truth for the UI's "why".

```sql
CREATE TABLE model_versions (
    id            INTEGER PRIMARY KEY,
    model_name    TEXT NOT NULL,          -- minutes|team_goals|player_goal_share|defcon|bonus|saves|cards
    version       TEXT NOT NULL,
    trained_at    TEXT NOT NULL,
    train_rows    INTEGER,
    train_seasons TEXT,
    metrics_json  TEXT NOT NULL,          -- {'log_loss':…, 'mae':…, 'calibration_ece':…}
    params_json   TEXT NOT NULL,
    artefact_path TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    UNIQUE(model_name, version)
);

CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES players(id),
    season_id       TEXT NOT NULL,
    gameweek        INTEGER NOT NULL,
    fixture_id      INTEGER REFERENCES fixtures(id),
    generated_at    TEXT NOT NULL,
    -- component expectations
    p_start REAL, p_appear REAL, exp_minutes REAL,
    exp_goals REAL, exp_assists REAL, p_clean_sheet REAL,
    exp_saves REAL, exp_defcon_points REAL, exp_bonus REAL,
    exp_cards_penalty REAL, exp_conceded_penalty REAL,
    -- aggregate distribution
    exp_points REAL NOT NULL,
    sd_points REAL, p10 REAL, p50 REAL, p90 REAL,
    p_haul_10 REAL,                       -- P(points >= 10)
    p_blank_2 REAL,                       -- P(points <= 2)
    -- adjustment layer
    base_exp_points REAL,                 -- before text/LLM adjustment
    adjustment REAL NOT NULL DEFAULT 0,
    adjustment_reason TEXT,
    model_run_id    INTEGER REFERENCES model_runs(id),
    UNIQUE(season_id, gameweek, player_id, COALESCE(fixture_id,0), generated_at)
);
CREATE INDEX ix_pred_gw ON predictions(season_id, gameweek, generated_at DESC);

CREATE TABLE model_runs (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT NOT NULL, finished_at TEXT,
    season_id     TEXT NOT NULL, gameweek INTEGER NOT NULL,
    models_json   TEXT NOT NULL,          -- {model_name: version_id}
    n_sims        INTEGER NOT NULL,
    notes         TEXT
);

CREATE TABLE squads (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    colour        TEXT,                   -- UI accent for the picker
    fpl_entry_id  INTEGER,                -- optional link to a real FPL team
    is_shadow     INTEGER NOT NULL DEFAULT 0,  -- shadow = hypothetical, not linked
    season_id     TEXT NOT NULL,
    settings_json TEXT NOT NULL,          -- see 11-config.md for the schema
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    archived      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE squad_leagues (
    squad_id   INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    league_id  INTEGER NOT NULL,
    league_name TEXT,
    league_type TEXT,                     -- classic|h2h
    target_rank INTEGER,
    rival_entry_ids_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (squad_id, league_id)
);

CREATE TABLE squad_states (
    id            INTEGER PRIMARY KEY,
    squad_id      INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    gameweek      INTEGER NOT NULL,
    source        TEXT NOT NULL,          -- fpl_sync|manual|planned|pushed
    bank          INTEGER NOT NULL,       -- tenths
    squad_value   INTEGER NOT NULL,
    free_transfers INTEGER NOT NULL,
    chips_used_json TEXT NOT NULL DEFAULT '[]',
    chip_active   TEXT,
    captured_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(squad_id, gameweek, source, captured_at)
);

CREATE TABLE squad_picks (
    squad_state_id INTEGER NOT NULL REFERENCES squad_states(id) ON DELETE CASCADE,
    player_id      INTEGER NOT NULL REFERENCES players(id),
    position       INTEGER NOT NULL,      -- 1..15, 12-15 = bench order
    is_captain     INTEGER NOT NULL DEFAULT 0,
    is_vice        INTEGER NOT NULL DEFAULT 0,
    purchase_price INTEGER, selling_price INTEGER,
    PRIMARY KEY (squad_state_id, player_id)
);

CREATE TABLE recommendations (
    id             INTEGER PRIMARY KEY,
    squad_id       INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    gameweek       INTEGER NOT NULL,
    generated_at   TEXT NOT NULL,
    variant        TEXT NOT NULL,         -- safe|balanced|aggressive|custom
    kind           TEXT NOT NULL,         -- initial_squad|transfer_plan|lineup|captain|chip
    horizon_gws    INTEGER NOT NULL,
    objective_value REAL,
    exp_points_gw  REAL, exp_points_horizon REAL, sd_points_gw REAL,
    hits_taken     INTEGER NOT NULL DEFAULT 0,
    chip_suggested TEXT,
    payload_json   TEXT NOT NULL,         -- full structured plan
    llm_rationale  TEXT,
    llm_critique   TEXT,
    model_run_id   INTEGER REFERENCES model_runs(id),
    accepted       INTEGER,               -- NULL unset, 1 accepted, 0 rejected
    accepted_at    TEXT
);

-- Provenance: everything the UI's evidence panel renders.
CREATE TABLE evidence_links (
    id              INTEGER PRIMARY KEY,
    subject_type    TEXT NOT NULL,        -- recommendation|prediction|adjustment
    subject_id      INTEGER NOT NULL,
    player_id       INTEGER REFERENCES players(id),
    evidence_type   TEXT NOT NULL,        -- claim|feature|odds|stat|model_component
    claim_id        INTEGER REFERENCES claims(id),
    raw_doc_id      INTEGER REFERENCES raw_documents(id),
    feature_name    TEXT,
    weight          REAL,                 -- contribution, e.g. SHAP value
    note            TEXT
);
CREATE INDEX ix_evidence_subject ON evidence_links(subject_type, subject_id);

-- Pundit / source accuracy scoreboard.
CREATE TABLE pundit_calls (
    id            INTEGER PRIMARY KEY,
    channel_id    TEXT REFERENCES channels(channel_id),
    source_id     TEXT REFERENCES sources(id),
    author_handle TEXT,
    claim_id      INTEGER NOT NULL REFERENCES claims(id),
    player_id     INTEGER NOT NULL REFERENCES players(id),
    gameweek      INTEGER NOT NULL,
    call_type     TEXT NOT NULL,          -- buy|sell|captain|avoid|start|bench
    made_at       TEXT NOT NULL,
    actual_points INTEGER,
    baseline_points REAL,                 -- position/price-matched baseline
    score         REAL,                   -- actual - baseline, signed by call direction
    resolved_at   TEXT
);

CREATE TABLE backtest_runs (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    seasons       TEXT NOT NULL,
    total_points  INTEGER, avg_gw_points REAL,
    vs_overall_avg REAL, vs_top10k_avg REAL,
    hits_taken INTEGER, transfers_made INTEGER,
    detail_json   TEXT NOT NULL
);

CREATE TABLE settings (
    scope       TEXT NOT NULL,            -- 'global' or 'squad:{id}'
    key         TEXT NOT NULL,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (scope, key)
);

CREATE TABLE llm_calls (
    id           INTEGER PRIMARY KEY,
    task         TEXT NOT NULL,
    model        TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    cost_usd     REAL,
    latency_ms   INTEGER,
    cached       INTEGER NOT NULL DEFAULT 0,
    ok           INTEGER NOT NULL,
    error_text   TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    response_json TEXT
);
CREATE UNIQUE INDEX ux_llm_cache ON llm_calls(task, model, prompt_hash) WHERE ok = 1;
```

## Retention

Nothing is deleted. Practical size estimate: raw JSON from FPL endpoints dominates (~50–150 MB/season compressed), transcripts ~30 MB/season, news ~200 MB/season, odds snapshots ~20 MB/season. Multi-year totals stay comfortably under 5 GB. A `VACUUM` job runs monthly; `data/raw/` is content-addressed so it self-deduplicates on disk too.
