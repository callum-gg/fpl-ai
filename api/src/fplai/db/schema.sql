-- FPL AI schema. See docs/03-database-schema.md.
-- L0 raw archive | L1 normalised entities | L2 derived features/predictions.

-- === L0: raw archive and dedup ==============================================

CREATE TABLE IF NOT EXISTS sources (
    id                  TEXT PRIMARY KEY,
    display_name        TEXT NOT NULL,
    category            TEXT NOT NULL,
    requires_key        INTEGER NOT NULL DEFAULT 0,
    enabled             INTEGER NOT NULL DEFAULT 1,
    base_url            TEXT,
    rate_limit_per_min  REAL,
    trust_weight        REAL NOT NULL DEFAULT 1.0,
    config_json         TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id              INTEGER PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES sources(id),
    job_name        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL,
    requests_made   INTEGER NOT NULL DEFAULT 0,
    docs_new        INTEGER NOT NULL DEFAULT 0,
    docs_duplicate  INTEGER NOT NULL DEFAULT 0,
    rows_upserted   INTEGER NOT NULL DEFAULT 0,
    error_text      TEXT,
    params_json     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_ingest_runs_source_time ON ingest_runs(source_id, started_at DESC);

CREATE TABLE IF NOT EXISTS raw_documents (
    id                INTEGER PRIMARY KEY,
    source_id         TEXT NOT NULL REFERENCES sources(id),
    doc_type          TEXT NOT NULL,
    external_id       TEXT,
    url               TEXT,
    content_hash      TEXT NOT NULL,
    simhash           INTEGER,
    payload_inline    TEXT,
    payload_path      TEXT,
    content_bytes     INTEGER NOT NULL,
    published_at      TEXT,
    fetched_at        TEXT NOT NULL DEFAULT (datetime('now')),
    first_seen_run    INTEGER REFERENCES ingest_runs(id),
    last_seen_at      TEXT NOT NULL DEFAULT (datetime('now')),
    seen_count        INTEGER NOT NULL DEFAULT 1,
    parsed_at         TEXT,
    parser_version    INTEGER,
    parse_status      TEXT NOT NULL DEFAULT 'pending',
    parse_error       TEXT,
    supersedes_id     INTEGER REFERENCES raw_documents(id),
    meta_json         TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_hash   ON raw_documents(source_id, content_hash);
CREATE INDEX IF NOT EXISTS ix_raw_external      ON raw_documents(source_id, doc_type, external_id);
CREATE INDEX IF NOT EXISTS ix_raw_type_time     ON raw_documents(doc_type, published_at DESC);
CREATE INDEX IF NOT EXISTS ix_raw_parse_pending ON raw_documents(parse_status, source_id) WHERE parse_status = 'pending';
CREATE INDEX IF NOT EXISTS ix_raw_simhash       ON raw_documents(simhash) WHERE simhash IS NOT NULL;

-- === L1: normalised football entities =======================================

CREATE TABLE IF NOT EXISTS seasons (
    id           TEXT PRIMARY KEY,
    start_date   TEXT,
    end_date     TEXT,
    is_current   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS teams (
    id            INTEGER PRIMARY KEY,
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

CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    first_name      TEXT, last_name TEXT, web_name TEXT,
    birth_date      TEXT,
    nationality     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_players_canon ON players(canonical_name);

CREATE TABLE IF NOT EXISTS player_external_ids (
    player_id     INTEGER NOT NULL REFERENCES players(id),
    system        TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    method        TEXT NOT NULL,
    verified      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (system, external_id)
);
CREATE INDEX IF NOT EXISTS ix_pei_player ON player_external_ids(player_id);

CREATE TABLE IF NOT EXISTS player_aliases (
    id          INTEGER PRIMARY KEY,
    player_id   INTEGER NOT NULL REFERENCES players(id),
    alias       TEXT NOT NULL,
    alias_norm  TEXT NOT NULL,
    origin      TEXT NOT NULL,
    UNIQUE(alias_norm, player_id)
);
CREATE INDEX IF NOT EXISTS ix_alias_norm ON player_aliases(alias_norm);

CREATE TABLE IF NOT EXISTS team_external_ids (
    team_key     TEXT NOT NULL,
    system       TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    PRIMARY KEY (system, external_id)
);

CREATE TABLE IF NOT EXISTS team_aliases (
    id         INTEGER PRIMARY KEY,
    team_key   TEXT NOT NULL,
    alias_norm TEXT NOT NULL,
    origin     TEXT NOT NULL DEFAULT 'seed',
    UNIQUE(alias_norm)
);

CREATE TABLE IF NOT EXISTS player_seasons (
    player_id      INTEGER NOT NULL REFERENCES players(id),
    season_id      TEXT NOT NULL REFERENCES seasons(id),
    team_id        INTEGER REFERENCES teams(id),
    fpl_element_id INTEGER,
    position       TEXT NOT NULL,
    start_price    INTEGER,
    PRIMARY KEY (player_id, season_id)
);
CREATE INDEX IF NOT EXISTS ix_ps_season_element ON player_seasons(season_id, fpl_element_id);

CREATE TABLE IF NOT EXISTS fixtures (
    id              INTEGER PRIMARY KEY,
    season_id       TEXT NOT NULL REFERENCES seasons(id),
    fpl_fixture_id  INTEGER,
    gameweek        INTEGER,
    kickoff_utc     TEXT,
    home_team_id    INTEGER NOT NULL REFERENCES teams(id),
    away_team_id    INTEGER NOT NULL REFERENCES teams(id),
    finished        INTEGER NOT NULL DEFAULT 0,
    home_score      INTEGER, away_score INTEGER,
    fdr_home        INTEGER, fdr_away INTEGER,
    competition     TEXT NOT NULL DEFAULT 'PL',
    UNIQUE(season_id, fpl_fixture_id)
);
CREATE INDEX IF NOT EXISTS ix_fixtures_gw   ON fixtures(season_id, gameweek);
CREATE INDEX IF NOT EXISTS ix_fixtures_kick ON fixtures(kickoff_utc);

CREATE TABLE IF NOT EXISTS gameweeks (
    season_id       TEXT NOT NULL REFERENCES seasons(id),
    gameweek        INTEGER NOT NULL,
    deadline_utc    TEXT NOT NULL,
    is_current      INTEGER NOT NULL DEFAULT 0,
    is_next         INTEGER NOT NULL DEFAULT 0,
    finished        INTEGER NOT NULL DEFAULT 0,
    data_checked    INTEGER NOT NULL DEFAULT 0,
    average_score   INTEGER,
    highest_score   INTEGER,
    chip_plays_json TEXT,
    PRIMARY KEY (season_id, gameweek)
);

CREATE TABLE IF NOT EXISTS player_fixture_stats (
    id                  INTEGER PRIMARY KEY,
    player_id           INTEGER NOT NULL REFERENCES players(id),
    fixture_id          INTEGER NOT NULL REFERENCES fixtures(id),
    team_id             INTEGER NOT NULL REFERENCES teams(id),
    was_home            INTEGER NOT NULL,
    minutes             INTEGER, goals_scored INTEGER, assists INTEGER,
    clean_sheets        INTEGER, goals_conceded INTEGER, own_goals INTEGER,
    penalties_saved     INTEGER, penalties_missed INTEGER,
    yellow_cards        INTEGER, red_cards INTEGER, saves INTEGER,
    bonus               INTEGER, bps INTEGER,
    defensive_contribution INTEGER,
    defcon_points       INTEGER,
    total_points        INTEGER,
    starts              INTEGER,
    xg REAL, xa REAL, npxg REAL, xgot REAL, shots INTEGER, shots_on_target INTEGER,
    key_passes INTEGER, big_chances INTEGER, big_chances_missed INTEGER,
    touches_in_box INTEGER, progressive_carries INTEGER, sca INTEGER, gca INTEGER,
    tackles INTEGER, interceptions INTEGER, blocks INTEGER, clearances INTEGER, recoveries INTEGER,
    sofascore_rating REAL, whoscored_rating REAL,
    source_ids_json     TEXT NOT NULL DEFAULT '[]',
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(player_id, fixture_id)
);
CREATE INDEX IF NOT EXISTS ix_pfs_fixture ON player_fixture_stats(fixture_id);
CREATE INDEX IF NOT EXISTS ix_pfs_player  ON player_fixture_stats(player_id);

CREATE TABLE IF NOT EXISTS player_prices (
    player_id   INTEGER NOT NULL REFERENCES players(id),
    season_id   TEXT NOT NULL REFERENCES seasons(id),
    observed_at TEXT NOT NULL,
    price       INTEGER NOT NULL,
    selected_by_percent REAL,
    transfers_in_event INTEGER, transfers_out_event INTEGER,
    net_transfers      INTEGER,
    PRIMARY KEY (player_id, season_id, observed_at)
);

CREATE TABLE IF NOT EXISTS availability (
    id           INTEGER PRIMARY KEY,
    player_id    INTEGER NOT NULL REFERENCES players(id),
    source_id    TEXT NOT NULL REFERENCES sources(id),
    raw_doc_id   INTEGER REFERENCES raw_documents(id),
    observed_at  TEXT NOT NULL,
    status       TEXT NOT NULL,
    chance_pct   INTEGER,
    issue        TEXT,
    expected_return TEXT,
    note         TEXT NOT NULL DEFAULT '',
    UNIQUE(player_id, source_id, observed_at, status, note)
);
CREATE INDEX IF NOT EXISTS ix_avail_player_time ON availability(player_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS lineups (
    id            INTEGER PRIMARY KEY,
    fixture_id    INTEGER NOT NULL REFERENCES fixtures(id),
    player_id     INTEGER NOT NULL REFERENCES players(id),
    source_id     TEXT NOT NULL REFERENCES sources(id),
    kind          TEXT NOT NULL,
    is_starting   INTEGER NOT NULL,
    formation     TEXT, position_slot TEXT,
    observed_at   TEXT NOT NULL,
    UNIQUE(fixture_id, player_id, source_id, kind, observed_at)
);
CREATE INDEX IF NOT EXISTS ix_lineups_fixture ON lineups(fixture_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id             INTEGER PRIMARY KEY,
    fixture_id     INTEGER NOT NULL REFERENCES fixtures(id),
    source_id      TEXT NOT NULL REFERENCES sources(id),
    bookmaker      TEXT,
    market         TEXT NOT NULL,
    selection      TEXT NOT NULL,
    player_id      INTEGER REFERENCES players(id),
    price_decimal  REAL NOT NULL,
    implied_prob   REAL NOT NULL,
    devig_prob     REAL,
    observed_at    TEXT NOT NULL,
    UNIQUE(fixture_id, source_id, bookmaker, market, selection, observed_at)
);
CREATE INDEX IF NOT EXISTS ix_odds_fixture_market ON odds_snapshots(fixture_id, market, observed_at DESC);

CREATE TABLE IF NOT EXISTS set_piece_roles (
    player_id   INTEGER NOT NULL REFERENCES players(id),
    season_id   TEXT NOT NULL REFERENCES seasons(id),
    role        TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    source_id   TEXT NOT NULL REFERENCES sources(id),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (player_id, season_id, role, source_id, observed_at)
);

CREATE TABLE IF NOT EXISTS weather_observations (
    fixture_id  INTEGER PRIMARY KEY REFERENCES fixtures(id),
    temp_c REAL, wind_kph REAL, precip_mm REAL, humidity REAL,
    is_forecast INTEGER NOT NULL DEFAULT 1,
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ownership_snapshots (
    player_id     INTEGER NOT NULL REFERENCES players(id),
    gameweek      INTEGER NOT NULL,
    season_id     TEXT NOT NULL,
    scope         TEXT NOT NULL,
    owned_pct     REAL, captained_pct REAL, effective_ownership REAL,
    observed_at   TEXT NOT NULL,
    PRIMARY KEY (season_id, gameweek, player_id, scope, observed_at)
);

CREATE TABLE IF NOT EXISTS referee_appointments (
    fixture_id  INTEGER PRIMARY KEY REFERENCES fixtures(id),
    referee     TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

-- === L1b: text ==============================================================

CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY,
    raw_doc_id    INTEGER NOT NULL UNIQUE REFERENCES raw_documents(id),
    title         TEXT, author TEXT, outlet TEXT,
    published_at  TEXT, url TEXT,
    body_text     TEXT NOT NULL,
    word_count    INTEGER,
    near_dupe_group INTEGER
);
CREATE INDEX IF NOT EXISTS ix_articles_time ON articles(published_at DESC);

CREATE TABLE IF NOT EXISTS videos (
    id             INTEGER PRIMARY KEY,
    raw_doc_id     INTEGER NOT NULL UNIQUE REFERENCES raw_documents(id),
    youtube_id     TEXT NOT NULL UNIQUE,
    channel_id     TEXT NOT NULL,
    channel_title  TEXT,
    title          TEXT, description TEXT,
    published_at   TEXT, duration_s INTEGER,
    view_count     INTEGER, like_count INTEGER,
    gameweek_hint  INTEGER,
    transcript_source TEXT,
    transcript_text TEXT,
    transcript_json TEXT,
    discovered_via TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_videos_time ON videos(published_at DESC);

CREATE TABLE IF NOT EXISTS channels (
    channel_id       TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    tracked          INTEGER NOT NULL DEFAULT 1,
    subscriber_count INTEGER,
    accuracy_score   REAL,
    accuracy_n       INTEGER NOT NULL DEFAULT 0,
    trust_weight     REAL NOT NULL DEFAULT 1.0,
    added_by         TEXT NOT NULL DEFAULT 'seed'
);

CREATE TABLE IF NOT EXISTS social_posts (
    id             INTEGER PRIMARY KEY,
    raw_doc_id     INTEGER NOT NULL UNIQUE REFERENCES raw_documents(id),
    platform       TEXT NOT NULL,
    external_id    TEXT NOT NULL,
    author_handle  TEXT, author_display TEXT,
    author_is_verified INTEGER,
    body_text      TEXT NOT NULL,
    posted_at      TEXT,
    likes INTEGER, reposts INTEGER, replies INTEGER, score INTEGER,
    parent_external_id TEXT,
    retrieval_method TEXT,
    UNIQUE(platform, external_id)
);
CREATE INDEX IF NOT EXISTS ix_social_time ON social_posts(posted_at DESC);

CREATE TABLE IF NOT EXISTS doc_chunks (
    id           INTEGER PRIMARY KEY,
    raw_doc_id   INTEGER NOT NULL REFERENCES raw_documents(id),
    ordinal      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    start_s      REAL,
    end_s        REAL,
    token_count  INTEGER,
    embedded     INTEGER NOT NULL DEFAULT 0,
    extracted    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(raw_doc_id, ordinal)
);
CREATE INDEX IF NOT EXISTS ix_chunks_pending ON doc_chunks(extracted) WHERE extracted = 0;

-- Embedding store. A sqlite-vec vec0 virtual table is created at runtime when the
-- extension is available (db/engine.py); otherwise cosine search runs over this
-- table in numpy. Same rows either way, so the fallback is not a second schema.
CREATE TABLE IF NOT EXISTS chunk_vec_fallback (
    chunk_id  INTEGER PRIMARY KEY REFERENCES doc_chunks(id),
    embedding BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(text, content='doc_chunks', content_rowid='id');

CREATE TABLE IF NOT EXISTS claims (
    id             INTEGER PRIMARY KEY,
    raw_doc_id     INTEGER NOT NULL REFERENCES raw_documents(id),
    chunk_id       INTEGER REFERENCES doc_chunks(id),
    player_id      INTEGER REFERENCES players(id),
    team_id        INTEGER REFERENCES teams(id),
    surface_form   TEXT,
    claim_type     TEXT NOT NULL,
    stance         TEXT,
    sentiment      REAL,
    confidence     REAL,
    is_reported    INTEGER NOT NULL DEFAULT 0,
    horizon_gw     INTEGER,
    text_span      TEXT NOT NULL,
    start_s        REAL,
    extracted_at   TEXT NOT NULL,
    extractor_model TEXT NOT NULL,
    semantic_group INTEGER,
    UNIQUE(raw_doc_id, chunk_id, player_id, claim_type, text_span)
);
CREATE INDEX IF NOT EXISTS ix_claims_player_time ON claims(player_id, extracted_at DESC);
CREATE INDEX IF NOT EXISTS ix_claims_type ON claims(claim_type, horizon_gw);

-- === L2: features, predictions, squads, recommendations =====================

CREATE TABLE IF NOT EXISTS feature_values (
    season_id   TEXT NOT NULL,
    gameweek    INTEGER NOT NULL,
    player_id   INTEGER NOT NULL REFERENCES players(id),
    fixture_key INTEGER NOT NULL DEFAULT 0,
    name        TEXT NOT NULL,
    value       REAL,
    computed_at TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    PRIMARY KEY (season_id, gameweek, player_id, fixture_key, name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS model_versions (
    id            INTEGER PRIMARY KEY,
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    trained_at    TEXT NOT NULL,
    train_rows    INTEGER,
    train_seasons TEXT,
    metrics_json  TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    artefact_path TEXT NOT NULL,
    feature_version INTEGER NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    UNIQUE(model_name, version)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT NOT NULL, finished_at TEXT,
    season_id     TEXT NOT NULL, gameweek INTEGER NOT NULL,
    models_json   TEXT NOT NULL,
    n_sims        INTEGER NOT NULL,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES players(id),
    season_id       TEXT NOT NULL,
    gameweek        INTEGER NOT NULL,
    fixture_id      INTEGER REFERENCES fixtures(id),
    fixture_key     INTEGER NOT NULL DEFAULT 0,
    generated_at    TEXT NOT NULL,
    p_start REAL, p_appear REAL, exp_minutes REAL,
    exp_goals REAL, exp_assists REAL, p_clean_sheet REAL,
    exp_saves REAL, exp_defcon_points REAL, exp_bonus REAL,
    exp_cards_penalty REAL, exp_conceded_penalty REAL,
    exp_points REAL NOT NULL,
    sd_points REAL, p10 REAL, p50 REAL, p90 REAL,
    p_haul_10 REAL,
    p_blank_2 REAL,
    base_exp_points REAL,
    adjustment REAL NOT NULL DEFAULT 0,
    adjustment_reason TEXT,
    model_run_id    INTEGER REFERENCES model_runs(id),
    UNIQUE(season_id, gameweek, player_id, fixture_key, generated_at)
);
CREATE INDEX IF NOT EXISTS ix_pred_gw ON predictions(season_id, gameweek, generated_at DESC);

CREATE TABLE IF NOT EXISTS squads (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    colour        TEXT,
    fpl_entry_id  INTEGER,
    is_shadow     INTEGER NOT NULL DEFAULT 0,
    season_id     TEXT NOT NULL,
    settings_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    archived      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS squad_leagues (
    squad_id    INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    league_id   INTEGER NOT NULL,
    league_name TEXT,
    league_type TEXT,
    target_rank INTEGER,
    rival_entry_ids_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (squad_id, league_id)
);

CREATE TABLE IF NOT EXISTS squad_states (
    id             INTEGER PRIMARY KEY,
    squad_id       INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    gameweek       INTEGER NOT NULL,
    source         TEXT NOT NULL,
    bank           INTEGER NOT NULL,
    squad_value    INTEGER NOT NULL,
    free_transfers INTEGER NOT NULL,
    chips_used_json TEXT NOT NULL DEFAULT '[]',
    chip_active    TEXT,
    captured_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(squad_id, gameweek, source, captured_at)
);
CREATE INDEX IF NOT EXISTS ix_states_squad ON squad_states(squad_id, gameweek DESC, captured_at DESC);

CREATE TABLE IF NOT EXISTS squad_picks (
    squad_state_id INTEGER NOT NULL REFERENCES squad_states(id) ON DELETE CASCADE,
    player_id      INTEGER NOT NULL REFERENCES players(id),
    position       INTEGER NOT NULL,
    is_captain     INTEGER NOT NULL DEFAULT 0,
    is_vice        INTEGER NOT NULL DEFAULT 0,
    purchase_price INTEGER, selling_price INTEGER,
    PRIMARY KEY (squad_state_id, player_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id             INTEGER PRIMARY KEY,
    squad_id       INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    gameweek       INTEGER NOT NULL,
    generated_at   TEXT NOT NULL,
    variant        TEXT NOT NULL,
    kind           TEXT NOT NULL,
    horizon_gws    INTEGER NOT NULL,
    objective_value REAL,
    exp_points_gw  REAL, exp_points_horizon REAL, sd_points_gw REAL,
    hits_taken     INTEGER NOT NULL DEFAULT 0,
    chip_suggested TEXT,
    payload_json   TEXT NOT NULL,
    llm_rationale  TEXT,
    llm_critique   TEXT,
    model_run_id   INTEGER REFERENCES model_runs(id),
    accepted       INTEGER,
    accepted_at    TEXT,
    reject_reason  TEXT
);
CREATE INDEX IF NOT EXISTS ix_recs_squad_gw ON recommendations(squad_id, gameweek, generated_at DESC);

CREATE TABLE IF NOT EXISTS evidence_links (
    id              INTEGER PRIMARY KEY,
    subject_type    TEXT NOT NULL,
    subject_id      INTEGER NOT NULL,
    player_id       INTEGER REFERENCES players(id),
    evidence_type   TEXT NOT NULL,
    claim_id        INTEGER REFERENCES claims(id),
    raw_doc_id      INTEGER REFERENCES raw_documents(id),
    feature_name    TEXT,
    weight          REAL,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS ix_evidence_subject ON evidence_links(subject_type, subject_id);

CREATE TABLE IF NOT EXISTS pundit_calls (
    id            INTEGER PRIMARY KEY,
    channel_id    TEXT REFERENCES channels(channel_id),
    source_id     TEXT REFERENCES sources(id),
    author_handle TEXT,
    claim_id      INTEGER NOT NULL REFERENCES claims(id),
    player_id     INTEGER NOT NULL REFERENCES players(id),
    gameweek      INTEGER NOT NULL,
    call_type     TEXT NOT NULL,
    made_at       TEXT NOT NULL,
    actual_points INTEGER,
    baseline_points REAL,
    score         REAL,
    resolved_at   TEXT,
    UNIQUE(claim_id, call_type)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    seasons       TEXT NOT NULL,
    total_points  INTEGER, avg_gw_points REAL,
    vs_overall_avg REAL, vs_top10k_avg REAL,
    hits_taken INTEGER, transfers_made INTEGER,
    detail_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    scope       TEXT NOT NULL,
    key         TEXT NOT NULL,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (scope, key)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY,
    task          TEXT NOT NULL,
    model         TEXT NOT NULL,
    prompt_hash   TEXT NOT NULL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    cost_usd      REAL,
    latency_ms    INTEGER,
    cached        INTEGER NOT NULL DEFAULT 0,
    ok            INTEGER NOT NULL,
    error_text    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    response_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_llm_cache ON llm_calls(task, model, prompt_hash) WHERE ok = 1;

CREATE TABLE IF NOT EXISTS push_records (
    id            INTEGER PRIMARY KEY,
    squad_id      INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    gameweek      INTEGER NOT NULL,
    pushed_at     TEXT NOT NULL,
    dry_run       INTEGER NOT NULL DEFAULT 1,
    ok            INTEGER NOT NULL,
    request_json  TEXT NOT NULL,
    response_text TEXT,
    pre_state_id  INTEGER REFERENCES squad_states(id)
);

CREATE TABLE IF NOT EXISTS entity_review_queue (
    id            INTEGER PRIMARY KEY,
    surface_form  TEXT NOT NULL,
    club_context  TEXT NOT NULL DEFAULT '',
    system        TEXT NOT NULL DEFAULT 'text',
    occurrences   INTEGER NOT NULL DEFAULT 1,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    resolved_player_id INTEGER REFERENCES players(id),
    resolved_at   TEXT,
    first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(surface_form, club_context, system)
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id            INTEGER PRIMARY KEY,
    squad_id      INTEGER REFERENCES squads(id) ON DELETE CASCADE,
    title         TEXT,
    messages_json TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS job_runs (
    id          INTEGER PRIMARY KEY,
    job_name    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS ix_job_runs ON job_runs(job_name, started_at DESC);
