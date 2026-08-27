# 04 — Ingestion

## The connector contract

Every source implements the same tiny interface. Claude Code should build `base.py` first and then each connector is a small, testable file.

```python
class Connector(ABC):
    id: str
    category: str
    requires_keys: list[str] = []
    default_cadence: str                 # cron expression

    def is_available(self, settings) -> bool: ...
    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]: ...
    def parse(self, doc: RawDoc) -> ParsedBatch: ...
    parser_version: int = 1
```

- `fetch` yields raw payloads only. It never touches normalised tables.
- The framework hashes, dedups, and persists each `RawDoc`, then calls `parse` **only for docs that are new or whose `parser_version` is stale**.
- `parse` returns typed rows (`PlayerStatRow`, `AvailabilityRow`, `OddsRow`…) that the framework upserts. A parser is a pure function of a raw doc — which is what makes "bump `parser_version`, reprocess everything from the archive, zero refetches" work. That capability is worth more than it sounds; you will change parsers often.

Cross-cutting behaviour lives in the framework, not the connectors: retry with exponential backoff + jitter, per-source token-bucket rate limiting, a shared `httpx.AsyncClient` with realistic headers, ETag/Last-Modified caching, circuit breaking after N consecutive failures (source auto-disabled, Discord alert), and full `ingest_runs` accounting.

## Entity resolution

The hardest unglamorous problem in the whole build. "Gabriel", "Gabriel Magalhães", "Gabriel Dos Santos", "Gabby Jesus" and "Gabriel Martinelli" all appear in transcripts. Get this wrong and every text feature is noise.

Resolution ladder, first match wins, recorded in `player_external_ids.method`:

1. **Exact external id** — if a source gives an id we've mapped before, done.
2. **Deterministic join on backfill** — for structured sources, join on `(normalised_name, team, birth_date)`. The vaastav repo already carries Understat mappings; seed from those.
3. **Fuzzy** — `rapidfuzz.token_set_ratio` over accent-stripped names within the same club and position. Accept ≥ 92 automatically; 80–92 goes to a review queue.
4. **LLM disambiguation** — for text mentions, resolve *in context*: the prompt receives the surrounding sentences plus the candidate list restricted to players mentioned near that club in the same document. Returns `player_id` or `null`. Cached by `(surface_form, club_context)` so each nickname is resolved once, then promoted into `player_aliases`.
5. **Manual** — a Settings → Entity Review screen listing unresolved surface forms by frequency, so ten minutes of clicking fixes the long tail permanently.

Never resolve a bare surname when two players in the league share it and no club context is present — emit `player_id = NULL` and keep the claim as team-level. A wrong resolution is far worse than a dropped one.

## Scheduler jobs

APScheduler, all times UK. Cadences adapt to the deadline: `deadline_proximity()` returns `far|near|imminent` and jobs consult it.

| Job | Cadence | Notes |
|---|---|---|
| `fpl_bootstrap` | hourly; 10 min when imminent | prices, ownership, availability flags |
| `fpl_fixtures` | hourly | detects postponements, blanks, doubles |
| `fpl_element_summaries` | daily; hourly for watchlist | |
| `fpl_entry_sync` | 6-hourly | your squads, leagues, transfers |
| `fpl_post_lockdown_reconcile` | 09:30 the day after each GW's last match | re-pull `event/{gw}/live`, overwrite stats, **recompute anything derived** — the new lockdown rule means full-time data is provisional |
| `odds_poll` | 4-hourly; 30 min when imminent | store every snapshot, movement is signal |
| `injury_scrape` | 3-hourly | premierinjuries + physioroom |
| `lineups_poll` | hourly; 5 min from 90 min before kickoff | predicted → confirmed |
| `news_rss` | 20 min | ~60 feeds, conditional GET |
| `youtube_tracked` | 3-hourly | uploads playlists |
| `youtube_discovery` | daily | rotating search queries |
| `transcripts` | 15 min | queue drain for videos lacking transcripts |
| `social_x` | 30 min; 10 min when imminent | layered fallback |
| `bluesky` / `reddit` | 30 min | |
| `understat` / `fbref` | after each round, weekly | heavy, rate-limited |
| `sofascore_ratings` | after each round | |
| `transfermarkt` | weekly | |
| `weather` | daily; 6h before kickoff refresh | |
| `extract_claims` | 10 min | LLM queue drain over new chunks |
| `build_features` | after any material ingest, debounced 15 min | |
| `predict` | 3× daily + 90 min before deadline | |
| `optimise_all_squads` | after each predict | |
| `train_models` | after `data_checked` for a GW | |
| `resolve_pundit_calls` | after each GW finalises | scores the scoreboard |
| `discord_digest` | daily 08:00 + deadline-2h + on price-change alerts | |
| `vacuum_analyze` | monthly | |

## Backfill

`python -m fplai.cli backfill` runs an ordered plan:

1. Seasons, teams, gameweeks, fixtures from the FPL API + vaastav (all seasons available).
2. `vaastav` GW-level player stats for every season → `player_fixture_stats`.
3. Understat match+player data for every season (rate-limited, ~2h).
4. FBref advanced stats for as many seasons as it'll give without banning you (start 2 seasons, extend overnight).
5. Entity resolution pass + review queue population.
6. YouTube: current + one prior season, tracked channels only, transcripts, chunk, embed, extract claims.
7. News: RSS only reaches back a few days — accept that. Historic text is thin and that's fine; text features carry recency weight anyway.
8. Odds: no free historic source. **Accept that odds features only exist from install date forward**, and make the model tolerant of the whole odds feature block being absent for historic rows (see `06` on missing-block handling). This is a real limitation, not something to paper over.

Expect the first full backfill to take 3–6 hours wall-clock, mostly sleeping on rate limits.

## Rate-limit and politeness policy

Per-source token buckets in settings, defaults: FPL 60/min, Understat 20/min, FBref 15/min with 3s minimum gap, SofaScore 30/min, YouTube by quota units, RSS unlimited (conditional GET), X-scrape 10/min with jitter. All scrapers send a real `User-Agent` and honour `Retry-After`. A global `SCRAPE_ENABLED=false` kill switch stops everything ToS-adjacent in one flip if you ever host this somewhere you care about.
