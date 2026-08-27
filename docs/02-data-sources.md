# 02 — Data Sources

Every source has: an ID, an auth requirement, a cadence, a natural key for dedup, a documented failure mode, and a `enabled` flag in global settings. If a key is absent the connector self-disables and the UI shows it greyed out with "no key configured".

Legend: **CORE** = app is useless without it · **HIGH** = big predictive lift · **MED** · **LOW** = marginal, cheap to have · **FLAKY** = expect breakage.

---

## Tier 1 — Core FPL

### `fpl_official` — CORE, free, no key
Undocumented but stable public JSON. No auth for read.

| Endpoint | Contents | Cadence |
|---|---|---|
| `/api/bootstrap-static/` | all players, teams, positions, prices, ownership, form, chip metadata, gameweek list | hourly; every 10 min in the 6h before a deadline |
| `/api/fixtures/` | all fixtures, kickoff times, FDR, finished flags | hourly |
| `/api/fixtures/?event={gw}` | single GW | as above |
| `/api/element-summary/{player_id}/` | per-player history this season + past seasons summary | daily for all, hourly for owned/watchlist |
| `/api/event/{gw}/live/` | live per-player stats incl. BPS | every 2 min during matches (phase 2), plus **a mandatory re-fetch after the 09:00 next-day lockdown** to capture Opta review corrections |
| `/api/entry/{entry_id}/` | manager metadata, leagues | daily |
| `/api/entry/{entry_id}/history/` | past GW scores, chips used, transfers | daily |
| `/api/entry/{entry_id}/event/{gw}/picks/` | your picks for a GW | daily |
| `/api/entry/{entry_id}/transfers/` | your transfer history | daily |
| `/api/leagues-classic/{league_id}/standings/?page_standings=N` | mini-league standings | daily — **this is how squad settings get mini-league data**, answering your Q9 |
| `/api/dream-team/{gw}/`, `/api/element-types/`, `/api/team/set-piece-notes/` | extras | weekly |

Rate limiting: no published limit, but be polite — 1 req/sec, exponential backoff on 429/503. Add a browser-like `User-Agent`.

Dedup key: `(endpoint, entity_id, gameweek, payload_hash)`.

### `fpl_write` — CORE for "push changes from the app"
Not a public API. The flow the site itself uses:

1. Authenticate. **`users.premierleague.com` no longer resolves** — FPL moved to PingOne SSO, and the hosted login is a DaVinci JS widget behind Cloudflare bot management, so there is no headless password login to reimplement. What does work headlessly is the refresh-token grant against FPL's public SPA client: `POST https://account.premierleague.com/as/token` with `grant_type=refresh_token`, `client_id=bfcbaf69-aade-4c1b-8f00-c1cb8a193030` and your `refresh_token` (no client secret — it is a public PKCE client). Authenticate every later call with `X-API-Authorization: Bearer {access_token}`, *not* a cookie jar. Get the initial refresh token by logging in with a browser and reading localStorage key `oidc.user:https://account.premierleague.com/as:bfcbaf69-aade-4c1b-8f00-c1cb8a193030`; set it as `FPL_REFRESH_TOKEN`. PingOne rotates it on every use, so the replacement is persisted to `settings(global, fpl_refresh_token)`.
2. `GET /api/me/` to confirm the token and get your entry id.
3. `GET /api/my-team/{entry_id}/` — returns picks, chips, transfer state (only works authenticated). This is also the **only** way to read a squad before its gameweek deadline: `/entry/{id}/event/{gw}/picks/` 404s until the deadline passes, so `sync_squad` falls back to `my-team` when a token is configured. It is the better source regardless — it states `purchase_price` and `selling_price` outright instead of making you reconstruct them from transfer history.
4. `POST /api/transfers/` with `{entry, event, transfers: [{element_in, element_out, purchase_price, selling_price}], chip, confirmed: true}`.
5. `POST /api/my-team/{entry_id}/` with `{chip, picks: [{element, position, is_captain, is_vice_captain}]}` to set the lineup/captain.

**Honest warnings.** This is fragile (the login flow has changed repeatedly — it has now changed out from under this doc once already), it stores a long-lived FPL credential in `.env`, and it is not something FPL sanctions. Gate it behind `FPL_WRITE_ENABLED=false` by default, require an explicit typed confirmation in the UI ("PUSH GW12"), always dry-run first showing the exact diff, snapshot the pre-push squad state to the DB so you can see what changed, and never let a scheduled job push — human-initiated only.

### `vaastav_history` — CORE for training, free
`github.com/vaastav/Fantasy-Premier-League` — season-by-season `players_raw.csv`, `gws/merged_gw.csv`, `understat/` merges, from 2016/17. This is the backbone of your training set and the only clean way to get pre-current-season GW-level FPL data. Pull via `raw.githubusercontent.com`; refresh weekly for the current season, once for history.

You asked for one prior season of *text* backfill, but take **all available seasons for numeric training data** — it costs nothing and the minutes/goals models want every row they can get. Weight recent seasons higher (see `06`).

### `livefpl` — HIGH, free
`livefpl.net` for effective ownership, top-10k ownership, and live rank estimation. Scrape the ownership tables. EO is what makes the risk/differential setting meaningful. FLAKY: HTML structure changes; write the parser defensively and alert on zero-row parses.

---

## Tier 2 — Underlying performance data

### `understat` — HIGH, free scrape
xG, xA, shots, key passes, npxG, per player per match since 2014. Data is embedded in the page as JSON inside `JSON.parse('...')` blocks. Use `understat` python lib or parse directly. Cadence: after each match round. Natural key `(understat_match_id, understat_player_id)`.

### `fbref` — HIGH, free scrape, FLAKY
Via `soccerdata` (`sd.FBref`). Gives progressive carries, touches in box, SCA/GCA, tackles/interceptions/blocks/clearances/recoveries — **the raw components of DefCon**, which is the single most valuable non-FPL dataset now that DefCon points exist. Aggressive rate limits (3s+ between requests, they will 429 and then 403 you). Cache hard, fetch weekly, never in a loop.

### `sofascore` — MED, free-ish scrape, FLAKY
Player ratings (your "rating" requirement), heatmaps, positional data. Endpoints under `api.sofascore.com/api/v1/...` are JSON but Cloudflare-protected; needs realistic headers and sometimes `curl_cffi` with browser TLS impersonation. Treat ratings as a soft feature, not a dependency.

### `whoscored` — MED, FLAKY
Harder than SofaScore (Incapsula). Only attempt via `soccerdata`'s WhoScored scraper with Selenium; mark as opt-in, default off. If it breaks, shrug — SofaScore covers the same feature.

### `transfermarkt` — MED, free scrape
Market value, contract expiry, transfer rumours, and a decent injury history table. Market value is a surprisingly good prior for a newly promoted club's player quality where you have no PL history — genuinely useful for Coventry, Ipswich and Hull players this season.

### `football_data_org` — LOW, free key
Fixtures, results, standings, competition calendars. Mostly redundant with FPL, but useful as a cross-check and as the source for European competition dates.

### `api_football` / `sportmonks` — HIGH if enabled, paid, key-gated
Both give **predicted/confirmed lineups**, which is the single biggest edge available pre-deadline: confirmed lineups land ~1h before kickoff, but *predicted* lineups and press-conference-derived availability land days earlier. API-Football also gives injuries, sidelined lists, and player statistics per fixture. Sportmonks has better historical depth and a proper "lineups + probable lineups" endpoint.

Implement both behind one `LineupProvider` interface so either key works. If neither is present, fall back to LLM extraction of lineup hints from news/YouTube (see `08`).

---

## Tier 3 — Odds (the highest-signal cheap data there is)

### `odds_api` — HIGH, paid key
`the-odds-api.com`, `/v4/sports/soccer_epl/odds/?regions=uk&markets=h2h,totals,spreads`. Match odds → implied win/draw/loss → team goal expectations, better than any FDR. Free tier is 500 req/month which is genuinely enough at one poll per day plus deadline-day refresh.

### `betfair` — HIGH, free-ish
Betfair Exchange API needs an account + application key + certificate login. Gives exchange prices (sharper than bookmaker odds) and, on some markets, **anytime goalscorer** and **team clean sheet** — which map *directly* onto FPL scoring. If you get these markets, they should be a first-class feature: implied anytime-scorer probability is close to a free, market-calibrated goal model.

Devig with the multiplicative or Shin method before use; store both raw and devigged.

### Odds-derived features
`p_win`, `p_clean_sheet`, `expected_team_goals` (from Poisson fit to 1X2 + over/under 2.5), `p_anytime_scorer` per player, and the **movement** of these over the days before kickoff (steam = information).

---

## Tier 4 — Team news, injuries, availability

### `premier_injuries` — HIGH, scrape
`premierinjuries.com` injury table: player, injury type, expected return, status. Best structured injury source in the UK, updated frequently.

### `physioroom` — MED, scrape
Second opinion; useful for disagreement detection (two sources disagreeing on a return date is itself a signal to surface).

### FPL's own `chance_of_playing_next_round`
From `bootstrap-static`. Authoritative but lagging — often updated only after the press conference. **A key derived feature: the gap between what news/YouTube says and what FPL's flag says.** That gap is where edges live before a deadline.

### `rss_news` — HIGH, free
Free RSS/Atom only, per your call. Starter feed list (all editable in global settings):

- BBC Sport football + per-club feeds (`feeds.bbci.co.uk/sport/football/teams/{club}/rss.xml`)
- Sky Sports Premier League (`feeds.skynews.com`, `skysports.com/rss/12040`)
- Guardian football (`theguardian.com/football/rss`, plus `/football/{club}/rss`)
- talkSPORT, Football365, 90min, Metro Sport, Mirror Football, Express Sport
- Every official club site feed (20 of them) — these carry the actual team news
- `fantasyfootballscout.co.uk/feed` (public posts only), `fplstatistics.co.uk` blog feed
- Google News RSS as a catch-all: `news.google.com/rss/search?q=%22{player+name}%22+injury&hl=en-GB` — one query per watchlist player, run daily. Cheap, no key, surprisingly complete.

Full-text extraction with `trafilatura` (better than readability for football sites), then dedup — syndicated wire copy appears on six sites at once, which is what the near-dupe detector in `03` is for.

### `setpieces` — MED
The community-maintained set-piece takers sheet (published as a Google Sheet each season, exportable as CSV via `/export?format=csv`). Plus FPL's own `/api/team/set-piece-notes/`. Penalty-taker status is worth ~0.5 pts/game to a forward; corner/free-kick duty drives assists.

### `euro_fixtures` — MED
UEFA Champions League / Europa / Conference dates + EFL Cup + FA Cup rounds, from football-data.org or Wikipedia scrape. Feeds the **congestion and rotation-risk features**, which is where your "time since last match" requirement lives.

### `weather` — LOW (explicitly low weight, as you said)
Open-Meteo (free, no key) by stadium lat/long at kickoff time: wind speed, precipitation, temperature. Effect on goals is small but real (high wind suppresses xG conversion slightly). Store stadium coordinates as a static seed table.

---

## Tier 5 — Community, social, video

### `youtube` — HIGH, free key
YouTube Data API v3 (10,000 quota units/day, plenty). Two modes:

1. **Tracked channels** — `search.list` or `playlistItems.list` on the uploads playlist of each configured channel, daily.
2. **Discovery** — `search.list` with rotating queries: `FPL GW{n} team selection`, `FPL {n} transfer tips`, `fantasy premier league {player} injury`, `FPL captain gameweek {n}`. Surface newly-found channels in the UI as "suggested channels" for you to promote into the tracked list, and auto-ingest any video above a view/recency threshold.

Starter tracked list (all editable in Settings → Global → YouTube):
`Let's Talk FPL`, `FPL Mate`, `FPL Harry`, `FPL Raptor`, `Above Average FPL`, `Focal Fantasy`, `FPL Family`, `Planet FPL`, `FPL BlackBox (FPL Wire)`, `Fantasy Football Scout`, `FPL Andy`, `Elite FPL`, `FPL Tips`, `Fantasy Football Hub`, `FPL Kiwi`, `FPL Sonaldo`, `Let's Talk Transfers`, `The FPL Show (Sky)`.

Transcripts: `youtube-transcript-api` first (manual captions preferred over auto-generated, `en-GB` then `en`). Paid fallback behind `SUPADATA_API_KEY` or `APIFY_TOKEN` when the free lib is blocked or the video has no listed transcript. Whisper on downloaded audio is **deferred to future enhancements** per your answer — but leave `transcript_source` as an enum column so it slots in later without a migration.

Store: full transcript with timestamps, plus chunked+embedded segments. LLM then extracts structured claims (see `08`).

### `reddit` — MED, free key
PRAW on `r/FantasyPL`. The daily/GW megathreads, "Team Reveal" threads, and injury news posts. Sort by new every 30 min in the 48h before a deadline. Also `r/soccer` filtered for team news flairs. Score comments by upvotes as a crude credibility weight.

### `bluesky` — MED, free
AT Protocol public API, no key needed for reads: `app.bsky.feed.searchPosts`. Follow the football journalists who've migrated (Ornstein-adjacent accounts, club beat reporters) plus FPL community accounts. Cheap and unrestricted — worth doing well.

### `twitter_scrape` — MED, FLAKY, ToS-violating (you've okayed this)
X is still where team news breaks first, so it's worth the pain. Layered fallback, each attempted in order until one works:

1. **Syndication endpoint** — `cdn.syndication.twimg.com/timeline/profile?screen_name={handle}` (the embedded-timeline JSON). No auth, no login, returns recent tweets. Historically the most durable unauthenticated route; expect it to break periodically.
2. **Nitter instances** — rotate over a configurable list of public instances, RSS per handle (`/{handle}/rss`). Most instances are dead; keep a health-checked pool and drop failing ones automatically.
3. **`twscrape`** — needs burner accounts with cookies in `.env`. Best coverage, highest ban risk. Off by default.
4. **Playwright headless with a logged-in burner** — last resort, heaviest.

Handle list in global settings: club beat reporters, `@OptaJoe`, `@FPLStatistics`, `@FPL_Salah`-tier community accounts, and the official club accounts (which post confirmed lineups).

Every layer writes into the same `social_posts` table with a `retrieval_method` column so you can see what's actually working. If all four fail, the app logs it and carries on — nothing downstream may hard-depend on X.

---

## Cost summary

| | Cost | Verdict |
|---|---|---|
| FPL API, Understat, FBref, Reddit, Bluesky, RSS, Open-Meteo, vaastav, YouTube Data API | £0 | Take all of it |
| The Odds API | £0 (500/mo free) → ~£25/mo | **Best value paid source.** Start free tier |
| API-Football | ~£15–30/mo | Worth it for predicted lineups alone |
| Sportmonks | ~£30+/mo | Only if API-Football's lineup quality disappoints |
| Betfair | £0 + a £/one-off account activation | Highest quality odds if you'll do the cert setup |

Everything paid is key-gated and off until you add a key, so you can decide later exactly as you asked.
