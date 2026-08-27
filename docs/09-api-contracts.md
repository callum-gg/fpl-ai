# 09 — API Contracts

FastAPI, all routes under `/api`. OpenAPI schema is generated and `openapi-typescript` produces `web/src/lib/api.d.ts` — so the frontend types are never hand-written or out of date.

Conventions: prices in tenths of a million as integers (`105` = £10.5m), timestamps ISO-8601 UTC, all list endpoints paginated `?limit=&offset=`, all errors as `{"error": {"code": "...", "message": "...", "detail": {...}}}`.

## Squads

```
GET    /api/squads                          → [SquadSummary]
POST   /api/squads                          {name, colour, fpl_entry_id?, clone_from?} → Squad
GET    /api/squads/{id}                     → Squad (incl. settings, current state)
PATCH  /api/squads/{id}                     {name?, colour?, settings?} → Squad
DELETE /api/squads/{id}                     (soft archive)
POST   /api/squads/{id}/sync                pull latest from FPL API → SquadState
GET    /api/squads/{id}/state?gameweek=     → SquadState + picks
PUT    /api/squads/{id}/state               manual squad entry → SquadState
GET    /api/squads/compare?ids=1,2,3&gw=14  → ComparisonView
```

### Squad state sources

A `squad_state` row carries a `source`, and only two of them mean "the squad you actually own":

| source | written by | is it your squad? |
|---|---|---|
| `fpl_sync` | `POST /{id}/sync` | yes |
| `manual` | `PUT /{id}/state`, draft commit, clone | yes |
| `planned` | accepting a recommendation | no — a plan |
| `draft` | the working copy below | no — a scratch pad |

`current_state()` reads only the first two. This matters: every state write is just another
row, so without the source filter the newest write wins and an accepted recommendation
silently becomes "your team".

### Working copy ("draft")

A scratch copy of the squad you can rearrange freely — swap players, pull in a
recommendation's picks, re-run safe/balanced/aggressive against it — with nothing counting
until you commit. One draft per squad; seeding again replaces it.

```
GET    /api/squads/{id}/draft               → Draft (404 if none started)
PUT    /api/squads/{id}/draft               {from_recommendation?, gameweek?} → Draft
PATCH  /api/squads/{id}/draft               {add?: [pid], drop?: [pid], captain?, vice?, bank?} → Draft
DELETE /api/squads/{id}/draft               discard
POST   /api/squads/{id}/draft/commit        promote to a `manual` state → SquadState
```

`Draft` is a `SquadState` plus `ok` and `errors` from `validate_squad`. A draft is allowed
to be illegal while you edit it (13 players, four keepers, over budget); `commit` is the
gate that refuses with 400 until it isn't. Each pick carries both `position` (its 1-15
slot) and `position_name` (GK/DEF/MID/FWD).

Pass `use_draft: true` to `/recommend` or `/whatif` to optimise from the working copy
instead of the set squad. Draft-based runs are never persisted or served from the
recommendation cache, so experimenting cannot pollute your real gameweek history.

`ComparisonView` is what powers side-by-side: aligned rows of expected points, risk profile, shared vs unique players, per-GW projections over the horizon, and each squad's recommended action.

## Recommendations

```
POST   /api/squads/{id}/recommend           {gameweek?, variants?, force_refresh?, use_draft?} → [Recommendation]
GET    /api/squads/{id}/recommendations?gw= → [Recommendation]
GET    /api/recommendations/{rid}           → Recommendation (full payload)
GET    /api/recommendations/{rid}/evidence  → [EvidenceItem]
POST   /api/recommendations/{rid}/accept    marks accepted, writes a `planned` SquadState
                                            (a plan — it does not change the squad you own)
POST   /api/recommendations/{rid}/reject    {reason?}  — feeds the critique pass's memory
POST   /api/squads/{id}/whatif              {constraints: {force_in:[], force_out:[], lock:[], budget_override?, chip?}} → Recommendation
```

`Recommendation.payload_json` shape:

```jsonc
{
  "variant": "balanced",
  "gameweek": 14,
  "horizon": {"gws": 5, "decay": 0.84},
  "transfers": [
    {"out": {"player_id": 412, "name": "…", "selling_price": 78},
     "in":  {"player_id": 88,  "name": "…", "price": 81},
     "delta_exp_points_gw": 1.8, "delta_exp_points_horizon": 4.2}
  ],
  "hits": 0,
  "chip": null,
  "lineup": {"xi": [...], "bench_order": [...], "captain": 88, "vice": 301, "formation": "3-4-3"},
  "totals": {"exp_points_gw": 61.4, "sd_points_gw": 12.1,
             "exp_points_horizon": 289.7, "p_haul_captain": 0.31},
  "alternatives": [{"label": "roll transfer", "delta": -0.4}, {"label": "no change", "delta": -1.9}],
  "chip_calendar": [{"chip": "bench_boost", "best_gw": 34, "gain": 11.2}]
}
```

## Players and predictions

```
GET  /api/players?position=&team=&max_price=&min_minutes=&sort=&owned_by_squad=
     → [PlayerRow]  (price, ownership, form, exp_points_gw, exp_points_horizon, value, risk)
GET  /api/players/{id}                       → PlayerDetail
GET  /api/players/{id}/history?season=       → per-fixture stats
GET  /api/players/{id}/predictions?gws=1-6   → [Prediction] with component breakdown
GET  /api/players/{id}/features?gw=          → [{name, value, percentile, contribution}]
GET  /api/players/{id}/claims?days=14        → [Claim] with source + deep link
GET  /api/players/compare?ids=1,2,3&gws=1-6  → ComparePayload
```

## Fixtures, teams, gameweeks

```
GET /api/gameweeks/current           → {gameweek, deadline_utc, seconds_remaining, chips_available}
GET /api/fixtures?gw=&team=          → [Fixture]
GET /api/fixture-ticker?gws=1-8      → matrix of team × gw with difficulty + dgw/bgw flags
GET /api/teams/{id}/strength         → model-derived attack/defence over time
```

## Content and evidence

```
GET /api/feed?types=article,video,social,claim&player_id=&since=  → paginated feed
GET /api/videos/{id}                 → video + transcript + extracted claims (with timestamps)
GET /api/search?q=&k=20              → hybrid vector+FTS search over chunks
GET /api/sources                     → [SourceStatus] — enabled, last run, health, rows
```

## Models and diagnostics

```
GET  /api/models                     → [ModelVersion] active + history + metrics
POST /api/models/train               {models: [...]} → job id
GET  /api/models/{name}/calibration  → calibration curve data
GET  /api/backtests                  → [BacktestRun]
POST /api/backtests                  {seasons, settings} → job id
GET  /api/pundits                    → channel/source accuracy scoreboard
GET  /api/jobs                       → running + recent scheduler jobs, with last error
POST /api/jobs/{name}/run            manual trigger
```

## LLM

```
POST /api/chat                       {squad_id, messages[], stream: true} → SSE stream
GET  /api/chat/sessions              → saved conversations
GET  /api/llm/usage?days=30          → per-task cost and volume
```

## Settings

```
GET   /api/settings/global           → merged view with .env-derived defaults marked read-only
PATCH /api/settings/global           {key: value, ...}
GET   /api/settings/squad/{id}
PATCH /api/settings/squad/{id}
GET   /api/settings/schema           → JSON schema driving the settings UI (so the UI never hardcodes fields)
```

## FPL push

```
POST /api/squads/{id}/push/preview    → {transfers_diff, lineup_diff, cost, warnings[]}
POST /api/squads/{id}/push/execute    {confirmation_text: "PUSH GW14", dry_run: false} → PushResult
GET  /api/squads/{id}/push/history    → [PushRecord]
```

`push/execute` refuses unless `FPL_WRITE_ENABLED=true`, the confirmation string matches exactly, and a preview was generated in the last 10 minutes. It snapshots the pre-push state first, and any failure returns the exact upstream response so you can see what FPL rejected.

## Realtime

`GET /api/events` — SSE stream for job progress, new predictions, ingest completions, and deadline countdown. The frontend uses this instead of polling.
