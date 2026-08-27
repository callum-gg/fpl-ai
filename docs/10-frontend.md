# 10 — Frontend

**Stack:** Vite + React 18 + TypeScript (strict), TailwindCSS, TanStack Query (server state), Zustand (active squad + comparison set), Recharts, `dnd-kit` for pitch drag-and-drop, `react-router`. No Next.js — the backend is Python, so an SPA served by nginx is simpler and there's no SSR benefit for a local single-user tool.

Types come from `openapi-typescript` against the FastAPI schema. No hand-written API types.

## Design direction

Dark by default, and specifically *not* the FPL purple-and-green palette — you'll be looking at this alongside the real site and they should be distinguishable at a glance. Proposed: near-black background (`#0B0D10`), elevated surfaces at `#14171C`, a single cool accent (`#4ADE80` for positive deltas, `#F87171` negative), and one warm accent per squad chosen from a fixed palette so the squad picker is colour-coded and you can tell two squads apart in the comparison view without reading labels.

Typography: one variable sans (Inter or Geist) for UI, tabular figures everywhere numbers align. Numbers are the product here — a points projection must never shift columns when it changes from 9.9 to 10.1.

Density: comfortable on mobile, dense on desktop. Two breakpoints that matter: `<640px` (phone, primary for viewing) and `≥1024px` (desktop, primary for planning). The middle is a stretched phone layout; don't over-engineer it.

## Global chrome

- **Squad picker** — persistent, top-left on desktop, a sticky pill at the top on mobile. Shows squad name + colour dot + this GW's projected points. Tapping opens a sheet listing squads with live projections, a "compare" multi-select, and "new squad".
- **Deadline countdown** — always visible, turns amber at 24h and red at 2h.
- **Command palette** (`⌘K`) — jump to player, squad, or screen; run "recommend now"; open chat with the player pre-loaded.
- **Bottom tab bar on mobile** (Dashboard / Squad / Players / Chat / More); left sidebar on desktop.

## Screens

### 1. Dashboard
The one screen you'd keep if you could only have one.

- Headline card: recommended action for the active squad this GW, in one sentence, with expected gain vs doing nothing.
- Three variant chips (Safe / Balanced / Aggressive) — tapping swaps the whole card.
- Projected XI with captain flagged, expected points and the P10–P90 band drawn as a bar.
- "What changed since you last looked" — new injuries, price changes, lineup news affecting *your* players only.
- Alert strip: chip expiry warnings (loud from GW15 onward), players flagged, price-fall risk on your holdings.

### 2. Squad view
- Pitch graphic, drag to reorder bench and set captain (writes to a *planned* state, never pushed automatically).
- Per-player card: predicted points with distribution, minutes probability as a small bar, next 5 fixtures as coloured pips, and a status dot for availability consensus.
- Toggle: "current" / "planned" / "recommended" overlays showing exactly what changes.
- Push-to-FPL button lives here, with the preview-diff modal and typed confirmation.

### 3. Comparison view
Because you want squads side by side. Columns = squads, rows = aligned metrics: projected points this GW and horizon, risk profile, transfers proposed, hits, chip plan, and a shared/unique player Venn. On mobile it becomes a horizontally-swipeable set of columns with the metric labels frozen on the left.

### 4. Player explorer
Sortable, filterable table (position, price band, club, ownership, minutes floor). Columns: price, EP this GW, EP horizon, value (EP/£m), P(start), P(haul), ownership, EO, form, and a sparkline of the last 6 GWs. Long-press/right-click → compare, add to watchlist, ask chat about them.

### 5. Player detail
Header with the model's numbers. Then tabs:
- **Projection** — component breakdown (minutes × goals × assists × CS × DefCon × bonus) as a waterfall to the final EP. This is the "why" people actually want.
- **Form** — per-fixture history with xG/xA overlays.
- **News & video** — the claim feed, grouped by day, each with source, trust weight, and for videos a timestamped deep link straight to the moment they said it (`youtube.com/watch?v={id}&t={start}s`). Near-duplicate news collapses into one entry with a "seen on 6 sites" badge.
- **Fixtures** — next 8 with model difficulty, not FDR.

### 6. Transfer planner
Multi-GW grid: rows = your players, columns = GW+1…GW+n, cells showing planned in/out. Editable — drag a player out, the optimiser re-solves the rest around your constraint and shows the cost of your idea in points. Free transfer and bank counters update live. Chip markers can be dragged onto a gameweek to force that chip and see the consequence.

### 7. Fixture ticker
Team × gameweek matrix, coloured by model-derived difficulty, with DGW/BGW badges. Filter to your squad's clubs. Tap a cell for the underlying odds and team-strength numbers.

### 8. Feed
Everything ingested, newest first: articles, videos, tweets, Reddit posts, claims. Filter by source type, player, or "affects my squad". Each item expands to the extracted claims. This is also where source health surfaces — a source that's returned nothing for 24h shows a warning here.

### 9. Model performance
Backtest results, calibration curves, ablation table (does the YouTube pipeline actually help?), per-position rank correlation, and the **pundit accuracy scoreboard** — channels ranked by how their recommendations actually performed against a price-and-position-matched baseline. Expect this screen to be quietly humbling for everyone involved, including the model.

### 10. Chat
Streaming, tool-call steps shown collapsed ("searched 412 transcripts", "ran optimiser with Haaland forced out"). Suggested prompts seeded from context. Any squad change it proposes appears as a card with an "apply to planned squad" button — never automatic.

### 11. Settings
Two clearly separated tabs, as you asked:
- **Global** — sources on/off and keys present/absent (values redacted), YouTube tracked channels (add/remove/discover), RSS feeds, LLM per-task model selection, scheduler cadences, Discord webhook, scrape kill switch, entity review queue.
- **Squad** — risk slider, horizon GWs + decay weight, differential/rank-chase toggles, mini-leagues (added by id, or picked from the ones the API reports for your entry), banned clubs, locked players, max hits per GW, bench weight, chip strategy preferences.

Both are rendered from `/api/settings/schema` so adding a setting in the backend needs no frontend change.

## Mobile specifics

- Every primary action reachable one-thumbed in the bottom third.
- Pitch view scales to viewport height with no scroll; player cards use a bottom sheet, not a modal.
- Tables become cards below 640px, keeping the two most important numbers visible.
- Charts get a simplified variant on small screens (no dual axes, fewer ticks).
- `content-visibility: auto` on long feeds; virtualised lists for the player table.
- PWA manifest so you can add it to the home screen — that's not "a mobile app", it's just a nicer bookmark, and it's ten lines.
- Optimistic UI for planned-squad edits; everything else shows real loading states because model runs genuinely take seconds.
