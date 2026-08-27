# 12 — Testing

`pytest` + `pytest-asyncio` + `respx` (HTTP mocking) + `hypothesis` for the rules engine. Vitest + Testing Library on the frontend. Target: fast unit suite under 30s, full suite under 5 minutes.

## Fixtures and golden data

`tests/fixtures/` holds captured real payloads: one `bootstrap-static`, a fixtures list, three `element-summary` responses, a YouTube transcript, five news articles (including two syndicated duplicates), an odds snapshot, an injury table HTML, and a Reddit thread. These are checked in and are the basis of every parser test. Recapture with `python -m fplai.cli capture-fixtures` when a source's shape changes — which it will.

## Layer 1 — Parsers and dedup

- Each connector's `parse` runs against its golden payload and asserts exact typed output. Parser changes that alter output must update the golden file, making the diff visible in review.
- **Dedup tests, the important ones:**
  - Same payload twice → one `raw_documents` row, `seen_count == 2`.
  - Same external id, changed content → two rows, second has `supersedes_id` set.
  - Volatile-field-only difference (timestamps, request ids) → treated as identical.
  - Two syndicated articles with reworded headlines → same `near_dupe_group`.
  - Property test: for any random ordering of an ingest batch, final row count is identical.

## Layer 2 — Entity resolution

A labelled set of ~300 real surface forms ("TAA", "Gabriel (Arsenal)", "the Brazilian", "Sánchez") with expected resolutions, including deliberate ambiguities that must resolve to `null`. Assert precision ≥ 0.98 (a wrong link is worse than a miss) and recall ≥ 0.85. This test is the early-warning system for the whole text pipeline.

## Layer 3 — FPL rules engine

Pure functions, exhaustively tested with `hypothesis`:

- Selling price: purchase 70, now 75 → sells at 72 (profit 5, half rounded down = 2). Cover every rounding edge.
- Free transfer accrual with the 5 cap, including "used 3 when you had 2" hit maths.
- Chip legality: set 1 unusable at GW 20, one chip per GW, Free Hit reverting the squad, Wildcard persisting.
- Autosub logic and bench ordering, including the GK special case.
- Formation validity across all legal shapes.
- Captain/vice fallback when the captain doesn't play.

Golden case: replay a full historic season's transfer/chip sequence for a known entry and assert the computed points match FPL's reported total exactly. If this passes, the rules engine is right.

## Layer 4 — Features

- **Leakage test (critical):** rebuild features for GW k using only data with `observed_at < deadline(k)` and assert byte-identical output to the values stored live at the time. Any future-peeking feature fails here.
- Missing-block test: with the odds source disabled, every feature still computes and the block indicator is set.
- Determinism: same inputs, same feature version → same values.

## Layer 5 — Models

- **Regression test with tolerance:** a frozen slice (season 2024/25, GWs 20–30) with committed expected metrics. New model versions must not degrade minutes log-loss by more than 3% or points rank-correlation by more than 0.02 without an explicit `--allow-regression` flag.
- Calibration: predicted probabilities binned; ECE below threshold for minutes and clean sheets.
- Sanity assertions that catch stupid bugs fast: expected points non-negative; a player with `P(start)=0` has EP < 1.0; a premium forward's EP exceeds a £4.0m bench defender's in ≥ 95% of gameweeks; simulated mean matches analytic expectation within Monte Carlo error.
- **Text-feature dominance guard:** assert no text-derived feature ranks top-3 by gain importance in the points model. If it does, the test fails loudly — that's a signal of leakage or of the pundit signal being a proxy for something the model should already know.

## Layer 6 — Optimiser

- Every returned squad satisfies all constraints (15 players, position counts, ≤3 per club, budget) — asserted on 200 randomised inputs.
- Given hand-crafted inputs with a known optimum, the solver finds it.
- Monotonicity: raising one player's EP never removes them from the XI, all else equal.
- Transfer continuity: planned squad at GW g+1 differs from GW g by exactly the planned transfers.
- Determinism given a fixed seed and solver.

## Layer 7 — LLM

Never assert on model prose. Assert on structure:

- Extraction output validates against the schema; `text_span` is a genuine substring of the input chunk; spans stay under the length cap.
- Adjustment layer respects its cap even when the model returns an absurd value.
- Critique output cannot mutate a plan (the applied plan is byte-identical before and after).
- Chat tool calls only ever hit read-only tools; a test asserts the tool registry exposes no mutating function.
- Cache: identical prompt hits the cache and makes zero HTTP calls.
- A recorded-cassette suite (`vcr`-style) for one real call per task, refreshed manually, so schema drift at the provider is caught.

## Layer 8 — End to end

`docker compose up` in CI, seed from golden fixtures, then: create squad → sync → predict → optimise → fetch recommendation → assert a valid 15 with evidence links present. Plus a Playwright pass over the dashboard at 390px and 1440px widths asserting no horizontal scroll and that the primary action is reachable.

## What isn't tested, deliberately

Live third-party endpoints. Nothing in CI touches the real internet — scrapers break on their own schedule and a red CI that means "SofaScore changed their HTML" trains you to ignore CI. Instead, source health lives in the app: the Sources screen and a daily Discord message report last-success time per connector, and a connector that returns zero rows twice in a row raises an alert.
