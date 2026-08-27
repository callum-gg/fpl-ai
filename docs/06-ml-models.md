# 06 — Prediction Models

## Why decomposed, not end-to-end

You could regress FPL points directly. Don't. Points are a lumpy mixture of a near-binary minutes process and several rare count processes; a single regressor learns "expensive attackers score more" and produces mush. Decomposing gives calibrated components, honest uncertainty, explanations you can actually show in the evidence panel, and — importantly for this app — a natural place to inject market odds.

```
                    ┌──────────────────────┐
odds ──────────────▶│ team goal model      │──▶ λ_home, λ_away
team form ─────────▶│ (bivariate Poisson / │    ▶ P(clean sheet), P(concede n)
                    │  Dixon–Coles + odds) │
                    └──────────────────────┘
                               │
availability ──┐    ┌──────────────────────┐
lineups ───────┼───▶│ minutes model        │──▶ P(start), P(sub), E[minutes | played]
congestion ────┤    │ (ordinal / 3-class)  │
claims ────────┘    └──────────────────────┘
                               │
player form ───────▶┌──────────────────────┐
set pieces ────────▶│ share models         │──▶ per-90 rates: goals, assists,
                    │ (GBM on per-90 rates)│    defcon actions, saves, cards
                    └──────────────────────┘
                               │
                    ┌──────────────────────┐
                    │ Monte Carlo (N=10k)  │──▶ full points distribution per player
                    │ correlated by fixture│    E, SD, P10/P50/P90, P(haul), P(blank)
                    └──────────────────────┘
```

## Model 1 — Team goals

Bivariate Poisson with Dixon–Coles low-score correction, attack/defence strengths as time-decayed latent parameters (exponential decay, half-life ≈ 60 matches), home advantage as a fitted parameter, fitted per team-season with shrinkage toward the league mean for promoted sides.

**Then blend with the market.** Where odds exist, fit λ_home/λ_away to the devigged 1X2 + over/under 2.5 surface, and take a weighted geometric blend: `λ_final = λ_model^(1-w) · λ_odds^w` with `w ≈ 0.65` (the market is better than your model at this and pretending otherwise is vanity). Learn `w` by backtest. Where odds are missing, `w = 0`.

Outputs: `P(clean sheet)` for each team, `P(goals_conceded = k)`, expected team goals, expected team shots.

## Model 2 — Minutes (the one that decides whether the app is good)

Three-class classifier per player-fixture: `start`, `bench_with_cameo`, `no_appearance`, plus a conditional minutes regressor for starters (accounting for early substitution) and for cameos.

- LightGBM multiclass, monotonic constraints where obviously correct (`start_streak` ↑ ⇒ P(start) ↑).
- Features: start streak, minutes trend, availability consensus, `predicted_lineup_prob`, congestion, manager rotation index, `news_signal_gap`, price/ownership (proxies for perceived nailedness), competition next/last 3 days.
- Hard overrides applied post-model, not learned: suspended ⇒ P(start)=0; confirmed lineup available ⇒ collapse to the truth; FPL flag 0% ⇒ P(appear) capped at 0.02.
- Calibrate with isotonic regression on held-out gameweeks and report ECE. An uncalibrated minutes model quietly destroys everything downstream.

## Model 3 — Per-90 rate models

Separate LightGBM regressors (Tweedie or Poisson objective) predicting per-90 rates, then scaled by expected minutes and by team attacking strength:

- `goals90` — target is smoothed npxG-informed goal rate, not raw goals; predicts a rate, then multiplied by `team_expected_goals / team_avg_xg` to condition on the fixture.
- `assists90` — same, with the 2025/26 simplified assist definition in mind.
- `defcon_actions90` → a **separate threshold classifier** `P(defcon_actions ≥ threshold | minutes)`, because the DefCon payoff is a step function and modelling the mean then thresholding is measurably worse. Fit as a count model (negative binomial) so you can integrate over the threshold exactly.
- `saves90` for GKs, conditioned on opponent expected shots.
- `cards90` (with referee strictness as a feature if you add a referee source — see the appendix).

Where an anytime-scorer market exists, do the same blend as Model 1: the market's `P(scores)` is a strong calibration target for `goals90 × minutes`.

## Model 4 — Bonus / BPS

Predict the BPS *distribution* per player, then simulate the within-fixture ranking to allocate 3/2/1. Simulating the ranking is essential — bonus is competitive, not absolute, and modelling `E[bonus]` directly ignores who else is on the pitch.

Given the 2026/27 BPS retune, run the blended-regime approach from `05`, and surface a banner in the model performance screen: "BPS model is on limited 2026/27 data (n = X fixtures); bonus predictions are wider than usual."

## Model 5 — Price changes

Separate, small, and genuinely useful for planning. Historical net-transfer counts vs observed rises/falls give a logistic model of `P(rise tonight)` per player. FPL now ships its own predictor, so ingest that too and use it as a feature/cross-check rather than competing with it.

## Simulation

For each gameweek, run N = 10,000 correlated simulations:

1. Sample match scorelines from the team model (home/away goals jointly).
2. Conditional on the scoreline, sample which players start (correlated within a team via a shared "rotation shock" factor — managers rotate in blocks, not independently).
3. Sample goals/assists allocation within the team's scored goals (multinomial over players' share weights) — this preserves the negative correlation between teammates that independent sampling destroys.
4. Sample DefCon, saves, cards.
5. Compute BPS, rank within fixture, allocate bonus.
6. Sum FPL points.

Store per-player E, SD, quantiles, P(haul ≥ 10), P(blank ≤ 2). **Keep the raw simulation draws for the current GW in memory (or a parquet in `data/models/sims/`)** — the optimiser needs the joint distribution, not just marginals, to evaluate squad-level variance and mini-league rank probabilities honestly.

## Training

- **Walk-forward only.** Train on GWs 1..k, predict k+1, roll. Never random k-fold — it leaks the future.
- Season weighting: `w = 0.72 ^ seasons_ago`, so 2025/26 counts ~0.72, 2024/25 ~0.52. Enough history to fit, recent enough to matter.
- Retrain after each GW's `data_checked` flips true. Full retrain, not incremental — it's minutes on your hardware.
- Hyperparameters via Optuna, 50 trials, on a fixed validation window; re-tuned monthly, not weekly.
- Every trained model writes a `model_versions` row with metrics, and is only promoted to `is_active` if it beats the incumbent on the held-out window. **Automatic promotion with a manual override toggle**, and a Discord message when a promotion happens or is blocked.

Metrics to track and display: minutes log-loss + ECE; points MAE and RMSE overall and by position; **Spearman rank correlation of predicted vs actual points within each position** (this matters more than MAE — you only ever act on the ranking); calibration curves for P(haul) and P(clean sheet).

## Backtesting

`python -m fplai.cli backtest --seasons 2024-25,2025-26 --settings safe,balanced,aggressive`

Replays a season deadline-by-deadline: rebuild features as-of, predict, run the optimiser under the season's real rules, apply real transfer/hit/chip constraints, then score against actual results. Reports total points, average GW score, comparison to the season's overall average and (where obtainable) the top-10k average, number of hits, and a chip-timing report.

Also run **ablations**, and show them in the UI: full model vs no-odds vs no-text vs no-minutes-model vs FPL-form-only baseline. This is how you find out whether the YouTube pipeline is earning its keep, and you should be genuinely willing to discover that it isn't.

**Caveats to state plainly in the report, not bury:** odds features are unavailable historically, so backtests before install date understate the model; the 2026/27 BPS change means older bonus results don't transfer; and a backtest tuned repeatedly against the same seasons will overfit them, so keep 2023/24 as a locked holdout you touch at most twice a season.

## What "enough data" looks like

You asked to use ML only if enough historic data exists. It does: ~10 seasons × 38 GWs × ~500 players ≈ 190k player-gameweek rows, of which ~100k have non-trivial minutes. That is comfortably enough for the minutes and rate models. It is *thin* for rare events (penalties saved, own goals) — those stay as fixed empirical rates rather than fitted models, and that's the right call.
