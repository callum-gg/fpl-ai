# 05 — Feature Store

Features are declared, not scattered. A registry gives each one a name, a version, a dependency list and a builder function, so the UI can explain any number and the trainer can guarantee no leakage.

```python
@feature("mins_last5_weighted", deps=["player_fixture_stats"], version=2)
def mins_last5_weighted(ctx: FeatureCtx) -> float: ...
```

**Leakage rule, enforced in code:** every builder receives `ctx.as_of` (the gameweek deadline) and may only read rows with `observed_at < as_of`. A test asserts that rebuilding features for a past GW produces byte-identical output to what was produced live. This is the difference between a backtest that means something and one that lies to you.

---

## A. Player form and underlying performance

| Feature | Definition |
|---|---|
| `mins_last{1,3,5,10}` | minutes played, plus exponentially-weighted variant (half-life 3 games) |
| `starts_last5`, `start_streak` | consecutive starts — the strongest single minutes predictor |
| `sub_appearance_rate`, `avg_sub_on_minute` | separates rotation risk from cameo risk |
| `xg90_last{3,5,10}`, `npxg90`, `xa90`, `xgi90` | per-90 with EWMA; shrunk toward positional mean for low-minute players |
| `shots90`, `sot90`, `touches_box90`, `big_chances90` | volume, which is more stable than conversion |
| `conversion_ratio` | goals / xG — used to *regress* hot streaks, not chase them |
| `xg_overperformance_last10` | explicit mean-reversion signal |
| `defcon_actions90`, `defcon_hit_rate_last{5,10}` | count of CBIT (DEF) or CBIRT (MID/FWD) per 90, and the share of games clearing the 10/12 threshold. **Threshold-crossing rate matters more than the mean** — model it directly |
| `defcon_margin` | mean distance above/below threshold, captures how safe a DefCon asset is |
| `bps90_last5`, `bonus_rate_last10` | with a **2026/27 recalibration flag** — see the BPS note below |
| `saves90`, `saves_per_shot_faced` | GK only |
| `pens_taken_share`, `is_first_pen_taker`, `corner_duty`, `direct_fk_duty` | from set-piece data |
| `card_rate90`, `fouls90` | small negative EV, real for some midfielders |
| `age`, `days_since_debut` | mild curve effects |
| `promoted_club_flag` + `transfermarkt_value_pct_of_squad` | the only useful prior for Coventry/Ipswich/Hull players with no PL history |

**BPS caveat.** The system was retuned for this season to reduce DefCon overlap and favour goalkeepers, full-backs and attackers. Any BPS coefficient learned from 2025/26 is therefore biased. Handling: keep historic BPS features but add `season_bps_regime` as a categorical, and for 2026/27 fit a *separate* lightweight bonus model on this season's rows only, blending with the historic model using a weight that shifts toward the new model as the season's sample grows (`w_new = n_new / (n_new + 40)` fixtures). Flag this prominently in the model performance screen for the first ~8 GWs.

## B. Fixture and opponent

| Feature | Definition |
|---|---|
| `is_home` | |
| `opponent_defence_rating`, `opponent_attack_rating` | from the team-strength model, not FPL's FDR |
| `fdr_official` | kept only as a comparison baseline |
| `opp_xga_per_game_last6`, `opp_xg_conceded_home/away_split` | |
| `opp_clean_sheet_rate`, `opp_goals_conceded_ewma` | |
| `team_expected_goals` | from odds (primary) or team model (fallback) |
| `p_clean_sheet_odds` | devigged market probability where available |
| `p_anytime_scorer_odds` | player-level market probability — treat as a *feature*, not the answer |
| `odds_movement_48h` | drift in team goal expectation; steam is information |
| `fixture_run_score_next{3,5,8}` | decayed sum of upcoming difficulty, the transfer-planner's headline number |
| `dgw_flag`, `bgw_flag`, `n_fixtures_this_gw` | |
| `kickoff_slot` | early Sat / late Sun / midweek — affects rotation |
| `derby_flag`, `dead_rubber_flag` | end-of-season motivation, crude but real |

## C. Congestion, rest and travel — your "time since last match"

| Feature | Definition |
|---|---|
| `days_since_last_match` | player-level, not team-level (a benched player is fresh) |
| `player_minutes_last_{7,14,21}_days` | the real fatigue proxy |
| `team_matches_next_14_days` | includes UCL/UEL/FA Cup/EFL — the rotation driver |
| `midweek_european_flag`, `european_competition_tier` | Thursday Europa is worse than Tuesday UCL for a Saturday lunchtime kickoff |
| `days_rest_diff_vs_opponent` | one team on 3 days rest against one on 7 is a genuine edge |
| `international_break_flag`, `intl_minutes_last_break`, `intl_travel_km` | long-haul returnees get benched; approximate distance from nationality → federation |
| `manager_rotation_index` | historical: variance in that manager's XI across congested vs normal weeks. Rebuild per manager per season |
| `manager_tenure_days`, `new_manager_flag` | nine new managers this season means historic team priors are shakier than usual — this feature lets the model widen its own uncertainty |

## D. Availability and news-derived

| Feature | Definition |
|---|---|
| `fpl_chance_of_playing` | official flag, 0–100 |
| `injury_status_consensus` | weighted vote across premierinjuries, physioroom, transfermarkt, claims |
| `days_since_injury_report`, `expected_return_gw` | |
| `source_disagreement_score` | variance across sources — high disagreement should widen the minutes distribution, not shift it |
| `claim_count_{injury,rotation,return}_7d` | volume of talk |
| `claim_sentiment_weighted_7d` | trust-weighted mean sentiment (channel accuracy × recency decay × near-dupe collapse) |
| `press_conference_recency` | hours since the manager last spoke — the single most informative timing feature before a deadline |
| `predicted_lineup_prob` | share of predicted XIs including the player, across available sources |
| `news_signal_gap` | consensus availability minus FPL's official flag. **The edge feature.** Positive gap = news says fit before FPL updates |

## E. Market and ownership

| Feature | Definition |
|---|---|
| `owned_pct`, `owned_pct_top10k`, `effective_ownership` | |
| `ownership_momentum_24h`, `net_transfers_24h` | |
| `price_change_progress` | estimated distance to the next rise/fall threshold |
| `is_template_flag` | EO > 40% |
| `differential_score` | predicted points percentile minus ownership percentile |
| `value_score` | expected points per £m over the horizon |

## F. Text-derived aggregates (per player per GW)

Built from `claims`, all trust-weighted and recency-decayed with a 3-day half-life:

`yt_mention_count`, `yt_buy_calls`, `yt_sell_calls`, `yt_captain_calls`, `yt_avoid_calls`, `yt_net_sentiment`, `yt_consensus_strength` (agreement across distinct channels — not distinct videos), `reddit_mention_volume_zscore`, `reddit_sentiment`, `x_breaking_news_flag`, `journalist_tier1_mention` (weighted by which handle), `narrative_novelty` (how different this week's claims are from last week's, by embedding distance).

**Deliberate design choice:** text features are *never* allowed to be top-3 by importance in the points model. If they are, something has gone wrong — pundit consensus is mostly a lagging summary of the same stats the model already has. Their real value is (a) injury/rotation information the structured feeds haven't published yet, and (b) explanation material. There's a test for this in `12`.

## G. Squad-context features (computed at optimisation time, not stored per player)

`is_currently_owned`, `selling_price_vs_current`, `transfer_cost_if_sold`, `bench_slot_probability`, `captaincy_share_expected`.

---

## Missing data policy

Every feature has a declared `missing_strategy`: `zero`, `positional_mean`, `shrink_to_prior`, or `indicator` (a companion `_is_missing` column). LightGBM handles NaN natively so the default is to pass NaN plus an indicator. Whole missing *blocks* (e.g. no odds before install date) get a block-level indicator so the model can learn "when odds are absent, lean harder on the team model" instead of silently treating absence as zero.
