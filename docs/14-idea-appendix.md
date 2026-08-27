# 14 — Idea Appendix

You asked for as many ideas as possible, good or bad. Here they all are, graded honestly. Grades are my opinion of expected value per hour of build time, not of how fun they'd be.

**A** = do it, it's in the plan or should be · **B** = worth building once the core works · **C** = interesting, probably won't pay for itself · **D** = fun but I'd bet against it · **F** = actively harmful, listed so you know why not.

---

## Data and signal ideas

**A — Referee assignment as a feature.** Referee appointments are published a few days before each round. Card rates vary meaningfully by official, and cards are direct negative FPL points. Cheap to scrape, small but real edge, especially for aggressive midfielders.

**A — Odds movement rather than odds level.** Everyone can see the current price. Movement in the 48h before kickoff often reflects team news the public hasn't priced. Store snapshots (the schema already does), feature the drift.

**A — The "news gap" feature.** Consensus availability from your text corpus minus FPL's official `chance_of_playing`. When the gap is positive and large, you know something the field's default tooling doesn't yet. This is the single most defensible edge in the whole design.

**A — Manager press conference timing.** Not the content — the *timing*. Knowing that Friday's presser happens two hours before your deadline tells the app when to re-run, and tells you when to wait.

**B — Ingest the manager's press conference transcripts directly.** Club YouTube channels post them. Direct quotes about fitness are the highest-quality availability signal that exists, and it's the same pipeline you've already built for pundit videos.

**B — Under-the-hood shot location data.** Understat gives shot coordinates. Player-level xG-per-shot from location beats aggregate xG for identifying who's getting *good* chances versus lots of chances.

**B — Track "returning from injury" minute ramps.** Players coming back rarely play 90 immediately. A small model of minutes-after-return by injury type and days out would improve the minutes model exactly where it's currently worst.

**B — Fantasy pundit *disagreement* as a variance signal.** Not consensus — disagreement. When channels split hard on a player, that's genuine uncertainty and should widen the distribution rather than shift the mean.

**B — Weather at the *stadium* rather than the city.** Trivial refinement of something already planned, and some grounds are notably windier.

**C — Betting market for "player to be subbed off" / "to start".** Some books offer starting-XI markets. If you find a reliable feed, it's a free minutes model. Coverage is patchy, hence C.

**C — Ingest FPL Twitter/X community sentiment as a contrarian indicator.** The idea that the crowd is systematically wrong is appealing and mostly untrue. Worth *measuring* with the pundit scoreboard; not worth building a strategy around until measured.

**C — Squad-value tracking and "team value" optimisation.** Early-season price rises compound into flexibility later. The optimiser has the hook (`price_bonus_weight`) but chasing value at the cost of points is a beginner's trap, so it's off by default.

**C — Historical head-to-head "hoodoo" features.** Player X always scores against Team Y. Sample sizes are tiny and it's mostly noise. Cheap to add, so build it, then let the ablation kill it.

**D — Ingest Football Manager or FIFA/EAFC player ratings.** Fun, and a genuinely useful prior for obscure new signings with no PL minutes. But Transfermarkt value does the same job with better provenance.

**D — Social media *emoji* sentiment.** "🔥🔥🔥 vs 🤢" as a signal. Amusing, and I'd guess it correlates with recent points, which the model already has directly.

**D — Player Instagram activity as a fitness proxy.** Posting from a gym vs. a beach. Real people have tried this. The signal-to-noise is dreadful and the creepiness-to-value ratio is worse.

**F — Scraping other managers' teams at scale to reverse-engineer the "template".** LiveFPL already publishes effective ownership. Hammering FPL's API for hundreds of thousands of entries to recompute it yourself is rude, slow, and gets your IP blocked.

---

## Modelling ideas

**A — Model DefCon as a threshold-crossing probability, not a mean.** Already in the plan, but it deserves repeating: this is the most commonly botched piece of modelling in post-2025 FPL tools.

**A — Correlated simulation.** Sampling teammates independently is the single most common modelling error in public FPL projections and it makes every variance number wrong. Doing it properly is what makes the risk slider mean something.

**A — Separate BPS regime handling for 2026/27.** The system was retuned this summer specifically to reduce DefCon overlap and favour keepers, full-backs and attackers. Any model trained naively on last season will systematically misprice full-backs for the first two months. Treat this as an opportunity: most public tools will get it wrong for a while.

**B — Hierarchical player priors.** New signings and promoted-club players have no PL history. A Bayesian hierarchical model that shrinks toward league/position/price priors handles cold starts far better than a GBM with NaNs.

**B — Quantile regression for the tails.** Expected points are for the optimiser; `P(haul)` is for captaincy. Fitting the upper quantile directly beats reading it off a simulated distribution assembled from means.

**B — Learn the odds-blend weight `w` per market and per horizon** rather than fixing it at 0.65. The market is sharper close to kickoff; your model may be better a week out.

**C — A "manager tendency" model.** Per-manager rotation, substitution timing, and formation priors. Real signal, but with nine new managers this season the training data for most of them is thin to nonexistent.

**C — Opponent-specific matchup modelling.** Pace-heavy winger vs. slow full-back. Requires positional/tracking data you can't get cheaply.

**C — Reinforcement learning for chip timing.** Framing the season as an MDP and learning a chip policy is genuinely elegant. It's also 38 decisions per season per episode with maybe 10 seasons of data — nowhere near enough. The MILP long-horizon planner gets 95% of the value.

**D — A neural net over the whole feature set.** With ~100k usable rows and mostly tabular features, gradient boosting wins. Deep learning here is résumé-driven development.

**D — LLM-as-forecaster.** Asking a strong model to predict a player's points directly. It'll produce confident, plausible, poorly-calibrated numbers anchored on name recognition. Worth running *once* as a baseline in the backtest so you can point at the result — I'd expect it to lose badly to the statistical model, and knowing that by how much is genuinely useful.

**F — Optimising directly on backtest results.** Tuning hyperparameters until the 2024/25 replay looks amazing. You'll build a model that would have won last season and loses this one. Hence the locked holdout in `06`.

---

## Product and UX ideas

**A — "Do nothing" as a first-class recommendation.** Most weeks the right move is no move. A tool that always finds a transfer will lose you points and money.

**A — Timestamped deep links from claims to video.** Being able to tap "Let's Talk FPL said this" and land at 14:32 in the video is the feature that makes the evidence panel trustworthy rather than decorative.

**A — Chip expiry nagging.** Set 1 dies at the GW19 deadline on 2 January. An unused Bench Boost expiring is a pure, avoidable loss, and it happens to thousands of people every year.

**B — Decision journal.** Log every recommendation you accepted or rejected, and what happened. After ten weeks the app can tell you whether *you* or *it* is the better decision-maker. Slightly confronting; very valuable.

**B — "Explain the difference" between two squads.** Not two lists — a generated paragraph on what strategic bet each squad represents.

**B — Regret analysis after each gameweek.** "Your accepted plan scored 58. The aggressive variant scored 71. The best possible transfer was X." Bounded honestly: hindsight isn't a scoreboard for decisions, and the UI should say so.

**B — Watchlist with alerting.** Discord ping when a watchlist player's price is about to change or their status flips.

**C — Natural-language squad setup.** "Build me something for a work league where I'm 40 points behind with a wildcard in hand" → proposed settings JSON, which you confirm. The `settings_assistant` task exists for this; it's a nice demo and a mediocre workflow.

**C — A "pundit consensus XI" shown alongside the model's XI.** Fun comparison, and the pundit scoreboard makes it meaningful over time. Purely informational.

**C — Voice input on mobile.** "Should I captain Haaland?" while walking. Web Speech API, half a day. Mostly a novelty.

**D — Gamified confidence betting.** Predicting your own gameweek score and scoring your calibration. Cute, distracting.

**D — An LLM-simulated pundit debate.** Three personas argue about your transfer and produce a verdict. Genuinely entertaining, completely epistemically worthless — the personas share one model's priors, so "consensus" among them is one opinion in three voices. If you build it, label it as entertainment, and never let its output touch a prediction.

**D — Auto-generated video or TTS gameweek preview.** You've already got the TTS pieces from the radio project. It would be fun. It would also be a weekly artefact you stop watching after three weeks.

**F — Auto-pushing transfers on a schedule.** The whole point of a pre-deadline tool is that you review it. An unattended job that makes real transfers based on a scraper that might have broken silently is how you wake up to a squad full of unavailable players.

**F — Letting the LLM directly edit the squad without the optimiser.** You already ruled this out and you were right. The failure mode isn't dramatic — it's a slow drift toward whichever players were most discussed this week.

---

## Infrastructure ideas

**B — A "reprocess from archive" command.** Change a parser, replay every stored raw doc, rebuild everything, zero refetches. The architecture already supports it; make it a first-class CLI verb with progress reporting because you'll use it constantly.

**B — Snapshot the whole DB before each retrain.** SQLite makes this a file copy. Cheap insurance and instant rollback.

**C — Export projections to CSV/Google Sheets.** For when you want to do something the UI doesn't support. Ten lines, occasionally invaluable.

**C — A read-only "season replay" mode.** Scrub back to any past deadline and see exactly what the app knew and recommended then. Excellent for debugging and for trust; moderate build cost.

**D — Multi-user support.** You said single user. Every feature that assumes one user is simpler; adding users later is a rewrite of the settings model. Don't pre-build it.

---

## The three things I'd actually prioritise

If the plan proves too big and you want the highest value per hour:

1. **Minutes model quality.** Everything downstream is multiplied by it. A great points model with a mediocre minutes model is a mediocre app.
2. **Odds integration.** The cheapest large accuracy gain available, and the free tier of The Odds API is enough to prove it.
3. **The news gap feature.** The only part of this design where you're likely to know something the rest of the field doesn't.

Everything else — the YouTube pipeline included — is worth building, but should be judged by the ablation table in `06`, not by how impressive it sounds. Be prepared for the honest answer that a well-calibrated minutes model plus market odds gets you 90% of the way, and the entire text corpus adds a couple of points a season. That would still be a good app, and knowing it would be worth more than assuming otherwise.
