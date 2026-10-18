# Tail program, test 1 — DraftKings ladder-shape diagnostic, locked rule

Committed 2026-10-18 (UTC) before running. Written from the user's instruction of 2026-10-18; nothing below may change after this
commit. No game outcomes enter the diagnostic, so it burns no test season. Output: `docs/data/ladder_shape_diagnostic.json` and the
Backtest page only; never `data/paper_log.csv` or `docs/data/results.json`. Script: `scripts/ladder_shape_diagnostic.py`.

## Question

Does DraftKings' alternate-threes ladder price a player's "shape" (how variable his minutes and his made threes are), or only his
level (main line and main price)? If the shape is unpriced at DK, shape-based angles could exist at DK; if DK prices it, they
could live only at a venue that does not.

## 1. Shape score (locked; prior games only; >= 10 prior games required)

For player-game i on date t (2024-25 season, the games in the Part A DraftKings cache), from raw ESPN box scores strictly before t:

- **Minutes CV**: standard deviation / mean of minutes over the player's trailing 15 games before t (fewer if he has 10–14; at least
  10 games played with minutes > 0, otherwise the player-game is not scored).
- **3PM variance-to-mean ratio (VMR)**: sample variance / mean of made threes over the same trailing window; if the mean is 0 the
  VMR is set to the league prior.
- **Empirical-Bayes shrinkage of the VMR** toward a league prior estimated by method of moments on raw 2023-24 box scores (2022-23
  would be preferred as a season further from the test data, but no 2022-23 box scores exist in this repository; stated in the
  report): over every 2023-24 player with >= 10 games played, compute each player's season VMR_j from his raw per-game 3PM and his
  game count n_j; prior mean = the n-weighted mean of VMR_j; sampling variance of a VMR estimate taken as 2·VMR_j² / (n_j − 1);
  between-player variance tau² = max(var(VMR_j) − mean(sampling variance), 0.01·prior mean²); prior strength K = mean(n_j · sampling
  variance_j) / tau². Shrunk VMR_i = (n_i · VMR_i + K · prior mean) / (n_i + K) with n_i the window length. Raw box scores only: no
  Part B parameter, code output or dispersion estimate is reused.
- **Shape score** = mean of the two standardized components, z(minutes CV) and z(shrunk VMR), standardized with the mean and SD
  over all scored 2024-25 player-games in the diagnostic (no outcomes involved).

## 2. DraftKings ladder regression

- Data: the Part A closing snapshots (`data/odds_close_2025/`, `data/odds_alt_2025/`), DraftKings only. Per player-game: DK's
  **main line** = its two-sided player_threes line nearest even money, with the Over price there (**main price**, as implied
  probability and its logit); DK's **tail rungs** = Over rungs in player_threes_alternate at lines main + 1, main + 2, main + 3 and
  main + 4 (**rung distance** d = 1..4). Implied tail P = raw implied probability of the Over price (no de-vig; the ladder's margin is
  part of what is being modelled). Player-games need a scored shape score (section 1) to be included.
- Split: DK games (the Odds API events) are split in half at random by game, seed 20261018, into FIT and HOLDOUT.
- Model on FIT, ordinary least squares: logit(implied tail P) ~ main_line * main_logit * d (all main effects, two-way and three-way
  interactions; 8 coefficients). Nothing else enters.
- Holdout residuals: r = observed logit − fitted logit on HOLDOUT rungs. Reported: residual SD (logit units) on HOLDOUT, the FIT R²,
  and the Pearson correlation between r and the shape score at the rung level (primary) and between the per-player-game mean residual
  and the shape score (secondary), with 95% CIs by game-cluster bootstrap (1,000 resamples). Spearman is reported for information.

## 3. Outcome (locked, read from the primary rung-level figures on HOLDOUT)

- **F (formulaic ladder)**: residual SD < 0.10 logit AND |corr| < 0.10 -> shape is unpriced at DK; test 3 proceeds.
- **P (priced)**: corr >= +0.20 (positive: DK prices high-shape players' tails higher than the formula) -> DK prices shape; shape
  angles live only via Underdog.
- Anything else -> in-between: report; run test 3 at low prior.

Report and stop.
