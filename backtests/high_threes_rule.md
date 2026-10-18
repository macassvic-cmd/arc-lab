# Dedicated high-threes model — locked rule

Committed 2026-10-18 (UTC) before any part of the new model was written or run. Written from the user's instruction of 2026-10-18;
nothing below may change after this commit. Goal: accurate P(k+) for k = 4, 5, 6, 7 threes. No prices anywhere. Output goes to the
Backtest page only (`docs/data/high_threes_backtest.json`), never to `data/paper_log.csv` or `docs/data/results.json`.

## Data and seasons

- Box scores: 2023-24, 2024-25 (`data/player_games_history.csv`, backfilled by `.github/workflows/backfill-history.yml` with the live
  fetcher's validation: regular season, completed, both competitors NBA teams, All-Star weekend rejected) and 2025-26
  (`data/player_games.csv`, the live file). The live model reads only the live file; the backtest concatenates both.
- **Fit all dispersion / shrinkage parameters on 2023-24 ONLY.**
- **PASS / FAIL gate on 2024-25 ONLY**, walk-forward: every projection for a date uses games strictly before that date (2023-24
  history counts as prior games; 2024-25 games before the date count; nothing after).
- **2025-26 is contaminated** (the usage bump, the 39-minute cap, the DvP clamp and structural choices of the current model were
  tuned on it): it is reported as a consistency check with the same frozen parameters, never as a gate.
- 2024-25 was used once before: the failed threes closing-line backtest (`backtests/nba_threes_2024_25_rule.md`, graded FAIL) that
  prompted this pivot. No 2024-25 results were used to design or tune the new model; the only 2024-25 numbers read before this
  commit are the step-2 baseline figures below, which set the bar and nothing else.

## Baselines

1. Current Arc Lab model: the walk-forward rows of `scripts/backtest.py --season 2025` (P(k+) columns `p_tpm_k`), scored on the
   same player-games.
2. Trailing-10 Poisson: Poisson with mean = the player's trailing-10 threes per game (prior games only; the rows' `naive_tpm`).

## Step 2 bar (current model, 2024-25, 18,682 player-games with a projection and minutes > 0; `scripts/high_threes_baseline.py`)

| | 4+ | 5+ | 6+ |
|---|---|---|---|
| Base rate | 12.3% | 5.7% | 2.5% |
| Log loss, current | 0.3127 [0.3043, 0.3218] | 0.1847 [0.1766, 0.1931] | 0.0998 [0.0928, 0.1069] |
| Log loss, trailing-10 Poisson | 0.3284 [0.3190, 0.3388] | 0.1962 [0.1868, 0.2062] | 0.1073 [0.0993, 0.1158] |
| Worst calibration gap at n >= 200, current | 5.0 pts (>=35% bucket) | 12.2 pts (>=35%) | 9.1 pts (20-35%) |

Top-decile capture of actual 5+ games, current model: **40.1%** [37.2%, 42.6%] (1,066 actual 5+ games; top decile = 1,871 rows).
Trailing-10 Poisson: 36.8% [33.9%, 39.8%]. The current model over-predicts the high buckets for every k (hit rates 3–12 points
under the predicted probability from 10% up).

## Gate on 2024-25 (all three required for PASS)

- (a) **Log loss** for 4+ and for 5+ is lower than both baselines (point estimates; the bootstrap 95% CI of each difference is
  reported).
- (b) **Calibration** within ±2.0 pp in every predicted-probability bucket with n >= 200, for 4+, 5+ and 6+. Buckets (fixed): <2%,
  2–5%, 5–10%, 10–20%, 20–35%, >=35% of the model's predicted P(k+). Gap = actual hit rate − mean predicted probability.
- (c) **Top-decile capture** of actual 5+ games >= **50.1%** (the current model's 40.1% + 10 pp, which exceeds the 35% floor).
  Top decile = the 10% of scored player-games with the highest predicted P(5+) that season (ties by rank).
- Every figure carries a 95% confidence interval: 1,000 game-cluster bootstrap resamples for log loss, log-loss differences and
  capture; normal approximation per calibration bucket. CIs are reported; the gate reads the point estimates as written above.
- Scored rows: every 2024-25 player-game with minutes > 0 for which the new model, the current model and the Poisson baseline all
  produce a probability (the intersection), so the three are compared on identical rows.
- **FAIL means report and stop. No tuning on 2024-25 or 2025-26.** A second attempt needs a new rule file.

## The model to build (step 4, fixed in advance)

- 3PA mean = projected minutes × 3PA per minute. Projected minutes: the current model's walk-forward minutes projection for the
  player-game (an input, not re-fitted). 3PA/min blends last 5, last 15 and season-to-date rates (prior games only; 2023-24 serves
  as the season-to-date start for early 2024-25) with empirical-Bayes shrinkage toward the position average; blend weights and the
  shrinkage strength are fit on 2023-24 only.
- 3PA distribution: negative binomial; dispersion fit per player group (groups by 3PA-rate tercile within position) on 2023-24.
- Makes | 3PA: beta-binomial with the player's 3P% shrunk toward the league / position rate; the concentration is fit on 2023-24.
- Multiplicative inputs on the 3PA mean, fit on 2023-24 by Poisson regression with log(minutes × rate) as offset: pace (team and
  opponent possessions per 48 from box-score totals, FGA − OREB + TOV + 0.44 FTA), home / away, rest (back-to-back flag, days of rest
  capped at 3), starters out (count of the team's usual starters absent; the usage bump applies to players who are starters that
  night only). Spread / blowout: no historical spreads exist in this repository, so that input is absent in the backtest and
  reported as such.
- Output: the full 0–12+ distribution per player per game (13+ folded into the top cell); P(k+) read from it.

## Reporting (step 5)

The new model next to both baselines, per season (2024-25 gate; 2025-26 consistency check; 2023-24 in-sample fit only, labelled),
with this rule's commit hash and timestamp, the three criteria each marked PASS / FAIL with its CI, and the overall verdict.
