# Arc Lab v2 — ladder engine and recalibration, locked rule

Committed 2026-10-20 (UTC) before any isotonic fit and before the first live projection is logged (season opens 2026-10-20 ET).
Written from the user's instruction of 2026-10-19 ("ARC LAB v2 — LADDER ENGINE + PERFORMANCE DASHBOARD"). Nothing below may change
after this commit. No prices anywhere: grading is hit / miss only. Live-season results are never used to tune anything.

## 1. Engine (what is projected)

- **Minutes**: the shipped minutes model unchanged (`build_model.py`: 39-minute cap, blowout shave, 20% vacated-minutes
  redistribution, usage bump for remaining starters only).
- **3PM, attempts first**: 3PA = projected minutes x 3PA-per-minute, where the per-minute rate is the equal-weight mean (1/3 each,
  fixed a priori, not fitted) of the player's 3PA/min over his last 5 games, his last 15 games and the current season to date
  (fewer than 5 current-season games: prior-season games fill the windows at weight 0.5, the engine's existing prior-season rate
  weight), then empirical-Bayes shrunk toward the league per-minute 3PA rate at the player's position: rate = (M x blend +
  K x position rate) / (M + K), M = the player's minutes inside his season-to-date window (prior-season minutes at weight 0.5 when
  they are used), K = 120 minutes (fixed a priori). Makes and P(3PM >= k) come from the existing simulation ladder on the Threes
  page (Poisson attempts at that rate, binomial makes at the shrunk 3P%); none of the Part B negative-binomial / beta-binomial
  machinery and no Part B fitted parameter is used anywhere.
- **PTS / AST / REB**: the engine's mean projection mu, predictive distribution Gaussian with continuity correction,
  P(X >= k) = 1 - Phi((k - 0.5 - mu) / sigma_stat). sigma_stat = the standard deviation of (projected mean - actual) over the
  2023-24 walk-forward rows of this engine (players who played), one number per stat, measured once and written into the code
  with its provenance. It is not a tuning knob.
- **Rungs**: 3PM >= 3, 4, 5, 6, 7; PTS >= 10, 15, 20, 25, 30; AST >= 4, 6, 8, 10; REB >= 6, 8, 10, 12.

## 2. Projection log (immutable)

- `data/ladder_log.csv`, append-only: date, event_id, tip (UTC), player_id, player, team, opp, pos, stat, rung, p_raw, p (the
  shipped probability: recalibrated where adopted under section 3, else raw), logged_at (UTC). Every projection run before tip
  appends its rows; nothing is ever edited or deleted. The **graded snapshot** of a (date, event, player, stat, rung) is the last
  row logged before that game's tip; earlier rows stay in the file.
- Grading, after the game, from the box score: hit = actual >= rung, binary. A logged player with no box-score minutes is graded
  `void` (did not play), counted and shown on the page, excluded from hit rates. Nothing else is filtered.
- Live-season data feeds the Performance page only; historical walk-forwards stay on the Backtest page, labelled backtest.

## 3. Recalibration (isotonic, gated)

- **Fit** (2023-24 only): for every stat and rung, isotonic regression (pool-adjacent-violators, monotone non-decreasing) of hit on
  p_raw over the 2023-24 walk-forward rows of the section-1 engine, players who played; outputs clipped to [0.005, 0.995]. The
  walk-forward is generated with `scripts/backtest.py --season 2024` from box scores before each game date (minutes, rates, DvP
  and absorber history rebuilt per date). No Part B parameter enters.
- **Gate** (2024-25 only, the engine's 2024-25 walk-forward, same script), decided **per stat**; a stat's maps ship only if both
  hold, otherwise that stat ships raw P(k+) and the page reports the fail:
  (a) mean log loss over the stat's rungs, recalibrated < raw, with a paired bootstrap over player-games (1,000 resamples, seed
      20261019) whose 95% interval for (raw - recalibrated) lies above 0;
  (b) worst calibration gap shrinks: over the stat's rungs and the ten probability deciles [0, .1), ..., [.9, 1], cells with
      n >= 200, max |mean predicted - hit rate| is smaller recalibrated than raw.
- The 2024-25 season is read once for this gate. 2025-26 walk-forward numbers, if shown, are consistency only.
- Nothing is refit on live results. A future refit of the maps needs a new committed rule.

## 4. First basket (parallel track, must not block opening night)

- Label: the first play with `scoringPlay` true in period 1 of the ESPN summary `plays` array; the scorer is its first participant
  athlete id. Backfilled for 2024-25 and 2025-26 to `data/first_basket_history.csv`; label coverage (games with a clean label /
  games) is reported before any model is built.
- Model (after coverage): P(player scores first) over the ten starters, from jump-ball matchup / tip-win rate, first-quarter usage
  and starter status; logged before tip, graded binary, calibration buckets, own section on the Performance page.

## 5. Decision ownership

Isotonic maps adopted or rejected by section 3 exactly as written; the engine of section 1 ships regardless (it is the product
being graded). Report every stat, pass or fail.
