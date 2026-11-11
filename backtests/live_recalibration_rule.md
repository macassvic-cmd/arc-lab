# Live recalibration of the v2 ladder probabilities: pre-commitment (locked rule)

Committed 2026-10-23 under the CEO directive "PERFORMANCE PAGE + LIVE RECALIBRATION PRE-COMMITMENT", with 4 graded game dates
in the log (2026-10-20 to 2026-10-23) and before any more games are graded. Nothing below changes after this commit except by a
dated addendum that leaves these sections untouched. This is the ONLY permitted change to the live ladder engine from live data;
everything else stays as shipped under `backtests/ladder_v2_rule.md`.

## 0. Definitions

- **Graded record**: the rows `ladder_log.py grade` grades (last pre-tip snapshot per key, hit = actual >= rung, void excluded).
  Nothing is re-derived elsewhere; every number below comes from that code path.
- **Day k**: the k-th distinct graded game date, day 1 = 2026-10-20. The clock is game dates, not calendar days.
- **The band**: graded rungs whose shipped probability `p` satisfies `0.55 <= p < 0.80`.
- **Gap**: mean shipped `p` minus hit rate over the band (positive = the model says more than it delivers).
- **Player-game cluster**: (date, event_id, player_id). Rungs of one player-game are correlated; every CI below is a cluster
  bootstrap over player-games (resample clusters with replacement, 2,000 resamples, seed 20261023, percentile 2.5/97.5).

## 1. Trigger (evaluated once, when day 21 is graded)

Recalibration is ALLOWED only if, over days 1-21, the band's hit rate is below its mean predicted probability with a 95%
cluster-bootstrap CI for the gap that excludes 0. The trigger is evaluated per stat (3PM, PTS, AST, REB) and for all stats
combined; a stat may be refit only if its own gap CI excludes 0 (the combined read is reported, never used alone). If the
trigger fails, nothing is refit and the report says so.

## 2. Refit (only if triggered)

- Isotonic regression (pool-adjacent-violators, the same `fit_map` as `scripts/ladder_calibrate.py`) of hit on the engine's
  **raw** probability `p_raw`, per triggered stat, fit on the graded rungs of days 1-21 only.
- Nothing else changes: no sigma, rate, minutes or usage constant, no feature, no re-weighting.

## 3. Validation (days 22-35, games the refit never saw) and the shipping decision

- For each refit stat, log loss of the refit map applied to `p_raw` vs log loss of the currently shipped `p`, on the graded
  rungs of days 22-35.
- SHIP the refit map for that stat only if the paired difference (current minus refit) is positive with a 95% CI that
  excludes 0, CI from the same cluster-by-player-game bootstrap of section 0. Otherwise the current calibration stays and the
  result is reported as written.
- Shipping = writing the map into `data/ladder_calibration.json` with the stat added to `adopted`, dated; the rows of days
  1-35 that produced it stay in the append-only log untouched. The decision table (per stat: gap, gap CI, triggered, log loss
  current vs refit, CI, decision) is published on the Backtest page.

## 4. Nothing else

- No other change to the live engine or its probability layer from live data is permitted while this rule runs. The model is
  left alone until the clock reaches day 35 and the section 3 decision is published.
- Diagnostics (for example the minutes-error breakdown of the band's gap) are report-only and never trigger a change.
- A refit that ships is itself a model change: its own dated entry in `data/ladder_calibration.json`, and any further
  recalibration needs a new rule file with a new clock.

## Addendum, 2026-10-24: count correction (sections 0-4 untouched)

At commit the log held 3 graded game dates (2026-10-20 to 2026-10-22); 2026-10-23 was logged but not yet graded. Day 1 =
2026-10-20 as written; the clock and every rule above are unchanged. First read at day 3 (report-only, published on the
Backtest page): band gap +15.2 pts, cluster CI [+9.2, +21.0]; the trigger is still evaluated only when day 21 is graded.

## Addendum, 2026-11-10: evaluation pulled to day 20 by CEO order (sections 0-4 untouched)

- The one-time trigger evaluation of section 1 was run on graded days 1-20 (2026-10-20 .. 2026-11-09, 20 distinct graded dates)
  instead of day 21, by CEO order of 2026-11-10. Recorded plainly: the order was given after the day-20 diagnostic block of
  `docs/data/performance.json` had been viewed (PTS gap +0.0727, CI [0.0493, 0.0981]; AST, REB and 3PM CIs include 0). The
  numbers used were recomputed at build time from `data/ladder_log.csv` by `scripts/ladder_recal.py evaluate` with the grader's
  own functions (band, gap, player-game cluster bootstrap, 2,000 resamples, seed 20261023); the diagnostic's values were not
  copied. Result: PTS gap +0.0727, CI [0.0483, 0.0977] -> REFIT; AST +0.0384 [-0.0005, 0.0810], REB +0.0392 [-0.0027, 0.0803],
  3PM -0.0189 [-0.1039, 0.0668] -> no refit; combined +0.0546 [0.0359, 0.0751] (reported, never used alone).
- The one-time evaluation is SPENT. There is no second evaluation under this rule; `ladder_recal.py evaluate` refuses to run
  again while `data/ladder_recal.json` exists.
- Refit as section 2, on days 1-20: isotonic regression (`ladder_calibrate.fit_map`) of hit on p_raw over the graded PTS rungs
  of days 1-20 (11,655 rungs, 2,331 player-games, 925 isotonic blocks). The map is versioned `recal_v1` in `data/ladder_recal.json` and logged in SHADOW as `p_recal` next
  to p_raw and the shipped p in `data/ladder_shadow_log.csv` from the first snapshot after this commit (2026-11-10 slate, logged
  00:02Z 2026-11-11, before that slate's first tip). Picks and display keep consuming the shipped p; p_raw stays in the log
  untouched.
- Ship condition UNCHANGED from section 3: validation on the 14 graded days after the fit window, now graded days 21-34 (the
  committed text said 22-35 after a 21-day fit; the window is the same length and still consists only of games the refit never
  saw). Per stat, the map ships only if the log loss of the map applied to p_raw beats the log loss of the currently shipped p
  on the graded rungs of days 21-34, paired difference with the 95% player-game cluster-bootstrap CI excluding 0. Otherwise the
  map is dropped for that stat and the result is written to `data/failure_log.jsonl`. No partial ship, no ship on the fit. The
  decision is computed once, when day 34 is graded, by `ladder_recal.py grade` in the morning grade pass; a ship writes the map
  into `data/ladder_calibration.json` (previous maps kept under `history`), dated.
- The validation clock, the interim table (information only) and the decision surface in the performance diagnostic block
  (`performance.json` -> diagnostic.recalibration, also `docs/data/ladder_recal.json`, rendered on the Backtest page) either way.
- Part B of the same CEO order (minutes uncertainty) is a separate engine version under its own rule file,
  `backtests/ladder_minutes_sim_rule.md`; it runs in shadow and may not ship before this rule's validation window closes.
