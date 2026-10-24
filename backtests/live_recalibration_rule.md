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
