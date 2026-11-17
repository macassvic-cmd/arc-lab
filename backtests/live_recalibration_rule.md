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

## Addendum (2026-11-16): how the recal_v1 map was actually constructed, and its freeze

Written before the day-34 readout, at the CEO's instruction (order dated 2026-11-11). Nothing in this addendum changes the map,
the shadow log, the validation window or the ship condition. Every number below was recomputed read-only from the committed
files on 2026-11-16; `data/ladder_recal.json` was not written.

### Freeze

`data/ladder_recal.json` (recal_v1) is frozen as of 2026-11-11. SHA-256 of the file on main:
`049f22add58b737fd485b364c5cd9a4b6d2c1253ff3cc3e6f527db8d696da4bc`. It is not modified, regenerated or re-fitted before the
day-34 readout. `scripts/ladder_recal.py` now refuses to run its shadow and grade steps if the file's hash differs while the
validation is pending (the step fails and is reported), and the day-34 decision records the hash it found and a hash of each
map's arrays before it writes its decision into the file.

Verified 2026-11-16: all 9,755 `p_recal` values logged in `data/ladder_shadow_log.csv` (PTS rows, 2026-11-10 .. 2026-11-16)
equal the frozen map applied to the logged `p_raw`, to the fourth decimal.

### Construction, step by step (the code is `ladder_calibrate.fit_map` / `pav`, called once by `ladder_recal.py evaluate`)

1. **Rows.** Every graded PTS rung of graded days 1-20 (2026-10-20 .. 2026-11-09): 11,655 rows from 2,331 player-games.
   x = the engine's raw probability `p_raw`, y = hit (0 or 1). There are 2,815 distinct `p_raw` values among those rows.
2. **Order.** Rows are sorted by `p_raw` with a stable sort. Rows that share a `p_raw` are NOT pooled first: they stay in the
   graded frame's own order (date, game, player, rung) and enter the scan one at a time.
3. **Scan (pool-adjacent-violators).** Each row is appended as its own block; while the previous block's mean is STRICTLY
   greater than the new block's mean, the two are merged (n-weighted mean of y, n-weighted mean of x). Equal neighbours are
   not merged.
4. **Output.** One knot per surviving block: x = the block's mean `p_raw` rounded to 5 decimals, y = the block's hit rate
   clipped to [0.005, 0.995] and rounded to 5 decimals, n = rows in the block.
5. **Application.** `p_recal = clip(numpy.interp(p_raw, x, y), 0.005, 0.995)`: linear interpolation between consecutive
   knots, constant outside the first and last knot. It is not a step function.

### What "925 blocks" means

The file's `blocks: 925` counts KNOTS, not distinct fitted levels. Because equal neighbours are never merged, every all-miss
row at the bottom and every all-hit row at the top stays a single-row block:

- 833 knots (1,480 rows) form the low tail: raw 0.0000 to 0.0040, fitted 0 clipped to 0.005;
- 43 knots (43 rows) form the high tail: raw 0.9991 to 1.0000, fitted 1 clipped to 0.995;
- 49 interior knots, one per level, carry the other 10,132 rows (largest 646 rows).

So the map has **51 distinct levels** on 925 knots; 874 of the knots are single rows. Between two neighbouring levels the map
ramps linearly from the last knot of the lower level to the first knot of the upper one (50 ramps; the widest spans raw 0.2460
to 0.3125). Several knots share an x after rounding (874 knots across 25 x values); in every such case they also share y, so
the interpolation is unambiguous.

### How it differs from a textbook isotonic fit

A textbook isotonic regression on the same rows (rows with identical `p_raw` pooled first, tied neighbours merged) gives
**53 blocks**. Compared at probes 0.01, 0.02, ..., 0.99, the built map minus the textbook map ranges from -0.036 (at raw 0.61)
to +0.033 (at raw 0.89), mean absolute difference 0.0057; 19 of the 99 probes differ by more than 0.01. On the 9,755 logged
shadow rows the two differ by 0.0031 on average and 0.039 at most. In-sample log loss on the fit rows is 0.34149 for the built
map and 0.34147 for the textbook one.

The construction detail is therefore load-bearing: the unpooled ties and the strict merge rule decide where the block
boundaries fall, and the ramps between knots decide the value between them. The validation grades the logged `p_recal`, i.e.
THIS map, not a textbook refit of the rule text.

### The Quant's 2026-11-10 finding

The Quant reported 77 blocks from a pooled fit of the rule text and deviations of up to +0.050 at individual probes. I could
not reproduce those two figures from the committed data: my textbook fit gives 53 blocks and a largest deviation of 0.036.
The direction of the finding stands (the build's "925" is knots, and the built map is not the pooled fit); the exact figures
depend on inputs or a pooling convention I do not have. What is reproduced exactly: re-running `fit_map` on the day 1-20 PTS
rows returns the frozen x, y and n arrays value for value.

### Pre-registered readout diagnostic: extreme map outputs (registered 2026-11-16, REPORT ONLY)

Registered before the day-34 readout, at the CEO's instruction. It changes nothing: not the frozen map, not the validation
window, not the ship condition, and its result cannot convert a ship into a drop or a drop into a ship. It is written into the
decision record (`validation.diagnostic_extremes` in `data/ladder_recal.json`) at the same moment as the decision and is not
computed on outcomes before then.

Reported at the readout, per stat with a map (PTS):

1. the map's minimum and maximum output over raw probabilities 0 to 1;
2. the number of validation rows (graded rungs of days 21-34) whose mapped probability is above 0.95 or below 0.05, each count
   given separately, with the number of player-games they come from;
3. the validation log-loss difference (shipped minus mapped, the ship condition's own statistic, same player-game cluster
   bootstrap) three ways: on all validation rows, on the rows WITHOUT those extreme rows, and on the extreme rows alone.

Why: the map's tails are 876 single-row knots clipped to 0.005 and 0.995, so a large share of rows can be mapped to near-certain
values; this shows how much of any validation result rests on them.

Outcome-free facts as of 2026-11-16 (no hit or miss was read to produce these): the map's output ranges from 0.005 to 0.995.
Raw probabilities at or below 0.0855 map under 0.05; raw probabilities at or above 0.9904 map over 0.95. Of the 4,255 PTS shadow
rows logged so far that the grader would use (last pre-tip snapshot per rung, 2026-11-10 .. 2026-11-16), 1,730 are mapped below
0.05 and 81 above 0.95: 42.6% of rows. By rung, below 0.05: 15+ 107, 20+ 334, 25+ 578, 30+ 711; above 0.95: 10+ 63, 15+ 12,
20+ 6. The log-loss differences (item 3) need outcomes and are computed only at the readout.
