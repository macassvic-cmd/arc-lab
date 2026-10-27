# Fantasy Points v2 — Pre-Registered Rule

Status: DRAFT for CEO sign-off, 2026-10-27. Amended in draft the same day
per reviewer Claude's review, three rounds (market test added,
recalibration refit on v2 outputs, split-window grid selection,
per-source check, known limits logged; then per-source baselines, the
market test sized by power calculation at n = 800, sanity-slice
clarification, minutes truncation; then reproducible market-test sizing,
log-time replay inputs, and a one-row-per-line grader assertion), before
any commit. Once committed to the repo
(backtests/fantasy_points_v2_rule.md) this file is frozen. Any change
after forward collection starts is an amendment: it gets its own commit,
its own dated section at the bottom of this file, and it restarts the
forward gate.

## 1. Why v2 exists

FP v1 failed its pre-registered gate on 380 combined lines (weighted mean
|gap| 0.0922 vs <= 0.05; worst decile 0.1846 vs <= 0.10). The failure was
decomposed on the 416 graded player-games (Oct 23-25 slates):

- Projections ran +2.7 FP hot. Minutes projections explain ~1/3 of the
  bias (mean minutes error +0.95, projected high in 54% of games). The
  other ~2/3 is per-minute production running below the frozen 2024-25
  rate priors.
- 26% of player-games had minutes over-projected by 5+; those games'
  mean PIT is 0.165. When minutes land within +/-2, PIT is 0.448.
- At 650 lines the decile shape is stable: the 40-60% bands are
  calibrated (gaps -0.2 and +0.9 pts), the 30-40% band is backwards
  (said 36.1%, hit 52.4%), and the 60-70% band is the worst honest row
  (said 63.8%, hit 48.8%).
- B2B: no adjustment. B2B player-games graded PIT 0.478 with minutes
  error -0.83; the badge stays display-only in v2.

v2 is a calibration repair of v1. The joint-sim machinery, the FP formula
(PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TO), the sources, and the gate
math are unchanged.

## 2. What v2 changes: four items, nothing else

### 2a. Minutes distribution: widen, and add a short-night component
- The minutes standard deviation used by the sim is multiplied by a
  factor from the grid {1.25, 1.50, 1.75}.
- A mixture component is added to the minutes draw: with probability q
  from the grid {0.10, 0.15, 0.20}, the draw comes from a reduced-minutes
  component centered at 0.55 x projected minutes (same widened sd). This
  prices the observed cliff (early hooks, blowout pulls, rest) instead of
  smearing it across a single bell.
- Minutes draws from EITHER component are truncated at 0. The sim can
  never produce negative minutes.
- The (factor, q) pair is selected ONLY by the backtest gate in §4, from
  this 3x3 grid. No other values may be tried. Ties break toward the pair
  closest to v1 (smallest factor, then smallest q).

### 2b. Rate priors: unfreeze on a fixed schedule
- Per-minute stat rates are no longer frozen at 2024-25 priors. Each
  player's rate is a blend: rate = (1 - w) * prior + w * current-season,
  with w = GP / (GP + 10), where GP = the player's games played this
  season at log time. (10 games -> 50/50; 20 -> 67%; the prior never
  fully leaves.)
- The blend formula and the +10 constant are part of this rule. They are
  not tuned on the backtest or the forward record.

### 2c. Recalibration layer on P(over)
- An isotonic regression maps raw P(over) -> calibrated P(over). It is
  NOT fitted on v1's outputs: 2a and 2b change the raw outputs, and a map
  learned on v1's errors would correct for errors v2 no longer has.
  Instead, once the grid pair is selected, v2 (2a + 2b, selected pair,
  pre-tip inputs ONLY) is replayed retroactively over the already-graded
  v1 lines, and the map is fitted on those replayed v2 raw P(over) values
  against the recorded hits.
- The replay takes minutes projections, starters, and games played from
  the v1 log rows as recorded at capture time. It does NOT recompute
  projections from later data. No information from after log time may
  enter the map.
- One combined map, both sources (the gate is combined; splits are
  displayed). After mapping, each source's calibration gap is reported
  with a confidence interval. Source-specific maps are permitted only
  under a separate future rule, and only if one source's gap CI
  excludes 0.
- The fitted map is FROZEN for the entire forward gate. No refits during
  collection. The map ships as a data file in the repo with its fit date
  and fit n recorded.
- The FP distribution itself (percentiles, PIT) is still produced from
  the raw sim. The recalibration touches only the P(over) used against
  lines.

### 2d. Grader hygiene
- A logged player-game whose box score shows 0 minutes (or who does not
  appear) is VOID at grade time and leaves the pending set. No silent
  stuck pendings.

## 3. Fitting discipline

- Anything fitted (grid selection, the isotonic map) is fitted only on
  data that precedes the forward window: the backtest replay and the
  graded v1 record. Nothing from the v2 forward window may influence any
  v2 constant. That is the entire point of the gate.
- No other live-data changes are permitted under this rule. A bug that
  changes outputs is a model change: it follows the quality loop and
  restarts the forward gate if collection has begun.

## 4. Backtest gate (must pass before any forward collection)

- Replay v2 distributions (items 2a + 2b only; no lines exist
  historically, so no recalibration) over historical player-games NOT
  used to fit the 2024-25 priors' dispersion: the 2025-26 season replay
  already used by the ladder backtests, plus the graded v1 live record as
  a sanity slice. The v1 live record slice is reported for sanity ONLY
  and never counts toward the verdict (those same lines also fit the §2c
  map).
- SPLIT WINDOWS (the pair may not be selected and judged on the same
  data): the (factor, q) pair is SELECTED on the Oct-Jan slice of the
  replay; the verdict is rendered on the Feb-Apr slice only.
- Verdict on PIT deciles of realized FP against the v2 distribution, on
  the verification window: weighted mean |gap| <= 0.05 AND no decile
  share off by more than 0.10, the same bars v1 failed.
- All 9 grid pairs are reported on both windows, WITH mean FP bias per
  pair (the short-night mixture also lowers mean minutes; at q = 0.15 it
  cuts them ~7%, so its overlap with the rate blend must be visible).
  The selected pair is the one that passes verification with the
  tie-break in §2a. If no pair passes, v2 does not ship: the result goes
  in the failure log and the design returns to the CEO.

## 5. Forward gate

- v2 logs every standard UD and PP Fantasy Points line from the same
  pre-tip snapshots v1 uses, into its own log (data/fp_v2_log.csv),
  priced with the calibrated P(over) of §2c. v1 logging continues
  unchanged in parallel.
- ONE ROW PER LINE: the grader asserts uniqueness on (player, game,
  source, line) and grades only the last pre-tip snapshot of each line.
  The assertion runs on every grade; a violation fails the run loudly.
- CALIBRATION VERDICT at 200 combined graded lines, on the calibrated
  P(over) decile table: weighted mean |gap| <= 0.05 AND no decile
  |gap| > 0.10. Per-source (UD / PP) split tables are displayed; the
  verdict is combined, exactly as in the amended v1 rule.
- MARKET TEST (also required). A calibration pass alone proves nothing:
  on this record a model that said ~50% on every line would pass the
  decile bars while carrying zero information, because the books' lines
  sit near the median and the base hit rate is 50%. v2's calibrated
  P(over) must beat the book's baseline P(over) on log loss, paired per
  line; the confidence interval on the log-loss difference (player-game
  cluster bootstrap, the house standard) must exclude 0 in v2's favor.
  Reported combined and per source.
- BASELINE PER SOURCE: UD = the book's no-vig P(over) from both sides'
  captured prices (the v2 logger records BOTH sides, american/decimal,
  for every UD line at capture time). PP = 0.5. PrizePicks standard lines
  are fixed-payout pick'em with no two-sided price, so there is nothing
  to de-vig; the line stands as PP's median estimate and 0.5 is the
  honest baseline. Per-source reports name the baseline used, and the PP
  bar is understood to be the weaker one.
- MARKET TEST n: the calibration verdict renders at 200 lines; the market
  verdict renders at 800 lines, with collection continuing past 200 until
  both can render. Sizing, from a power calculation on the v1 graded
  record (2026-10-27): with probabilities recalibrated by the same
  isotonic procedure (split-sample: fit Oct 23-24, evaluated on 484 rows
  from Oct 25-26), the per-line paired log-loss difference vs the
  baseline has sd 0.116 nats. The independent-lines formula,
  n = ((z_{alpha/2} + z_beta) * sd / delta)^2 with delta = 0.02
  nats/line, alpha = 0.05 two-sided, power 0.80, gives ~264 lines. The
  differences cluster within player-games: the cluster-robust SE of the
  mean on the evaluation slice was 0.0091 vs 0.0053 unclustered (design
  effect ~3), and sizing from the cluster-robust SE gives ~790 lines.
  n = 800 is the cluster-adjusted figure, rounded up: deliberately
  conservative, about a week of collection at current line volume. (On
  v1's RAW probabilities the per-line sd is 0.276; the forward test
  grades calibrated probabilities, so the recalibrated variance is the
  right basis.)
- Note on the design effect: ~3 exceeds the theoretical maximum of 2 for
  at most 2 lines (UD + PP) per player-game, so the evaluation slice
  likely contained more than one snapshot row per line. Before forward
  collection, the power-calc slice is re-counted on unique lines (per
  the one-row-per-line rule above) and the unique count and recomputed
  design effect are reported. n = 800 stands either way.
- The raw-distribution PIT table is reported alongside as a diagnostic.
  It does not carry the verdict.

## 6. Consequences

- PASS requires BOTH the calibration bars (verdict at 200 lines) and the
  market test (verdict at 800 lines); the gate fully resolves only when
  both have rendered. Passing calibration while failing the market test
  means "calibrated but not useful": v2 remains a paper pricing study,
  and that exact verdict is what gets reported. A market verdict of "not
  shown" is the honest reading if the CI includes 0: the test detects an
  edge of 0.02 nats/line or larger; a smaller real edge is neither proven
  nor disproven by it (see §7).
- PASS (both): v2 becomes the priced FP product. Any staking conversation
  is a separate CEO decision, after the pass.
- FAIL: v2 remains a paper pricing study, like v1. No conclusions drawn
  from it, no quiet re-tunes. A v3 requires a new rule file and a new CEO
  commission.
- The FP tab labels v2 as a paper study until the forward gate resolves.

## 7. Known limits (logged, not acted on)

- The recalibration map is fitted on early-season data and frozen; the
  rate decay it partly corrects is an early-season effect and may have
  faded by midseason, leaving the map correcting a bias that is gone. No
  mid-gate remedy exists under this rule.
- The backtest gate is a screen (distribution calibration only; no
  historical lines exist). It is a minimum bar, not evidence of
  usefulness. The forward gate with the market test is the real test.
- The market test is powered for an edge of 0.02 nats/line. A smaller
  true edge would most likely render "not shown" at n = 800. That outcome
  means the edge is below the size this study can certify, not that no
  edge exists. Certifying a smaller edge is a larger study and a new
  rule.

## 8. Amendments

None at signing. Amendments append below this line with date and commit.
