# Ladder minutes-uncertainty engine (Part B of the 2026-11-10 ladder fix): pre-registration, locked rule

Committed 2026-11-11 (UTC; 2026-11-10 ET, before any 2026-11-10 game is graded) under the CEO directive "LADDER FIX — PART B:
MINUTES UNCERTAINTY". Shadow only: nothing below changes a shipped probability until section 5 says SHIP and the Part A freeze
(section 6) has lifted. Nothing in sections 0-6 changes after this commit except by a dated addendum.

## 0. Diagnosis on file (why)

The day-20 diagnostic (`docs/data/performance.json`, graded days 1-20 = 2026-10-20 .. 2026-11-09): in the 0.55-0.80 band the
gap is -0.02 when projected minutes land within +/-2 of actual, +0.295 at +5 to +10 minutes over projection and +0.423 at +10 or
more (promised ~66%, delivered ~24%). The live sim prices PTS / AST / REB off the projected mean with one fixed sigma and treats
the minutes input as near-certain; when minutes fall short every rung price is wrong. That diagnosis used graded data through
day 20, which includes Nov 3-9, so Nov 3-9 can never grade this build.

## 1. Minutes-error distribution (fit once, on these rows only)

- Rows: the last pre-tip row of `data/fp_log.csv` per player-game (projected minutes, the same build as the ladder snapshot)
  joined to the box score in `data/player_games.csv`, dates **2026-10-23 .. 2026-11-09 only** (the log starts 2026-10-23;
  2026-11-09 is graded day 20), players who played (voids are not graded), projected minutes >= 4 (the engine's rotation floor).
- Error = projected minutes minus actual minutes (positive = played less than projected).
- Split: **starter / bench**, starter = started at least half of the last 5 box scores before the date (`build_model`'s
  starter_rate, pre-tip information only); a player with no earlier box score uses the pooled distribution.
- Representation: the 50-point equal-probability quantile grid of each group's errors (q = (i + 0.5) / 50). Written once to
  `data/ladder_minsim.json` (version `minsim_v1`) with n, mean, sd and the tail shares; the script refuses to refit.

## 2. The engine change (the only change)

For every player, stat and rung the probability is the mean over the group's error grid e of the rung probability at minutes
m(e) = clip(projected minutes - e, 0, 48), with the projected per-minute production held fixed:

- PTS / AST / REB: P(X >= k | e) = 1 - Phi((k - 0.5 - mu(e)) / sigma_cond), mu(e) = projected mean x m(e) / projected minutes.
  sigma_cond = the minutes-conditional residual SD per stat, measured once on the 2023-24 walk-forward rows the shipped sigma was
  measured on (`data/v2_rows_2024.csv.gz`, players who played, projected minutes >= 4): SD of (projected mean x actual minutes /
  projected minutes - actual). The shipped sigma is the unconditional residual of the same rows; convolving the minutes error on
  top of it would count the minutes variance twice. Measured: pts 5.4762, ast 1.8564, reb 2.3064 (unconditional on the same
  rows 6.5024 / 2.0178 / 2.6507).
- 3PM: attempts Poisson at the projected 3PA per minute x m(e), makes binomial at the projected make probability, so
  P(3PM >= k | e) is the upper tail of Poisson(lambda(e) x make_p). The empirical error grid replaces the sim's Normal(projected,
  sd_min) minutes draw; attempts and makes are the engine's.
- Then the convolved raw probability goes through the probability layer the engine ships at snapshot time (the adopted isotonic
  maps in `data/ladder_calibration.json`, applied as `build_model.apply_map` applies them) exactly as the fixed-sigma raw does
  today. That mapped value, `p_minsim_cal`, is the Part B candidate; the unmapped `p_minsim` is logged as information.
- Nothing else: rates, priors, the rate blend, DvP, the usage bump, the minutes projection, the adopted maps and the grading code
  are untouched.

## 3. Shadow log

`data/ladder_shadow_log.csv`, append-only, written by `scripts/ladder_recal.py shadow` right after every ladder snapshot (both data
workflows and the news re-projection): date, event_id, tip, player_id, stat, rung, p_raw, shipped p, p_recal (Part A), p_minsim,
p_minsim_cal, cal_applied, projected minutes, group, versions, the build's stamp, logged_at, backfill flag. Pre-tip rows only;
the evaluated row of a key is the last one logged before tip, the grader's rule. Picks and display never read this file.

## 4. Evaluation (pre-registered, read once)

- Rows: graded rungs with shipped P in [0.55, 0.80) dated **2026-11-10 or later** that carry a last-pre-tip shadow row, ordered
  by date, tip, event_id, player_id, stat, rung. The 2026-11-10 rows were logged live at 00:02Z on 2026-11-11, before that
  slate's first tip (00:30Z); no backfill was needed.
- **n = 2,000** band rows, written here before the window opened (registration stamp in `data/ladder_minsim.json`). Read ONCE
  when the n-th row is graded; the numbers on those n rows are the evaluation, whatever comes after. Interim numbers are published
  as information and are never the decision.
- Power: about 143 band rungs grade per day (2,854 over days 1-20), so n = 2,000 is about 14 graded days, the length of Part A's
  validation window. With a per-rung paired log-loss sd near 0.08 and a design effect near 1.7 (rungs per player-game) the 95%
  half-width at n = 2,000 is about 0.0045 nats; removing a +0.07 gap on half the band (PTS) is worth about +0.006 nats pooled.
- Compared against the shipped p as logged at the time: today the raw engine through the 2024-25 maps; if a Part A map ships at
  graded day 34, the rows from then on carry it, so Part B is judged against the then-current engine and never stacked blindly
  on top of a map it was not evaluated with.

## 5. Ship condition (both, pooled over the four stats)

1. Band log loss of p_minsim_cal < band log loss of the shipped p on the n rows, paired difference (shipped - minsim) > 0 with the
   95% cluster-bootstrap CI by player-game (2,000 resamples, seed 20261023, the grader's code) excluding 0.
2. |mean p_minsim_cal - hit rate| < |mean shipped p - hit rate| on the same rows (the band gap moves toward 0).

One engine change, one decision: per-stat numbers are reported, never used to ship a subset. PASS = ship as its own engine version
(`minsim_v1`) with the fit / evaluation split recorded, a new rule clock for anything after it. FAIL = stays in shadow, the result
is written to `data/failure_log.jsonl` and reported; no retune, no second n.

## 6. Freeze

Part B may not ship before Part A's validation window (graded days 21-34 under `backtests/live_recalibration_rule.md`) closes; a
read at n before then is recorded and held, never re-read. Parts A and B grade on overlapping days, and a mid-window Part B ship
would change the raw P that Part A is being validated against. If Part A's map has shipped by the time Part B ships, that map is
refit or dropped on Part B's evaluation rows, never stacked blindly on top.
