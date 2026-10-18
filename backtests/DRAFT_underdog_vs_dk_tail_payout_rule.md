# DRAFT — Underdog vs DraftKings tail payout comparison (locked rule, for review; not run)

Drafted 2026-10-18 after the 2024-25 alternate-threes calibration test. Nothing here has been run. Once approved it is committed
with a timestamp before any capture starts and cannot change afterwards. Output: Backtest page only, never `data/paper_log.csv`
or `docs/data/results.json`.

## Question

For the same player, game and threes rung, does an Underdog higher/lower leg at the tail pay more or less than DraftKings' alternate
rung, in expected-return terms, after both are measured against the same outcome? The calibration test showed DK alt tail rungs
carry heavy margin; this test asks whether Underdog's leg multipliers carry less.

## Definitions (fixed)

- **Capture window.** Live captures during 2025-26 regular-season game days: Underdog pick'em board (`scripts/underdog_lines.py`)
  and The Odds API `player_threes_alternate` for DraftKings (region us, bookmaker filter draftkings), both logged with the capture
  timestamp. A pair is valid only when the two captures are within **10 minutes** of each other and both precede tip by at least
  30 minutes. The later capture's timestamp is the pair time.
- **Tail rung.** Threes line >= 4.5 (5+ made), or >= 2.0 above the player's DraftKings main line in the same capture, on the Higher /
  Over side. (Same definition as the calibration test.)
- **Matching.** Same player (`common.norm_name`), same game, same line. Underdog rungs with no DraftKings quote at that exact line
  (or vice versa) are not pairs and are counted separately.
- **Underdog leg multiplier -> implied probability.** Underdog pays a stated multiplier `m` per leg in its "Higher/Lower" payout
  table (Standard entries: non-correlated legs multiply; a leg at `m` means an entry of `L` legs pays the product of the leg
  multipliers). The implied probability of the leg is `p_UD = 1 / m` (the probability at which the leg returns stake, fee-free).
  For legs priced through a flat entry payout (e.g. 4 legs pay 12x with no per-leg multiplier), the per-leg multiplier is
  `m = 12 ^ (1/4) = 1.861` and `p_UD = 0.537`; the rule uses the per-leg multiplier the app shows for that leg when it shows one
  (`boost` / `scorcher` legs excluded).
- **DraftKings implied probability.** `p_DK = implied(American price)` for the Over at the rung, raw (with margin). The DK main-line
  hold `h_DK` is recorded alongside, so a de-vigged `p_DK / h_DK` can also be shown, but the comparison below is on raw prices,
  because the bettor pays the raw price at both venues.
- **Expected return per unit staked.** With the realised outcome `y` (threes made > line) from the ESPN box score:
  `R_DK = y * (decimal(DK price) - 1) - (1 - y)` and `R_UD = y * (m - 1) - (1 - y)`. Over pairs, `ER_DK = mean(R_DK)`,
  `ER_UD = mean(R_UD)`; the difference `D = ER_UD - ER_DK` with a cluster-robust 95% CI (clusters = game).
  Also reported: the price-only gap `p_DK - p_UD` (positive = Underdog pays more at the same rung), the realised hit rate against each
  implied probability, and the same by line and by bucket of `p_DK`.
- **Correlation caveat.** Underdog Standard entries require legs from different games to be non-correlated and pay a flat table;
  the leg multiplier comparison is therefore "per leg as if independent". Entries that mix a tail leg with other legs are out of
  scope.

## Sample and pass bar (fixed)

- **n >= 400 valid tail pairs**, spread over **>= 20 game dates** and **>= 60 distinct players**, before any grade is read. Pairs are
  collected until the first date on which all three minimums are met, then the capture stops for grading (further pairs are a
  separate, pre-announced extension).
- **PASS ("Underdog pays more at the tail")**: `D >= +0.05` per unit (five cents per dollar) with the 95% CI excluding 0, and
  `D >= +0.05` still holds after dropping the best game date by `D` (dates with >= 15 pairs).
- **FAIL**: anything else. In particular, a positive `D` whose CI includes 0 is "no evidence", not a pass.
- Secondary (reported, not graded): `p_DK - p_UD` mean and share of pairs with `p_UD < p_DK`; the de-vigged DK comparison; results on
  non-tail pairs for context.
- No model probabilities enter this test; it compares two venues' prices against the same outcomes. A pass does not by itself
  authorise bets: it feeds the venue-ranking decision.

## Costs and logistics

- DraftKings captures: one Odds API event call per game at capture time with `markets=player_threes_alternate&bookmakers=draftkings`
  (1 credit per event when a single bookmaker is requested; 10 if the full region is pulled). Budget: 5,000 credits, stop and report
  past it. Underdog captures: the existing scraper, no cost.
- Capture cadence: every 30 minutes from 4 hours before the first tip to the last tip of the day; the 10-minute pairing window is
  checked on timestamps, never assumed from the schedule.

## Open points for review

1. Per-leg multiplier source: the app's displayed leg multiplier vs the flat Standard payout table. The draft uses the displayed leg
   multiplier when present, the flat table otherwise.
2. Whether to require the DK quote to be the *best* US book or DK specifically (the draft says DK specifically, matching the
   calibration test).
3. Pass bar magnitude: +0.05 per unit is arbitrary but large enough to matter after Underdog's correlation rules; raise or lower
   before locking.
