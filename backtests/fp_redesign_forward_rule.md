# FP redesign, Builds 1 and 3: forward logger, integrity audit and the anchored gap price in shadow (pre-registration)

Committed 2026-11-11 (UTC; 2026-11-10 ET, before any clock slate was graded) under the CEO queue "ARC LAB REDESIGN — BUILDS 1–3".
Paper only. Nothing live changes: ladders keep their frozen rules, first basket is untouched, the FP v2 record stays closed exactly
as graded, and the projection sim's pricing weight is ZERO (logged as a zero-weight deviation column only). Config, set before this
registration: GATE_CALIBRATION = one-sided (CEO decision #1, 2026-11-10). Nothing in sections 1-7 changes after this commit except
by a dated addendum.

## 0. What this build did not have

The redesign folder (MASTER, claude_verdicts.md, spec DS-1, R4, R5) is not in this repository or on this machine; the definitions
below are written from the queue text and the data on file, and every place that fills a gap says so. Build 2's eight Stage-0
items are NOT registered here: their kill numbers must come from claude_verdicts.md, and registering substitutes would be
registering a different test. What Build 2 needs from this repository, the PrizePicks single-stat capture, is stood up (section 7).

## 1. Capture (Build 1)

- Underdog: every standard ("balanced") Fantasy Points line of every committed board copy (`data/board_latest.json`, the scraper
  hook's own `captured_at` / `source_updated_at` / `board_sha1`, about every 30 minutes), one row per line per copy, with both
  sides' prices and NBA.com's lineup status for the player as of that capture (starter:Confirmed / starter:Expected /
  bench:Active / bench:Inactive / absent / no_file) -> `data/fp_forward_ud.csv`, append-only. Written by `fp_forward.py capture`
  on every health pass (push-triggered by each board copy under Amendment 2) and in both update-data passes.
- PrizePicks: every Fantasy Score history row of the repo copy (`data/pp_fantasy_lines.json` is the scraper's append-only line
  history: a row when a projection first appears and again on every move, with the fetch time) -> `data/fp_forward_pp.csv`,
  deduplicated on (source_id, captured_at, line).
- Backfill: the same rows rebuilt once from git history for 2026-10-28 .. 2026-11-10 (flag backfill=1, lineup status
  not_captured_backfill). The slates before 2026-11-10 are PRE-CLOCK: they feed the power report and the expanding offset, never the gate.

## 2. The priced feature (frozen)

- PP MORNING pull (DS-1 as read from the data: the pull whose median capture is about 9 hours before tip) = the LAST repo copy
  of PrizePicks captured on the slate date inside 10:00-17:00 UTC (06:00-13:00 ET) that carries rows for the slate (set on
  pre-clock data, section 6: that copy lands 13:00-16:57Z, a median of about 9 h before tip, and pairs 95-100% of UD lines; the
  earliest window copy pairs only 20-40%). A line's morning value = its last history row captured at or before that copy. No copy
  in the window = no morning pull for that slate; the late pull is NEVER substituted (the gap arm has no priced line that day).
- UD line = the line at its first capture on or after the morning pull (the first moment both numbers exist); if a line's last
  pre-tip capture precedes the morning pull it is used as is. The last pre-tip UD line is a secondary column.
- gap = PP morning line - UD line. No PP pair = gap 0 and gap_missing = 1 (about 18% of UD lines in the queue's estimate; the
  observed share is reported per slate). The PP PRE-TIP line (last history row before tip) is logged as a secondary column only;
  it never enters a price (this is the R5 file-level bug: v2-era pricing ran off a late PP pull, median 39 minutes pre-tip; the
  v2-era UD lines on file were captured a median 115 minutes pre-tip and their PP pairs 28).

## 3. The shadow price (Build 3)

- P(over UD line) = clip(0.5 + beta x (gap - offset), 0.05, 0.95), beta = 0.034 per FP point, fixed at the queue's cap (the queue
  says "linear and capped at 0.034"; without the redesign folder's fitted slope the cap is registered as the slope, so the price
  can never exceed it). offset = mean gap over PRIOR slate dates' priced lines with a PP pair (expanding; no same-slate or future
  information; pre-clock slates count as prior). The hinge form and the per-bucket table are dead and are not implemented.
- Logged at zero weight on every record, never priced: dev_engine = FP v2's pre-capture mean minus the UD line (the projection
  sim; its log is read, never written), and the matchup column d_match = the mean DvP factor (pts / reb / ast) of the opponent
  at the player's listed position, with the three factors beside it (SHADOW ONLY: the position data under it is broken).
- One priced record per UD line (slate date, player, team) in `data/fp_forward_log.csv`, written once both inputs are on file;
  graded / void records are NEW rows with the same key (hit = box-score FP > line, FP = pts + 1.2 reb + 1.5 ast + 3 stl + 3 blk -
  tov; push, 0 minutes or absent from the box score = void). Corrections are new records; originals stay.

## 4. Integrity audit (Build 1)

Reported slate by slate in `docs/data/fp_forward.json` (the first five clock slates in full): UD FP lines captured / lines the
board history carries (bar >= 95%); PP morning pull present (bar >= 90% of slates; below it the gap arm dies by its own rule and NO
priced arm exists; the late pull is not substituted); graded lines whose hit re-derives from the box score (bar 100%).

## 5. Gate (Stage 3) and the reads

- Clock: UD lines on slates dated 2026-11-10 or later. n = 750 graded clock lines (CEO ruling 2026-11-10, cut from 1,500 before
  the clock started), ordered by tip then player. Not shown at 750 = FAIL. No re-bar, no extension.
- PASS requires BOTH: (a) paired gain = mean(log loss at 0.5 - log loss at P) >= +0.003 nats with the 95% CI from the two-way
  (player + date) cluster-robust SE excluding 0; (b) the calibration prong, one-sided: by confidence bucket of the priced side
  (0.50-0.55, 0.55-0.60, 0.60-0.65, 0.65-0.70, 0.70+), realized directional hit rate >= claimed - 0.05 in every bucket with
  n >= 50, and the n-weighted mean over-confidence <= 0.05. Under-confidence never fails.
- Futility look at n = 375 (halfway): point estimate of the gain <= 0 stops the arm. The stop can only stop, never pass; 750 is
  the only pass/fail read.
- Rolling-slope regime-decay monitor throughout: slope of (hit - 0.5) on (gap - offset) over the last 150 graded paired lines,
  per slate, next to the assumed 0.034.
- Money ledger, DESCRIPTIVE from the first slate: picks = sides with P >= 0.55 (|gap - offset| >= 1.47 FP), one unit each at
  Underdog's standard 2-leg payout (3x, break-even 57.7%, the ~55-58% band). Selected-pick hit rate < 52% at 400 picks = no stakes
  even on a gate PASS. A PASS certifies an edge over a coin-flip price, not profit, and is labelled that way wherever displayed.

## 6. Power report (recorded before the clock's first slate was graded)

From the pre-clock lines (slates 2026-10-28 .. 2026-11-09 rebuilt from the committed board and PrizePicks copies with this logger,
priced exactly as section 3 and graded from the box scores):

- Lines: 394 graded pre-clock UD lines with a PP morning pair (2026-10-28 .. 2026-11-09; 433 graded in all, 41 of 490 priced lines
  without a pair = 8.4%, 30 void). Over hit rate 0.508; mean |gap - offset| 1.191 FP.
- Per-line sd of the paired log-loss difference (flat 0.5 minus the gap price): **0.1047 nats**.
- Two-way (player + date) cluster-robust SE of the mean on these lines 0.00461 vs iid 0.00528 (design effect
  0.764; below 1 on this sample, so the iid figure is the conservative one for planning).
- Expected 95% CI half-width at n = 750: **+/-0.0066 nats** (iid-conservative: +/-0.0075).
- MINIMUM observed gain that passes at 750 (CI excluding 0 and >= the +0.003 bar): **+0.0066 nats**, about
  2.2x the bar. The honest n for detecting +0.003 at 80% power is about **7,299** lines.
- Pre-clock point estimate (information, never the gate): mean gain +0.00687 nats; empirical slope of (hit - 0.5) on
  (gap - offset) 0.0377 per FP point against the registered 0.034.
- Morning-pull definition check (why the last copy before 17:00Z): per pre-clock slate the earliest window copy (12:44-15:30Z) paired
  19-100% of UD lines (median 38%) at a median lead of 9.5-12.8 h; the last window copy (13:00-16:57Z) paired 48-100% (median 100%)
  at a median lead of 6.7-11.0 h, about 9 h on the typical slate, which is DS-1's description. Under the earliest-copy reading the
  same 213 paired lines gave sd 0.1133 and mean gain -0.0029; the definition was fixed on coverage and lead, not on that sign.

The registration carries these computed numbers; "not shown at 750 = FAIL" stands as a deliberate CEO decision with the power on
paper, not an accident of arithmetic.

## 7. Build 2 (Stage-0 remainder battery): what this build stood up

- The PrizePicks single-stat capture the story-(g) reconstruction test needs: the scraper's hook pre-step now also writes
  `board_snapshots/pp_stat_lines.jsonl` (Points, Rebounds, Assists, Steals, Blocked Shots, Turnovers, 3-PT Made, Pts+Rebs+Asts;
  standard odds; same history format, 2-day retention; off with PP_STAT_EXTRACT=0), and `scripts/underdog_lines.py` copies it into
  this repo as `data/pp_stat_lines.json` with every board copy. The extract takes effect at the next scraper start (not restarted
  by this build). The 10-slate count starts at the first copy that lands.
- The eight items' kill numbers, the (f)/(d)/(e)/(c)/(b) tests and R4's ambiguous-band rule are not registered here (section 0).
  Items 1-2 (story (g)) must land before the n = 375 futility look; the futility read is published either way and the report
  says whether (g) is on file.

## Addendum (2026-11-11): verified scraper downtime ("outage" lines)

CEO text, verbatim, received 2026-11-11 before any 2026-11-11 line was graded (first tip 2026-11-12T00:00Z):

> Addendum (2026-11-11): Lines priced without a morning pair because of verified scraper downtime are excluded from the
> morning-pair share and from graded forward-test results. They are logged separately and tagged 'outage.' This applies to
> today and to any future outage.

Sections 1-7 above are unchanged. How the addendum is applied (written before the code that applies it):

- **Verified downtime** = an entry in `data/fp_forward_outages.json` (id, start, end in UTC, evidence, who verified and when).
  Entries are added by hand with evidence from outside the test's own outcomes (machine event log, the gap in board commits);
  nothing in the code adds an entry. An entry may only be added before the lines it covers are graded.
- **An outage line** = a priced clock line with gap_missing = 1 on a slate that has NO PrizePicks morning pull (section 2) where
  a verified downtime window overlaps that slate's morning window (10:00-17:00 UTC on the slate date). A line with a missing pair
  on a slate whose morning pull exists is NOT an outage line (that is PrizePicks not carrying the player, section 2's ~18%).
- **Tag**: the original priced record stays; a NEW record with the same key, kind = `outage`, void = 1,
  void_reason = `outage:<id>` is appended to `data/fp_forward_log.csv`, and the same row goes to the separate
  `data/fp_forward_outage_log.csv`. An outage record is final: the line is never graded, never enters the gate's n, the
  futility look, the monitor, the ledger or the expanding offset (it had no pair, so it never fed the offset).
- **Morning-pair share** (section 4, bar >= 90% of slates): a slate whose morning pull is missing under a verified outage is
  left out of the numerator and the denominator. The line-level missing-pair share is likewise reported without outage lines.
  The figures as they stood before the exclusion are published next to the new ones (`*_before_outage_exclusion`).
- The gate's n = 750, the futility look at 375, "not shown at 750 = FAIL" and every bar are unchanged; outage lines simply do
  not count toward n.
- First entry: `2026-11-11-reboot`, 2026-11-11T08:59:40Z (last capture) to 2026-11-11T22:02:23Z (first capture after the
  restart). Evidence: Windows System event 1074 at 01:05 PT (TrustedInstaller restart for the 2026-11 security update); no
  board commit on main between 08:38Z and 22:02Z; both scraper processes absent until the restart. It covers the 43 Underdog FP
  lines of the 2026-11-11 slate, all priced at 22:02:58Z with gap 0 and no morning pull.

### Addendum text as finally worded (2026-11-11, received after the first wording above was committed; still before any 2026-11-11 line was graded)

> Addendum (2026-11-11): Lines priced without a PrizePicks morning pair because of verified scraper downtime are excluded from
> the morning-pair share and from graded forward-test results. They are logged separately and tagged 'outage.' This applies to
> today and to any future verified outage.

This wording governs; the application notes above are unchanged. Tagging condition restated from the CEO's instruction: ONLY
lines whose morning pair is genuinely missing because of the downtime are tagged; a line with a valid morning pair stays in the
test untouched. Capture inventory printed before tagging (2026-11-11 slate): PrizePicks copies inside 10:00-17:00Z = 0 in the
logger's store and 0 in the repo file (copies that day: 00:00Z .. 07:37Z, then 22:02Z; the 22:02Z copy was built from the
pre-reboot 08:59Z snapshot); Underdog captures of the slate = 1 (22:02:23Z, 43 lines). 20 of the 43 players carry a PrizePicks
row captured at 07:34Z, before the downtime and outside the window: under section 2 that is not a morning pair and it is not
substituted.
