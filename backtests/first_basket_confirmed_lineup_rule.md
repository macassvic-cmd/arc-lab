# First-Basket Confirmed-Lineup Rule — ACCEPTED (2026-11-25)

Status: ACCEPTED 2026-11-25, with four review edits incorporated and accepted by the CEO (runner named in §2, "tip" defined in §2, no re-grading in §3, void exclusion + second read in §5), plus the push-and-landing clause in §2 added at review before build. Build via Claude Code paste. The build stamps the effective date in §3. Nothing in this rule changes the model.

Origin: the 2026-11-25 lineup-miss measurement (hidden_files/quant/2026-11-25_fb_lineup_miss_measurement.md, repo main commit 12b8a60): projected starters miss 0.8876 slots per game (229 / 258 games); the 80 games graded on confirmed lineups carried only 4 of those 229 misses; in all 19 scorer-miss games the scorer was in the confirmed five. Most of the misses are the model being fed the wrong five players. The correct five is already in the pipeline for many games.

## 1. Scope
Input and grading-order fix only. The model's parameters stay frozen; nothing is refit, recalibrated, or re-tuned. This rule covers first basket only.
Definitions:
- Confirmed lineup: the NBA.com daily-lineups Confirmed five for that game (the lab's lineup source since 2026-11-02; its Confirmed stamp lands a median of ~24 min before tip, measured).
- Confirmed row: a pre-tip log row built on a confirmed lineup.
- Projected row: any pre-tip log row not built on a confirmed lineup.

## 2. Watcher
- For every game on the slate, the watcher checks the lineup source at T-22, T-15, T-8, and T-3 minutes before tip.
- The watch list is built from the schedule itself — every slate game is on it by construction; there is no separate list to fall off of.
- When a Confirmed stamp is present, the watcher appends a confirmed row for that game.
- A snapshot stamped after tip is never graded.
"Tip" means the scheduled tip time. A row is pre-tip only if its log timestamp is before the scheduled tip.
Runner: the checks run on the CEO's PC via Windows Task Scheduler, every 2 minutes across the pre-tip window, pushing under the same account as the board scraper. GitHub Actions cron is NOT the runner — its measured start delay in this repo is 4–7 hours, which is what broke the Sam captures. Claude Code registers the task and reports it. If the PC is off or blocked, games grade on the projected row, tagged, and count against the 90% mark. The build report must show, for the first full slate after the build lands, the actual time of every check against its target; a check more than 3 minutes late counts as missed. That first slate is the acceptance check on this runner clause.
Push and landing: the log the watcher appends to lives on GitHub, so the watcher entry point appends AND pushes the confirmed row itself: fetch + rebase + retry, 3 attempts. If the push does not land, it exits non-zero and posts to the Discord webhook. A confirmed row counts for the pass mark only if it is on main with a log timestamp before the scheduled tip. (This repo has already had a push fail on a conflict while the run reported success — that is how the Oct 30 slate went missing. A row computed on time that never lands must not look fine until the 100-game read.) The build report confirms the watcher pushes under the same account as the board scraper. No token appears in any chat or log.

## 3. Grading order — dated amendment to the grading rule
Effective the first slate graded after the build lands (the build stamps the exact date here):

**EFFECTIVE DATE (stamped by the build): slate 2026-11-25.** The build landed on main on 2026-11-26 (UTC), after every game of the 2026-11-25 slate had tipped and before any of them was graded (all nine were still pending in `data/first_basket_grades.csv`); the next slate with games is 2026-11-27 (no NBA games on 2026-11-26). The 2026-11-25 slate is therefore the first slate graded after the build landed. Every game dated 2026-11-25 or later is graded under this section; every game dated 2026-11-24 or earlier is graded as before.

**Amendment note (CEO ruling, 2026-11-25): the coverage count starts 2026-11-27.** The effective grading date stays 2026-11-25: the games of that slate were graded under the new order, keep their era marker, and stand exactly as written (no re-grading, no row changes). They are outside the section 5 coverage count, because the watcher never covered that slate (it was registered after those games had tipped). The 100-game coverage read counts games dated 2026-11-27 or later, the first slate the watcher actually covered.

- The graded row for a game is the last pre-tip row built on a CONFIRMED lineup.
- If no confirmed row exists, the graded row is the last pre-tip projected row, and the grade row is tagged "projected".
- A confirmed row can never be superseded by a projected row, regardless of append order. (This fixes game 401909941, 2026-11-07, where a projected snapshot appended after the confirmed one restored the wrong five in grading order.)
- No game graded before the effective date is re-graded under this amendment — including 401909941 (2026-11-07). The amendment replaces the old "last pre-tip row" rule going forward only.

## 4. Era marker and the split record
- Every grade row from the fix date carries an era marker. Pre-fix rows stand exactly as recorded.
- Records are reported split by era. The 80 pre-fix confirmed-lineup games are not evidence the model is better — those games were not chosen at random.

## 5. Pass mark — coverage, not accuracy
Read once, at the first 100 graded games after the fix date. Games with a void grade do not count toward the 100. There is no option to re-read outside the second-read clause below.
- ≥ 90% of games graded on a confirmed row, AND
- ≤ 0.10 wrong starter slots per game on the confirmed-graded games.
Miss either mark: report it and fix the watcher. No model conclusion is drawn either way. Whether the model beats an even split cannot be settled this season and is explicitly not this test.
Second read: on a miss, once the watcher fix is reported, ONE further read is taken at the next 100 graded games, same marks. A second miss goes to the CEO as a decision — there is no third read.

## 6. Reporting
- v2 vs the strict 1/10 baseline on post-fix games is reported with the game-bootstrap CI, and never quoted without it.
- Coverage (the pass-mark numbers) is reported on the same read.

## 7. Delivery to chat
The fix lands ~24 minutes before tip, so the page only helps someone looking that late. When a confirmed lineup changes a game's first-basket picks, the corrected picks post to the Arc Lab chat at that time.

## 8. Out of scope
No parameter change, no refit, no change to the gate definitions or the baseline definitions from the 2026-11-25 baseline audit. The staleness alarm (health.py job 7) is a separate, already-approved paste.

---

## Build notes (2026-11-25 build; how the text above is implemented, nothing here changes it)

- **What marks a row as confirmed.** The log (`data/first_basket_log.csv`) has no lineup column and is not given one. A snapshot is a confirmed row when `data/first_basket_why.jsonl` carries a line for the same game and `logged_at` with `lineup_status: "confirmed"`. That line is written in the same append as the rows, by the watcher (`scripts/first_basket_watch.py`) and by the older health job 6 refresh (`scripts/first_basket_lineups.py`). Every other snapshot is a projected row.
- **Selection code.** `first_basket_model.graded_rows` is the one selector for grading and for the page's picks. Games dated before the effective date go through the old "last pre-tip snapshot" path untouched.
- **Era marker.** `data/first_basket_grades.csv` gains two columns: `era` (`confirmed_lineup_rule` from the effective date, blank before) and `graded_on` (`confirmed` or `projected` from the effective date, blank before). No pre-fix value changes.
- **Wrong starter slot** (§5): a graded-snapshot row whose player did not start (the existing "didn't start" void row), the same count the 2026-11-25 measurement used.
- **The 100** (§5): the first 100 games dated 2026-11-27 or later (amendment note in §3) with a grade, in order of scheduled tip (ties by event id); games without a clean label (void grade) are not counted.
- **Watcher checks.** Every check is logged with its target time and the actual time NBA.com answered, in `data/first_basket_watch_checks.csv` (append-only). A target the watcher never reached is logged as missed. A confirmed row is appended at the first check that sees the Confirmed stamp, and again at a later check only if the confirmed ten has changed since.
- **Discord.** No webhook is stored on the PC. The watcher posts through the `fb-watch-notify` workflow (dispatched with the board scraper's GitHub login), which uses the repo's existing Discord secret.
- **Not changed.** The METHOD layer (`first_basket_method.py`, its own rule and log) still reads the last pre-tip snapshot; health job 6 still runs.
