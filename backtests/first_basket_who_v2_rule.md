# First basket WHO-model v2: offensive-rebound continuation, locked rule

Committed 2026-10-29 BEFORE any v2 quantity was fitted and before the pre-ship check was run. Nothing below changes after this
commit; the check's numbers are reported exactly as this file says. Ordered by the CEO 2026-10-29.

**Naming.** The CEO's directive calls the live attempt-based model "v1" and the continuation model "v2". In the repo the live
model is `backtests/first_basket_v2_rule.md` / `first_basket_model.py` (array name `v2` in the code) and the code's `v1` is the
makes-only comparator of that rule's section 3. This file uses the directive's labels: **who-v1** = the live attempt-based model
(frozen `data/first_basket_params.json`, CEO override 2026-10-21, gate FAILED and standing), **who-v2** = the model below,
**makes-only** = the old section-3 comparator. Files of this version carry `who_v2` in their names so nothing of who-v1's record
is touched.

## 0. Diagnosis (data only; from `data/first_basket_history.csv` and `data/first_shot_sequence.csv`)

Putbacks and tips (play type containing "tip" or "putback") are 4.9% of 2024-25 and 3.8% of 2025-26 first made field goals
league-wide, but the main channel for a class of bigs: Jarrett Allen 5 of 21 FG first baskets since 2024-25 (24%), 12 of 56
starts scored first in 2025-26 (21.4%) while who-v1 priced him near 9%. After a missed attempt in the opening sequence, the next
attempt belongs to the SAME team 30.9% of the time over all labelled seasons (952 of 3,079); on the 2024-25 fit window 29.9%
(472 of 1,580).

## 1. How who-v1 handles a missed attempt (read from `first_basket_model.py`, `score_game`, section 2d of its rule)

The chain state is one number per team, P(the next attempt is team T's). Attempt 1 is the home team's with probability
p_tip*r + (1-p_tip)*(1-r). At every attempt, team T's attempt is taken by starter i with the FIRST-ATTEMPT share s_i (2b) and
made with his FIRST-ATTEMPT make rate m_i (2c); the team's miss mass is P(T attempts) * (1 - sum_i s_i m_i). After a miss the
same team takes the next attempt with the league constant **o = 0.2987** (fit window; frozen in `first_basket_params.json`),
otherwise the other team does; `o` is estimated as the share of opening-sequence attempts after the first (every one follows a
miss) taken by the same team as the previous attempt. The chain runs N = 8 attempts and the remaining mass is spread pro-rata.
So who-v1 ALREADY has a continuation probability. What it lacks is any difference between a continuation attempt and a fresh
possession: the continuation is shot by the same first-attempt shares (a guard-heavy mix) with the same first-attempt make rates.
Measured on the fit window, continuation attempts are rim 42.8% / three 35.4% / mid 21.8% and go in at 50.6%, against
32.4% / 39.8% / 27.8% and 43.6% for first attempts; by position (starters file) a centre's continuation attempt is at the rim
78% of the time, a wing's 39%, a guard's 27%.

## 2. The change (one structural addition; everything else of who-v1 is reused unchanged)

2a. **States.** The chain carries four masses: P(next attempt by home, fresh), P(away, fresh), P(home, continuation),
P(away, continuation). Attempt 1 is a fresh home attempt with who-v1's probability. A fresh attempt by team T is shot and made
exactly as in who-v1 (s_i, m_i, frozen K, frozen constants r, M, eps). After a MISSED attempt by team T (fresh or continuation
alike) the miss mass goes to T's continuation state with probability **p_cont** and to the other team's FRESH state with
1 - p_cont (possession passes as in who-v1). Continuations chain under the same rule. N = 8 attempts, pro-rata remainder and
the bench share eps as in who-v1.

2b. **Continuation attempt by team T.** Shooter i with probability c_i (2c), made with probability
mc_i = sum over types of pi_cont(pos_i, type) * M_cont(type) (2d). Put differently, P(i scores first on a continuation) uses a
continuation shooter distribution and continuation type/make tables, NOT the first-attempt shares and make rates.

2c. **Continuation shooter distribution.** For starter i of team T: k_i = continuation attempts (opening-sequence attempts
after the first, same team as the previous attempt) taken by i in labelled regular-season and postseason games he started;
n_i = continuation attempts by i's team in those games (his opportunities). Position prior base_cont(pos) = sum of k over all
starters at that position / sum of their n (per-starter share of the team's continuation attempts, pooled by position).
c_i = (k_i + 20 * base_cont(pos_i)) / (n_i + 20), prior 20 pseudo-attempts, normalised over the five starters. A position with
no history takes 0.2. Tallies accrue walk-forward exactly as who-v1's 2b(a) share tallies do (every labelled regular-season and
postseason game dated before the game scored; cold start at the prior); preseason games are not used here.

2d. **Continuation tables (league constants, fit window).** p_cont = share of opening-sequence attempts after the first taken
by the same team as the previous attempt (the same definition and window as who-v1's o). pi_cont(pos, type) = type mix
(rim / mid / 3) of continuation attempts taken by starters at that position; a position with fewer than 20 continuation attempts
in the window uses the league continuation mix. M_cont(type) = make rate of continuation attempts by type. All from
`data/first_shot_sequence.csv` restricted to the gradable games of the window, regular season + postseason, as who-v1's
constants are. Continuation attempts by non-starters (0.2%) are dropped from the tables. No hand-set values: the only fixed
numbers are the prior 20 (the directive's), the 20-attempt position floor, and who-v1's N = 8.

2e. **Positions** come from `data/first_shot_starters.csv` (box-score starters, `common.norm_pos`), never from the sequence
file's `pos` column (corrupt: PG appears 6 times in 2,891 first attempts).

2f. **Unchanged:** the tip model, first-attempt shares and make rates, the frozen K and constants, FT handling (free throws are
not attempts; the target stays the first made field goal), the bench share, the page, who-v1's frozen parameter file and its
record. P(first attempt) keeps its who-v1 definition.

## 3. Pre-ship check (bars fixed here)

- Fit: who-v2's constants (2d) on the 2024-25 season only (season 2025, regular season + postseason, gradable games). who-v1's
  projections are re-derived on the same fit: frozen K from `data/first_basket_params.json` and constants from the same 2025
  window (number for number the gate's fit block). The walk-forward tallies (who-v1's shares, who-v2's 2c) accrue through the
  evaluation season as the gate harness does; the makes-only comparator rides along.
- Evaluation set: the gate harness's 2025-26 regular-season games with a gradable first made field goal and a full ten-starter
  box score (`first_basket_model.walk(games, st, {2026})`; the gate graded 1,230 such games, the clean 2025-26 label count is
  1,231). Per game, every model's P over the ten starters, clipped at 0.005, scores the actual scorer; a non-starter scorer scores
  eps for every model. who-v1's log loss on this exact set is reported.
- **BAR 1:** mean per-game log loss of who-v2 is lower than who-v1's by at least 0.002 nats AND the paired game-bootstrap 95%
  CI of the difference (1,000 resamples over games, seed 20261029) excludes 0.
- **BAR 2:** the putback class, pre-specified from the FIT season only (players with >= 10 FG first baskets in 2024-25 and a
  putback/tip share >= 10% of them, `data/first_basket_history.csv` play type): **Alperen Sengun (4871144; 3 of 18) and Evan
  Mobley (4432158; 2 of 13)**. The directive expected Allen and Wembanyama as well: in the fit season Allen has 8 FG first
  baskets (2 putbacks) and Wembanyama 6 (1), both under the 10-basket floor, so by the stated criteria they are NOT in the class;
  their lines are reported alongside (report-only). For each class member, over the evaluation-set games he started: mean model
  P and his actual rate (first FGs / starts); the bar is the class mean of |mean P - actual rate| being strictly smaller under
  who-v2 than under who-v1.
- Report-only: makes-only log loss on the set; top-1 and top-3 accuracy who-v1 vs who-v2; Allen's line (starts, actual rate,
  mean P who-v1, mean P who-v2), Wembanyama's likewise; the fitted continuation quantities with their fit-window n's; the
  bar-2 class members' lines.
- Sanity stop: if the fit-window p_cont is far from 0.31, stop and report before anything else runs (it is 0.2987, 472 of
  1,580: not far).
- Output: `docs/data/first_basket_who_v2_preship.json`, rendered on the Backtest page.

## 4. Ship rule

- Both bars pass -> who-v2 goes live: the pre-tip snapshot prices the ten starters with who-v2 (same nightly windows, same
  log, same columns; `top_pick` from who-v2), using the check's fit-2025 constants unchanged plus the walk-forward tallies
  through yesterday, exactly as who-v1 runs today. `data/first_basket_who_v2_params.json` records the constants, the check
  commit and `live_from` (UTC); it is written once and never edited. Grade rows gain an era marker derived from `logged_at`
  against `live_from` (`who-v1` before, `who-v2` at or after), in `data/first_basket_grades.csv`, the performance JSON (per-era
  record, hit rate and log loss) and the Performance page. who-v1's record, including its failed gate and the override
  addendum, stands unedited; the log is append-only as before.
- Either bar fails -> who-v1 stays live, the numbers go in the record (the preship JSON and the Backtest page) and an override
  is the CEO's explicit call only.
- No who-v2 parameter changes after the check is seen. A changed who-v2 is a new version with a new rule file and a new check.

## 5. Report

who-v1's miss handling as found in the code (section 1); commits; fitted continuation quantities with fit-window n's; both bars
with CIs; report-only numbers; the ship decision and what changed.

## 6. Amendments

None. Dated amendments only, each its own commit; the check is never rerun under this file.
