# First basket METHOD layer: locked rule (2026-10-28)

Committed before any profile was fit and before the pre-ship check ran. A bolt-on conditional layer on the shipped first-basket
who-model: P(player, method) = P_model(player scores first) x P(method | player). Display and grading only. Paper study; no
staking claims, no prices, no edges, no betting language. Accuracy framing only, like the rest of the First Basket tab.

## 0. Hard constraints

- The who-model is untouched: its params (`data/first_basket_params.json`), attempt-share chain, tip model, log format and grading
  do not change. Its live record and its FAILED backtest gate stay as recorded. This layer reads the model's published per-game
  probabilities (the last pre-tip rows of `data/first_basket_log.csv`) read-only.
- The taxonomy and the shrinkage constants below are frozen at build. Any later change is a model change: a dated note in section 8
  of this file, in its own commit, and the pre-ship check (section 6) rerun before anything ships.
- Nothing in this layer is tuned on live results, ever.

## 1. Method taxonomy

8-way classifier on `first_basket_history.csv` (play_type, is_ft, points), applied in order, first match wins:

1. Free throw: is_ft = 1, or play_type contains "free throw"
2. Dunk: play_type contains "dunk"
3. Tip-in: contains "tip"
4. Layup: contains "layup" or "finger roll"
5. Floater: contains "float"
6. Hook: contains "hook"
7. 3-pointer: points = 3 (among the remaining plays)
8. Jumper (2pt): everything else

**Book buckets (amendment of 2026-10-28, the published, displayed and graded taxonomy):** five buckets mapped from the 8-way:
Dunk -> Dunk; Layup -> Layup; 3-pointer -> 3-Point; Free throw -> Free Throw; Tip-in + Hook + Floater + Jumper (2pt) -> Other
(a tip-in grades as Other, not Layup). The 8-way split is kept as a diagnostic (profiles file, grades CSV); it is not published on
the tab and is not graded.

Sanity targets, frozen history at build (all 2,518 first scores; 2,286 are field goals): a materially different mix is a classifier
bug to fix before shipping.

- 8-way, FG first baskets only (%): Dunk 11.5, Layup 25.2, Tip-in 1.9, Hook 4.2, Floater 8.5, Jumper 17.8, 3-pointer 31.0.
  Centers: Dunk 23.6, Layup 24.1, Hook 12.3, 3pt 16.1. Guards: 3pt 37.7, Dunk 4.8.
- Book buckets, all 2,518 first scores (%): league Dunk 10.4, Layup 22.8, 3-Point 28.1, Free Throw 9.2, Other 29.4.
  Centers: Dunk 21.7, Layup 22.1, 3-Point 14.8, Other 33.2. Guards: 3-Point 34.2, Dunk 4.3.

Position group = the last character of the position in `data/first_shot_starters.csv` (G / F / C); the scorer's position is his
starters-file position for that game.

## 2. Player profile P(method | player): pseudo-count pooling

Fitted from games strictly before the projection date only. Three components, summed as counts over the 8-way methods and
normalised:

- **Position prior: 20 pseudo-counts** distributed as the 8-way method mix of first scorers at the player's position group
  (first_basket_history rows, free throws included).
- **Attempt profile: up to 40 pseudo-counts** from his field-goal attempts in `data/first_shot_sequence.csv` in games he started,
  attempts before the game's first made field goal (the sequence's missed attempts; the made first basket itself is counted
  through own makes, not here). Map each attempt to the 8-way: `rim` -> Dunk / Layup / Tip-in / Hook by the position group's
  rim-bucket shares among its FG first baskets; `mid` -> Floater / Jumper by the position group's shares; `3` -> 3-pointer.
  Scale so the total attempt weight = min(n, 40), n = number of such attempts.
- **Own makes:** his first_basket_history first scores (8-way, free throws included) at face count.

No attempts and no makes = the pure position prior. The profiles are rebuilt each morning into
`data/first_basket_method_profiles.json` with a fit stamp and the component counts per player.

Reference (position + own makes only, prior 20, frozen history, book buckets): Victor Wembanyama Dunk 24.6, Layup 15.3,
3-Point 14.2, Free Throw 11.1, Other 34.9. The attempt component moves this somewhat; wildly different = bug.

## 3. Projection (per game, at snapshot time)

- Starters and P(first) = the game's last pre-tip rows of the who-model's log. Renormalise the ten P(first) to sum to 1.
- joint(player, method) = P(player) x P(method | player); game distribution = sum over the ten starters.
- Published in two forms: (a) over all first scores (Free Throw bucket in); (b) FG-only (Free Throw removed, the four remaining
  buckets renormalised): books settle method on the first field goal.
- New sibling file `docs/data/first_basket_method.json` keyed by event_id: both game distributions, the method splits for the
  model's top-3 picks (both forms, with each pick's profile coverage), the profile fit stamp. `first_basket_today.json` is not
  modified. Every projection is also appended to `data/first_basket_method_log.csv` (append-only) with the two baseline
  distributions of section 5 as of the same moment, so grading reads what was published before tip and nothing else.
- Starter coverage: the share of history first scores by a player who started (`scorer_started = 1`) is reported; under ~97% the
  residual is explained before shipping.

## 4. Display (First Basket tab)

Per game, one compact line under the picks: the top three book buckets with percentages, FG-only form, e.g.
"Method: Layup 28% · 3-Point 24% · Dunk 13%". For the top pick, his top three buckets in the why area. The page reads the
published JSON only; no re-derived maths in display code.

## 5. Grading (report-only, no gate)

In the existing first-basket grade step (isolated, continue-on-error, Discord on failure). For each graded game with a clean
label: the actual method = the classifier on the history row (8-way recorded for diagnostics; the book bucket is graded).
Game-level method log loss = -log P(actual bucket) for the model's form (a) distribution as logged before tip, against two
baselines logged at the same moment:

- **League mix, season to date:** the bucket mix of the current season's first scores strictly before the game date, with one
  pseudo-count per bucket (Laplace) so early-season baselines are finite.
- **Who-model x position-only profiles:** the same who-model probabilities with each starter's profile replaced by his position
  group's prior mix (section 2's prior, as of the same fit).

Probabilities are floored at 1e-4 before the log. Accumulated in `data/first_basket_method_grades.csv` (one row per graded game)
and summarised in `docs/data/first_basket_method_perf.json` (n, mean log loss per model, paired differences with a game-level
bootstrap 95% CI, predicted-vs-actual bucket shares). **No verdicts or threshold claims for at least the first 200 graded games;**
the page says "no read before 200 games" until then.

## 6. Pre-ship check (thresholds stated here, before running)

Walk-forward: profiles fit on the 2025 season only (history, sequence and starters rows with season = 2025, all phases); project
every 2026-season first score (1,231 games: 1,120 field goals + 111 free-throw rows) whose who-model record exists (the who-model's
own walk-forward code with the frozen constants, scored on the game's actual ten starters, exactly as its gate did). League-mix
baseline for the check = the 2025 season's bucket mix; position-only ablation = who-model x 2025 position priors. Score the
game-level book-bucket log loss of form (a).

Ship the full blend only if its mean log loss beats (a) the league-mix baseline by >= 0.010 nats AND (b) the position-only
ablation by >= 0.003 nats. Otherwise ship position-only profiles and say so plainly in the report. Verdict and all three numbers
go to `docs/data/first_basket_method_preship.json` (Backtest page) and are reported exactly as written here.

## 7. Report

Commits; the three pre-ship log losses and the verdict; the starter-coverage share; profile coverage for tonight's starters
(own makes / attempt data vs position-only); one worked example game (full distribution in both forms and the favourite's split).

## 8. Amendments

(None at commit. Dated sections only; each in its own commit; the pre-ship check reruns before any amended layer ships.)
