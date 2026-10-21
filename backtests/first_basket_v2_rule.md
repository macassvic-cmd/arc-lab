# First basket v2: attempt-based model, locked rule

Committed before any label beyond the v1 coverage table was built and before any model was fit. Nothing below changes after this
commit; results are reported exactly as this file says. Supersedes section 4 of `backtests/ladder_v2_rule.md` for the model
(the v1 coverage pipeline stays as shipped).

## 0. Target

The event graded is **the first made field goal of the game** (free throws excluded, the market convention). Its scorer is the
first participant athlete id of the first period-1 play with `shootingPlay`, `scoringPlay` and `scoreValue` >= 2. A game is
gradable when that id is a player in the game's box score. The v1 label (first scoring play of any kind) stays in
`data/first_basket_history.csv` and is not the graded event.

## 1. Labels (built from the raw ESPN play store, zero Odds API credits)

Scope: every 2024-25 and 2025-26 game, regular season and postseason (play-in included), plus the 2024, 2025 and 2026
preseasons. Postseason and preseason event ids come from the ESPN scoreboard with `seasontype` 3 / 1, day by day.

- `data/first_shot_history.csv`, one row per regular-season or postseason game: each team's first field-goal attempt (player,
  shot type, clock, made), the game's first attempt overall, the first made field goal (the target), jump-ball jumpers (home /
  away jumper ids), the possession gainer and the team that gained possession. Column `phase` = regular | postseason.
- `data/first_shot_preseason.csv`: the same columns for preseason games, kept as a separate table (small sample, current
  starters and scripts; it carries real weight in the model, see 2b).
- `data/first_shot_sequence.csv`: one row per field-goal attempt from the opening tip up to and including the first made field
  goal (game, attempt index, team, player, shot type, made, clock), all phases. The chain terms of section 2 are estimated from
  this file and nothing else.
- Shot type from the ESPN play: `3` when `pointsAttempted` is 3; otherwise `rim` when the play type contains layup, dunk, tip,
  putback, finger roll, alley oop or hook; otherwise `mid`.
- Coverage is reported per season and phase (games, games with a play file, gradable games, share) before the model runs.

## 2. Model (per game, over the ten starters)

2a. **Tip.** Each team's jumper is the starter who has jumped most often for that team in the labelled games to date (ties: the
starter at C, then the first listed starter). A jumper's win rate = share of his jumps where his team gained possession, shrunk
toward 0.5 with K_tip pseudo-jumps. The matchup prior is the log-odds combination p0 = wA(1-wB) / (wA(1-wB) + wB(1-wA)); the
head-to-head record of the same two jumpers is shrunk toward p0 with K_h2h pseudo-jumps. If a team has no jumper history, its
jumper's rate is 0.5. p_tip = P(home team gains possession).

2b. **Attempt share.** For starter i of team T, with f = games (as a starter, labelled, to date) in which i took T's first
field-goal attempt and n = such games played:
- (a) overall: share_a = (f + K_a * base_pos) / (n + K_a), base_pos = the position's mean share of first attempts among starters
  in the labelled games to date (fallback 0.2);
- (b) vs tonight's opponent, postseason included: share_b = (f_opp + K_b * share_a) / (n_opp + K_b);
- (c) current preseason (the preseason immediately before the season being scored): share_c = (f_pre + K_c * share_b) /
  (n_pre + K_c).
The team's five starters are normalised: s_i = share_c(i) / sum over the five starters. A departed starter therefore leaves his
share to the others pro-rata; a starter with no history enters at the position baseline.

2c. **Make rate.** m_i = sum over shot types of pi_i(type) * M(type): pi_i is i's shot-type mix on his first attempts to date,
shrunk toward his position's mix with K_type = 10 pseudo-attempts; M(type) is the league make rate of opening-sequence attempts
by type, estimated on the fit season.

2d. **Chain.** Attempt 1 is taken by the home team with probability p_tip * r + (1 - p_tip) * (1 - r), where r = P(the game's
first attempt belongs to the team that gained possession). An attempt by team T is taken by starter i with probability s_i and
made with probability m_i. After a miss the same team takes the next attempt with probability o (offensive-rebound continuation),
otherwise the other team does. The chain runs for N = 8 attempts; the mass left after attempt 8 is distributed in proportion to
the accumulated probabilities. r, o and M are league constants from `data/first_shot_sequence.csv` (fit season for validation;
the two complete seasons for the live model, frozen for the season). A bench share eps = share of fit-season games whose first
made field goal was scored by a non-starter is reserved; the ten starters share 1 - eps.

2e. **Output per starter:** P(first basket), P(first attempt) = P(attempt 1 by his team) * s_i, and `top_pick` = 1 for the
game's highest P(first basket). This is the format a price comparison consumes; no prices are used or shown.

2f. **Constants.** K_tip, K_h2h, K_a, K_b, K_c are chosen on the 2024-25 walk-forward (section 4) by coordinate search, two
passes from the start point (K_tip 20, K_h2h 5, K_a 20, K_b 5, K_c 2) over the grid K_tip in {5, 10, 20, 40}, K_h2h in
{2, 5, 10}, K_a in {5, 10, 20, 40}, K_b in {2, 5, 10, 20}, K_c in {1, 2, 4, 8}, minimising mean log loss. They are then frozen
and never touched again. K_type = 10 and N = 8 are fixed here, not searched.

## 3. v1 makes-only comparator

Same tip model (2a), same bench share, same normalisation and same walk-forward. P(team T scores first) = p_tip * r1 +
(1 - p_tip) * (1 - r1), r1 = P(the first made field goal belongs to the team that gained possession) on the fit season. Starter
i's share of his team's first made field goals when he started is shrunk toward the position baseline with K = 20 (fixed) and
normalised over the five starters. P_v1(i) = P(T scores first) * share(i) * (1 - eps). No attempts, no make rates.

## 4. Validation (fit 2024-25, grade 2025-26)

- Walk-forward: every quantity for a game uses only labelled games dated before it. The 2024-25 fit starts cold (position
  baselines) and accrues 2024-25 games plus the 2024 preseason. The 2025-26 grade accrues 2024-25 regular season and
  postseason, the 2025 preseason, and 2025-26 games before the date. Starters are the box-score starters.
- Graded set: 2025-26 regular-season games with a gradable target. Per game, each model's P over the ten starters (clipped at
  0.005) scores the actual scorer; a non-starter scorer scores eps for every model.
- Metrics, reported for v2, v1 and the uniform 1/10 model ((1 - eps) / 10 per starter): mean log loss, top-1 hit rate (the
  model's top pick scored first), top-3 hit rate.
- Decision: **v2 ships only if its mean log loss is lower than both the uniform model's and v1's, each with the paired bootstrap
  95% CI of the difference (1,000 resamples over games, seed 20261021) excluding 0.** Any fail = report the numbers and stop:
  no live log, no page section, no retry with other constants. Pass = sections 5 and 6 go live unchanged.
- Everything above is reported in `docs/data/first_basket_gate.json` and on the Backtest page.

## 5. Live log (only after a pass)

- Before tip, every run that snapshots the ladder log also appends the ten projected starters of every game not yet tipped to
  `data/first_basket_log.csv` (date, event, tip, team, opp, player, position, P(first basket), P(first attempt), top_pick,
  logged_at). Append-only; nothing edited or deleted after tip. The graded snapshot of a (date, event, player) is the last row
  logged before tip. Projected starters = per team, the five active players with the highest start share over their last five
  games, projected minutes breaking ties; the box score decides who actually started and the grading reports the match rate.
- Grade: hit = the player scored the game's first made field goal (section 0). Rows for a game whose target is not gradable are
  void (shown, not hidden). The model's top pick per game is the headline record; calibration buckets over every logged starter
  row, cells with n < 30 flagged thin; top-3 rate and mean log loss against the uniform model on the live games.
- No retroactive entries: the log starts with the first pre-tip snapshot after this ships. 2026-10-20 stays blank.
- No tuning on live results, ever. The constants of 2f and the chain terms of 2d change only at a season boundary, and only
  by re-running this rule's estimation on complete past seasons.

## 6. Performance page

Own section: headline record and hit rate of the top pick, mean P of the top pick, games graded, calibration table, top-3 rate,
log loss vs uniform, starter-projection match rate, and the line "Hit rate measures accuracy, not profit. Edge requires
comparing to prices." Live data only; walk-forwards stay on the Backtest page.
