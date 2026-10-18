# Tail program, test 3 — conditional shape mispricing at DraftKings, locked rule

Committed 2026-10-18 (UTC) before running. Written from the user's instruction of 2026-10-18; nothing below may change after this
commit. Output: `docs/data/shape_mispricing.json` and the Backtest page only; never `data/paper_log.csv` or
`docs/data/results.json`. Script: `scripts/shape_mispricing.py`. Prior from test 1: low (ladder in-between, residual × shape 0.10).

## Contamination declared

- Part A (alternate-threes calibration) read 2024-25 outcomes only as aggregates (bucket hit rates); it never split by player shape.
- Part B gated the high-threes model on 2024-25 but never saw this split; nothing from Part B is reused here.
- Test 1 used no outcomes. Every feature below uses prior games only (strictly before the game date).

## 1. Universe and score

- **Universe**: 2024-25 DraftKings tail rungs from the Part A closing pulls (`data/odds_alt_2025/`, `data/odds_close_2025/`): DK Over
  rungs with line >= 4.5 or >= 2 above the player's main line (Part A's tail definition and its main line: the modal two-sided
  line across books, ties lower), DK's de-vigged probability by the Part A method (implied(Over) / DK's two-sided main-line hold
  for that player, DK's game-median hold when the player has no two-sided DK main line), outcome = box-score threes made > line,
  DNP rungs dropped. Each rung is one (player-game, line).
- **Shape score**: the locked test-1 score exactly as implemented in `scripts/ladder_shape_diagnostic.py` (trailing-15 minutes CV
  and EB-shrunk 3PM variance-to-mean ratio with the 2023-24 method-of-moments prior, standardized over the scored 2024-25
  player-games, mean of the two z-scores; >= 10 prior games with minutes > 0). Rungs whose player-game has no score are excluded.
- **Starter-out flag** (decided pre-game, never the ex-post absorber): a team's usual starters are the players who started >= 50% of
  the team's previous 10 games (prior games only) and appeared within the last 21 days. If at least one usual starter is absent
  tonight (not in the box score; the only availability record in historical data, as in every earlier backtest), **every remaining
  usual starter of that team who plays** gets +1 SD of the shape score (SD over the scored player-games) added to his score. No one
  else is adjusted; the absent player is not in the universe (DNP).

## 2. One comparison

Terciles of the adjusted shape score over the universe rungs (cuts at the 33.3rd and 66.7th percentiles of the rung-level score).
**Top tercile vs bottom tercile**, nothing else.

## 3. Statistic

gap_T = mean(hit) − mean(DK de-vigged P) over top-tercile rungs; gap_B the same over bottom-tercile rungs; **D = gap_T − gap_B**.
95% CI by bootstrap clustered by player-game (resample player-games with replacement, 1,000 resamples, seed 20261018). Also reported:
each tercile's n, hits, hit rate, mean de-vigged P and gap with its own clustered CI; the middle tercile; the same by rung kind (5+
vs >= 2 above main) and with/without the starter-out adjustment, for information only.

## 4. Decision (locked)

- **PASS**: D >= +0.015 AND the 95% CI of D excludes 0 (lower bound > 0).
- **KILL**: anything else -> seeds A / B / C are dead everywhere, including Underdog.

## 5. 2025-26

Consistency check only, reported, never gated: the same computation on 2025-26 DraftKings closing rungs **if such snapshots exist in
the repository**. None have been pulled (Part A covered 2024-25 only); pulling them costs ~10 credits per game per market from the
shared Odds API plan (~24.6K for main + alternates over 1,231 games), which is not spent under this rule. The report states what
exists and what the check would cost.

Report and stop.
