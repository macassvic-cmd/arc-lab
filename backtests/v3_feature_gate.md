# Arc Lab v3 — feature gate, locked rule

Committed 2026-10-20 (UTC) before any v3 feature is fitted. Written from the user's instruction of 2026-10-19 ("ARC LAB v3 — DATA
FOUNDATION"). Nothing below may change after this commit. Coverage reports precede modelling; a source that proves unusable is
reported in numbers and its families are not fitted.

## Baseline

The v2 ladder engine exactly as shipped for 2026-10-20 (`backtests/ladder_v2_rule.md`, section 1), run walk-forward by
`scripts/backtest.py` on 2023-24 and 2024-25 from box scores before each game date. v2 keeps logging live throughout; nothing
reaches the live model except through this gate.

## Feature families (one at a time, in this order)

(a) stint / rotation minutes model (replaces trailing-minutes); (b) lineup-context rates (with-whom adjustments); (c) touches / role
features for PTS and AST; (d) shot-profile features for 3PM; (e) rebound-chance features for REB; (f) game context (rest / travel /
pace matchup / altitude / referee crew). Each family is tested against the **then-current baseline** (v2 plus every family already
adopted), so every shipped combination has itself been tested; no family is stacked untested. Inputs are prior games only (strictly
before the game date), from `data/stints.csv`, `data/lineups.csv`, `data/tracking_game_logs.csv`, `data/game_context.csv`.

## Fit and gate

- Every parameter of a family is fitted on **2023-24 only** (fitting method stated in the family's report).
- The gate is read on **2024-25 only**, walk-forward, players who played. 2025-26 is reported as consistency only, never gated.
- Metrics, per family, by what it touches: ladders -> mean log loss over the v2 rungs of each affected stat (3PM: 3-7; PTS:
  10-30; AST: 4-10; REB: 6-12); means -> MAE of the projected mean for each affected stat (and minutes MAE for family (a)).
  Affected stats: (a) minutes and all four ladders; (b) all four; (c) PTS, AST; (d) 3PM; (e) REB; (f) all four.
- **Ships** only if its primary metric (ladder log loss for (a), (b), (d), (e); MAE and log loss both for (c) and (f) on their
  stats) improves with a paired bootstrap over player-games (1,000 resamples, seed 20261019) whose 95% interval for
  (baseline - candidate) lies above 0, and no other reported metric worsens with an interval that excludes 0.
- Every family is reported, pass or fail, on the Backtest page with its n, point estimates and intervals.
- No tuning on live results, ever. Adopted families reach the live engine only after this report is published.
