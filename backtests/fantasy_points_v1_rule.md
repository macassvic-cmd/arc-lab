# Fantasy Points v1: joint simulation, Underdog pricing study, grading (locked rule)

Committed 2026-10-22 under the CEO directive "SCOPE RESET + FANTASY POINTS BUILD" before any constant was estimated or any row
logged. Nothing below changes after this commit except by a dated addendum that leaves these sections untouched. Products in
scope after the reset: Fantasy Point lines, 3s, First Basket, PTS/AST/REB ladders. PRA and standalone Underdog pricing of
individual PTS/REB/AST lines are cut everywhere.

## 0. Target

Underdog NBA scoring, verified current by the directive: `FP = PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - 1*TOV`, no
double-double or triple-double bonus. Actual FP comes from the ESPN box score in `data/player_games.csv`.

## 1. Marginals (per player-game, given minutes)

Minutes are the shared driver: one draw per simulation from the shipped minutes model (`build_model`: Normal(min, sd_min)
clipped to 0-48, after the allocation, blowout and status rules already live). Given minutes `m`:

- **3PM and two-point makes**: exactly the shipped engine (Poisson 3PA at the v2 rate, Binomial makes, Poisson twos from the
  non-3 rate). PTS = 3*3PM + 2*twos. Unchanged, so the ladder engine's outputs are not touched.
- **REB, AST, STL, BLK, TOV**: Negative-binomial counts with mean `m * rate` and a per-stat dispersion `alpha` (variance
  `mu + alpha*mu^2`), the Poisson limit when `alpha = 0`. REB and AST keep the engine's rates and DvP factors; STL/BLK/TOV get
  no opponent adjustment in v1.
- **STL/BLK/TOV rate per minute**: the engine's own weighting (half-life 12 games, prior season at weight 0.5), EB-shrunk
  toward the league rate at the player's position with `K = 120` minutes (the v2 3PA constant, reused a priori).
- **Dispersion `alpha` per stat** (REB, AST, STL, BLK, TOV): the moment estimate on 2024-25 box scores (players with 20+ games,
  minutes >= 8, expectation = leave-one-out season rate x minutes): `alpha = sum((x-mu)^2 - mu) / sum(mu^2)`. Estimated once
  by `fantasy_points.py priors`, written to `data/fp_priors.json`, frozen.

## 2. Joint simulation (Gaussian copula)

`N = 10,000` simulations per player-game, seeded. For each simulation: one minutes draw; then a 7-vector of standard normals
with correlation matrix `R` (components 3PA, twos, REB, AST, STL, BLK, TOV) mapped through the normal CDF to uniforms and
through each marginal's inverse CDF (given the minutes draw). 3PM = Binomial(3PA, make_p) on top. FP from section 0.

`R` for a player = `(n * R_player + K_rho * R_pos) / (n + K_rho)`, `K_rho = 40` games (fixed a priori), where `R_pos` is the
league within-player residual correlation of per-minute rates at his position (2024-25, players with 20+ games, demeaned per
player, estimated once in `fantasy_points.py priors`, frozen in `data/fp_priors.json`) and `R_player` is the same quantity
from his own last two seasons (n games). The blend is projected to the nearest positive-definite matrix if needed.
Pre-registered expectation, stated so it cannot be "discovered" later: once minutes are shared, the residual rate
correlations measured on 2024-25 are small (|rho| <= 0.13 in every cell with 3,000+ games), so the copula will move FP's
variance little; the shared minutes draw carries most of the dependence.

Output per player: FP mean, sd, the 99 percentiles (1..99) of the simulated FP, and the STL/BLK/TOV means and pmfs.

## 3. Log (immutable)

`fantasy_points.py snapshot` appends, for every projected active player whose game has not tipped, one row to
`data/fp_log.csv`: date, event, tip, player, team, opp, pos, projected minutes, FP mean/sd, the 99 percentiles (JSON), the
STL/BLK/TOV means and pmfs (JSON), logged_at. Append-only, never edited or deleted; post-tip rows are never written. The graded
snapshot of a (date, event, player) is the last row logged before tip (same rule as the ladder log).

## 4. Pricing study (paper only)

Source, chosen openly: the pirate-bets Underdog board capture `reader/board_snapshots/underdog.json` (the scraper the user
already runs; the app itself is never scraped by this repo). Every run records which NBA market strings the capture carries.

- **Fantasy Points lines**: NBA entries whose market string is one of `Fantasy Points`, `Fantasy Score`, `Fantasy Pts` (case-
  insensitive; the list is extended only by addendum when the capture shows a new string). Each is joined to the last pre-tip
  FP log row of the same player (normalized name) and game date. `P(higher) = 1 - CDF(line)` interpolated on the logged
  percentiles; `P(lower) = CDF(line)`. Break-even per side = `0.55 / mult`, where `mult` is Underdog's per-leg payout
  multiplier for that side (`platform_ids.payout_multipliers`, `1.0` for a standard balanced leg) and 0.55 is the paper log's
  standard per-leg break-even (3-pick at 6x). Edge = P - break-even. Appended to `data/fp_prices.csv` with the capture
  timestamp. **If the capture carries no NBA Fantasy Points market, that fact is reported in `docs/data/fp_pricing.json`
  and nothing is priced**; no other source is substituted.
- **3PM lines**: NBA `3-Pointers Made` entries joined to tonight's projections: `P(higher) = P(3PM > line)` from the shipped
  3PM pmf (the calibrated ladder P when the line + 0.5 is a ladder rung), same break-even rule, appended to
  `data/ud_prices.csv` (3PM only). No other individual stat is priced.
- Nothing here feeds `data/paper_log.csv` or `docs/data/results.json`.

## 5. Grading

The morning run (`fantasy_points.py grade`) grades every logged player-game whose box score exists:

- `data/fp_grade.csv` (regenerated, the log stays the record): actual FP, actual STL/BLK/TOV, the PIT (`CDF(actual)` from the
  logged percentiles), and for every Underdog FP line priced before tip: side, line, P, break-even, hit. A player with no
  box-score minutes is void (shown, excluded).
- `docs/data/fp_performance.json`: lines graded, hit rate (reported, not the verdict), calibration by decile of P(higher) over
  the priced lines, PIT decile table for the FP distribution over all graded player-games, and the STL/BLK/TOV reads
  (P(X >= k) for k = 1, 2 on STL and BLK, k = 2, 3 on TOV, bucketed by predicted probability).

## 6. Gates (pre-registered)

- No conclusion, staking plan, or slip until **200 graded Underdog FP lines**. Lines graded against the model's own median
  are shown for calibration only and never count toward the 200.
- The verdict at n >= 200 is **calibration by decile**, not raw hit rate: PASS if, over the deciles of P(higher) with n >= 20
  lines, the n-weighted mean absolute gap |mean P - hit rate| is <= 0.05 and no such decile has |gap| > 0.10. Anything else
  FAILS. Hit rate and edge-weighted return are reported beside it, never as the verdict.
- STL, BLK and TOV marginals each get their own calibration read at n >= 200 graded player-games, same decile criterion on
  the P(X >= k) buckets above.
- Any constant change, new feature or re-weighting is a model change: new rule file, backtest gate, full quality loop. No
  tuning on live results, ever.

## 7. Display

Main page, "Fantasy Points" tab: tonight's projection per player (minutes, FP mean, 10th-90th percentile, STL/BLK/TOV means),
the Underdog FP line when the capture carries one, P(higher), break-even, edge, and the capture's market list when it does
not. Labeled "Pricing study, paper only" and "Hit rate measures accuracy, not profit." The Performance page shows the graded
record from `docs/data/fp_performance.json`.
