# NBA 2024-25 threes closing-line backtest — locked rule

Committed 2026-10-17 (UTC) before any closing-line pull (step 4). Written by the user; nothing below may change after this commit.
The scripts that run it: `scripts/backtest.py --season 2025` (projections, walk-forward), `scripts/backtest_threes_close.py`
(closing lines, grading, report). Outputs go to the Backtest page only (`docs/data/backtest_threes_2024_25.json`), never to
`data/paper_log.csv` or `docs/data/results.json`.

Run in this order; nothing after step 1 runs until step 1 passes.

1. TEST CALL (~11 credits). New Odds API key, basketball_nba:
   `GET /v4/historical/sports/basketball_nba/events?date=2025-02-10T23:59:59Z`, take one eventId and its commence_time, then
   `GET /v4/historical/sports/basketball_nba/events/{eventId}/odds?regions=us&markets=player_threes&date={commence_time}&oddsFormat=american`.
   PASS: 200, bookmakers present with player_threes outcomes, snapshot timestamp at or before the date param. Report pass/fail
   before continuing.

2. BACKFILL (free). Pull 2024-25 NBA box scores from ESPN into the lab's backfill format. Also backfill 2023-24 box scores, used
   ONLY as prior-season input for early 2024-25 projections, mirroring how the live model starts a season. Then confirm and
   state: no model parameter, prior, cap, clamp, or calibration was fit on 2024-25 data (the same no-tuning rule applies to
   2023-24). If anything was, stop and report what.

3. PROJECTIONS. Run the walk-forward projection pipeline over every 2024-25 regular-season game (1,230 games) exactly as the
   live model runs — no 2024-25 tuning, no injury-news inputs (they don't exist historically; the test is harder for it).

4. BACKTEST (~12.5K credits). For each game, pull player_threes at the closing snapshot (date = tip time, regions=us) via the
   per-event historical endpoint. 10 credits per game. Budget guard: stop and report if spend passes 15K credits. Use the
   ODDS_API_KEY environment variable; never print the key.

5. GRADE against this locked rule (all five must pass):
   Pick = model prob >= no-vig close prob + 5pp AND model prob >= (1/payout)^0.25, using Underdog's live 4-pick payout table
   (never hardcoded).
   (a) Encompassing: logit(outcome) ~ logit(model) + logit(no-vig close), SEs clustered by game; model coefficient > 0, p < 0.05.
   (b) n >= 300 graded threes props.
   (c) Drop the best week by pick win rate; (a) and (e) still pass.
   (d) No calibration bucket off by >10pp at n >= 30.
   (e) Picks win at a rate above the break-even from the pick definition; n >= 200 picks.
   Report the full grade. Pass unlocks nothing by itself — the other four markets (~49K) need a separate go-ahead.

## Implementation definitions fixed with the rule (so the grade cannot drift)

- Season tags: ESPN labels 2024-25 as season 2025 and 2023-24 as 2024. The walk-forward sees only seasons <= 2025; 2025-26
  rows are dropped before anything is computed. Prior-season inputs (DvP weight 0.35, player-rate weight 0.5, minutes blend
  for the first 5 games, prior-tail exclusion) come from 2023-24 exactly as the live model takes them from the season before.
- Closing line per player-game: the per-event snapshot at or before tip (date = commence_time). Books: every `us` book in the
  snapshot. The graded line is the line offered by the most books (ties -> the lower line). No-vig close P(over) = median over
  books quoting both sides at that line of over / (over + under) in implied probability; a book quoting one side only is
  ignored. Model P(over x.5) = P(k+) with k = x + 0.5 from the walk-forward pmf (THREES_K extended to 1..8).
- Outcome = actual threes made > line (box score). Rows where the player did not play are dropped (DNP, no projection).
- Payout: Underdog's live 4-pick Standard payout multiplier, fetched at grading time and recorded in the report with its
  source and timestamp; break-even per pick = (1/payout)^(1/4).
- Weeks for (c): ISO weeks of the game date (UTC). Calibration buckets for (d): deciles of model P(over) at the graded line,
  bucket error = |mean model P - hit rate|, judged at n >= 30.
- Encompassing (a): logistic regression of hit on [1, logit(model P), logit(no-vig close P)] by IRLS; cluster-robust (CR1)
  sandwich SEs by game event_id; one-sided test on the model coefficient (> 0) at p < 0.05.
