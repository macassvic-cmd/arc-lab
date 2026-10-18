# NBA 2024-25 alternate threes — market-calibration test, locked rule

Committed 2026-10-18 (UTC) before any data was pulled. Written from the user's instruction of 2026-10-18; nothing below may change
after this commit. Runs alongside the box-score dispersion test; both report to the Backtest page only
(`docs/data/backtest_threes_alt_calibration.json`), never to `data/paper_log.csv` or `docs/data/results.json`.
Script: `scripts/backtest_threes_alt_calibration.py pull|grade|all --season 2025`.

## The user's rule (verbatim intent)

1. Pull 2024-25 historical `player_threes_alternate` (closing snapshot <= tip, region us) via the per-event historical endpoint.
   15K credit stop.
2. De-vig each rung (method stated below), bucket by fair probability (<5%, 5–10%, 10–20%, 20–35%, 35–50%), compare to the
   actual hit rate.
3. Locked rule: "Market underprices tails" = rungs at 5+ threes OR >= 2 above the player's main line hit at least 3pp more often
   than the de-vigged price, n >= 500 rungs, holding after dropping the best week. Otherwise: market tails are calibrated ->
   stop tail modeling, report.
4. Report alongside the dispersion test. Backtest page only, never paper log or results.

## Implementation definitions fixed with the rule

- **Games.** The same 1,231 regular-season 2024-25 games as the closing-line backtest (ESPN event ids; the Odds API event ids and
  tip times are already cached in `data/odds_close_2025/events_by_date.json`, so no listing credits are spent).
- **Pull.** `GET /v4/historical/sports/basketball_nba/events/{id}/odds?regions=us&markets=player_threes_alternate&date=<commence_time>`
  — the snapshot at or before tip (the close). 10 credits per game, ~12,310 expected. Raw responses cached under
  `data/odds_alt_2025/<espn_event_id>.json`; every call logged to `data/odds_alt_2025/usage.jsonl`; the run stops once this pull's
  spend passes 15,000 credits and reports. Key from `ODDS_API_KEY` in the environment, never printed.
- **Rung.** One (game, player, line) from the alternate market, Over side, pooled across books. Alternate ladders are quoted
  one-sided (Over only), so the de-vig must come from each book's two-sided main line.
- **De-vig (multiplicative, per book).** For book b and player i: overround h = implied(Over) + implied(Under) at b's main line for i
  in the cached closing `player_threes` snapshot of the same game (same `date` parameter, so the same or an adjacent snapshot;
  the two timestamps are recorded). Fair P(rung) = implied(Over price at the rung) / h, clipped to [0, 1] (clips counted). If b
  quotes no two-sided main line for i, h = the median overround of b's two-sided main lines in that game; if b has none, b's
  rungs for i are dropped. Pooled fair P = the median over books quoting the rung; `n_books` kept. Multiplicative de-vigging
  removes the whole margin proportionally, which is the standard and, for long prices, a conservative choice for this test
  (it lowers fair P at the tail less than a power de-vig would, so it is harder, not easier, to find "underpriced" tails).
- **Outcome.** Hit = box-score threes made (`tpm`, ESPN) > line. Player-games with no box-score row or 0 minutes are dropped
  (DNP; the book would have voided). Names join on `common.norm_name`.
- **Main line** for the ">= 2 above" criterion: the player's modal two-sided line in the closing `player_threes` snapshot (ties
  -> lower), as in the closing-line backtest. Without one, the lowest two-sided line in the alternate market; without either the
  rung can qualify as a tail only through the 5+ criterion.
- **Tail rung.** line >= 4.5 (5+ threes) OR line >= main line + 2.
- **Buckets** by pooled fair P: <5%, 5–10%, 10–20%, 20–35%, 35–50%, and >= 50% (reported, outside the user's list). Bucket error
  = hit rate − mean fair P; also per-line (0.5 … 7.5+) and by n_books.
- **Grade.** gap = mean(hit) − mean(fair P) over tail rungs. "Market underprices tails" iff gap >= +0.03 with n >= 500 tail rungs,
  AND after dropping the best ISO week (UTC game date; the week with the largest tail gap among weeks with >= 30 tail rungs) the
  gap is still >= +0.03 with n >= 500. Anything else: "market tails are calibrated" -> tail modeling stops. A cluster-robust
  standard error of the gap (clusters = game) and a 95% interval are reported for information; they are not part of the rule.
- **Dispersion test.** Reported in the same page section when present; this rule neither depends on nor changes it.
