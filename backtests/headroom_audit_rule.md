# Tail program — headroom audit, locked rule (report-only, no tuning)

Committed 2026-10-18 (UTC) before anything below was run. Written from the user's instruction of 2026-10-18; nothing here may change
after this commit. Output: `docs/data/headroom_audit.json` and the Backtest page only; never `data/paper_log.csv` or
`docs/data/results.json`. Script: `scripts/headroom_audit.py`. No prices enter part 1; part 2 uses DraftKings' de-vigged prices
descriptively.

## 1. Oracle decomposition on 2023-24 (walk-forward rows, untouched by any gate)

Rows: `scripts/backtest.py --season 2024` (current model, prior games only), every 2023-24 player-game with minutes > 0, re-run
with two extra reporting columns (`proj_tpa` = the simulation's mean attempts, `make_p` = the make probability the simulation used,
i.e. the player's shrunk 3P% x the opponent make/attempt DvP factor, clipped to [0.12, 0.60]). The model itself is unchanged: in
`build_model.simulate` attempts ~ Poisson(minutes x rate), makes ~ Binomial(attempts, make_p).

Top-decile capture of actual 5+ games (share of actual 5+ games whose score ranks in the top 10% of that season's scored rows;
ties by rank), with 1,000-resample game-cluster bootstrap 95% CIs on each level and on each step difference, for:

- (a) **current model as-is**: `p_tpm_5` from the rows.
- (b) **oracle minutes**: attempts ~ Poisson(actual minutes x rate), rate = `proj_tpa` / `proj_min` (the per-minute attempt rate the
  model used, factors included), makes ~ Binomial(attempts, `make_p`); P(5+) summed analytically over attempts 0..40. Only the
  minutes distribution is collapsed to the truth.
- (c) **oracle 3PA**: P(5+) = Binomial tail(n = actual attempts, p = `make_p`). Minutes and attempt-rate error removed; the 3P%
  estimate is still the model's.
- (d) **oracle 3P% talent**: P(5+) = Binomial tail(n = actual attempts, p = the player's full-season 2023-24 3P% (made / attempted
  over the whole season, players with < 20 season attempts take the league 2023-24 rate)). This is shooting talent, not the game's
  shooting: actual attempts x actual makes would be the outcome itself.

Step differences: a->b = minutes error, b->c = attempt-rate error, c->d = 3P%-estimation error; what remains below 100% at (d) is
shooting luck plus any residual from the binomial form. Log loss for 5+ is reported at each level for information.

## 2. DraftKings vs the current model, head to head (2024-25, descriptive)

The gate season is used here knowingly: this is report-only and nothing is fit; noted on the page. Subset: 2024-25 player-games where
DraftKings quoted an Over 4.5 alternate rung at the close (Part A cache, `data/odds_alt_2025/`, DraftKings' own de-vigged
probability by the Part A method: implied(Over 4.5) / DK's two-sided main-line hold for that player, same snapshot) AND the current
model has `p_tpm_5` for the same player-game AND the player logged minutes. Rank the subset by DK's de-vigged P(5+) and by the
current model's P(5+); top-decile capture of actual 5+ games for each, bootstrap CIs by game, and the DK − model difference with
its CI. Also reported: the subset's size, base rate of 5+, and the pooled-books (median across books) ranking for context.

## 3. Decision table (pre-registered; applied to the point estimates)

- oracle-3PA capture (c) <= current (a) + 5 pp -> **physical ceiling**: kill feature work on ranking, fold back to recalibration.
- (c) − (a) between +5 pp and +15 pp -> **partial headroom**: report; no model build without a separate pre-registered rule.
- (c) − (a) >= +15 pp -> **attempts are the lever**: ranking headroom exists on the attempts side.
- DK >= current + 3 pp on the 2024-25 quoted subset -> **market has timing / information we lack**: investigate what (movement
  timing, not price).
- DK within ±3 pp of current -> **ranking is commoditized**: no model beats the market at ranking tails.
- (DK below current − 3 pp is reported as is; the table does not assign it an action.)

Report and stop. Whatever the table says determines whether model #3 gets built at all.
