# Arc Lab

NBA prop research site. Threes first, plus points, rebounds, assists and PRA, defense vs position, and injury news forwarded from your phone.

Live at `https://macassvic-cmd.github.io/arc-lab/` once Pages is on.

## How it fits together

| Piece | Runs on | What it does |
|---|---|---|
| `scripts/fetch_games.py` | GitHub Actions, 10am and 5:30pm ET | Pulls box scores, tonight's slate, rosters and ESPN injury tags |
| `scripts/positions.py` | Same workflow | Assigns PG/SG/SF/PF/C from play style into `data/positions.csv` (ESPN only gives G/F/C) |
| `scripts/build_model.py` | Same workflow | Defense vs position, minutes, and 8,000 simulated games per player |
| `scripts/health.py` | GitHub Actions, every 15 min 5pm-1am ET | Discord alert when `lines.json` goes stale during games |
| `scripts/ingest_news.py` | GitHub Actions, on every phone alert | Tags the news, drops OUT players, moves their minutes, re-projects |
| `docs/index.html` | GitHub Pages | The site |

## Setup (about 15 minutes)

1. Create a repo named `arc-lab` and push this folder to it.
2. **Settings → Pages:** deploy from branch `main`, folder `/docs`.
3. **Settings → Actions → General → Workflow permissions:** read and write.
4. **Backfill last season** (gives the model its priors before opening night): Actions → *Update data and projections* → Run workflow with start `2025-10-21`, end `2026-04-12`. Takes 15–25 minutes.
5. Set up the phone (below).

## Phone → Discord + site (Android, MacroDroid)

Turn on post notifications for Underdog NBA in the X app, then create one macro.

**Trigger:** Notification received → app *X* → title contains `Underdog NBA`.

**Action 1, text cleanup:** use Text manipulation to replace `"` with `'` in the notification text, saved to a variable such as `newsText`. Unescaped quotes break the JSON below.

**Action 2, Discord (instant):** HTTP request, POST to your Discord webhook URL, content type `application/json`, body:
```
{"content": "{newsText}"}
```

**Action 3, site:** HTTP request, POST to `https://api.github.com/repos/macassvic-cmd/arc-lab/dispatches`
Headers: `Authorization: Bearer <token>` and `Accept: application/vnd.github+json`
Body:
```
{"event_type": "injury_news", "client_payload": {"text": "{newsText}"}}
```
Use a fine-grained personal access token scoped to only this repo with **Contents: read and write**. Insert `{newsText}` with MacroDroid's variable button.

Exempt MacroDroid from battery optimization, or Android will kill it overnight.

`fetch_games.py` stores all 30 rosters each run. `build_model.py` posts to Discord (once per player per day) when a regular starter from last season (80%+ starts, 20+ games) whose team plays tonight is tagged inactive by ESPN with no news item in 7 days, or is on no ESPN roster at all; those players stay out of the projections and are listed under `alerts` in `projections.json`.

The site header shows the phone's last alert time next to the Underdog lines' age. `scripts/health.py` (workflow *Lines watchdog*) posts to Discord via the `DISCORD_WEBHOOK_URL` secret when `docs/data/lines.json` is missing or more than 45 minutes old between 90 minutes before the first tip and 3 hours after the last: once when it goes stale, once when it recovers, state in `data/health_state.json`.

Timing: Discord gets the alert in a second or two, which is what you bet off. The site updates 45–90 seconds later (Action run plus Pages deploy), which is fine for research.

## Underdog lines

Lines come from the Underdog board scraper in `pirate-bets-pc` (`python -m reader.underdog_scraper`). Set these in that repo's `.env` and the scraper hands every fresh board to `scripts/underdog_lines.py` here, which writes `docs/data/lines.json` and pushes it:

```
ARC_LAB_DIR=C:\Users\vmora\Downloads\arc-lab\arc-lab
ARC_LAB_LINES_SECONDS=600   # at most one write/push per 10 minutes
ARC_LAB_PUSH=1              # 0 = write the file, don't commit
ODDSBLAZE_API_KEY=...       # optional, see below
```

One-off, from this repo:
```
python scripts/underdog_lines.py --snapshot ..\..\pirate-bets-pc\reader\board_snapshots\underdog.json --push
```

Stats kept: `3PM`, `PTS`, `REB`, `AST`, `PRA`, main lines only, games not yet started. Season-long markets are dropped. Each line carries `team`, `opp`, `start`, `ud_id` and, when an OddsBlaze key is present, `sharp`: the first book in `ODDSBLAZE_SHARP_BOOKS` (default Pinnacle, Circa, DraftKings, FanDuel) with its `line`, `over`/`under` American prices, devigged `fair_over`, and `exact` (false when the book's nearest line differs from Underdog's). The site reads `player`, `stat`, `line` (the shape of `lines.example.json`); the rest is for you and your validation agent.

## OddsBlaze

`scripts/odds.py` is the one client. Key: `ODDSBLAZE_API_KEY` in the environment or a `.env` here (see `.env.example`), plus the repo secret of the same name for Actions (`gh secret set ODDSBLAZE_API_KEY`). Without it everything still runs: lines have no `sharp`, and the model skips blowout risk and says so in the log.

## Model notes

- **Positions:** ESPN's box scores, rosters and athlete pages only say G/F/C, so `positions.py` sorts each player within ESPN's group by per-36 rates over the backfill: guards with 5.5+ assists are PG, forwards with 7.5+ rebounds (or 6.5+ with under 3.5 3PA) are PF, forwards with 9.5+ rebounds and 2+ blocks are C. Under 150 minutes keeps the group default. The table is `data/positions.csv`; put a position in the `override` column to pin a player and it survives regeneration.
- **Minutes baseline:** with 5+ games this season, the last 10 weighted (half-life 4). With fewer, `(n * recent + (5 - n) * last_season_avg) / 5`, where last season's average skips its final 14 days (rest and tank games). Those players carry a `Minutes blend n/5` or `Last season minutes` flag; a player whose last game was for another team last season gets `Offseason move (OLD->NEW)`.
- **Minutes:** recent games weighted (half-life 4 games), with the roster trimmed to 240 from the top of the depth chart down. Cap 39 (95th percentile of regular starters' regulation minutes). Questionable widens the minutes range; a minutes limit caps at 24.
- **Redistribution when a player is out:** 20% of the vacated minutes (`ROTATION_SHARE`) are spread over the rotation by each teammate's probability of being the top absorber: from past games this player missed with the same team (3+ games), otherwise a role-similarity prior (bigger role, same position 2.5x). Backtested on 1,235 absence team-games from 2025-26: per-player MAE 5.22 vs 5.34 with no redistribution, 5.74 at a 67% share, and 6.07 for concentrating 40/27 on the predicted top two. Only about a quarter of vacated minutes land on rotation players; the rest goes to deep bench and call-ups, so projected team minutes drop below 240 when someone is out. `REDISTRIBUTION = "concentrate"` switches to the 40/27 rule. Since redistribution trims minutes error by only ~2%, the site shows `may gain minutes (X out)` and keeps the modeled gain (`min_gain`) in a tooltip.
- **Usage bump:** per absent regular starter (capped at two), remaining players' rates rise 5.5% points, 3.5% 3PA, 4.5% assists, 1.5% rebounds, measured on the backfill.
- **Threes:** attempts per minute (half-life 12 games) × minutes × opponent's attempt rate allowed to that position, then makes = binomial on attempts at the shooter's 3P%, blended with 150 attempts of league average.
- **Defense vs position:** per-minute stats allowed to each position vs league average, with last season at 35% weight and 600 minutes of league average blended in so early-season numbers don't swing wildly. It already reflects pace, so pace isn't applied twice. Validation (2026-10-08) found position adds nothing predictive beyond the team, so the simulation clamps the factor to 0.98–1.02 (`DVP_CLAMP`); the DvP tab keeps the full numbers, labeled descriptive.
- **Points, rebounds, assists, PRA** come from the same simulated minutes, so they're correlated the way real games are.
- **Blowout risk:** game spreads from OddsBlaze. Final margins run about N(spread, 13.5); starters sit roughly the last 8 minutes of a 20+ point game. The extra blowout chance a spread adds over a pick'em, times those 8 minutes, comes off each regular starter (scaled by their minutes) and goes to the bench, on both teams. A 7.5-point spread shaves about 0.9 minutes, 14.5 about 2.2. Those players get a `Blowout risk (-12.5)` flag and a wider minutes range; `spread`, `total` and `odds_book` sit on each game in `projections.json`.

## Known limits to fix next

- ESPN positions are sometimes generic (G, F); those map to SG and SF.
- Blowout shave is a fixed curve, not fit to data. Check it against actual starter minutes by spread after a month.
- OddsBlaze NBA market names are matched by keyword (`stat_for_market` in `scripts/odds.py`), written against the docs rather than a live response. Verify once with a key.
- Usage bump when a star sits is only through minutes for now, not shot share.
- Track closing-line value from day one before trusting edges.
