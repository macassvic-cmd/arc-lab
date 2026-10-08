# Arc Lab

NBA prop research site. Threes first, plus points, rebounds, assists and PRA, defense vs position, and injury news forwarded from your phone.

Live at `https://macassvic-cmd.github.io/arc-lab/` once Pages is on.

## How it fits together

| Piece | Runs on | What it does |
|---|---|---|
| `scripts/fetch_games.py` | GitHub Actions, 10am and 5:30pm ET | Pulls box scores, tonight's slate, rosters and ESPN injury tags |
| `scripts/build_model.py` | Same workflow | Defense vs position, minutes, and 8,000 simulated games per player |
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

Timing: Discord gets the alert in a second or two, which is what you bet off. The site updates 45–90 seconds later (Action run plus Pages deploy), which is fine for research.

## Underdog lines

Have your Underdog board scraper write `docs/data/lines.json` in the shape of `lines.example.json`. Stats accepted: `3PM`, `PTS`, `REB`, `AST`, `PRA`. The site then shows over chances and edge against your break-even.

## Model notes

- **Minutes:** recent games weighted (half-life 4 games), scaled so each team totals 240. Ruled-out players' minutes go to teammates, 2.5x weighted toward the same position. Questionable widens the minutes range; a minutes limit caps at 24.
- **Threes:** attempts per minute (half-life 12 games) × minutes × opponent's attempt rate allowed to that position, then makes = binomial on attempts at the shooter's 3P%, blended with 150 attempts of league average.
- **Defense vs position:** per-minute stats allowed to each position vs league average, with last season at 35% weight and 600 minutes of league average blended in so early-season numbers don't swing wildly. It already reflects pace, so pace isn't applied twice.
- **Points, rebounds, assists, PRA** come from the same simulated minutes, so they're correlated the way real games are.

## Known limits to fix next

- ESPN positions are sometimes generic (G, F); those map to SG and SF.
- No blowout risk yet. That needs spreads from your odds API to shave minutes in lopsided games.
- Usage bump when a star sits is only through minutes for now, not shot share.
- Track closing-line value from day one before trusting edges.
