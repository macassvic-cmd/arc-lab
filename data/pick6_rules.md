# DraftKings Pick6: settlement rules for the eight opening markets (summary)

Source: https://pick6.draftkings.com/pick6-rules-and-scoring-nba ("Pick6 Rules and Scoring NBA"), read logged out on
2026-11-14 in the same pass as the first full board capture. This is a summary of the published text for our own grading; the
page itself governs. Quoted phrases are the page's wording.

## The eight markets (Pick6 name on the board -> the rules page's stat name)

| Board pill | Rules-page stat | Definition | Eligibility |
|---|---|---|---|
| 1st Point | 1st Points Scorer | "Player who scores the first point(s) of the game." | "Player must start the game to be eligible; void if not" |
| Team 1st FG | First Team Field Goal Scorer / Team First Basket | "Player who makes the first field goal for their team." | must start; void if not |
| 1st Three | First Three Point Field Goal Scorer / Game 1st 3PT Made | "Player who makes the first 3-point field goal of the game." | must start; void if not |
| Team 1st Three | Team First Three Point Field Goal Scorer | "Player who makes the first 3-point field goal for their team." | must start; void if not |
| 1st Dunk | First Dunk Scorer | "Player who scores the first dunk of the game." | must start; void if not |
| Points 1st 3 Mins | Player Total Points – First 3 Minutes | "Total points scored by a player in the first 3 minutes of the game." | see below |
| Rebounds 1st 3 Mins | Player Total Rebounds – First 3 Minutes | "Total rebounds by a player in the first 3 minutes of the game." | see below |
| Assists 1st 3 Mins | Player Total Assists – First 3 Minutes | "Total assists by a player in the first 3 minutes of the game." | see below |

Window-market eligibility is stated two ways on the page: the scoring table says "Must enter before 9:00 on clock; void if
not"; the minimum-play table says "Player must start in the game". A starter satisfies both. Our grader treats a player who
did not start as VOID for all eight markets (the stricter reading), and flags it.

## Sides on the board (observed in the capture, not in the rules text)

- The five firsts are YES-ONLY: one Yes button per card, no No side.
- The three window markets are MORE-ONLY at the default posted line (0.5, 1.5 or 2.5). Other lines exist behind the card's line
  adjuster; we use the default line only.

## Voids, ties, did-not-play

- **Did not start / minimum play not met:** "If a player does not meet the defined minimum play requirements for their pick,
  it will be scored as void, regardless of selecting more or less for the pick."
- **Tie with the line:** "if a pick's final result exactly matches the projection line, that pick will be graded as a void".
  The window lines are half-points, so a tie cannot occur at the default line.
- **A slip with void picks:** it drops to the lower pick level: "the lineup will receive the Base Payout in the sport, pick
  level, and payout tier of the non-void picks". With all picks void, or all but one void, the entry fee is refunded.
- **Postponed or cancelled game:** rescheduled inside the scoring period (first scheduled start to 11:59 PM ET the day after
  the last scheduled game) = graded normally; rescheduled outside it = the game's picks are void.
- **Suspended game:** stats before the suspension count when the Pick Group has two or more games.
- **Pardon policy:** applies to "More" picks on FULL-GAME stats of a player who exits early. It is not stated for the firsts
  or the first-3-minutes markets; we do not assume it applies.

## Not stated on the page (open, do not assume)

- **A game with no dunk, or no made three:** the page defines the winner ("the first dunk of the game") and says nothing about
  a game where none occurs. Every Yes on that card then has no winner; whether those picks lose or void is not written.
  Our grader marks such legs `no_event` and counts them as MISSES in the record, labelled, until the rule is confirmed.
- **First point scored on a free throw:** "first point(s) of the game" includes free throws by the wording; our store's true
  first-point column (`wg_first_point_*`) counts free throws the same way.
- **Overtime** does not affect any of these markets (all are decided in the first quarter or by the first event).

## Payout mechanics (context only: nothing here is priced)

Each side carries a multiplier (the capture stores it). "Winning entries receive a share of the total prize pool proportional
to their share of Standings Points amongst all winners." Multipliers under 1x exist on likely outcomes.

## Payouts: what IS published (added 2026-11-14)

Pick6 is a prize pool, not a fixed-odds product. Two published pieces define what an entry receives.

**1. Base Multiplier (the guaranteed minimum, "Base Payout").** Source: DraftKings support article KB0010250, "Picks with
Multipliers and Standings Points" (v13.0, last modified 2026-08-25),
https://support.draftkings.com/dk/en-us/picks-with-multipliers-and-standings-points?id=kb_article_view&sysparm_article=KB0010250

> "Your Base Multiplier is calculated by multiplying the Standard Base Multiplier of the sport and pick tier by the Pick
> Multiplier of all the picks in your lineup, capped at a Max Base Multiplier."

Its worked example: a $1 3-pick NFL lineup, Standard Multiplier 5x, pick multipliers 1x, 1.2x, 1.3x, Max Base Multiplier 80x:
"Your Base Multiplier for this lineup is 7.8x (5x * 1x * 1.2x * 1.3x)."

**The Standard Base Multiplier table for NBA** is not in the article; the Pick6 page publishes it in its own entry details
for the pick group. As captured logged out on 2026-11-14 (NBA pick group 156466); the capture now stores it with every
board (`payout_table` in data/pick6_board.json), because it can differ by sport and may change:

| Pick level | Correct picks | Standard Base Multiplier | Max Base Multiplier |
|---|---|---|---|
| 2 | 2 of 2 | 3x | 50x |
| 3 | 3 of 3 | 6x | 100x |
| 4 | 4 of 4 | 10x | 150x |
| 5 | 5 of 5 | 12x | 200x |
| 5 | 4 of 5 | 1x | 10x |
| 6 | 6 of 6 | 25x | 400x |
| 6 | 5 of 6 | 1.5x | 15x |
| 7 | 7 of 7 | 40x | 600x |
| 7 | 6 of 7 | 2x | 20x |
| 8 | 8 of 8 | 80x | 800x |
| 8 | 7 of 8 | 3x | 30x |
| 8 | 6 of 8 | 1x | 10x |

**2. Standings Points and the prize-pool share ("Extra Winnings").** Same article:

> "Standings Points are calculated by multiplying your entry fee by the Pick Multipliers of your correct picks."

> "Your Extra Winnings are calculated by dividing the Standings Points for this entry by the total Standings Points of all
> winning lineups in the contest and multiplying by the guaranteed prize pool."

In the article's example the lineup's contest winnings are $10, shown "as $7.80 in Base Payout and $2.20 in Extra Winnings".
So the Base Payout is a floor that can be computed in advance; the amount above it depends on the other winners and cannot.

**Void picks.** Source: support article KB0010365, "Pick6 Minimum Play Requirements and Voided Pick Policy",
https://support.draftkings.com/dk/en-us/pick6-minimum-play-requirements-and-voided-pick-policy?id=kb_article_view&sysparm_article=KB0010365

> "Voided picks will reduce the size (i.e., number of picks) of your entry by the number of voided picks when calculating the
> associated Base Payout and contest standing points for a finished lineup, but the entry will still compete in the original
> contests it was entered into."

Its examples: a 4-pick with one void and three correct "would be scored as 3/3"; a 6-pick with one void, four correct and one
incorrect "will be scored as a 4/5". "If a lineup settles with all but one (1) voided pick, or with all voided picks, that
lineup will be paid its entry fee".

## Payouts: what is NOT published or not explicit

- **Partial tiers.** The Base Multiplier sentence says "all the picks in your lineup", and the only worked example is an
  all-correct lineup. Whether a 5-of-6 finish multiplies by all six pick multipliers or only the five correct ones is not
  stated. Our grader applies the sentence literally (all non-void picks) and labels those results "partial tier (literal
  reading)"; their total is shown separately in the record.
- **Adjusted multipliers.** "Specific pick combinations may result in adjusted multipliers, impacting your Base Multiplier as
  well as your Standings Points calculation." Which combinations, and by how much, is not published. Our numbers use the
  per-card multipliers as captured and ignore any such adjustment.
- **Extra Winnings** cannot be computed without the contest's other winners. Nothing in our files estimates them.
- Multipliers can change after posting; the logged slip keeps the multipliers of the last pre-tip board.

How our files use this: the "multiplier product" of a slip is the Standings Points basis per $1 (never called a payout); the
"Base Payout multiple" is DraftKings' guaranteed floor by the formula above; the record's "Base Payout returned" sums that
floor over graded slips at one unit each and leaves Extra Winnings out.
