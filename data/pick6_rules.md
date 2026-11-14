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
