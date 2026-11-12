# Sam Method: probability source and outlier guard (pre-registration)

Committed 2026-11-12, before the first gate-counting run (the first scheduled capture is 2026-11-13 13:00 PT) and before any
logged Sam Method leg has been graded. CEO rulings 2026-11-12. PRE-GATE SHADOW, PAPER ONLY. This file adds to the locked
optimizer spec (v5 + Amendment A1); it changes none of its other rules. Method version from the first counted run:
**sam_v1 + guard_v1**. Nothing below is tuned after a graded run exists without a version bump (guard_v2, ...), and a bumped
version starts its own record.

## 1. Probability source (one rule for every stat)

P = the LIVE PUBLISHED probability of the ladder model for that stat and rung: the number the page shows and the ladder record
grades (`ladder[stat][rung].p` in the projection snapshot). A recalibration map that is in shadow prices nothing: the Part A
points map (recal_v1, data/ladder_recal.json) stays out of the Sam Method until its day-34 ship condition passes. If a map
ships, it becomes part of the published P on its ship date and is used from then on, with no change to this tool.

Record of the switch: the 2026-11-12 off-schedule dry run priced POINTS legs with recal_v1 (MAP). From this commit on, points
use the published P like every other stat. That dry run is off schedule and counts toward no gate; its logged rows are left as
logged, and the runs log marks its points run `p_source = recal_v1_map_pre_switch`.

## 2. Why a guard

The search keeps the largest model-versus-market gaps. On a liquid two-sided market a very large gap is far more often the
model's own projection error than a mispriced contract, so an unguarded search selects the model's worst projections. The
2026-11-12 dry run showed it: Kyshawn George (WSH), projected 36.2 minutes, topped three of the four runs.

## 3. The guard (guard_v1), applied to every candidate before eligibility, base rate and tail checks

A candidate leg is EXCLUDED, and counted in the runs log, when either holds:

- **Edge cap, X = 10.0 points.** The leg's claimed edge (P of the logged side minus its fee-adjusted break-even) is greater
  than 10.0 percentage points. Anchor: 10 points is the depth rule's own maximum yes spread ($0.10). A model P that sits beyond
  the fee-adjusted ask by more than the widest spread the tool will trade into is outside the market's entire two-sided quote.
- **Minutes move, Y = 3.0 minutes.** The player's projected minutes in the capture snapshot differ from the slate's MORNING
  build (the first committed projection build dated the slate) by more than 3.0 minutes in either direction, or the player has
  no row in the morning build, or the morning build cannot be read. (Fail closed: a leg whose minutes move cannot be checked
  is excluded and counted as `guard_minutes`.)

The two numbers were set from the 2026-11-12 dry-run capture, priced with the section-1 P, and from nothing else:

- 590 side-legs searched; 143 candidates at or above the +3 point floor. Candidate edges: median 7.2, 75th percentile 11.0,
  90th 14.0, 95th 15.7, largest 21.8 points. 45 of the 143 exceed 10.0 points.
- Projected-minutes move, morning build (15:53Z) to the capture snapshot (21:26Z), 54 players: 90th percentile 0.4 minutes,
  95th 0.8, largest 1.3. Y = 3.0 is above everything seen that day; it would have excluded nothing on 2026-11-12.
- What this guard does NOT catch, stated now: Kyshawn George's minutes did not move that day (36.2 in both builds). His
  over-projection was already in the morning build, so only the edge cap removes his legs. A projection that is wrong all day
  and produces a gap under 10 points passes this guard.
- One slate is a thin basis. The anchor for X (the spread limit) does not depend on that slate; Y does.

## 4. Logging

`data/sam_method/runs_log.csv` carries, per run: `method_version`, `p_source`, `guard_edge` (candidates excluded by the edge
cap), `guard_minutes` (excluded by the minutes move). Excluded legs are never written to `shadow_log.csv`. The capture stores
each player's morning-build minutes next to the snapshot minutes.

## 5. What does not change

The 35 locked columns, the +3 point candidate floor, eligibility, base rates, tail flags, one run per stat, top three clusters,
per-leg z ranking, the grading and void rules, and the off-schedule exclusion from every gate count.
