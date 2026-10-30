"""Ladder Tonight graded overlay (display only, 2026-10-30): docs/data/ladder_graded.json.

  python ladder_graded.py        # after `ladder_log.py grade`; rewrites the overlay from the grader's published rows

Marks come ONLY from the grading pipeline's own output: the graded ladder-log rows the Performance page reads
(docs/data/performance.json `recent`: date, player, stat, rung, p, actual, hit), joined to the strip sidecar
(data/strip_log.csv) by date + player + stat; the main line is the one recorded in data/main_lines.csv at snapshot time.
Nothing here re-derives a threshold, an actual or an outcome: hit is the grader's hit, actual is the grader's actual, the
main-line verdict is actual vs the recorded line (> OVER, < UNDER, = PUSH). A lined card whose game is ingested (its event is
in data/player_games.csv, the grader's own "ingested" test) but has no graded rows is VOID: the grader voided the player-game
(no minutes). A date is published only when every one of its lined games is ingested (the slate is final and graded).
If this overlay and the graded log ever disagree, the log wins and it is a bug to report.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json  # noqa: E402

OUT = SITE_DATA / "ladder_graded.json"
PERF = SITE_DATA / "performance.json"
STRIP = DATA_DIR / "strip_log.csv"
MAIN = DATA_DIR / "main_lines.csv"
BOX = DATA_DIR / "player_games.csv"
STATS = ["reb", "pts", "ast", "tpm"]
KEEP_DATES = 3   # the grader publishes its last three graded dates; the page shows the latest final one


def now():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def verdict(actual, line):
    return "OVER" if actual > line else ("UNDER" if actual < line else "PUSH")


def build():
    perf = read_json(PERF, None)
    if not perf or perf.get("empty") or not STRIP.exists():
        write_json(OUT, {"generated": now(), "dates": {}, "note": "no graded rows or no strip sidecar yet"}); print("[ladder-graded] nothing to overlay"); return
    g = pd.DataFrame(perf.get("recent") or [])
    strip = pd.read_csv(STRIP, dtype={"event_id": str, "player_id": str})
    main = pd.read_csv(MAIN, dtype={"event_id": str, "player_id": str})
    box = pd.read_csv(BOX, dtype={"event_id": str}, usecols=["event_id"])
    ingested = set(box.event_id)
    main_line = {(r.date, r.event_id, r.player_id, r.stat): float(r.line) for r in main.itertuples()}
    dates, disagreements = {}, []
    for date in sorted(strip.date.unique())[-KEEP_DATES:]:
        s = strip[strip.date == date]
        events = sorted(s.event_id.unique())
        final = all(e in ingested for e in events)
        if not final:
            continue   # the slate is not final: the pre-tip strip stays up untouched
        gd = g[g.date == date] if len(g) else g
        cards = []
        for r in s.itertuples():
            line = main_line.get((r.date, r.event_id, r.player_id, r.stat))
            if line is None:
                continue   # a card without a recorded main line is not a strip card
            rows = gd[(gd.player == r.player) & (gd.stat == r.stat)] if len(gd) else gd
            card = {"player": r.player, "player_id": r.player_id, "event_id": r.event_id, "team": r.team, "opp": r.opp, "home": int(r.home), "pos": r.pos, "stat": r.stat,
                    "line": line, "p_over": round(float(r.p_over), 4)}
            if len(rows):
                actuals = sorted(set(float(x) for x in rows.actual))
                if len(actuals) != 1:
                    disagreements.append({"date": date, "player": r.player, "stat": r.stat, "actuals": actuals})
                actual = actuals[0]
                card.update({"status": "graded", "actual": actual, "main": verdict(actual, line),
                             "rungs": [{"rung": int(x.rung), "p": float(x.p), "hit": int(x.hit)} for x in rows.sort_values("rung").itertuples()]})
            else:
                card.update({"status": "void", "actual": None, "main": None, "rungs": []})
            cards.append(card)
        graded = [c for c in cards if c["status"] == "graded"]
        by_stat = {}
        for st in STATS:
            cs = [c for c in graded if c["stat"] == st]
            by_stat[st] = {"cards": sum(c["stat"] == st for c in cards), "graded": len(cs), "void": sum(c["stat"] == st and c["status"] == "void" for c in cards),
                           "main_over": sum(c["main"] == "OVER" for c in cs), "main_under": sum(c["main"] == "UNDER" for c in cs), "main_push": sum(c["main"] == "PUSH" for c in cs),
                           "rungs": sum(len(c["rungs"]) for c in cs), "rung_hits": sum(x["hit"] for c in cs for x in c["rungs"])}
        dates[date] = {"date": date, "final": True, "events": events, "cards": cards, "all_lined_cards": by_stat,
                       "note": "all_lined_cards counts every lined player-game-stat of the slate; the strip's own record is over the cards it displayed (the pre-tip ranking)"}
    doc = {"generated": now(), "source": {"marks": "docs/data/performance.json recent (ladder_log.py grade): actual and hit per graded rung", "lines": "data/main_lines.csv (board capture at snapshot time)",
                                           "cards": "data/strip_log.csv (lined player-game-stats with the pre-tip P(over) the strip ranked by)", "final": "every lined game of the date ingested in data/player_games.csv"},
           "grader_generated": perf.get("generated"), "dates": dates, "disagreements": disagreements,
           "rules": "graded: actual and hit from the grader; main OVER when actual > line, UNDER when actual < line, PUSH when equal; void: game ingested, no graded rows (grader void); display never re-derives a threshold, actual or outcome"}
    write_json(OUT, doc)
    for d, v in dates.items():
        a = v["all_lined_cards"]
        print(f"[ladder-graded] {d}: {len(v['cards'])} lined cards over {len(v['events'])} games, " + ", ".join(f"{st} {a[st]['graded']} graded/{a[st]['void']} void, mains {a[st]['main_over']}-{a[st]['main_under']}, rungs {a[st]['rung_hits']}/{a[st]['rungs']}" for st in STATS))
    if disagreements:
        print(f"[ladder-graded] BUG: {len(disagreements)} player-stats with inconsistent actuals in the graded rows: {disagreements[:5]}")
    if not dates:
        print("[ladder-graded] no final slate among the strip sidecar's dates")


if __name__ == "__main__":
    build()
