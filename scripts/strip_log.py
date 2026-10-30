"""Ladder Tonight strip sidecar (display only, 2026-10-30): what each Underdog-lined card said before tip.

  python strip_log.py                       # record tonight's lined cards from the current build + board capture (pre-tip only)
  python strip_log.py backfill PROJ.json    # one-off: the same from a saved pre-tip projections.json, lines from data/main_lines.csv

data/strip_log.csv: date, event_id, player_id, player, team, opp, home, pos, stat, line, p_over, captured_at. One row per
player-game-stat with a recorded Underdog main line; the LAST capture before that game's tip wins (as data/main_lines.csv).
p_over = the strip's own P(over the main line) = sum of the build's pmf above the line (docs/index.html `over(pmf, line)`),
recorded so the graded strip can rank its cards exactly as the pre-tip strip did after the build has rolled to the next slate.
Nothing here is a model, grade, log or recalibration input: the ladder log, the grader and the Performance record never read it.
The main line itself is the one main_lines.py records from the same board capture (the display's single source for lines).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_lines as ML  # noqa: E402
from common import DATA_DIR, SITE_DATA, read_json  # noqa: E402

OUT = DATA_DIR / "strip_log.csv"
FIELDS = ["date", "event_id", "player_id", "player", "team", "opp", "home", "pos", "stat", "line", "p_over", "captured_at"]
KEY = ["date", "event_id", "player_id", "stat"]


def over(pmf, line):
    """P(X > line): the page's `over` (index.html), k > line over the integer pmf."""
    return float(sum(v for k, v in enumerate(pmf or []) if k > float(line)))


def with_p_over(rows, P):
    """Attach the player's identity and the build's P(over) to main-line rows (date, event_id, player_id, stat, line, captured_at)."""
    players = {str(p["id"]): p for p in (P or {}).get("players", [])}
    out, skipped = [], 0
    for r in rows:
        p = players.get(str(r["player_id"]))
        pmf = ((p or {}).get(r["stat"]) or {}).get("pmf")
        if not p or not pmf:
            skipped += 1; continue
        out.append({**{k: r[k] for k in ("date", "event_id", "player_id", "stat", "line", "captured_at")},
                    "player": p["name"], "team": p["team"], "opp": p["opp"], "home": int(bool(p.get("home"))), "pos": p.get("pos"), "p_over": round(over(pmf, r["line"]), 4)})
    if skipped:
        print(f"[strip-log] {skipped} main-line rows without a pmf in the build (skipped)")
    return out


def upsert(new, out=OUT):
    old = pd.read_csv(out, dtype={"event_id": str, "player_id": str}) if out.exists() else pd.DataFrame(columns=FIELDS)
    d = pd.concat([old, pd.DataFrame(new, columns=FIELDS)], ignore_index=True)
    if not len(d):
        print("[strip-log] nothing to record"); return d
    d["line"] = d.line.astype(float); d["p_over"] = d.p_over.astype(float)
    d = d.sort_values("captured_at", kind="stable").drop_duplicates(KEY, keep="last").sort_values(KEY + ["captured_at"])
    d[FIELDS].to_csv(out, index=False)
    print(f"[strip-log] {out.name}: {len(d)} lined player-game-stats, {d.date.nunique()} dates ({d.date.min()} .. {d.date.max()}); last pre-tip capture wins")
    return d


def record():
    P = read_json(SITE_DATA / "projections.json", None)
    rows = ML.from_lines(read_json(ML.LINES, None), P)   # the same matching, pre-tip filter and capture time as main_lines.py
    return upsert(with_p_over(rows, P))


def backfill(proj_path):
    """One-off for slates before this sidecar existed: p_over from a saved pre-tip build, lines from data/main_lines.csv for
    that date, captured_at = the build time (only games not yet tipped at that build time)."""
    P = read_json(Path(proj_path), None)
    m = pd.read_csv(ML.OUT, dtype={"event_id": str, "player_id": str})
    m = m[m.date == P["date"]]
    built = ML.parse_ts(P["updated"]); tips = {str(g["event_id"]): ML.parse_ts(g["start"]) for g in P["games"]}
    rows = [{"date": r.date, "event_id": r.event_id, "player_id": r.player_id, "stat": r.stat, "line": float(r.line), "captured_at": ML.iso(built)}
            for r in m.itertuples() if r.event_id in tips and built < tips[r.event_id]]
    print(f"[strip-log] backfill from build {ML.iso(built)}: {len(rows)} lined player-game-stats of {P['date']} not yet tipped")
    return upsert(with_p_over(rows, P))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "backfill":
        backfill(sys.argv[2])
    else:
        record()
