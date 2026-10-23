"""v2 projection log and grading (backtests/ladder_v2_rule.md, section 2). No prices anywhere.

  python ladder_log.py snapshot   append every current pre-tip ladder projection to data/ladder_log.csv (append-only, never edited)
  python ladder_log.py grade      grade the last pre-tip snapshot of every logged rung against the box scores -> docs/data/performance.json

The log is immutable: rows are only ever appended. The graded snapshot of a (date, event, player, stat, rung) is the last row
logged before that game's tip. Hit = actual >= rung. A logged player with no box-score minutes is `void` (shown, not hidden,
excluded from hit rates). Live-season data only; walk-forwards stay on the Backtest page.
"""
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json

LOG = DATA_DIR / "ladder_log.csv"
FIELDS = ["date", "event_id", "tip", "player_id", "player", "team", "opp", "pos", "stat", "rung", "p_raw", "p", "logged_at"]
TIERS = [0.55, 0.60, 0.65, 0.70]
THIN = 30
STAT_ACT = {"tpm": "tpm", "pts": "pts", "ast": "ast", "reb": "reb"}
HEADER = "Every pick graded against real results. Nothing filtered, nothing hidden."


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse_ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def snapshot():
    P = read_json(SITE_DATA / "projections.json", None)
    if not P or not P.get("games"):
        print("[ladder-log] no slate"); return 0
    ev = {}
    for g in P["games"]:
        ev[(g["home"], g["away"])] = g; ev[(g["away"], g["home"])] = g
    t, ts = now(), now().strftime("%Y-%m-%dT%H:%M:%SZ")
    new_file = not LOG.exists()
    rows, skipped = [], 0
    for p in P["players"]:
        if p.get("inactive") or "ladder" not in p:
            continue
        g = ev.get((p["team"], p["opp"]))
        if not g or not g.get("start"):
            continue
        tip = parse_ts(g["start"])
        if t >= tip:
            skipped += 1; continue   # post-tip: never logged
        for stat, rungs in p["ladder"].items():
            for k, v in rungs.items():
                rows.append({"date": P["date"], "event_id": g["event_id"], "tip": tip.strftime("%Y-%m-%dT%H:%M:%SZ"), "player_id": p["id"], "player": p["name"], "team": p["team"], "opp": p["opp"],
                             "pos": p["pos"], "stat": stat, "rung": int(k), "p_raw": v["raw"], "p": v["p"], "logged_at": ts})
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"[ladder-log] appended {len(rows)} rows for {P['date']} ({skipped} players skipped: game already tipped)")
    return len(rows)


def graded_snapshot(log):
    """Last row logged before tip per key."""
    d = log[pd.to_datetime(log.logged_at, utc=True) < pd.to_datetime(log.tip, utc=True)].copy()
    d = d.sort_values("logged_at").drop_duplicates(["date", "event_id", "player_id", "stat", "rung"], keep="last")
    return d


def block(d):
    n = int(len(d)); h = int(d.hit.sum())
    return {"n": n, "hits": h, "misses": n - h, "hit_rate": round(h / n, 4) if n else None, "record": f"{h}-{n - h}"}


def bucket_table(d):
    out = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (d.p >= lo) & (d.p < hi) if i < 9 else (d.p >= lo)
        g = d[m]; n = int(len(g))
        out.append({"lo": lo, "hi": hi, "n": n, "pred": round(float(g.p.mean()), 4) if n else None, "actual": round(float(g.hit.mean()), 4) if n else None, "thin": n < THIN})
    return out


def grade():
    if not LOG.exists():
        write_json(SITE_DATA / "performance.json", {"header": HEADER, "empty": True, "generated": now().strftime("%Y-%m-%dT%H:%M:%SZ")}); print("[ladder-log] no log yet"); return
    log = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str})
    snap = graded_snapshot(log)
    box = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})
    done_events = set(box.event_id)
    snap["ingested"] = snap.event_id.isin(done_events)
    b = box.set_index(["event_id", "player_id"])
    act, mins = [], []
    for r in snap.itertuples():
        key = (r.event_id, r.player_id)
        if r.ingested and key in b.index:
            row = b.loc[key]; act.append(float(row[STAT_ACT[r.stat]])); mins.append(float(row["min"]))
        else:
            act.append(np.nan); mins.append(0.0 if r.ingested else np.nan)
    snap["actual"], snap["minutes"] = act, mins
    snap["status"] = np.where(~snap.ingested, "pending", np.where(snap.minutes > 0, "graded", "void"))
    g = snap[snap.status == "graded"].copy(); g["hit"] = (g.actual >= g.rung).astype(int)
    g["p"] = g.p.astype(float)
    tiers = []
    for t in TIERS:
        s = g[g.p >= t]
        tiers.append({"min_p": t, **block(s), "by_stat": {st: block(s[s.stat == st]) for st in STAT_ACT}})
    by_stat = {}
    for st in STAT_ACT:
        s = g[g.stat == st]
        by_stat[st] = {**block(s), "brier": round(float(((s.p - s.hit) ** 2).mean()), 4) if len(s) else None,
                       "by_rung": {str(k): {**block(r), "mean_p": round(float(r.p.mean()), 4), "brier": round(float(((r.p - r.hit) ** 2).mean()), 4), "buckets": bucket_table(r),
                                            "tiers": {str(t): block(r[r.p >= t]) for t in TIERS}} for k, r in s.groupby("rung")}}
    by_pos = {pos: {**block(s), "by_stat": {st: block(s[s.stat == st]) for st in STAT_ACT}} for pos, s in g.groupby("pos")}
    players = [{"player": name, "player_id": pid, "team": s.team.iloc[-1], "pos": s.pos.iloc[-1], **block(s), "by_stat": {st: block(s[s.stat == st]) for st in STAT_ACT if (s.stat == st).any()}}
               for (pid, name), s in g.groupby(["player_id", "player"])]
    players.sort(key=lambda x: -x["n"])
    recent_dates = sorted(g.date.unique())[-3:]
    recent = g[g.date.isin(recent_dates)].sort_values(["date", "p"], ascending=[False, False])
    recent_rows = [{"date": r.date, "player": r.player, "team": r.team, "opp": r.opp, "pos": r.pos, "stat": r.stat, "rung": int(r.rung), "p": round(float(r.p), 3), "actual": r.actual, "hit": int(r.hit)}
                   for r in recent.head(4000).itertuples()]
    void = snap[snap.status == "void"]
    doc = {"header": HEADER, "generated": now().strftime("%Y-%m-%dT%H:%M:%SZ"), "season": "2026-27",
           "log": {"rows_in_file": int(len(log)), "keys": int(len(snap)), "graded": int(len(g)), "pending": int((snap.status == "pending").sum()), "void": int(len(void)),
                   "dates": int(g.date.nunique()), "first_date": str(g.date.min()) if len(g) else None, "last_date": str(g.date.max()) if len(g) else None,
                   "void_by_stat": {st: int((void.stat == st).sum()) for st in STAT_ACT}},
           "overall": block(g), "tiers": tiers, "by_stat": by_stat, "by_pos": by_pos, "players": players, "recent": recent_rows, "recent_dates": [str(x) for x in recent_dates],
           "calibration_adopted": read_json(DATA_DIR / "ladder_calibration.json", {}).get("adopted", []),
           "method": ["Graded snapshot = the last projection logged before tip; every earlier row stays in data/ladder_log.csv, which is append-only.",
                      "Hit = actual >= rung, from the ESPN box score. Void = the player logged no minutes; counted here, excluded from hit rates.",
                      f"Calibration cells with n < {THIN} are flagged thin. Nothing is filtered or re-run after tip; no tuning on these results, ever.",
                      "Shipped P is the raw engine probability except for stats whose isotonic maps passed the 2024-25 gate (backtests/ladder_v2_rule.md); those are listed."]}
    fb = SITE_DATA / "first_basket_performance.json"
    if fb.exists():
        doc["first_basket"] = read_json(fb, None)
    fpp = SITE_DATA / "fp_performance.json"
    if fpp.exists():
        doc["fantasy_points"] = read_json(fpp, None)
    write_json(SITE_DATA / "performance.json", doc)
    print(f"[ladder-log] graded {len(g)} rungs ({doc['overall']['record']}, {doc['overall']['hit_rate']}), {doc['log']['pending']} pending, {len(void)} void")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    snapshot() if cmd == "snapshot" else grade()
