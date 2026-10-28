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
FP_LOG = DATA_DIR / "fp_log.csv"          # projected minutes logged pre-tip by fantasy_points.py snapshot (same build as the ladder snapshot)
MAIN_LINES = DATA_DIR / "main_lines.csv"  # Underdog main line per player-game-stat (main_lines.py; display-only playable-rungs view, 2026-10-28)
BAND = (0.55, 0.80)                       # backtests/live_recalibration_rule.md section 0
BOOT_ITERS, BOOT_SEED, BOOT_CHUNK = 2000, 20261023, 100
TRIGGER_DAY, VALIDATION_END = 21, 35
MIN_ERR_EDGES = [(-np.inf, -10), (-10, -5), (-5, -2), (-2, 2), (2, 5), (5, 10), (10, np.inf)]
RECAL_RULE = "backtests/live_recalibration_rule.md"
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


def cluster_ids(d):
    """Player-game cluster per row: rungs of one player-game are correlated (rule section 0)."""
    return pd.factorize(d.date.astype(str) + "|" + d.event_id.astype(str) + "|" + d.player_id.astype(str))[0]


def player_games(d):
    return int(cluster_ids(d).max() + 1) if len(d) else 0


def boot_ci(d, kind="rate", iters=BOOT_ITERS, seed=BOOT_SEED):
    """95% cluster-bootstrap CI by player-game (resample clusters with replacement, percentile 2.5/97.5).
    kind = "rate": hit rate; "gap": mean p minus hit rate. Returns [lo, hi] or None when fewer than 2 clusters."""
    if len(d) == 0:
        return None
    cid = cluster_ids(d); C = int(cid.max() + 1)
    if C < 2:
        return None
    cn = np.bincount(cid, minlength=C).astype(float)
    ch = np.bincount(cid, weights=d.hit.to_numpy(dtype=float), minlength=C)
    cp = np.bincount(cid, weights=d.p.to_numpy(dtype=float), minlength=C)
    rng = np.random.default_rng(seed); vals = []
    for start in range(0, iters, BOOT_CHUNK):
        idx = rng.integers(0, C, size=(min(BOOT_CHUNK, iters - start), C))
        n = cn[idx].sum(axis=1); h = ch[idx].sum(axis=1)
        vals.append(h / n if kind == "rate" else (cp[idx].sum(axis=1) - h) / n)
    v = np.concatenate(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def block(d):
    n = int(len(d)); h = int(d.hit.sum())
    return {"n": n, "hits": h, "misses": n - h, "hit_rate": round(h / n, 4) if n else None, "record": f"{h}-{n - h}",
            "mean_p": round(float(d.p.mean()), 4) if n else None, "player_games": player_games(d)}


def bucket_table(d, ci=False):
    out = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (d.p >= lo) & (d.p < hi) if i < 9 else (d.p >= lo)
        g = d[m]; n = int(len(g))
        row = {"lo": lo, "hi": hi, "n": n, "pred": round(float(g.p.mean()), 4) if n else None, "actual": round(float(g.hit.mean()), 4) if n else None, "thin": n < THIN}
        if ci:
            row["player_games"] = player_games(g); row["ci"] = boot_ci(g, "rate")
        out.append(row)
    return out


def gap_block(d):
    b = block(d)
    b["gap"] = round(b["mean_p"] - b["hit_rate"], 4) if b["n"] else None
    b["gap_ci"] = boot_ci(d, "gap")
    return b


def diagnostic(g):
    """CEO directive 2026-10-23 part 2, report-only: the 55-80% band's calibration gap by stat and by minutes error
    (projected minus actual minutes). Projected minutes come from data/fp_log.csv (pre-tip, same build as the ladder
    snapshot); rows without a logged minutes projection are counted but excluded from the minutes-error buckets."""
    band = g[(g.p >= BAND[0]) & (g.p < BAND[1])].copy()
    band["min_proj"] = np.nan
    if FP_LOG.exists() and len(band):
        f = pd.read_csv(FP_LOG, dtype={"event_id": str, "player_id": str}, usecols=["date", "event_id", "player_id", "tip", "min", "logged_at"])
        f = f[pd.to_datetime(f.logged_at, utc=True) < pd.to_datetime(f.tip, utc=True)].sort_values("logged_at").drop_duplicates(["date", "event_id", "player_id"], keep="last")
        band = band.drop(columns=["min_proj"]).merge(f[["date", "event_id", "player_id", "min"]].rename(columns={"min": "min_proj"}), on=["date", "event_id", "player_id"], how="left")
    band["min_err"] = band.min_proj.astype(float) - band.minutes.astype(float)
    cov = band[band.min_err.notna()]
    def stat_cells(d):
        return {st: {k: v for k, v in gap_block(d[d.stat == st]).items() if k in ("n", "player_games", "mean_p", "hit_rate", "gap")} for st in STAT_ACT}
    buckets = []
    for lo, hi in MIN_ERR_EDGES:
        d = cov[(cov.min_err >= lo) & (cov.min_err < hi)]
        label = f"< {hi:+.0f}" if lo == -np.inf else f">= {lo:+.0f}" if hi == np.inf else f"{lo:+.0f} to {hi:+.0f}"
        buckets.append({"lo": None if lo == -np.inf else lo, "hi": None if hi == np.inf else hi, "label": label, **gap_block(d),
                        "mean_min_err": round(float(d.min_err.mean()), 2) if len(d) else None, "by_stat": stat_cells(d)})
    days = int(g.date.nunique())
    return {"label": "DIAGNOSTIC: report-only, no model change", "band": list(BAND), "rule": RECAL_RULE,
            "days_graded": days, "trigger_day": TRIGGER_DAY, "validation_end": VALIDATION_END,
            "clock": (f"day {days} of the rule's clock: the trigger is evaluated once, when day {TRIGGER_DAY} is graded; nothing is refit before then"
                      if days < TRIGGER_DAY else f"day {days}: trigger day reached; the section 1 read and any refit are published here"),
            "all": gap_block(band), "by_stat": {st: gap_block(band[band.stat == st]) for st in STAT_ACT},
            "minutes": {"rows_in_band": int(len(band)), "rows_with_projected_minutes": int(len(cov)), "player_games_with_projected_minutes": player_games(cov),
                        "mean_min_err": round(float(cov.min_err.mean()), 2) if len(cov) else None, "share_projected_high": round(float((cov.min_err > 0).mean()), 4) if len(cov) else None,
                        "source": "projected minutes = the pre-tip row of data/fp_log.csv for the same player-game (logged by fantasy_points.py snapshot from the same build as the ladder snapshot, available from 2026-10-23); actual minutes = the box score the grader uses"},
            "min_err_buckets": buckets,
            "method": ["Band = graded rungs with shipped P in [0.55, 0.80). Gap = mean P minus hit rate; positive means the model said more than it delivered.",
                       "Minutes error = projected minus actual minutes for the player-game; buckets are fixed a priori. Rows whose player-game has no logged minutes projection are counted in the band totals but not in the buckets.",
                       "95% CIs are cluster bootstraps by player-game (2,000 resamples, seed 20261023): rungs from the same player-game are correlated and are never treated as independent.",
                       "If the gap concentrates where minutes ran high, the cause is early-season rotations, not the probability layer. Report-only: nothing here changes the model; the only permitted live change is the pre-committed rule."]}


def playable_view(g):
    """Display-only slice (2026-10-28): the graded record restricted to PLAYABLE rungs, i.e. rung >= floor(main line) + 1 for
    that player-game-stat (the Underdog main line's over plus the true alts above it; everything at or below the line is a
    negative alt and is excluded). Player-game-stats with no recorded main line are excluded and counted. Main lines come from
    data/main_lines.csv (main_lines.py: the board capture at snapshot time, 3PM backfilled from the pricing history; last pre-tip
    capture wins). Same block / tier / bucket code as the full record; nothing here feeds the model or the recalibration rule."""
    keys = ["date", "event_id", "player_id", "stat"]
    if MAIN_LINES.exists():
        ml = pd.read_csv(MAIN_LINES, dtype={"event_id": str, "player_id": str})
    else:
        ml = pd.DataFrame(columns=keys + ["line", "captured_at"])
    ml = ml.drop_duplicates(keys, keep="last")
    d = g.merge(ml[keys + ["line"]].rename(columns={"line": "main_line"}), on=keys, how="left")
    has = d.main_line.notna()
    d["playable"] = has & (d.rung >= np.floor(d.main_line.astype(float)) + 1)
    p = d[d.playable].copy()
    def counts(x):
        return {"rungs": int(len(x)), "player_game_stats": int(x.groupby(keys).ngroups) if len(x) else 0, "player_games": player_games(x)}
    def split(x):
        return {**counts(x), "by_stat": {st: counts(x[x.stat == st]) for st in STAT_ACT}}
    tiers = [{"min_p": t, **block(p[p.p >= t]), "by_stat": {st: block(p[(p.p >= t) & (p.stat == st)]) for st in STAT_ACT}} for t in TIERS]
    calibration = {"all": bucket_table(p, ci=True), "by_stat": {st: bucket_table(p[p.stat == st], ci=True) for st in STAT_ACT},
                   "n": int(len(p)), "player_games": player_games(p)}
    return {"label": "Playable rungs only: rung >= floor(Underdog main line) + 1",
            "definition": "A rung counts only if rung >= floor(main line) + 1 for that player-game-stat: the main line's over and the true alts above it. Rungs at or below the line are negative alts and are excluded. Player-game-stats with no recorded main line are excluded and counted below.",
            "source": "data/main_lines.csv (main_lines.py): the Underdog board capture lines.json is built from, recorded at ladder snapshot time; 3PM history backfilled from data/ud_prices.csv; last pre-tip capture per player-game-stat wins.",
            "overall": block(p), "by_stat": {st: block(p[p.stat == st]) for st in STAT_ACT}, "tiers": tiers, "calibration": calibration,
            "with_line": split(d[has]), "dropped_missing_line": split(d[~has]), "excluded_below_line": split(d[has & ~d.playable]),
            "lines_file": {"rows": int(len(ml)), "first_date": str(ml.date.min()) if len(ml) else None, "last_date": str(ml.date.max()) if len(ml) else None,
                           "by_stat": {st: int((ml.stat == st).sum()) for st in STAT_ACT}},
            "note": "Display and reporting only. The default view (all rungs, all stats) is the record as logged; the model, the ladder log and the recalibration rule's statistics (computed on the full log) do not change."}


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
    calibration = {"all": bucket_table(g, ci=True), "by_stat": {st: bucket_table(g[g.stat == st], ci=True) for st in STAT_ACT},
                   "n": int(len(g)), "player_games": player_games(g),
                   "note": "Buckets of shipped P x hit rate x n, computed by this grader from the graded record (no thresholds or filters in display code). "
                           "95% CIs are cluster bootstraps by player-game: rungs from the same player-game are correlated."}
    diag = diagnostic(g)
    play = playable_view(g)
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
           "log": {"rows_in_file": int(len(log)), "keys": int(len(snap)), "graded": int(len(g)), "player_games": player_games(g), "pending": int((snap.status == "pending").sum()), "void": int(len(void)),
                   "dates": int(g.date.nunique()), "first_date": str(g.date.min()) if len(g) else None, "last_date": str(g.date.max()) if len(g) else None,
                   "void_by_stat": {st: int((void.stat == st).sum()) for st in STAT_ACT}},
           "overall": block(g), "tiers": tiers, "calibration": calibration, "diagnostic": diag, "playable": play, "by_stat": by_stat, "by_pos": by_pos, "players": players, "recent": recent_rows, "recent_dates": [str(x) for x in recent_dates],
           "calibration_adopted": read_json(DATA_DIR / "ladder_calibration.json", {}).get("adopted", []),
           "method": ["Graded snapshot = the last projection logged before tip; every earlier row stays in data/ladder_log.csv, which is append-only.",
                      "Hit = actual >= rung, from the ESPN box score. Void = the player logged no minutes; counted here, excluded from hit rates.",
                      f"Calibration cells with n < {THIN} are flagged thin. Nothing is filtered or re-run after tip; no tuning on these results, ever.",
                      "Sample sizes are shown in player-games as well as rungs: the rungs of one player-game (3PM 3+, 4+, 5+ ...) are correlated, so rung counts overstate the evidence. Every 95% CI is a cluster bootstrap by player-game.",
                      f"The only permitted live change to the probability layer is the pre-committed {RECAL_RULE} (trigger evaluated at graded day {TRIGGER_DAY}, validation through day {VALIDATION_END}).",
                      "Shipped P is the raw engine probability except for stats whose isotonic maps passed the 2024-25 gate (backtests/ladder_v2_rule.md); those are listed.",
                      "Playable rungs only (a display toggle, 2026-10-28): rung >= floor(Underdog main line) + 1 for that player-game-stat, from data/main_lines.csv (the board capture at snapshot time; 3PM backfilled from the pricing history). Player-games with no recorded main line are dropped from that view and counted. The default view is the full record."]}
    fb = SITE_DATA / "first_basket_performance.json"
    if fb.exists():
        doc["first_basket"] = read_json(fb, None)
    fpp = SITE_DATA / "fp_performance.json"
    if fpp.exists():
        doc["fantasy_points"] = read_json(fpp, None)
    fpv2 = SITE_DATA / "fp_v2_performance.json"
    if fpv2.exists():
        doc["fantasy_points_v2"] = read_json(fpv2, None)
    write_json(SITE_DATA / "performance.json", doc)
    print(f"[ladder-log] graded {len(g)} rungs / {doc['log']['player_games']} player-games ({doc['overall']['record']}, {doc['overall']['hit_rate']}), {doc['log']['pending']} pending, {len(void)} void; "
          f"band 55-80%: n {diag['all']['n']}, gap {diag['all']['gap']} CI {diag['all']['gap_ci']}; {diag['clock']}")
    pt = next((t for t in play["tiers"] if t["min_p"] == 0.55), None)
    print(f"[ladder-log] playable view: {play['overall']['n']} rungs / {play['overall']['player_games']} player-games ({play['overall']['record']}); >=55%: {pt['record'] if pt else '-'}; "
          f"{play['dropped_missing_line']['player_game_stats']} player-game-stats dropped for missing main lines, {play['excluded_below_line']['rungs']} rungs below the line excluded; main_lines.csv {play['lines_file']['rows']} rows")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    snapshot() if cmd == "snapshot" else grade()
