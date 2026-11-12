"""Slips tonight (selection / display / logging layer, canonical v2 2026-11-04): three nested Underdog slips (3 in 6 in 8)
built every data build from the model's existing outputs. No model math here, nothing upstream is changed.

  python slips.py                       # build tonight's slips (pre-tip only), append new records, publish docs/data/slips.json
  python slips.py replay DATE --strip S --fplog F --lines L [--now T]   # the same selection from pinned inputs; prints, writes nothing

Inputs (read-only):
  * data/strip_log.csv           stat legs (PTS / REB / AST / 3PM): the last pre-tip snapshot per player-game-stat (the file keeps
                                 exactly that), line = the Underdog main line, p_over = the strip's own P(over). Side = More when
                                 p_over >= 0.5 else Less; raw P = max(p_over, 1 - p_over).
  * docs/data/lines.json         Underdog FP lines (stat "FP"), matched to the player's FP v2 row by normalized name + team.
  * data/fp_v2_log.csv           FP legs priced off the player's v2 distribution: P(More) = 1 - cdf(fp_q, line) with the 99 logged
                                 percentiles interpolated linearly (fantasy_points.cdf_from_q, the FP graders' own interpolation).
                                 Projected minutes = `min` of the player's latest logged row for the slate date.
Rules (hard): Underdog lines only; projected minutes >= 20 (no exceptions); every leg's raw P >= 0.57; at most one leg per game
and one per player; rank by raw P descending (ties: player name, then stat order FP, PTS, REB, AST, 3PM) and take greedily; the
3-leg is the top 3, the 6-leg adds the next 3, the 8-leg the next 2 (nested by construction). A size with too few legs is not
built and says so. The slate's slips regenerate on every data build until the earliest tip, then freeze: a build at or after the
earliest tip appends nothing and rewrites nothing. data/slips_log.jsonl is append-only: one JSON record per built slip per build
whose selection differs from the last logged one for that date (an identical rebuild appends nothing); a correction is a new
record, never an edit. Grading (display only): the ESPN box score already ingested in data/player_games.csv; More hits when
actual > line, Less when actual < line; absent / 0 minutes = VOID leg (a push is void too), and a slip with a void leg grades as
not hit with the void marked. FP actuals use the FP graders' scoring (fantasy_points.fp_of) and are checked against
data/fp_v2_grade.csv when that row exists. The slips record is its own line, never merged into the ladder / FP records.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fantasy_points as fp  # noqa: E402  (cdf_from_q, fp_of, parse_ts: read-only helpers)
from common import DATA_DIR, SITE_DATA, norm_name, read_json, write_json  # noqa: E402
from teams import team_abbr  # noqa: E402

STRIP = DATA_DIR / "strip_log.csv"
FPLOG = DATA_DIR / "fp_v2_log.csv"
LINES = SITE_DATA / "lines.json"
PROJ = SITE_DATA / "projections.json"
BOX = DATA_DIR / "player_games.csv"
FP_GRADE = DATA_DIR / "fp_v2_grade.csv"
LOG = DATA_DIR / "slips_log.jsonl"
OUT = SITE_DATA / "slips.json"

MIN_FLOOR = 20.0
P_FLOOR = 0.57
SIZES = (3, 6, 8)
STATS = {"pts": "PTS", "reb": "REB", "ast": "AST", "tpm": "3PM"}      # strip stat code -> display stat
BOX_COL = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "tpm"}    # display stat -> box-score column
STAT_ORDER = {"FP": 0, "PTS": 1, "REB": 2, "AST": 3, "3PM": 4}
HONESTY = ("Raw, uncalibrated model probabilities: FP v2 is mid-gate and the ladder recalibration read is pending. "
           "The joint number is a ranking aid, not a promised hit rate.")
KEEP_DATES = 5


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t):
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    return fp.parse_ts(s)


# ---------------- candidate legs ----------------
def fp_rows(fplog, date):
    """The latest logged FP v2 row per player for the slate date (minutes + distribution)."""
    f = pd.read_csv(fplog, dtype={"event_id": str, "player_id": str})
    f = f[f.date == date].sort_values("logged_at", kind="stable").drop_duplicates("player_id", keep="last")
    return f


def candidates(date, strip_df, fp_df, lines_doc):
    """Every Underdog-lined leg of the slate with its raw P and projected minutes, plus why it is out (if it is)."""
    mins = {r.player_id: float(r.min) for r in fp_df.itertuples()}
    tips = {r.event_id: r.tip for r in fp_df.itertuples()}
    by_name = {}
    for r in fp_df.itertuples():
        by_name.setdefault((norm_name(r.player), team_abbr(r.team) or r.team), r)
    legs = []
    for r in strip_df[strip_df.date == date].itertuples():
        p = round(float(r.p_over), 4); side = "More" if p >= 0.5 else "Less"
        legs.append({"player": r.player, "player_id": str(r.player_id), "team": r.team, "opp": r.opp, "event_id": str(r.event_id), "tip": tips.get(str(r.event_id)),
                     "stat": STATS[r.stat], "side": side, "line": float(r.line), "p": round(max(p, 1 - p), 4), "min": mins.get(str(r.player_id)),
                     "source": "strip_log", "captured_at": r.captured_at})
    cap = (lines_doc or {}).get("captured_at") or (lines_doc or {}).get("updated")
    unmatched = 0
    for ln in (lines_doc or {}).get("lines") or []:
        if ln.get("stat") != "FP":
            continue
        row = by_name.get((norm_name(ln.get("player")), team_abbr(ln.get("team")) or ln.get("team")))
        if row is None or (ln.get("opp") and (team_abbr(ln["opp"]) or ln["opp"]) != (team_abbr(row.opp) or row.opp)):
            unmatched += 1; continue
        try:
            line = float(ln["line"])
        except (KeyError, TypeError, ValueError):
            continue
        p_more = round(1 - fp.cdf_from_q(json.loads(row.fp_q), line), 4); side = "More" if p_more >= 0.5 else "Less"
        legs.append({"player": row.player, "player_id": str(row.player_id), "team": row.team, "opp": row.opp, "event_id": str(row.event_id), "tip": row.tip,
                     "stat": "FP", "side": side, "line": line, "p": round(max(p_more, 1 - p_more), 4), "min": float(row.min),
                     "source": "fp_v2_log", "captured_at": cap, "fp_logged_at": row.logged_at})
    for lg in legs:
        if lg["min"] is None:
            lg["out"] = "no minutes projection"
        elif lg["min"] < MIN_FLOOR:
            lg["out"] = "minutes floor"
        elif lg["p"] < P_FLOOR:
            lg["out"] = "probability floor"
        else:
            lg["out"] = None
    return legs, unmatched


def rank_key(lg):
    return (-lg["p"], lg["player"], STAT_ORDER.get(lg["stat"], 9), lg["line"])


def select(legs):
    """Greedy pick under the one-per-game / one-per-player rules from the qualifying legs ranked by raw P."""
    ranked = sorted([lg for lg in legs if not lg["out"] and lg["p"] >= P_FLOOR and (lg["min"] or 0) >= MIN_FLOOR], key=rank_key)   # floors re-checked here: rules 2-3 hold whatever the caller marked
    picks, games, players = [], set(), set()
    for lg in ranked:
        if lg["event_id"] in games or lg["player_id"] in players:
            continue
        picks.append(lg); games.add(lg["event_id"]); players.add(lg["player_id"])
        if len(picks) == max(SIZES):
            break
    slips, not_built = {}, {}
    for n in SIZES:
        if len(picks) >= n:
            slips[n] = picks[:n]
        else:
            not_built[n] = f"{n}-leg: only {len(picks)} qualifying legs (after one per game, one per player)"
    return ranked, picks, slips, not_built


def joint(legs):
    j = 1.0
    for lg in legs:
        j *= lg["p"]
    return round(j, 4)


def leg_record(lg):
    return {k: lg[k] for k in ("player", "player_id", "team", "opp", "event_id", "tip", "stat", "side", "line", "p", "min", "source", "captured_at")}


def records(date, slips, built_at, earliest_tip, lines_cap):
    return [{"slip_id": f"{date}-{n}-{built_at.replace('-', '').replace(':', '')}", "date": date, "size": n, "kind": "build", "built_at": built_at,
             "earliest_tip": earliest_tip, "lines_captured_at": lines_cap, "legs": [leg_record(lg) for lg in legs], "joint_p": joint(legs),
             "floors": {"min": MIN_FLOOR, "p": P_FLOOR}, "note": "nested: the 3-leg is the top 3, the 6-leg adds the next 3, the 8-leg the next 2"}
            for n, legs in sorted(slips.items())]


# ---------------- append-only log ----------------
def read_log(path=LOG):
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sig(rec):
    return tuple((lg["player_id"], lg["stat"], lg["side"], lg["line"], lg["p"]) for lg in rec["legs"])


def latest_build(recs, date):
    """The last build's records for a date (the frozen set once the slate has tipped)."""
    d = [r for r in recs if r.get("date") == date and r.get("kind", "build") == "build"]
    if not d:
        return []
    last = max(r["built_at"] for r in d)
    return sorted([r for r in d if r["built_at"] == last], key=lambda r: r["size"])


def append_if_new(new, path=LOG):
    """Append this build's records unless the last logged build of the date selected exactly the same slips."""
    recs = read_log(path)
    if not new:
        return 0
    last = latest_build(recs, new[0]["date"])
    if last and {r["size"]: sig(r) for r in last} == {r["size"]: sig(r) for r in new}:
        return 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(new)


# ---------------- grading (display only) ----------------
def grade_legs(recs, now=None, box_path=BOX, fp_grade_path=FP_GRADE):
    """Per-leg hit / miss / void / pending from the ingested box score (a leg whose game has not tipped is pending whatever the
    box file holds); returns {slip_id: [graded legs]} and the FP actual cross-checks against data/fp_v2_grade.csv."""
    now = now or now_utc()
    if not Path(box_path).exists():
        return {r["slip_id"]: [{**lg, "status": "pending", "actual": None} for lg in r["legs"]] for r in recs}, []
    box = pd.read_csv(box_path, dtype={"event_id": str, "player_id": str})
    events = set(box.event_id); b = box.set_index(["event_id", "player_id"])
    fg = {}
    if Path(fp_grade_path).exists():
        g = pd.read_csv(fp_grade_path, dtype={"event_id": str, "player_id": str}, usecols=["date", "event_id", "player_id", "fp_actual", "void"])
        fg = {(r.event_id, r.player_id): (None if pd.isna(r.fp_actual) else float(r.fp_actual), int(r.void)) for r in g.drop_duplicates(["event_id", "player_id"]).itertuples()}
    out, checks = {}, []
    for r in recs:
        legs = []
        for lg in r["legs"]:
            key = (lg["event_id"], lg["player_id"])
            g = {**lg, "actual": None, "minutes": None, "status": "pending", "reason": None}
            if lg["event_id"] not in events or (lg.get("tip") and now < parse_ts(lg["tip"])):
                legs.append(g); continue
            if key not in b.index:
                g.update(status="void", reason="absent from the box score", minutes=0.0); legs.append(g); continue
            row = b.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            mins = float(row["min"]); g["minutes"] = mins
            if mins <= 0:
                g.update(status="void", reason="0 minutes"); legs.append(g); continue
            actual = round(float(fp.fp_of(row)), 1) if lg["stat"] == "FP" else float(row[BOX_COL[lg["stat"]]])
            g["actual"] = actual
            if lg["stat"] == "FP" and key in fg and fg[key][0] is not None and abs(fg[key][0] - actual) > 0.05:
                checks.append({"slip_id": r["slip_id"], "player": lg["player"], "fp_of_box": actual, "fp_v2_grade": fg[key][0]})
            if actual == lg["line"]:
                g.update(status="void", reason="push")
            else:
                hit = actual > lg["line"] if lg["side"] == "More" else actual < lg["line"]
                g["status"] = "hit" if hit else "miss"
            legs.append(g)
        out[r["slip_id"]] = legs
    return out, checks


def slip_status(legs):
    st = [lg["status"] for lg in legs]
    if "pending" in st:
        return "pending"
    if all(s == "hit" for s in st):
        return "hit"
    return "miss"


# ---------------- publish ----------------
def publish(date, now, built, legs_all, unmatched, ranked, picks, not_built, frozen, earliest_tip, lines_cap, log_recs, appended, out=OUT, write=True):
    graded, checks = grade_legs(log_recs, now)
    dates = {}
    for d in sorted({r["date"] for r in log_recs})[-KEEP_DATES:]:
        cur = latest_build(log_recs, d)
        slips = []
        for r in cur:
            gl = graded.get(r["slip_id"], [{**lg, "status": "pending", "actual": None} for lg in r["legs"]])
            slips.append({**r, "legs": gl, "status": slip_status(gl), "voids": sum(lg["status"] == "void" for lg in gl)})
        final = bool(slips) and all(s["status"] != "pending" for s in slips)
        dates[d] = {"date": d, "final": final, "slips": slips, "builds": len({r["built_at"] for r in log_recs if r.get("date") == d})}
    record = {}
    for n in SIZES:
        done = [s for d in dates.values() if d["final"] for s in d["slips"] if s["size"] == n]
        hits = sum(s["status"] == "hit" for s in done)
        record[str(n)] = {"hits": hits, "misses": len(done) - hits, "with_void": sum(s["voids"] > 0 for s in done), "n": len(done), "record": f"{hits}-{len(done) - hits}"}
    excluded_min = sorted([lg for lg in legs_all if lg["out"] in ("minutes floor", "no minutes projection") and lg["p"] >= P_FLOOR], key=rank_key)[:8]
    today = {"date": date, "frozen": frozen, "earliest_tip": earliest_tip, "built_this_run": built, "appended": appended, "checked_at": iso(now),
             "lines_captured_at": lines_cap, "legs_considered": len(legs_all), "fp_lines_unmatched": unmatched,
             "qualifying": len(ranked), "pickable": len(picks), "not_built": not_built,
             "excluded_by_minutes": [{k: lg[k] for k in ("player", "team", "opp", "stat", "side", "line", "p", "min")} for lg in excluded_min],
             "counts": {"minutes_floor": sum(lg["out"] == "minutes floor" for lg in legs_all), "no_minutes": sum(lg["out"] == "no minutes projection" for lg in legs_all),
                        "probability_floor": sum(lg["out"] == "probability floor" for lg in legs_all)}}
    doc = {"generated": iso(now), "today": today, "dates": dates, "record": record, "honesty": HONESTY, "fp_actual_checks": checks,
           "rules": {"stats": ["FP", "PTS", "REB", "AST", "3PM"], "book": "Underdog", "min_floor": MIN_FLOOR, "p_floor": P_FLOOR, "sizes": list(SIZES),
                     "one_per_game": True, "one_per_player": True, "rank": "raw P descending; ties by player name, then FP, PTS, REB, AST, 3PM",
                     "freeze": "regenerated on each data build until the slate's earliest tip, then frozen; the log is append-only",
                     "grading": "ESPN box score (data/player_games.csv): More hits when actual > line, Less when actual < line; absent, 0 minutes or a push = VOID leg; a slip with a void leg is not hit",
                     "record": "slips only; never merged into the ladder or FP records"},
           "source": {"stat_legs": "data/strip_log.csv (last pre-tip snapshot per player-game-stat; Underdog main line; strip P(over))",
                      "fp_legs": "docs/data/lines.json FP lines priced off data/fp_v2_log.csv fp_q (99 percentiles, linear interpolation)",
                      "minutes": "data/fp_v2_log.csv min, latest logged row per player for the slate date", "log": "data/slips_log.jsonl"}}
    if write:
        write_json(out, doc)
    return doc


# ---------------- build ----------------
def slate_date(strip_df, proj):
    if proj and proj.get("date"):
        return proj["date"]
    return str(strip_df.date.max()) if len(strip_df) else None


def earliest_tip_of(date, proj, fp_df):
    tips = [g["start"] for g in (proj or {}).get("games", []) if proj and proj.get("date") == date and g.get("start")]
    tips += [t for t in fp_df.tip.dropna().unique()]
    return iso(min(parse_ts(t) for t in tips)) if tips else None


def compute(now, strip=STRIP, fplog=FPLOG, lines=LINES, proj=PROJ, date=None):
    """The selection from the pinned inputs, nothing written: the build and the replay share this."""
    cols = ["date", "event_id", "player_id", "player", "team", "opp", "stat", "line", "p_over", "captured_at"]
    strip_df = pd.read_csv(strip, dtype={"event_id": str, "player_id": str}) if strip and Path(strip).exists() else pd.DataFrame(columns=cols)
    P = read_json(proj, None) if proj else None
    date = date or slate_date(strip_df, P)
    c = {"date": date, "legs": [], "unmatched": 0, "ranked": [], "picks": [], "slips": {}, "not_built": {n: f"{n}-leg: no slate" for n in SIZES}, "earliest_tip": None, "frozen": False, "lines_captured_at": None}
    if not date:
        return c
    fp_df = fp_rows(fplog, date) if Path(fplog).exists() else pd.DataFrame(columns=["event_id", "player_id", "player", "team", "opp", "tip", "min", "fp_q", "logged_at"])
    lines_doc = read_json(lines, None) if lines else None
    c["lines_captured_at"] = (lines_doc or {}).get("captured_at") or (lines_doc or {}).get("updated")
    c["earliest_tip"] = earliest_tip_of(date, P, fp_df)
    c["frozen"] = bool(c["earliest_tip"]) and now >= parse_ts(c["earliest_tip"])
    c["legs"], c["unmatched"] = candidates(date, strip_df, fp_df, lines_doc)
    c["ranked"], c["picks"], c["slips"], c["not_built"] = select(c["legs"])
    return c


def build(now=None, strip=STRIP, fplog=FPLOG, lines=LINES, proj=PROJ, log=LOG, out=OUT, date=None):
    now = now or now_utc()
    c = compute(now, strip, fplog, lines, proj, date)
    date = c["date"]; log_recs = read_log(log)
    appended, built = 0, False
    if not date:
        print("[slips] no slate date (no strip sidecar, no projections)")
    elif c["frozen"]:
        print(f"[slips] {date}: earliest tip {c['earliest_tip']} has passed; slips frozen (nothing appended)")
    elif not c["legs"]:
        print(f"[slips] {date}: no lined legs yet (strip sidecar / FP lines empty)")
    else:
        built = True
        appended = append_if_new(records(date, c["slips"], iso(now), c["earliest_tip"], c["lines_captured_at"]), log)
        log_recs = read_log(log)
        print(f"[slips] {date}: {len(c['legs'])} lined legs, {len(c['ranked'])} qualify (min >= {MIN_FLOOR:g}, P >= {P_FLOOR}), {len(c['picks'])} pickable; built " +
              ", ".join(f"{n}-leg joint {joint(l):.4f}" for n, l in sorted(c["slips"].items())) + (f"; {'; '.join(c['not_built'].values())}" if c["not_built"] else "") +
              f"; {appended} records appended" + ("" if appended else " (same selection as the last build)"))
        for i, lg in enumerate(c["picks"], 1):
            print(f"[slips]   {i:2d} {lg['player']:24s} {lg['stat']:4s} {lg['side']:4s} {lg['line']:6.2f} P {lg['p']:.4f} min {lg['min']:.1f} {lg['team']} v {lg['opp']}")
    doc = publish(date, now, built, c["legs"], c["unmatched"], c["ranked"], c["picks"], c["not_built"], c["frozen"], c["earliest_tip"], c["lines_captured_at"], log_recs, appended, out=out)
    rec = doc["record"]
    print("[slips] record: " + " · ".join(f"{n}-leg {rec[str(n)]['record']}" for n in SIZES) + f"; {len(doc['dates'])} dates published")
    return doc


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="build", choices=["build", "replay"])
    ap.add_argument("date", nargs="?")
    ap.add_argument("--strip", default=str(STRIP)); ap.add_argument("--fplog", default=str(FPLOG)); ap.add_argument("--lines", default=str(LINES))
    ap.add_argument("--proj", default=None); ap.add_argument("--now", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(); return 0
    c = compute(parse_ts(a.now) if a.now else now_utc(), a.strip, a.fplog, a.lines, a.proj, a.date)
    print(json.dumps({"date": c["date"], "earliest_tip": c["earliest_tip"], "frozen": c["frozen"], "legs": len(c["legs"]), "qualifying": len(c["ranked"]), "not_built": c["not_built"],
                      "slips": {n: [f"{lg['player']} {lg['stat']} {lg['side']} {lg['line']} P {lg['p']}" for lg in legs] for n, legs in c["slips"].items()},
                      "joint": {n: joint(legs) for n, legs in c["slips"].items()},
                      "excluded_by_minutes": [f"{lg['player']} {lg['stat']} P {lg['p']} min {lg['min']}" for lg in sorted([x for x in c["legs"] if x["out"] in ("minutes floor", "no minutes projection") and x["p"] >= P_FLOOR], key=rank_key)[:5]]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
