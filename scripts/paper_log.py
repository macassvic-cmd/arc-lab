"""Paper log, closing line value, and results.

  python paper_log.py snapshot   record entry + latest pre-tip ("close") line/prob for every
                                 player prop that has an Underdog line; freeze at tip-off
  python paper_log.py settle     grade finished rows against box scores, rebuild results.json

Every prop with a line is logged (for calibration). A row is a *pick* when the model's edge
on one side cleared MIN_EDGE at some snapshot before tip; the first such snapshot is the entry.
"""
import csv
import datetime as dt
import sys

from common import DATA_DIR, SITE_DATA, norm_name, read_json, write_json

LOG = DATA_DIR / "paper_log.csv"
BREAKEVEN = 0.55   # 3-pick at 6x; change to match how you actually play
MIN_EDGE = 0.03
STAT_KEYS = {"3PM": "tpm", "3PT": "tpm", "PTS": "pts", "POINTS": "pts", "REB": "reb", "REBOUNDS": "reb",
             "AST": "ast", "ASSISTS": "ast", "PRA": "pra", "PTS+REB+AST": "pra"}
FIELDS = ["date", "tip", "player_id", "player", "team", "opp", "stat",
          "first_ts", "first_line", "first_p", "mean",
          "pick", "entry_ts", "entry_line", "entry_p", "entry_odds_p",
          "close_ts", "close_line", "close_p", "close_odds_p",
          "actual", "result"]


def now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)


def p_over(pmf, line):
    return sum(v for k, v in enumerate(pmf) if k > line)


def p_under(pmf, line):
    return sum(v for k, v in enumerate(pmf) if k < line)


def novig_over(o, u):
    """No-vig over probability from American odds, if a sharp price was supplied."""
    def imp(a):
        a = float(a)
        return 100 / (a + 100) if a > 0 else -a / (-a + 100)
    try:
        po, pu = imp(o), imp(u)
        return po / (po + pu)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def sharp_over(l):
    """Sharp no-vig P(over) for a lines.json row. underdog_lines.py writes it devigged as sharp.fair_over,
    only trusted when the sharp book's line is the Underdog line (sharp.exact); otherwise fall back to raw
    over_odds / under_odds if a different producer supplied those."""
    sh = l.get("sharp") or {}
    if sh.get("exact") and sh.get("fair_over") is not None:
        return float(sh["fair_over"])
    return novig_over(l.get("over_odds"), l.get("under_odds"))


def load_log():
    if not LOG.exists():
        return []
    with open(LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_log(rows):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def snapshot():
    proj = read_json(SITE_DATA / "projections.json", None)
    lines = read_json(SITE_DATA / "lines.json", {"lines": []}).get("lines", [])
    if not proj or not lines:
        print("[paper] no projections or no lines.json; nothing to log")
        return
    tips = {}
    for g in proj["games"]:
        for t in (g["home"], g["away"]):
            tips[t] = g["start"]
    players = {norm_name(p["name"]): p for p in proj["players"] if not p.get("inactive")}
    rows = load_log()
    index = {(r["date"], r["player_id"], r["stat"]): r for r in rows}
    ts = now()
    touched = 0
    for l in lines:
        stat = STAT_KEYS.get(str(l.get("stat", "")).upper())
        p = players.get(norm_name(l.get("player")))
        if not stat or not p:
            continue
        tip = tips.get(p["team"])
        if tip and ts >= dt.datetime.fromisoformat(tip.replace("Z", "+00:00")).replace(tzinfo=None):
            continue  # frozen: game started, last snapshot was the close
        line = float(l["line"])
        po, pu = p_over(p[stat]["pmf"], line), p_under(p[stat]["pmf"], line)
        odds_p = sharp_over(l)
        key = (proj["date"], p["id"], stat)
        r = index.get(key)
        if r is None:
            r = {f: "" for f in FIELDS}
            r.update(date=proj["date"], tip=tip or "", player_id=p["id"], player=p["name"], team=p["team"],
                     opp=p["opp"], stat=stat, first_ts=ts.isoformat(), first_line=line,
                     first_p=round(po, 4), mean=p[stat]["mean"])
            rows.append(r)
            index[key] = r
        if not r["pick"]:
            side = "over" if po - BREAKEVEN >= MIN_EDGE and po >= pu else \
                   "under" if pu - BREAKEVEN >= MIN_EDGE else ""
            if side:
                r.update(pick=side, entry_ts=ts.isoformat(), entry_line=line,
                         entry_p=round(po if side == "over" else pu, 4),
                         entry_odds_p="" if odds_p is None else round(odds_p if side == "over" else 1 - odds_p, 4))
        side = r["pick"] or "over"
        same_line = r["entry_line"] == "" or float(r["entry_line"]) == line
        r.update(close_ts=ts.isoformat(), close_line=line,
                 close_p=round(po if side == "over" else pu, 4),
                 close_odds_p="" if odds_p is None or not same_line else
                 round(odds_p if side == "over" else 1 - odds_p, 4))
        touched += 1
    save_log(rows)
    print(f"[paper] {touched} props snapshotted, {sum(1 for r in rows if r['pick'])} picks in log")


def settle():
    rows = load_log()
    if not rows:
        build_results(rows)
        return
    import pandas as pd  # only settle needs pandas; the game-window snapshot runs on a requests-only runner
    games = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"player_id": str})
    games["pra"] = games.pts + games.reb + games.ast
    actual = {(r.date, r.player_id): r for r in games.itertuples()}
    today = (now() - dt.timedelta(hours=5)).date().isoformat()
    graded = 0
    for r in rows:
        if r["result"] or r["date"] >= today:
            continue
        g = actual.get((r["date"], r["player_id"]))
        if g is None:
            r["result"] = "void"  # DNP or game missing: no action
            continue
        val = float(getattr(g, r["stat"]))
        r["actual"] = val
        line = float(r["entry_line"] or r["close_line"] or r["first_line"])
        side = r["pick"] or "over"
        if val == line:
            r["result"] = "push"
        else:
            r["result"] = "win" if (val > line) == (side == "over") else "loss"
        graded += 1
    save_log(rows)
    print(f"[paper] graded {graded} rows")
    build_results(rows)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(picks):
    decided = [r for r in picks if r["result"] in ("win", "loss")]
    wins = sum(r["result"] == "win" for r in decided)
    units = sum((1 / BREAKEVEN - 1) if r["result"] == "win" else -1 for r in decided)
    line_clv = []
    for r in picks:
        e, c = f(r["entry_line"]), f(r["close_line"])
        if e is not None and c is not None:
            line_clv.append((c - e) if r["pick"] == "over" else (e - c))
    mkt = [f(r["close_odds_p"]) for r in picks if f(r["close_odds_p"]) is not None]
    return {
        "n": len(picks), "decided": len(decided), "wins": wins,
        "hit_rate": round(wins / len(decided), 4) if decided else None,
        "units": round(units, 2),
        "avg_line_clv": round(sum(line_clv) / len(line_clv), 3) if line_clv else None,
        "beat_close_rate": round(sum(x > 0 for x in line_clv) / len(line_clv), 4) if line_clv else None,
        "line_moved_rate": round(sum(x != 0 for x in line_clv) / len(line_clv), 4) if line_clv else None,
        "market_close_p": round(sum(mkt) / len(mkt), 4) if mkt else None, "market_n": len(mkt),
    }


def build_results(rows):
    picks = [r for r in rows if r["pick"]]
    settled = [r for r in rows if r["result"] in ("win", "loss")]
    # calibration of the model's over probability at the first logged line, all props
    buckets = {}
    brier, brier_n = 0.0, 0
    for r in rows:
        if r["result"] not in ("win", "loss", "push") or r["actual"] == "":
            continue
        p, line, val = f(r["first_p"]), f(r["first_line"]), f(r["actual"])
        if p is None or val == line:
            continue
        hit = 1.0 if val > line else 0.0
        brier += (p - hit) ** 2
        brier_n += 1
        b = min(int(p * 10), 9)
        bk = buckets.setdefault(b, {"lo": b / 10, "hi": (b + 1) / 10, "n": 0, "p": 0.0, "hits": 0.0})
        bk["n"] += 1
        bk["p"] += p
        bk["hits"] += hit
    calib = [{"lo": b["lo"], "hi": b["hi"], "n": b["n"], "pred": round(b["p"] / b["n"], 3),
              "actual": round(b["hits"] / b["n"], 3)} for _, b in sorted(buckets.items())]
    # projection error (mean vs actual) by stat
    mae = {}
    for r in rows:
        if r["actual"] != "" and f(r["mean"]) is not None:
            mae.setdefault(r["stat"], []).append(abs(f(r["mean"]) - f(r["actual"])))
    by_day = {}
    for r in sorted(settled, key=lambda r: r["date"]):
        if r["pick"]:
            by_day[r["date"]] = by_day.get(r["date"], 0) + ((1 / BREAKEVEN - 1) if r["result"] == "win" else -1)
    cum, series = 0.0, []
    for d, u in by_day.items():
        cum += u
        series.append({"date": d, "units": round(cum, 2)})
    write_json(SITE_DATA / "results.json", {
        "updated": now().isoformat() + "Z", "breakeven": BREAKEVEN, "min_edge": MIN_EDGE,
        "overall": summarize(picks),
        "by_stat": {s: summarize([r for r in picks if r["stat"] == s]) for s in sorted({r["stat"] for r in picks})},
        "calibration": calib, "brier": round(brier / brier_n, 4) if brier_n else None, "brier_n": brier_n,
        "mae": {s: {"n": len(v), "mae": round(sum(v) / len(v), 2)} for s, v in mae.items()},
        "series": series,
        "recent": [{k: r[k] for k in ("date", "player", "team", "opp", "stat", "pick", "entry_line",
                                      "entry_p", "close_line", "close_odds_p", "actual", "result")}
                   for r in sorted(picks, key=lambda r: r["entry_ts"], reverse=True)[:60]],
    })


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "snapshot":
        snapshot()
    elif cmd == "settle":
        settle()
    else:
        sys.exit("usage: paper_log.py snapshot|settle")
