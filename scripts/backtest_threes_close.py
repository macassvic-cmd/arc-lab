"""NBA threes closing-line backtest (steps 4 and 5 of backtests/nba_threes_2024_25_rule.md). HISTORICAL. Writes only to its own
files and the Backtest page; never to data/paper_log.csv or docs/data/results.json.

  python backtest_threes_close.py pull   --season 2025 --rows data/backtest_rows_2025.csv.gz      # step 4: closing player_threes per game
  python backtest_threes_close.py grade  --season 2025 --rows data/backtest_rows_2025.csv.gz      # step 5: the locked grade -> docs/data/backtest_threes_2024_25.json
  python backtest_threes_close.py all    ...

Step 4 (The Odds API, historical, ODDS_API_KEY from the environment, never printed):
  per game date: GET /historical/sports/basketball_nba/events?date=<first tip of the day>  (1 credit) -> event ids and tips;
  per event:     GET /historical/sports/basketball_nba/events/{id}/odds?regions=us&markets=player_threes&date=<commence_time>
                 (10 credits: the snapshot at or before tip = the close). Raw responses cached under data/odds_close_<season>/ so a
                 rerun never re-spends. Every call is logged (data/odds_close_<season>/usage.jsonl); the run stops once the
                 season's spend passes BUDGET_STOP credits (15,000) and reports.
Step 5 grading follows the locked definitions in the rule file (no-vig close at the modal line, model P(k+) from the walk-forward
rows, Underdog's live 4-pick payout fetched at grading time, encompassing regression with CR1 cluster-robust SEs by game,
best-week drop, decile calibration, pick win rate vs break-even).
"""
import argparse
import datetime as dt
import gzip
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, SITE_DATA, NBA_TEAMS, norm_name, read_json, write_json

BASE = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
BUDGET_STOP = 15_000          # rule: stop and report past this many credits spent in step 4
EDGE_PP = 0.05                # pick = model >= no-vig close + 5pp ...
PICKS_IN_ENTRY = 4            # ... and model >= (1/payout)^(1/4) with Underdog's live 4-pick payout
MIN_GRADED, MIN_PICKS, MIN_BUCKET, MAX_BUCKET_OFF = 300, 200, 30, 0.10
TEAM_NAMES = {  # The Odds API full names -> ESPN abbreviations used in player_games.csv
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GS",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
    "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
}


# ---------------- step 4: closing lines ----------------
def _key():
    k = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not k:
        raise SystemExit("ODDS_API_KEY is not set in the environment")
    return k


def _get(path, params, usage_log, tag):
    q = urllib.parse.urlencode(dict(apiKey=_key(), **params))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{BASE}/{path}?{q}", timeout=60) as r:
                h = {k: r.headers.get(k) for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
                rec = dict(ts=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), tag=tag, path=path, params={k: v for k, v in params.items()},
                           cost=int(h.get("x-requests-last") or 0), used=int(h.get("x-requests-used") or 0), remaining=int(h.get("x-requests-remaining") or 0))
                with open(usage_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return json.load(r), rec
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            if e.code in (401, 403, 429) and "quota" in body.lower():
                raise SystemExit(f"Odds API refused the call ({e.code}): {body}")
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(3 * (attempt + 1)); continue
            raise
    raise RuntimeError("unreachable")


def spent(usage_log):
    tot = 0
    if Path(usage_log).exists():
        for ln in open(usage_log):
            if ln.strip():
                tot += json.loads(ln).get("cost", 0)
    return tot


def pull(season, rows_path, out_dir, budget=BUDGET_STOP, max_games=None):
    """Closing player_threes for every game in the walk-forward rows (one snapshot per game, cached)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = out_dir / "usage.jsonl"
    rows = pd.read_csv(rows_path, dtype={"player_id": str, "event_id": str}, usecols=["event_id", "date", "team", "opp", "home"])
    games = rows.drop_duplicates("event_id")
    games = games.assign(home_team=np.where(games.home == 1, games.team, games.opp), away_team=np.where(games.home == 1, games.opp, games.team))
    print(f"[close] {len(games)} games in the rows, {games.date.min()}..{games.date.max()}; spent so far {spent(usage):,} credits")
    events_cache = out_dir / "events_by_date.json"
    ev_by_date = read_json(events_cache, {})
    done = 0
    for date, g in games.groupby("date"):
        if date not in ev_by_date:
            # snapshot at 23:59Z of the US date: lists that day's events (tip times are ET evenings = 23:00Z-03:00Z next day)
            snap = f"{date}T23:59:59Z"
            s = spent(usage)
            if s > budget:
                print(f"[close] BUDGET STOP: {s:,} credits spent > {budget:,}; {done} games pulled this run"); return False
            d, rec = _get("events", dict(date=snap), usage, "nba_threes_events")
            ev_by_date[date] = dict(snapshot=d.get("timestamp"), events=[dict(id=e["id"], home=TEAM_NAMES.get(e["home_team"], e["home_team"]),
                                                                              away=TEAM_NAMES.get(e["away_team"], e["away_team"]), commence=e["commence_time"]) for e in d.get("data", [])])
            write_json(events_cache, ev_by_date)
        by_pair = {(e["home"], e["away"]): e for e in ev_by_date[date]["events"]}
        for r in g.itertuples():
            fn = out_dir / f"{r.event_id}.json"
            if fn.exists():
                continue
            e = by_pair.get((r.home_team, r.away_team))
            if not e:
                write_json(fn, dict(event_id=r.event_id, missing="no Odds API event for this matchup on this date", date=date, home=r.home_team, away=r.away_team)); continue
            # the close: the latest snapshot at or before the tip (date = commence_time)
            if spent(usage) > budget:
                print(f"[close] BUDGET STOP: {spent(usage):,} credits spent > {budget:,}; {done} games pulled this run"); return False
            d, rec = _get(f"events/{e['id']}/odds", dict(regions="us", markets="player_threes", date=e["commence"], oddsFormat="american"), usage, "nba_threes_close")
            write_json(fn, dict(event_id=r.event_id, odds_event=e, snapshot=d.get("timestamp"), data=d.get("data", d), cost=rec["cost"], remaining=rec["remaining"]))
            done += 1
            if max_games and done >= max_games:
                print(f"[close] stopped after {done} games (--max-games)"); return True
            if done % 50 == 0:
                print(f"[close] {done} games pulled, {spent(usage):,} credits spent, {rec['remaining']:,} remaining", flush=True)
            time.sleep(0.15)
    print(f"[close] done: {done} games pulled this run, {spent(usage):,} credits spent in total")
    return True


# ---------------- step 5: grade ----------------
def implied(price):
    price = float(price)
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def closing_rows(out_dir):
    """One row per (event, player): modal line, no-vig close P(over) (median over two-sided books), books, snapshot."""
    out = []
    for fn in sorted(out_dir.glob("*.json")):
        if fn.name in ("events_by_date.json",):
            continue
        d = read_json(fn, None)
        if not d or d.get("missing"):
            continue
        quotes = {}   # player -> line -> [(book, over_p, under_p)]
        for b in d["data"].get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] != "player_threes":
                    continue
                sides = {}
                for o in m["outcomes"]:
                    sides.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = o["price"]
                for (player, line), s in sides.items():
                    if line is None or "Over" not in s or "Under" not in s:
                        continue
                    po, pu = implied(s["Over"]), implied(s["Under"])
                    quotes.setdefault(player, {}).setdefault(float(line), []).append((b["title"], po / (po + pu)))
        for player, lines in quotes.items():
            line, qs = max(lines.items(), key=lambda kv: (len(kv[1]), -kv[0]))   # modal line, ties -> lower
            out.append(dict(event_id=d["event_id"], player=player, name_key=norm_name(player), line=line, close_p=float(np.median([p for _, p in qs])),
                            n_books=len(qs), books=",".join(sorted(b for b, _ in qs)), snapshot=d.get("snapshot"), tip=d["odds_event"]["commence"]))
    return pd.DataFrame(out)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logistic_cluster(y, X, groups, iters=100):
    """IRLS logistic MLE and CR1 cluster-robust sandwich covariance (clusters = groups). Returns coef, se, z, one-sided p for > 0."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]); g = X.T @ (y - p)
        step = np.linalg.solve(H, g); w = w + step
        if np.abs(step).max() < 1e-9: break
    p = 1 / (1 + np.exp(-X @ w)); resid = y - p
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    G = pd.Series(groups).astype(str).values; meat = np.zeros((X.shape[1], X.shape[1]))
    for gid in np.unique(G):
        m = G == gid; s = (X[m] * resid[m][:, None]).sum(axis=0); meat += np.outer(s, s)
    n, k, ng = len(y), X.shape[1], len(np.unique(G))
    cr1 = (ng / (ng - 1)) * ((n - 1) / (n - k))
    V = cr1 * Hinv @ meat @ Hinv
    se = np.sqrt(np.diag(V)); z = w / se
    from math import erf, sqrt
    p_one = [0.5 * (1 - erf(zz / sqrt(2))) for zz in z]      # one-sided, H1: coef > 0
    return dict(coef=[round(float(v), 4) for v in w], se=[round(float(v), 4) for v in se], z=[round(float(v), 3) for v in z], p_one_sided=[round(float(v), 5) for v in p_one], n=int(n), clusters=int(ng))


def fetch_underdog_payout(picks=PICKS_IN_ENTRY):
    """Underdog's live payout for a Standard entry of `picks` picks. Tries the public API; the source and timestamp are recorded.
       Returns (multiplier, source) or (None, reason). Never hardcoded: without a live value grading reports the gap."""
    src = os.environ.get("UNDERDOG_PAYOUT_JSON")           # optional: a saved live capture {"picks":4,"multiplier":x,"source":...,"captured":...}
    if src and Path(src).exists():
        d = read_json(src, {});
        if d.get("picks") == picks and d.get("multiplier"):
            return float(d["multiplier"]), f"{d.get('source', src)} captured {d.get('captured')}"
    for url in ("https://api.underdogfantasy.com/v2/pickem_pool_styles", "https://api.underdogfantasy.com/v1/pickem_pool_styles"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.load(r)
            for ps in d.get("pickem_pool_styles", d if isinstance(d, list) else []):
                tbl = ps.get("payout_multipliers") or ps.get("payouts") or {}
                if str(picks) in {str(k) for k in tbl} and "standard" in str(ps).lower():
                    return float(tbl[str(picks)] if str(picks) in tbl else tbl[picks]), f"{url} {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ}"
        except Exception:
            continue
    return None, "no live Underdog payout source reachable (public pool-style endpoints 404; the estimate endpoint needs a session)"


def grade(season, rows_path, out_dir, report_path, season_label, payout=None, payout_source=None):
    rows = pd.read_csv(rows_path, dtype={"player_id": str, "event_id": str})
    rows = rows[rows.act_min > 0].copy(); rows["name_key"] = rows.player.map(norm_name)
    close = closing_rows(out_dir)
    if close.empty:
        raise SystemExit("no closing rows pulled yet")
    d = rows.merge(close, on=["event_id", "name_key"], how="inner")
    d["k"] = (d.line + 0.5).round().astype(int)
    kcols = {k: f"p_tpm_{k}" for k in range(1, 9)}
    d = d[d.k.isin(kcols)].copy()
    d["model_p"] = [float(r[kcols[int(r.k)]]) for _, r in d.iterrows()]
    d["hit"] = (d.act_tpm > d.line).astype(int)
    d["week"] = pd.to_datetime(d.date).dt.strftime("%G-W%V")
    if payout is None:
        payout, payout_source = fetch_underdog_payout()
    be = (1 / payout) ** (1 / PICKS_IN_ENTRY) if payout else None
    d["edge"] = d.model_p - d.close_p
    d["pick"] = (d.edge >= EDGE_PP) & ((d.model_p >= be) if be is not None else False)
    graded = d

    def encompassing(g):
        X = np.column_stack([np.ones(len(g)), logit(g.model_p), logit(g.close_p)])
        r = logistic_cluster(g.hit.values, X, g.event_id.values)
        r["model_coef_positive"] = r["coef"][1] > 0; r["model_p_value"] = r["p_one_sided"][1]
        r["pass"] = bool(r["coef"][1] > 0 and r["p_one_sided"][1] < 0.05)
        return r

    def picks_stat(g):
        p = g[g.pick]
        if not len(p): return dict(n=0, won=0, win_rate=None, break_even=be, pass_rate=False, pass_n=False)
        wr = float(p.hit.mean())
        return dict(n=int(len(p)), won=int(p.hit.sum()), win_rate=round(wr, 4), break_even=round(be, 4) if be else None,
                    pass_rate=bool(be is not None and wr > be), pass_n=bool(len(p) >= MIN_PICKS), avg_edge=round(float(p.edge.mean()), 4))

    a = encompassing(graded)
    b = dict(n=int(len(graded)), **{"pass": bool(len(graded) >= MIN_GRADED)})
    e = picks_stat(graded); e["pass"] = bool(e["pass_rate"] and e["pass_n"])
    # (c) drop the best week by pick win rate
    wk = graded[graded.pick].groupby("week").hit.agg(["mean", "size"]).reset_index()
    best_week = wk.sort_values(["mean", "size"], ascending=[False, False]).week.iloc[0] if len(wk) else None
    g2 = graded[graded.week != best_week] if best_week else graded
    a2, e2 = encompassing(g2), picks_stat(g2); e2["pass"] = bool(e2["pass_rate"] and e2["pass_n"])
    c = dict(best_week=best_week, dropped_rows=int(len(graded) - len(g2)), encompassing=a2, picks=e2, **{"pass": bool(a2["pass"] and e2["pass"])})
    # (d) decile calibration of model P at the graded line
    bins = pd.cut(graded.model_p, np.linspace(0, 1, 11), include_lowest=True)
    cal = []
    for b_, g in graded.groupby(bins, observed=True):
        cal.append(dict(bucket=str(b_), n=int(len(g)), model=round(float(g.model_p.mean()), 4), actual=round(float(g.hit.mean()), 4), off=round(abs(float(g.model_p.mean() - g.hit.mean())), 4)))
    dd = dict(buckets=cal, worst_off_at_n30=max([x["off"] for x in cal if x["n"] >= MIN_BUCKET], default=0.0))
    dd["pass"] = bool(dd["worst_off_at_n30"] <= MAX_BUCKET_OFF)
    overall = bool(a["pass"] and b["pass"] and c["pass"] and dd["pass"] and e["pass"])
    close_only = dict(brier_model=round(float(((graded.model_p - graded.hit) ** 2).mean()), 5), brier_close=round(float(((graded.close_p - graded.hit) ** 2).mean()), 5),
                      brier_half=round(float(((0.5 * graded.model_p + 0.5 * graded.close_p - graded.hit) ** 2).mean()), 5),
                      mean_model=round(float(graded.model_p.mean()), 4), mean_close=round(float(graded.close_p.mean()), 4), hit_rate=round(float(graded.hit.mean()), 4))
    doc = dict(kind="closing-line backtest", market="player_threes", season=season_label, rule="backtests/nba_threes_2024_25_rule.md",
               games=int(graded.event_id.nunique()), games_with_close=int(close.event_id.nunique()), rows=int(len(graded)), books_median=float(graded.n_books.median()),
               payout=dict(picks=PICKS_IN_ENTRY, multiplier=payout, source=payout_source, break_even_per_pick=round(be, 4) if be else None),
               pick_rule=f"model P >= no-vig close + {EDGE_PP:.0%} and model P >= (1/payout)^(1/{PICKS_IN_ENTRY})",
               grade=dict(a_encompassing=a, b_sample=b, c_drop_best_week=c, d_calibration=dd, e_picks=e, overall_pass=overall),
               summary=close_only, by_week=[dict(week=w, n=int(len(g)), picks=int(g.pick.sum()), pick_wins=int(g[g.pick].hit.sum()), hit_rate=round(float(g.hit.mean()), 3)) for w, g in graded.groupby("week")],
               credits=dict(spent=spent(out_dir / "usage.jsonl"), budget_stop=BUDGET_STOP),
               notes=["Positions table (data/positions_2025.csv) is built from full-season 2023-24 and 2024-25 play-style data: a minor lookahead that affects the defense-vs-position factor only (clamped to +/-2%).",
                      "No 2024-25 or 2023-24 data was used to fit any model parameter, prior, cap, clamp or calibration (constants measured on the 2025-26 backfill or hand-set; no calibration.json on main).",
                      "No injury-news inputs and no spreads (blowout shave off); absences read off the box score."],
               updated=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
    write_json(report_path, doc)
    graded[["event_id", "date", "player", "team", "opp", "line", "model_p", "close_p", "n_books", "hit", "pick", "week"]].to_csv(out_dir / "graded_rows.csv", index=False)
    print(f"[grade] {doc['rows']} graded rows over {doc['games']} games | (a) coef {a['coef'][1]} p {a['model_p_value']} {'PASS' if a['pass'] else 'FAIL'} | (b) n {b['n']} {'PASS' if b['pass'] else 'FAIL'} | "
          f"(c) drop {best_week} {'PASS' if c['pass'] else 'FAIL'} | (d) worst bucket {dd['worst_off_at_n30']} {'PASS' if dd['pass'] else 'FAIL'} | (e) picks {e['n']} win {e['win_rate']} vs BE {e['break_even']} {'PASS' if e['pass'] else 'FAIL'} | overall {'PASS' if overall else 'FAIL'}")
    return doc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "grade", "all"])
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--budget", type=int, default=BUDGET_STOP)
    ap.add_argument("--max-games", type=int, default=None, help="pull at most this many new games (smoke test)")
    ap.add_argument("--payout", type=float, default=None, help="Underdog 4-pick payout multiplier from a live capture (with --payout-source)")
    ap.add_argument("--payout-source", default=None)
    args = ap.parse_args()
    out_dir = DATA_DIR / f"odds_close_{args.season}"
    label = f"{args.season - 1}-{str(args.season)[2:]}"
    report_path = SITE_DATA / f"backtest_threes_{args.season - 1}_{str(args.season)[2:]}.json"
    ok = True
    if args.cmd in ("pull", "all"):
        ok = pull(args.season, Path(args.rows), out_dir, args.budget, args.max_games)
    if args.cmd in ("grade", "all") and ok:
        grade(args.season, Path(args.rows), out_dir, report_path, label, args.payout, args.payout_source)
