"""Walk-forward backtest of the projection model over one season. HISTORICAL SIMULATION.

For every game date, the model is rebuilt from the games BEFORE that date only (defense vs
position, player rates, minutes baselines, absorber history), each playing team is projected,
and the projections are scored against that night's box scores. Nothing here touches the live
paper log (data/paper_log.csv) or results.json; the outputs are their own files:

  data/backtest_rows.csv.gz     one row per projected player-game (means, actuals, key probabilities)
  docs/data/backtest.json       the report the Backtest page reads

  python backtest.py run      [--start 2025-11-01] [--end YYYY-MM-DD]   rebuild the rows (slow: ~10 min)
  python backtest.py report                                              rebuild backtest.json from the rows
  python backtest.py          both

What the simulation knows at "tip-off" of each date, and what it does not:
  - Rosters: players who appeared for the team in the previous ROSTER_DAYS days, plus anyone
    in tonight's box score (a returning player is on the injury report as available).
  - Absences: rostered players missing from tonight's box score are treated as OUT. That is the
    pre-tip injury report, which books also price from, but it is read off the box score, so a
    late scratch counts as "known" here when it may not have been known at close.
  - No spreads (no historical odds), so the blowout shave is off. No injury news text, so no
    QUESTIONABLE widening or minutes limits. Play-style positions come from data/positions.csv,
    which is built from the whole season (positions move the sim by at most +-2%).
  - Players are only scored when they actually played; a projection for a DNP is dropped.
"""
import argparse
import datetime as dt
import gzip
import io
import sys
import time

import numpy as np
import pandas as pd

import build_model as bm
from common import DATA_DIR, SITE_DATA, write_json
from positions import position_map

ROWS = DATA_DIR / "backtest_rows.csv.gz"
REPORT = SITE_DATA / "backtest.json"
ROSTER_DAYS = 21
DEFAULT_START = "2025-11-01"
SCORED = ["min", "tpm", "pts", "reb", "ast", "pra"]
THREES_K = [1, 2, 3, 4]
LINE_STATS = ["pts", "reb", "ast", "pra"]
OFFSETS = [-3, -2, -1, 0, 1, 2, 3]   # alt lines around the fair line used for the calibration buckets
BUCKETS = [(i / 10, (i + 1) / 10) for i in range(10)]
# Checked 2026-10-08 (docs.oddsblaze.com, the-odds-api.com). OddsBlaze's site is script-rendered, so its prices
# come from its own pricing page as indexed by search, not from a page I could render.
MARKET_COST = ("What it would cost: OddsBlaze's published plans are $299/month (300 requests/min, Rewind, CLV and OLV) "
               "and $999/month (2,000 requests/min, plus line movement). Its Rewind endpoint returns a whole book at any "
               "timestamp and its Historical Odds endpoint returns opening and closing prices per odds id, so one month "
               "covers a season pull, but how far back the archive reaches is not documented: confirm it holds October "
               "2025 before paying. Cheaper route: The Odds API keeps historical player props from May 2023 at 5-minute "
               "snapshots for 10 credits per market per region per game. About 1,230 games x 5 markets is roughly 62,000 "
               "credits, inside its $59/month 100K plan for a single region, or the $119/month 5M plan if a second region "
               "is added for Pinnacle.")


# ---------------- run ----------------
def p_at_least(pmf, k):
    return float(sum(pmf[k:])) if k < len(pmf) else 0.0


def cdf_at(pmf, m):
    return float(sum(pmf[:m + 1]))


def fair_line(pmf):
    """The x.5 line whose P(over) is closest to 50%."""
    best, best_gap = 0.5, 9
    for m in range(len(pmf)):
        p_over = 1 - cdf_at(pmf, m)
        if abs(p_over - 0.5) < best_gap:
            best, best_gap = m + 0.5, abs(p_over - 0.5)
        if p_over < 0.5 - best_gap:
            break
    return best


def p_over(pmf, line):
    return 1 - cdf_at(pmf, int(np.floor(line)))


def trailing_means(df):
    """Naive comparison: each player's trailing-10 average before each game (NaN for the first game)."""
    d = df.sort_values(["player_id", "date"])
    out = {}
    for s in SCORED:
        col = (d.pts + d.reb + d.ast) if s == "pra" else d[s]
        out[s] = col.groupby(d.player_id).transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    t = pd.DataFrame(out, index=d.index)
    t.columns = [f"naive_{s}" for s in SCORED]
    return pd.concat([d[["player_id", "event_id"]], t], axis=1)


def project_team(team, opp, home, hist_by_id, played, recent_ids, pos_map, cur, lg_pct, dvp, history):
    """Rebuild one team the way build_model.main does for a live slate, from history only."""
    roster_ids = set(recent_ids) | set(played)
    team_players, outs = [], []
    for pid in roster_ids:
        g = hist_by_id.get(pid)
        if g is None or len(g) < 3:
            continue
        prof = bm.profile(g, cur, lg_pct, None)
        entry = {"id": pid, "name": g.iloc[0].player, "team": team, "opp": opp, "home": home,
                 "pos": pos_map.get(pid) or g.iloc[-1].pos, "status": None, "flags": [], **prof}
        if pid not in played:
            entry["status"] = "OUT"
            outs.append(entry)
            continue
        if prof["base_min"] < 6:
            continue
        entry["min"] = prof["base_min"]
        team_players.append(entry)
    bm.allocate_minutes(team_players, outs, team, history)
    team_players = [p for p in team_players if p["min"] >= bm.ROTATION_FLOOR]
    absent_starters = sum(1 for o in outs if o["starter_rate"] >= 0.5)
    rows = []
    for p in team_players:
        sim = bm.simulate(p, opp, dvp, absent_starters)
        row = {"player_id": p["id"], "player": p["name"], "team": team, "opp": opp, "home": home, "pos": p["pos"],
               "proj_min": round(p["min"], 2), "absent_starters": absent_starters, "n_cur": p["n_cur"],
               "baseline": p["baseline"]}
        for s in ["tpm", "pts", "reb", "ast", "pra"]:
            row[f"proj_{s}"] = sim[s]["mean"]
        for k in THREES_K:
            row[f"p_tpm_{k}"] = round(p_at_least(sim["tpm"]["pmf"], k), 4)
        for s in LINE_STATS:
            pmf = sim[s]["pmf"]
            fl = fair_line(pmf)
            row[f"fair_{s}"] = fl
            for off in OFFSETS:
                line = fl + off
                row[f"po_{s}_{off:+d}"] = round(p_over(pmf, line), 4) if line > 0 else np.nan
        rows.append(row)
    return rows


def run(start, end=None):
    df = bm.load_games()
    cur = int(df.season.max())
    df["pra"] = df.pts + df.reb + df.ast
    pos_map = position_map()
    naive = trailing_means(df)
    dates = sorted(d for d in df.date.unique() if d >= pd.Timestamp(start) and (end is None or d <= pd.Timestamp(end)))
    print(f"[backtest] {len(dates)} dates {dates[0].date()} .. {dates[-1].date()}, season {cur}")
    all_rows = []
    t0 = time.time()
    for i, day in enumerate(dates, 1):
        hist = df[df.date < day]
        today = df[df.date == day]
        dvp, _ = bm.build_dvp(hist, cur)
        lg_pct = float(hist.tpm.sum() / max(hist.tpa.sum(), 1))
        history = bm.AbsorberHistory(hist, day)
        teams_today = set(today.team)
        recent = hist[(hist.date >= day - pd.Timedelta(days=ROSTER_DAYS)) & hist.team.isin(teams_today)]
        recent_ids = recent.groupby("team").player_id.apply(set).to_dict()
        hist_by_id = {pid: g.sort_values("date", ascending=False)
                      for pid, g in hist[hist.player_id.isin(set(recent.player_id) | set(today.player_id))].groupby("player_id")}
        day_rows = []
        for event_id, game in today.groupby("event_id"):
            sides = {int(h): t for t, h in game[["team", "home"]].drop_duplicates().values}
            if len(sides) != 2:
                continue
            for home, team in sides.items():
                opp = sides[1 - home]
                box = game[game.team == team]
                played = set(box.loc[box["min"] > 0, "player_id"])
                rows = project_team(team, opp, home, hist_by_id, played, recent_ids.get(team, set()), pos_map, cur, lg_pct, dvp, history)
                actual = box.set_index("player_id")
                for r in rows:
                    a = actual.loc[r["player_id"]]
                    r.update({"date": day.date().isoformat(), "event_id": event_id, "starter": int(a.starter),
                              **{f"act_{s}": float(a[s]) for s in SCORED}})
                day_rows.append(pd.DataFrame(rows))
                day_rows[-1]["n_played"] = len(played)
        all_rows.extend(day_rows)
        if i % 10 == 0 or i == len(dates):
            print(f"[backtest] {i}/{len(dates)} {day.date()}  {sum(len(x) for x in all_rows)} rows  {time.time() - t0:.0f}s")
    out = pd.concat(all_rows, ignore_index=True)
    out = out.merge(naive, on=["player_id", "event_id"], how="left")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROWS, "wt", encoding="utf-8", newline="") as f:
        out.to_csv(f, index=False)
    print(f"[backtest] wrote {len(out)} rows to {ROWS.name}")
    return out


# ---------------- report ----------------
def mae_table(d):
    out = {}
    for s in SCORED:
        err = d[f"proj_{s}"] - d[f"act_{s}"]
        nv = d[f"naive_{s}"] - d[f"act_{s}"]
        big = d[d.proj_min >= 24]
        out[s] = {"mae": round(float(err.abs().mean()), 2), "bias": round(float(err.mean()), 2),
                  "naive_mae": round(float(nv.abs().mean()), 2),
                  "mae_24plus": round(float((big[f"proj_{s}"] - big[f"act_{s}"]).abs().mean()), 2),
                  "naive_mae_24plus": round(float((big[f"naive_{s}"] - big[f"act_{s}"]).abs().mean()), 2),
                  "n": int(len(d))}
    return out


def buckets(pred, hit):
    rows = []
    for lo, hi in BUCKETS:
        m = (pred >= lo) & (pred < hi) if hi < 1 else (pred >= lo)
        if m.sum() == 0:
            continue
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "pred": round(float(pred[m].mean()), 3),
                     "actual": round(float(hit[m].mean()), 3)})
    return rows


def brier(pred, hit):
    return round(float(((pred - hit) ** 2).mean()), 4)


def threes_report(d):
    out = {}
    for k in THREES_K:
        pred = d[f"p_tpm_{k}"].values
        hit = (d.act_tpm >= k).astype(float).values
        base = float(hit.mean())
        out[str(k)] = {"n": int(len(d)), "pred": round(float(pred.mean()), 3), "actual": round(base, 3),
                       "brier": brier(pred, hit), "brier_base": round(float(((base - hit) ** 2).mean()), 4),
                       "buckets": buckets(pred, hit)}
    return out


def lines_report(d):
    out = {}
    for s in LINE_STATS:
        fair = d[f"fair_{s}"]
        at = {"n": int(len(d)), "pred": round(float(d[f"po_{s}_+0"].mean()), 3),
              "over": round(float((d[f"act_{s}"] > fair).mean()), 3)}
        preds, hits = [], []
        for off in OFFSETS:
            p = d[f"po_{s}_{off:+d}"]
            ok = p.notna()
            preds.append(p[ok].values)
            hits.append((d.loc[ok, f"act_{s}"] > (fair[ok] + off)).astype(float).values)
        pred, hit = np.concatenate(preds), np.concatenate(hits)
        out[s] = {"at_fair": at, "alt_n": int(len(pred)), "brier": brier(pred, hit), "brier_base": brier(np.full_like(pred, 0.5), hit),
                  "buckets": buckets(pred, hit),
                  "by_offset": [{"offset": off, "n": int(d[f"po_{s}_{off:+d}"].notna().sum()),
                                 "pred": round(float(d[f"po_{s}_{off:+d}"].mean()), 3),
                                 "over": round(float((d[f"act_{s}"] > fair + off)[d[f"po_{s}_{off:+d}"].notna()].mean()), 3)}
                                for off in OFFSETS]}
    return out


def monthly(d):
    out = []
    for m, g in d.groupby(d.date.str[:7]):
        row = {"month": m, "n": int(len(g)), "dates": int(g.date.nunique())}
        for s in SCORED:
            row[f"mae_{s}"] = round(float((g[f"proj_{s}"] - g[f"act_{s}"]).abs().mean()), 2)
        row["brier_tpm_1"] = brier(g.p_tpm_1.values, (g.act_tpm >= 1).astype(float).values)
        row["brier_tpm_2"] = brier(g.p_tpm_2.values, (g.act_tpm >= 2).astype(float).values)
        row["over_pts_fair"] = round(float((g.act_pts > g.fair_pts).mean()), 3)
        out.append(row)
    return out


def by_absent(d):
    """Bias split by how many regular starters the model thought were out (usage bump and redistribution both key off it)."""
    out = []
    for g, x in d.groupby(d.absent_starters.clip(upper=2)):
        r_proj, r_act = x.proj_pts.sum() / x.proj_min.sum() * 36, x.act_pts.sum() / x.act_min.sum() * 36
        out.append({"absent": int(g), "n": int(len(x)),
                    "bias_min": round(float((x.proj_min - x.act_min).mean()), 2),
                    "bias_pts": round(float((x.proj_pts - x.act_pts).mean()), 2),
                    "pts36_proj": round(float(r_proj), 2), "pts36_act": round(float(r_act), 2),
                    "over_pts_fair": round(float((x.act_pts > x.fair_pts).mean()), 3),
                    "p1_pred": round(float(x.p_tpm_1.mean()), 3), "p1_act": round(float((x.act_tpm >= 1).mean()), 3)})
    return out


def report(rows=None):
    d = rows if rows is not None else pd.read_csv(ROWS, dtype={"player_id": str, "event_id": str})
    d = d[d.act_min > 0].copy()
    played_total = int(d.groupby("event_id").n_played.first().sum() * 1)  # per team-game count of players who played
    team_games = d.groupby(["event_id", "team"]).n_played.first()
    doc = {
        "kind": "historical simulation",
        "season": "2025-26",
        "start": d.date.min(), "end": d.date.max(),
        "dates": int(d.date.nunique()), "games": int(d.event_id.nunique()),
        "rows": int(len(d)), "played_rows": int(team_games.sum()),
        "coverage": round(float(len(d) / team_games.sum()), 3),
        "notes": {"roster_days": ROSTER_DAYS, "absences": "rostered players missing from the box score are treated as OUT (injury-report proxy)",
                  "no_spreads": True, "no_news": True, "positions_full_season": True},
        "mae": mae_table(d),
        "threes": threes_report(d),
        "lines": lines_report(d),
        "monthly": monthly(d),
        "by_absent": by_absent(d),
        "market": {"available": False,
                   "reason": "No historical closing lines: the OddsBlaze trial key expired 2026-09-16 and no paid plan is active, "
                             "so what the model would have bet, its hit rate and ROI against close cannot be computed.",
                   "cost": MARKET_COST},
        "updated": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0).isoformat() + "Z",
    }
    write_json(REPORT, doc)
    print(f"[backtest] report: {doc['rows']} player-games over {doc['dates']} dates; MAE pts {doc['mae']['pts']['mae']} "
          f"(naive {doc['mae']['pts']['naive_mae']}), min {doc['mae']['min']['mae']}; Brier 1+ threes {doc['threes']['1']['brier']}")
    return doc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["run", "report", "all"])
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    rows = run(args.start, args.end) if args.cmd in ("run", "all") else None
    if args.cmd in ("report", "all"):
        report(rows)
