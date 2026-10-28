"""Fantasy Points v2 (backtests/fantasy_points_v2_rule.md). A calibration repair of v1: the joint-sim machinery, FP formula,
sources and gate math are v1's. Four changes, nothing else:

  2a  minutes: sd x sd_factor, short-night mixture (probability q, centred at 0.55 x projected minutes), truncated at 0
      -> fantasy_points.simulate_fp(sd_factor=, q=), v1 at the defaults (1.0, 0.0)
  2b  rates: each per-minute rate = (1 - w) * prior + w * current-season rate, w = GP / (GP + 10), GP = games played this
      season at log time; prior = the rate v1 uses (its engine weighting with the frozen position shrink), so GP = 0 is v1
  2c  isotonic recalibration of P(over), fitted on the v2 replay of the graded v1 lines, frozen in data/fp_v2_recal.json
  2d  grader hygiene: 0-minute or absent player-games void at grade time

Phase A (rule section 4):
  python fp_v2.py replay [--start 2025-11-01] [--end 2026-04-12] [--season 2026]   walk-forward 2025-26 replay of the 3x3 grid
                                                                                   -> data/fp_v2_replay.csv.gz
  python fp_v2.py grid                                                            PIT deciles, weighted |gap|, worst decile,
                                                                                   mean FP / minutes bias per pair per window;
                                                                                   SELECTION on Oct-Jan, VERDICT on Feb-Apr
                                                                                   -> docs/data/fp_v2_backtest.json
  python fp_v2.py select                                                          freeze the mechanically selected pair
                                                                                   -> data/fp_v2_params.json (only on PASS)
Phase B (rule sections 2c, 5):
  python fp_v2.py retro                 replay v2 (frozen pair, log-time inputs) over the graded v1 lines -> data/fp_v2_retro.csv
  python fp_v2.py fit                   isotonic P(over) map on the retro rows -> data/fp_v2_recal.json (frozen), per-source gaps
  python fp_v2.py recount               rule 5 re-count of the power-calc slice on unique lines (design effect)
  python fp_v2.py snapshot              append tonight's v2 distributions (projections.json fp_v2 block) -> data/fp_v2_log.csv
  python fp_v2.py price <underdog.json> UD standard FP lines, BOTH sides' prices recorded -> data/fp_v2_prices.csv, docs/data/fp_v2_pricing.json
  python fp_v2.py price_pp [copy.json]  PrizePicks standard Fantasy Score lines -> data/fp_v2_pp_prices.csv
  python fp_v2.py grade                 one row per line, DNP void, calibration verdict at 200, market test at 800
                                        -> data/fp_v2_grade.csv, docs/data/fp_v2_performance.json
Nothing here touches the v1 code path, log, grades or tab. Every JSON goes through common.write_json (strict).
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_model as bm  # noqa: E402
import fantasy_points as fp  # noqa: E402
from common import DATA_DIR, SITE_DATA, norm_name, read_json, write_json  # noqa: E402
from ladder_calibrate import apply_map, fit_map, logloss  # noqa: E402  (fit_map/apply_map: the rule-5 recount keeps the house interpolating map for the v1 power slice)
from positions import position_map  # noqa: E402

RULE = "backtests/fantasy_points_v2_rule.md"
REPLAY = DATA_DIR / "fp_v2_replay.csv.gz"
BACKTEST = SITE_DATA / "fp_v2_backtest.json"
PARAMS = DATA_DIR / "fp_v2_params.json"
RETRO = DATA_DIR / "fp_v2_retro.csv"
RECAL = DATA_DIR / "fp_v2_recal.json"
RECOUNT = SITE_DATA / "fp_v2_recount.json"
LOG = DATA_DIR / "fp_v2_log.csv"
PRICES = DATA_DIR / "fp_v2_prices.csv"
PP_PRICES = DATA_DIR / "fp_v2_pp_prices.csv"
GRADE = DATA_DIR / "fp_v2_grade.csv"
PRICING = SITE_DATA / "fp_v2_pricing.json"
PERF = SITE_DATA / "fp_v2_performance.json"
SD_GRID = (1.25, 1.50, 1.75)
Q_GRID = (0.10, 0.15, 0.20)
GRID = [(s, q) for s in SD_GRID for q in Q_GRID]          # tie-break order: smallest factor, then smallest q (rule 2a)
V1_REF = (1.0, 0.0)                                         # reference row only, never a candidate
K_BLEND = 10.0                                              # rule 2b: w = GP / (GP + 10)
BLEND_COMPONENTS = ("tpa", "non3", "reb", "ast")           # engine rates the FP sim reads from p["rates"]
FP_COMPONENTS = ("stl", "blk", "tov")                       # rates the FP sim reads from the FP profile
REPLAY_START, REPLAY_END, REPLAY_SEASON = "2025-11-01", "2026-04-12", 2026   # the ladder backtests' 2025-26 replay window
SELECT_END = "2026-01-31"                                   # selection window Oct-Jan (replay starts Nov 1); verdict Feb-Apr
WMAE_BAR, WORST_BAR = 0.05, 0.10                            # the bars v1 failed (rule 4)
CHECKPOINT_N, GATE_N, MARKET_N, MIN_DECILE, THIN = 200, 800, 800, 20, 30   # rule 5 as amended (Amendment 1): checkpoint at 200, both verdicts at 800
MIN_BLOCK = 20                                              # Amendment 1, 2c: minimum isotonic block size
GATE_EPOCH, PRE_GATE = "A1", "pre-A1"                       # log rows: the forward gate that counts them; pre-A1 = before the Amendment 1 refit, excluded
FIT_DATES = ("2026-10-23", "2026-10-26")                    # the graded v1 slates of the original map fit; the Amendment 1 refit uses the same lines
BOOT_ITERS, SEED = 2000, 20261027
USAGE_FLAG = {"Usage +6% pts": 1, "Usage +11% pts": 2}     # build_model's flag for 1 / 2 absent starters (USAGE_BUMP pts 5.5%)
LABEL = "FP v2 - paper study. Hit rate measures accuracy, not profit."
LOG_FIELDS = ["date", "event_id", "tip", "player_id", "player", "team", "opp", "pos", "min", "gp", "w", "sd_factor", "q", "fp_mean", "fp_sd", "fp_q", "logged_at", "gate"]
PRICE_FIELDS = ["captured_at", "date", "event_id", "player", "team", "opp", "market", "line", "side", "mult", "decimal", "american", "convention", "breakeven",
                "decimal_higher", "decimal_lower", "american_higher", "american_lower", "p_book_novig", "p_raw", "p_cal", "edge", "source", "source_id", "fp_logged_at", "map_fit"]
PP_PRICE_FIELDS = ["captured_at", "date", "event_id", "player", "team", "opp", "market", "line", "side", "p_book_novig", "p_raw", "p_cal", "scheduled_at", "game_status", "source", "source_id", "fp_logged_at", "map_fit"]
GRADE_FIELDS = ["date", "event_id", "tip", "player_id", "player", "team", "opp", "pos", "min_proj", "min_actual", "fp_mean", "fp_actual", "pit", "void", "void_reason",
                "source", "market", "line", "p_raw", "p_cal", "p_book", "hit", "ll_v2", "ll_book", "logged_at", "captured_at"]
ET = fp.ET


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_params():
    """The frozen (sd_factor, q) pair, or None before phase B."""
    p = read_json(PARAMS, None)
    return p if p and p.get("selected") else None


def load_recal():
    return read_json(RECAL, None)


CLIP = (0.05, 0.95)   # Amendment 1 (2026-10-28): was (0.005, 0.995)


def pav_step(x, y):
    """Pool-adjacent-violators isotonic regression returning the fitted STEP function: blocks as (x_lo, x_hi, mean y, n), y
    unclipped (the clip is applied where the map is used). The step over block ranges is the isotonic regression itself;
    interpolating between block means (the ladder maps' display convention) can ramp through sparse tail blocks and is not used."""
    order = np.argsort(x, kind="stable"); x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    lo, hi, by, bn = [], [], [], []
    for xi, yi in zip(x, y):
        lo.append(xi); hi.append(xi); by.append(yi); bn.append(1)
        while len(by) > 1 and by[-2] > by[-1]:
            n = bn[-2] + bn[-1]
            by[-2] = (by[-2] * bn[-2] + by[-1] * bn[-1]) / n; hi[-2] = hi[-1]; bn[-2] = n
            by.pop(); lo.pop(); hi.pop(); bn.pop()
    return {"x_lo": [round(float(v), 5) for v in lo], "x_hi": [round(float(v), 5) for v in hi], "y": [round(float(v), 5) for v in by], "n": [int(v) for v in bn]}


def merge_small_blocks(m, min_n=MIN_BLOCK):
    """Amendment 1, 2c: after PAV, any block with n < min_n is merged into its nearest neighbour (by the x gap between the block
    ranges; an end block into its only neighbour; a tie toward the lower block), the merged block taking the pooled hit rate.
    Repeated, smallest block first, until every block has n >= min_n (or one block remains). No re-pooling for monotonicity:
    the amendment does not ask for it, and `monotone` records whether the result still is."""
    lo, hi, y, n = list(m["x_lo"]), list(m["x_hi"]), [float(v) for v in m["y"]], [int(v) for v in m["n"]]
    merges = 0
    while len(n) > 1 and min(n) < min_n:
        i = int(np.argmin(n))
        if i == 0:
            j = 1
        elif i == len(n) - 1:
            j = i - 1
        else:
            j = i - 1 if (lo[i] - hi[i - 1]) <= (lo[i + 1] - hi[i]) else i + 1
        a, b = min(i, j), max(i, j)
        tot = n[a] + n[b]
        y[a] = (y[a] * n[a] + y[b] * n[b]) / tot; hi[a] = hi[b]; n[a] = tot
        del lo[b], hi[b], y[b], n[b]
        merges += 1
    return {"x_lo": lo, "x_hi": hi, "y": [round(v, 5) for v in y], "n": n, "y_cal": [round(float(np.clip(v, *CLIP)), 5) for v in y],
            "merges": merges, "min_block": min_n, "monotone": bool(all(y[k] <= y[k + 1] + 1e-12 for k in range(len(y) - 1)))}


def apply_step(m, p):
    """Step map: a point inside a block takes the block's value; between two blocks the step sits at the midpoint of the gap;
    beyond the ends the end block's value. Clipped to CLIP."""
    p = np.asarray(p, dtype=float)
    lo, hi, y = np.array(m["x_lo"]), np.array(m["x_hi"]), np.array(m["y"])
    edges = (hi[:-1] + lo[1:]) / 2 if len(lo) > 1 else np.array([])
    idx = np.searchsorted(edges, p, side="right")
    return np.clip(y[idx], *CLIP)


def cal(m, p_over):
    """Calibrated P(over) through the frozen map (identity when there is no map yet)."""
    if not m:
        return float(p_over)
    return float(apply_step(m["map"], np.array([p_over], dtype=float))[0])


# ---------------- rule 2b: rate blend ----------------
def current_season_rates(g, cur):
    """(per-minute rates this season to date, GP). GP counts this season's games with minutes, as of the rows in g (log time)."""
    gc = g[(g.season == cur) & (g["min"] > 0)]
    gp = int(len(gc)); m = float(gc["min"].sum())
    if gp == 0 or m <= 0:
        return None, 0
    r = {s: float(gc[s].sum() / m) for s in ("tpa", "reb", "ast", "stl", "blk", "tov")}
    r["non3"] = float((gc.pts - 3 * gc.tpm).sum() / m)
    return r, gp


def blend_rates(p, prof, g, cur):
    """v2 copies of the engine player dict and FP profile with rule 2b applied: rate = (1 - w) * v1 rate + w * current-season
    rate, w = GP / (GP + K_BLEND). The v1 objects are not modified. Returns (p2, prof2, GP, w)."""
    rc, gp = current_season_rates(g, cur)
    w = gp / (gp + K_BLEND)
    p2 = {**p, "rates": dict(p["rates"])}
    prof2 = {**prof, "rates": dict(prof["rates"])}
    if rc is not None and w > 0:
        for s in BLEND_COMPONENTS:
            if s in p2["rates"]:
                p2["rates"][s] = (1 - w) * float(p["rates"][s]) + w * rc[s]
        for s in FP_COMPONENTS:
            prof2["rates"][s] = (1 - w) * float(prof["rates"][s]) + w * rc[s]
    return p2, prof2, gp, w


def expected_minutes(mu, sd, sd_factor, q, rng, n=20000):
    """Mean of the 2a minutes draw (truncated, capped like the sim) for the bias report; the sim itself draws its own."""
    m = np.clip(rng.normal(mu, sd * sd_factor, n), 0, 48)
    if q > 0:
        short = rng.random(n) < q
        m[short] = np.clip(rng.normal(fp.SHORT_NIGHT * mu, sd * sd_factor, int(short.sum())), 0, 48)
    return float(m.mean())


def simulate_live(p, prof_fp, g, cur, opp, dvp, absent, pri, rng, params):
    """The live v2 distribution for one player (build_model.main calls this next to the v1 sim, own generator): 2a with the
    frozen pair, 2b from the player's games g. Returns the projections.json block {"fp_v2": {...}}."""
    sel = params["selected"]
    p2, prof2, gp, w = blend_rates(p, prof_fp, g, cur)
    sim = fp.simulate_fp(p2, prof2, opp, dvp, absent, pri, rng, sd_factor=float(sel["sd_factor"]), q=float(sel["q"]))
    return {"fp_v2": {**sim["fp"], "gp": gp, "w": round(w, 4), "sd_factor": sel["sd_factor"], "q_mix": sel["q"]}}   # "q" stays the percentile list


# ---------------- phase A: walk-forward replay ----------------
def configs():
    return [("v1", *V1_REF)] + [(f"s{s:.2f}_q{q:.2f}", s, q) for s, q in GRID]


def replay(start=REPLAY_START, end=REPLAY_END, season=REPLAY_SEASON, out_path=REPLAY, max_days=None):
    """One row per projected player-game (players who played, as in the ladder replay), with the v1 reference distribution and
    every grid pair's v2 distribution (2a + 2b). Mirrors backtest.project_team / build_model.main: profiles from history before
    the day only (current + prior season, the live engine's window), minutes allocation, absent starters from the box score."""
    pri = fp.load_priors()
    if pri is None:
        raise SystemExit("data/fp_priors.json missing")
    df = bm.load_games(include_history=True)
    df = df[df.season <= int(season)].copy()
    cur = int(season)
    prior_dates = df.loc[df.season == cur - 1, "date"]
    prior_cutoff = (prior_dates.max() - pd.Timedelta(days=bm.PRIOR_TAIL_DAYS)) if len(prior_dates) else None
    pos_map = position_map()
    cfgs = configs()
    rngs = {name: np.random.default_rng(SEED + i) for i, (name, _, _) in enumerate(cfgs)}
    rng_min = np.random.default_rng(SEED + 999)
    dates = sorted(d for d in df.date.unique() if d >= pd.Timestamp(start) and d <= pd.Timestamp(end))
    if max_days:
        dates = dates[:max_days]
    print(f"[fp-v2 replay] {len(dates)} dates {dates[0].date()} .. {dates[-1].date()}, season {cur}, {len(cfgs)} configs ({len(GRID)} grid pairs + v1 reference)")
    rows, t0 = [], time.time()
    for i, day in enumerate(dates, 1):
        hist = df[df.date < day]
        today = df[df.date == day]
        dvp, _ = bm.build_dvp(hist, cur)
        lg_pct = float(hist.tpm.sum() / max(hist.tpa.sum(), 1))
        history = bm.AbsorberHistory(hist, day)
        pos_rates = bm.pos_tpa_rates(hist, cur)
        teams_today = set(today.team)
        recent = hist[(hist.date >= day - pd.Timedelta(days=21)) & hist.team.isin(teams_today)]
        recent_ids = recent.groupby("team").player_id.apply(set).to_dict()
        pool = hist[(hist.season >= cur - 1) & hist.player_id.isin(set(recent.player_id) | set(today.player_id))]
        hist_by_id = {pid: g.sort_values("date", ascending=False) for pid, g in pool.groupby("player_id")}
        for event_id, game in today.groupby("event_id"):
            sides = {int(h): t for t, h in game[["team", "home"]].drop_duplicates().values}
            if len(sides) != 2:
                continue
            for home, team in sides.items():
                opp = sides[1 - home]
                box = game[game.team == team].set_index("player_id")
                played = set(box.index[box["min"] > 0])
                team_players, outs = [], []
                for pid in set(recent_ids.get(team, set())) | played:
                    g = hist_by_id.get(pid)
                    if g is None or len(g) < 3:
                        continue
                    prof = bm.profile(g, cur, lg_pct, prior_cutoff)
                    entry = {"id": pid, "name": g.iloc[0].player, "team": team, "opp": opp, "home": home, "pos": pos_map.get(pid) or g.iloc[-1].pos,
                             "status": None, "flags": [], **prof}
                    entry["rates"] = {**prof["rates"], "tpa": bm.tpa_rate_v2(g, cur, entry["pos"], pos_rates)}
                    entry["g"] = g
                    if pid not in played:
                        entry["status"] = "OUT"; outs.append(entry); continue
                    if prof["base_min"] < 6:
                        continue
                    entry["min"] = prof["base_min"]
                    team_players.append(entry)
                bm.allocate_minutes(team_players, outs, team, history)
                team_players = [p for p in team_players if p["min"] >= bm.ROTATION_FLOOR]
                absent = sum(1 for o in outs if o["starter_rate"] >= 0.5)
                for p in team_players:
                    a = box.loc[p["id"]]
                    act_fp = float(fp.fp_of(a)); act_min = float(a["min"])
                    prof_fp = fp.profile_fp(p["g"], cur, p["pos"], pri)
                    p2, prof2, gp, w = blend_rates(p, prof_fp, p["g"], cur)
                    row = {"date": day.date().isoformat(), "event_id": str(event_id), "player_id": p["id"], "player": p["name"], "team": team, "opp": opp, "pos": p["pos"],
                           "proj_min": round(p["min"], 2), "sd_min": round(p["sd_min"], 2), "gp": gp, "w": round(w, 4), "absent_starters": absent,
                           "act_min": act_min, "act_fp": round(act_fp, 1), "starter": int(a.starter)}
                    for name, sdf, q in cfgs:
                        if name == "v1":
                            sim = fp.simulate_fp(p, prof_fp, opp, dvp, absent, pri, rngs[name])
                        else:
                            sim = fp.simulate_fp(p2, prof2, opp, dvp, absent, pri, rngs[name], sd_factor=sdf, q=q)
                        row[f"mean_{name}"] = sim["fp"]["mean"]
                        row[f"pit_{name}"] = round(fp.cdf_from_q(sim["fp"]["q"], act_fp), 4)
                        row[f"emin_{name}"] = round(expected_minutes(p["min"], p["sd_min"], sdf, q, rng_min), 2)
                    rows.append(row)
        if i % 5 == 0 or i == len(dates):
            print(f"[fp-v2 replay] {i}/{len(dates)} {day.date()}  {len(rows)} rows  {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        out.to_csv(f, index=False)
    print(f"[fp-v2 replay] wrote {len(out)} rows to {out_path.name}")
    return out


# ---------------- phase A: grid report + mechanical selection ----------------
def pit_table(pit):
    pit = np.asarray(pit, dtype=float); n = len(pit)
    out = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (pit >= lo) & ((pit < hi) if i < 9 else True)
        k = int(m.sum()); share = k / n if n else None
        out.append({"lo": lo, "hi": hi, "n": k, "share": round(share, 4) if n else None, "gap": round(share - 0.1, 4) if n else None})
    return out


def pit_verdict(pit):
    """The rule-4 bars on PIT decile shares: weighted |gap| = sum(share_i * |share_i - 0.1|) (decile-n weighted), worst = max |share - 0.1|."""
    tab = pit_table(pit); n = len(pit)
    if not n:
        return tab, None, None, None, False
    shares = np.array([b["share"] for b in tab], dtype=float)
    wmae = float((shares * np.abs(shares - 0.1)).sum()); mae = float(np.abs(shares - 0.1).mean()); worst = float(np.abs(shares - 0.1).max())
    return tab, wmae, mae, worst, bool(wmae <= WMAE_BAR and worst <= WORST_BAR)


def window_stats(d, name):
    """PIT deciles, weighted |gap|, plain mean |gap|, worst decile, mean FP bias and mean minutes bias for one config on one
    window. Void rows (no minutes) are excluded before this is called."""
    pit = d[f"pit_{name}"].values; n = len(d)
    tab, wmae, mae, worst, ok = pit_verdict(pit)
    return {"n": int(n), "deciles": tab, "weighted_mean_abs_gap": round(wmae, 4) if n else None, "mean_abs_gap": round(mae, 4) if n else None,
            "worst_decile_gap": round(worst, 4) if n else None, "mean_pit": round(float(pit.mean()), 4) if n else None,
            "mean_fp_bias": round(float((d[f"mean_{name}"] - d.act_fp).mean()), 3) if n else None,
            "mean_minutes_bias": round(float((d[f"emin_{name}"] - d.act_min).mean()), 3) if n else None,
            "mean_proj_fp": round(float(d[f"mean_{name}"].mean()), 2) if n else None, "mean_act_fp": round(float(d.act_fp.mean()), 2) if n else None,
            "pass": ok}


def grid(rows_path=REPLAY, out_path=BACKTEST):
    d = pd.read_csv(rows_path, dtype={"event_id": str, "player_id": str})
    total = len(d)
    d = d[d.act_min > 0].copy()                    # rule 2d: no-minutes player-games are void
    sel = d[d.date <= SELECT_END]; ver = d[d.date > SELECT_END]
    cfgs = configs()
    table = []
    for name, sdf, q in cfgs:
        table.append({"config": name, "sd_factor": sdf, "q": q, "candidate": name != "v1",
                      "selection": window_stats(sel, name), "verdict": window_stats(ver, name)})
    cand = [t for t in table if t["candidate"]]
    sel_pass = [t for t in cand if t["selection"]["pass"]]            # rule 4: SELECTED on Oct-Jan, tie-break toward v1 (grid order)
    selected = sel_pass[0] if sel_pass else None
    verdict = None
    if selected is not None:
        v = selected["verdict"]
        verdict = {"config": selected["config"], "sd_factor": selected["sd_factor"], "q": selected["q"], "status": "PASS" if v["pass"] else "FAIL",
                   "weighted_mean_abs_gap": v["weighted_mean_abs_gap"], "worst_decile_gap": v["worst_decile_gap"], "mean_fp_bias": v["mean_fp_bias"], "n": v["n"], "window": "Feb-Apr"}
    ver_pass = [t["config"] for t in cand if t["verdict"]["pass"]]
    doc = {"generated": ts(now()), "rule_file": RULE, "label": LABEL, "kind": "fp_v2_backtest_grid",
           "replay": {"season": "2025-26", "start": REPLAY_START, "end": REPLAY_END, "rows": int(total), "rows_graded": int(len(d)), "rows_void_no_minutes": int(total - len(d)),
                      "selection_window": f"{REPLAY_START} .. {SELECT_END} (rule: Oct-Jan; the ladder replay starts Nov 1)", "verdict_window": f"2026-02-01 .. {REPLAY_END}",
                      "selection_rows": int(len(sel)), "verdict_rows": int(len(ver))},
           "bars": {"weighted_mean_abs_gap": WMAE_BAR, "worst_decile_gap": WORST_BAR, "note": "PIT decile shares vs 0.10; weighted |gap| = sum(share_i * |share_i - 0.1|); the bars v1 failed"},
           "grid": {"sd_factor": list(SD_GRID), "q": list(Q_GRID), "tie_break": "smallest sd_factor, then smallest q (closest to v1)"},
           "table": table,
           "selection": {"method": "rule 4: pairs passing both bars on the Oct-Jan slice, first in tie-break order", "passing_on_selection": [t["config"] for t in sel_pass],
                         "selected": selected["config"] if selected else None},
           "verdict": verdict if verdict else {"status": "NO PAIR SELECTED", "note": "no grid pair passed the bars on the selection window; v2 does not ship (rule 4)"},
           "passing_on_verdict_window_all_pairs": ver_pass,
           "notes": ["Replay rows: players who played, as in the ladder backtest (absent starters taken from the box score). Profiles, minutes allocation, DvP and usage from history before each day only; "
                     "FP priors are the frozen 2024-25 constants (data/fp_priors.json). The replay season (2025-26) was not used to fit them.",
                     "2b in the replay: GP and the current-season rates come from the season's games before each day; prior = the v1 rate (engine weighting, frozen position shrink).",
                     "PIT from the 99 logged-style percentiles via the grader's interpolation, so the backtest reads exactly like the forward grade.",
                     "The v1 row is a reference (same replay, v1 sim and rates); it is not a candidate.",
                     "The v1 live-record sanity slice (rule 4) is produced by the phase B retro replay and reported there; it never counts toward this verdict."]}
    write_json(out_path, doc)
    print(f"[fp-v2 grid] rows {total} ({len(d)} with minutes; selection {len(sel)}, verdict {len(ver)})")
    f4 = lambda x: f"{x:9.4f}" if x is not None else f"{'-':>9}"   # noqa: E731
    f3 = lambda x: f"{x:9.3f}" if x is not None else f"{'-':>9}"   # noqa: E731
    print(f"{'config':14} {'sel wmae':>9} {'sel worst':>9} {'sel bias':>9} | {'ver wmae':>9} {'ver worst':>9} {'ver bias':>9} {'min bias':>9}  sel/ver pass")
    for t in table:
        s, v = t["selection"], t["verdict"]
        print(f"{t['config']:14} {f4(s['weighted_mean_abs_gap'])} {f4(s['worst_decile_gap'])} {f3(s['mean_fp_bias'])} | {f4(v['weighted_mean_abs_gap'])} {f4(v['worst_decile_gap'])} {f3(v['mean_fp_bias'])} {f3(v['mean_minutes_bias'])}  {s['pass']}/{v['pass']}")
    print(f"[fp-v2 grid] selection-window passers (tie-break order): {doc['selection']['passing_on_selection']}; selected: {doc['selection']['selected']}; verdict: {doc['verdict']}")
    return doc


def select(doc_path=BACKTEST):
    """Freeze the mechanically selected pair (rule 4) in data/fp_v2_params.json. Refuses unless the verdict is PASS."""
    doc = read_json(doc_path, None)
    if not doc or doc.get("verdict", {}).get("status") != "PASS":
        raise SystemExit(f"[fp-v2 select] no PASS verdict in {doc_path.name}: nothing frozen, v2 does not ship")
    v = doc["verdict"]
    out = {"rule_file": RULE, "selected": {"sd_factor": v["sd_factor"], "q": v["q"], "config": v["config"]}, "selected_at": ts(now()), "k_blend": K_BLEND, "short_night": fp.SHORT_NIGHT,
           "backtest": {"selection_window": doc["replay"]["selection_window"], "verdict_window": doc["replay"]["verdict_window"], "verdict": v,
                        "passing_on_selection": doc["selection"]["passing_on_selection"], "generated": doc["generated"]},
           "note": "Frozen by rule 4 (selected on Oct-Jan, verdict on Feb-Apr, tie-break toward v1). Changing it is a model change: new rule, new gate."}
    write_json(PARAMS, out)
    print(f"[fp-v2 select] frozen {v['config']} (sd_factor {v['sd_factor']}, q {v['q']}) -> {PARAMS.name}")
    return out


# ---------------- phase B: retro replay over the graded v1 lines (rule 2c), log-time inputs ----------------
def _projections_at(logged_at, cache):
    """docs/data/projections.json as committed at or before logged_at (git history): minutes projections, flags (absent starters)
    and status as the v1 engine recorded them. None when git has nothing that early."""
    key = str(logged_at)[:16]
    if key in cache:
        return cache[key]
    root = Path(__file__).resolve().parent.parent
    t = pd.Timestamp(logged_at)
    doc, sha = None, None
    try:
        # Amendment 1 harness fix: the snapshot is the file whose own `updated` stamp is the latest at or before logged_at (the
        # file written by the same run, committed a minute later). The last COMMIT before logged_at is the previous run's file,
        # which for a day's first run is the previous slate and carries no flags for tonight's players.
        shas = subprocess.run(["git", "log", "--format=%h", f"--since={(t - pd.Timedelta(hours=36)).isoformat()}", f"--until={(t + pd.Timedelta(hours=3)).isoformat()}",
                               "--", "docs/data/projections.json"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
        for s in shas:   # newest first
            d = json.loads(subprocess.run(["git", "show", f"{s}:docs/data/projections.json"], cwd=root, capture_output=True, text=True, check=True).stdout)
            if d.get("updated") and pd.Timestamp(d["updated"]) <= t:
                doc, sha = d, s; break
    except (subprocess.CalledProcessError, ValueError):
        doc = None
    cache[key] = (sha, {str(p["id"]): p for p in (doc or {}).get("players", [])}) if doc else (None, {})
    return cache[key]


def retro(out_path=RETRO):
    """Rule 2c: replay v2 (frozen pair, 2a + 2b) over every graded v1 line with log-time inputs only: the v1 log row's projected
    minutes, position, team/opp and tip; absent starters, Q status from the projections.json committed at log time; sd_min, v1
    rates, FP profile, DvP, GP and current-season rates from box scores dated before the slate. Nothing after log time enters.
    Also replays v1 the same way (sanity slice for rule 4, and a fidelity check of the harness against the logged v1 means)."""
    params = load_params()
    if not params:
        raise SystemExit("[fp-v2 retro] no frozen pair (data/fp_v2_params.json): phase B has not been approved")
    sel = params["selected"]; sdf, q = float(sel["sd_factor"]), float(sel["q"])
    pri = fp.load_priors()
    grade = pd.read_csv(fp.GRADE, dtype={"event_id": str, "player_id": str})
    gl = grade[grade.line_source.isin(["underdog", "prizepicks"]) & grade.hit.notna() & (grade.date >= FIT_DATES[0]) & (grade.date <= FIT_DATES[1])].copy()
    log = fp.last_pretip(pd.read_csv(fp.LOG, dtype={"event_id": str, "player_id": str}))
    log = log.set_index(["date", "event_id", "player_id"])
    df = bm.load_games()
    pos_map = position_map(); cache = {}
    rng1, rng2 = np.random.default_rng(SEED + 100), np.random.default_rng(SEED + 200)
    rows, skipped = [], Counter()
    for date, dd in gl.groupby("date"):
        day = pd.Timestamp(date); cur = bm.season_of(date)
        hist = df[df.date < day]
        prior_dates = hist.loc[hist.season == cur - 1, "date"]
        prior_cutoff = (prior_dates.max() - pd.Timedelta(days=bm.PRIOR_TAIL_DAYS)) if len(prior_dates) else None
        dvp, _ = bm.build_dvp(hist, cur); lg_pct = float(hist.tpm.sum() / max(hist.tpa.sum(), 1)); pos_rates = bm.pos_tpa_rates(hist, cur)
        pool = hist[hist.season >= cur - 1]
        for (event_id, pid), lines in dd.groupby(["event_id", "player_id"]):
            try:
                lr = log.loc[(date, event_id, pid)]
            except KeyError:
                skipped["no v1 log row"] += len(lines); continue
            if isinstance(lr, pd.DataFrame):
                lr = lr.iloc[-1]
            g = pool[pool.player_id == pid].sort_values("date", ascending=False)
            if len(g) < 3:
                skipped["fewer than 3 games before the slate"] += len(lines); continue
            sha, proj = _projections_at(lr.logged_at, cache)
            pj = proj.get(str(pid), {})
            flags = pj.get("flags") or []
            absent = max([USAGE_FLAG.get(f, 0) for f in flags] + [0])
            prof = bm.profile(g, cur, lg_pct, prior_cutoff)
            pos = lr.pos or pos_map.get(pid) or g.iloc[-1].pos
            p = {"id": pid, "pos": pos, "min": float(lr["min"]), "sd_min": prof["sd_min"] + (4 if "Q" in flags else 0), "pct": prof["pct"],
                 "rates": {**prof["rates"], "tpa": bm.tpa_rate_v2(g, cur, pos, pos_rates)}}
            prof_fp = fp.profile_fp(g, cur, pos, pri)
            p2, prof2, gp, w = blend_rates(p, prof_fp, g, cur)
            s1 = fp.simulate_fp(p, prof_fp, lr.opp, dvp, absent, pri, rng1)
            s2 = fp.simulate_fp(p2, prof2, lr.opp, dvp, absent, pri, rng2, sd_factor=sdf, q=q)
            actual = float(lines.iloc[0].fp_actual) if pd.notna(lines.iloc[0].fp_actual) else None
            base = {"date": date, "event_id": event_id, "player_id": pid, "player": lr.player, "team": lr.team, "opp": lr.opp, "pos": pos, "min_proj": float(lr["min"]),
                    "gp": gp, "w": round(w, 4), "absent_starters": absent, "proj_commit": sha or "", "fp_actual": actual,
                    "v1_logged_mean": float(lr.fp_mean), "v1_retro_mean": s1["fp"]["mean"], "v2_mean": s2["fp"]["mean"], "v2_sd": s2["fp"]["sd"],
                    "pit_v1_retro": round(fp.cdf_from_q(s1["fp"]["q"], actual), 4) if actual is not None else None,
                    "pit_v2": round(fp.cdf_from_q(s2["fp"]["q"], actual), 4) if actual is not None else None}
            for x in lines.itertuples():
                line = float(x.line)
                c1, c2 = fp.cdf_from_q(s1["fp"]["q"], line), fp.cdf_from_q(s2["fp"]["q"], line)
                rows.append({**base, "source": x.source, "line": line, "side": x.side, "hit": int(x.hit), "p_v1_logged": float(x.p_model),
                             "p_v1_retro": round((1 - c1) if x.side == "higher" else c1, 4), "p_raw_v2": round((1 - c2) if x.side == "higher" else c2, 4)})
    R = pd.DataFrame(rows)
    R.to_csv(out_path, index=False)
    fid = (R.drop_duplicates(["date", "event_id", "player_id"]))
    mae = float((fid.v1_retro_mean - fid.v1_logged_mean).abs().mean()) if len(fid) else None
    print(f"[fp-v2 retro] {len(R)} graded lines replayed over {len(fid)} player-games with pair {sel['config']}; skipped {dict(skipped)}; "
          f"harness fidelity: mean |v1 retro mean - v1 logged mean| = {mae:.2f} FP; projections.json found at log time for {int((fid.proj_commit != '').sum())} player-games")
    return R


def cluster_ci(values, clusters, iters=BOOT_ITERS, seed=SEED):
    """95% CI of the mean of `values` by cluster bootstrap over `clusters` (player-games), the house standard."""
    values = np.asarray(values, dtype=float)
    codes, uniq = pd.factorize(pd.Series(clusters).astype(str))
    s = np.bincount(codes, weights=values, minlength=len(uniq)); n = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed); out = []
    for _ in range(iters):
        pick = rng.integers(0, len(uniq), len(uniq)); out.append(s[pick].sum() / n[pick].sum())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def fit(retro_path=RETRO):
    """Rule 2c: one combined isotonic map of hit on replayed v2 raw P(over), all graded v1 lines (UD + PP), ONE ROW PER LINE
    (the over side: the under is its complement and would make every mean gap identically zero), frozen in
    data/fp_v2_recal.json with fit date and n. Per-source post-map calibration gap with a clustered CI."""
    params = load_params()
    if not params:
        raise SystemExit("[fp-v2 fit] no frozen pair")
    R_all = pd.read_csv(retro_path, dtype={"event_id": str, "player_id": str})
    R = R_all[R_all.side == "higher"].drop_duplicates(["date", "event_id", "player_id", "source", "line"]).reset_index(drop=True)
    p, h = R.p_raw_v2.values.astype(float), R.hit.values.astype(int)
    m_pav = pav_step(p, h)
    m = merge_small_blocks(m_pav)
    pc = apply_step(m, p)
    retro_sha = hashlib.sha256(Path(retro_path).read_bytes()).hexdigest()
    pg = (R.date + ":" + R.event_id + ":" + R.player_id).values
    def block(mask, label):
        d = R[mask]; pcm, hm = pc[mask], h[mask]
        gap = pcm - hm
        return {"label": label, "n": int(mask.sum()), "player_games": int(d.drop_duplicates(["date", "event_id", "player_id"]).shape[0]),
                "gap_post_map": round(float(gap.mean()), 4) if mask.sum() else None, "gap_ci": cluster_ci(gap, pg[mask]) if mask.sum() else None,
                "gap_pre_map": round(float((p[mask] - hm).mean()), 4) if mask.sum() else None,
                "by_decile_post_map": fp.buckets(pcm, hm), "by_decile_pre_map": fp.buckets(p[mask], hm),
                "logloss_raw": round(float(logloss(p[mask], hm).mean()), 4) if mask.sum() else None, "logloss_cal": round(float(logloss(pcm, hm).mean()), 4) if mask.sum() else None}
    pit = R.drop_duplicates(["date", "event_id", "player_id"])
    tab2, wm2, _, wo2, ok2 = pit_verdict(pit.pit_v2.dropna().values); tab1, wm1, _, wo1, ok1 = pit_verdict(pit.pit_v1_retro.dropna().values)
    out = {"rule_file": RULE, "fit_date": ts(now()), "n": int(len(R)), "n_rows_both_sides": int(len(R_all)), "n_player_games": int(len(pit)), "pair": params["selected"],
           "sources": {s: int((R.source == s).sum()) for s in ("UD", "PP")}, "dates": [str(R.date.min()), str(R.date.max())],
           "map": m, "map_pav_unmerged": m_pav, "retro_file": str(Path(retro_path).name), "retro_sha256": retro_sha, "amendment": "Amendment 1 (2026-10-28): minimum block n >= 20, clip [0.05, 0.95]; calibration verdict at 800",
           "clip": list(CLIP), "min_block": MIN_BLOCK, "fit_dates": list(FIT_DATES), "gate_start": "the first slate priced by this map; rows logged or priced earlier carry gate = pre-A1 and are excluded",
           "method": f"pool-adjacent-violators isotonic regression of hit on replayed v2 raw P(over), one row per line (over side); blocks with n < {MIN_BLOCK} merged into the nearest neighbour with the pooled hit rate (Amendment 1); applied as the step function over block ranges (step at the midpoint between adjacent blocks, end blocks extended), clipped to [{CLIP[0]}, {CLIP[1]}]",
           "inputs": "log-time only: v1 log minutes, projections.json flags at log time (absent starters, Q), box scores dated before each slate",
           "combined": block(np.ones(len(R), bool), "UD + PP"), "by_source": {s: block((R.source == s).values, {"UD": "Underdog", "PP": "PrizePicks"}[s]) for s in ("UD", "PP")},
           "sanity_slice": {"note": "rule 4: the v1 live record replayed with v2 (and v1) distributions; sanity only, never the verdict",
                            "v2": {"n": int(pit.pit_v2.notna().sum()), "deciles": tab2, "weighted_mean_abs_gap": round(wm2, 4) if wm2 is not None else None, "worst_decile_gap": round(wo2, 4) if wo2 is not None else None, "pass": ok2,
                                   "mean_fp_bias": round(float((pit.v2_mean - pit.fp_actual).mean()), 3)},
                            "v1_retro": {"n": int(pit.pit_v1_retro.notna().sum()), "deciles": tab1, "weighted_mean_abs_gap": round(wm1, 4) if wm1 is not None else None, "worst_decile_gap": round(wo1, 4) if wo1 is not None else None, "pass": ok1,
                                         "mean_fp_bias": round(float((pit.v1_retro_mean - pit.fp_actual).mean()), 3)},
                            "harness_fidelity_mae_vs_logged_v1_mean": round(float((pit.v1_retro_mean - pit.v1_logged_mean).abs().mean()), 3)},
           "frozen": "for the entire forward gate; no refits during collection (rule 2c)"}
    write_json(RECAL, out)
    c = out["combined"]
    print(f"[fp-v2 fit] retro {retro_path} sha256 {retro_sha}; PAV {len(m_pav['x_lo'])} blocks -> {len(m['x_lo'])} after {m['merges']} merges (monotone {m['monotone']})")
    for lo_, hi_, y_, yc_, n_ in zip(m["x_lo"], m["x_hi"], m["y"], m["y_cal"], m["n"]):
        print(f"    raw P(over) {lo_:.4f}-{hi_:.4f} -> {yc_:.4f} (pooled {y_:.4f}), n {n_}")
    print(f"[fp-v2 fit] map frozen: n {out['n']} lines ({out['sources']}) over {out['n_player_games']} player-games, {len(m['x_lo'])} blocks; "
          f"gap post-map combined {c['gap_post_map']} CI {c['gap_ci']}; UD {out['by_source']['UD']['gap_post_map']} CI {out['by_source']['UD']['gap_ci']}; "
          f"PP {out['by_source']['PP']['gap_post_map']} CI {out['by_source']['PP']['gap_ci']}; logloss raw {c['logloss_raw']} -> cal {c['logloss_cal']}")
    return out


# ---------------- rule 5 re-count of the power-calculation slice ----------------
def novig(dec_hi, dec_lo):
    try:
        a, b = 1 / float(dec_hi), 1 / float(dec_lo)
        return a / (a + b)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _book_baseline(x, prices):
    """P(over) baseline for a graded line: UD = no-vig from both captured sides (fp_prices.csv), PP = 0.5 (rule 5)."""
    if x.source == "PP":
        return 0.5
    d = prices[(prices.captured_at == x.captured_at) & (prices.event_id == x.event_id) & (prices.player == x.player) & (prices.line.astype(float) == float(x.line))]
    hi = d[d.side == "higher"]; lo = d[d.side == "lower"]
    if len(hi) and len(lo):
        return novig(hi.iloc[0].decimal, lo.iloc[0].decimal)
    return None


def recount():
    """Rule 5: the power-calculation slice re-counted on unique lines. Isotonic fitted on the v1 raw P(over) of Oct 23-24, evaluated
    on Oct 25-26; per-line log-loss difference vs the book baseline; unclustered vs cluster-robust (player-game) SE; design effect;
    on all graded rows (both sides, as the rule's 484 were counted) and on unique lines (one row per player, game, source, line)."""
    grade = pd.read_csv(fp.GRADE, dtype={"event_id": str, "player_id": str})
    gl = grade[grade.line_source.isin(["underdog", "prizepicks"]) & grade.hit.notna()].copy()
    prices = pd.read_csv(fp.FP_PRICES, dtype={"event_id": str}) if fp.FP_PRICES.exists() else pd.DataFrame(columns=fp.FP_PRICE_FIELDS)
    fit_rows = gl[gl.date <= "2026-10-24"]; ev = gl[(gl.date >= "2026-10-25") & (gl.date <= "2026-10-26")].copy()
    m = fit_map(fit_rows.p_model.values.astype(float), fit_rows.hit.values.astype(int))
    ev["p_cal"] = apply_map(m, ev.p_model.values.astype(float))
    ev["p_book_over"] = [_book_baseline(x, prices) for x in ev.itertuples()]
    ev = ev[ev.p_book_over.notna()].copy()
    ev["p_book"] = np.where(ev.side == "higher", ev.p_book_over, 1 - ev.p_book_over)
    ev["d"] = logloss(ev.p_cal.values, ev.hit.values.astype(int)) - logloss(ev.p_book.values.astype(float), ev.hit.values.astype(int))
    ev["pg"] = ev.date + ":" + ev.event_id + ":" + ev.player_id
    def stats(d):
        n = len(d); x = d.d.values; mean = x.mean(); se_u = x.std(ddof=1) / math.sqrt(n)
        resid = x - mean; by = pd.Series(resid).groupby(d.pg.values).sum(); se_c = math.sqrt((by ** 2).sum()) / n
        return {"rows": int(n), "player_games": int(d.pg.nunique()), "mean_diff_nats": round(float(mean), 5), "sd": round(float(x.std(ddof=1)), 4),
                "se_unclustered": round(float(se_u), 5), "se_cluster_robust": round(float(se_c), 5), "design_effect": round(float((se_c / se_u) ** 2), 3) if se_u else None,
                "n_for_delta_0.02_power_0.8": int(math.ceil(((1.96 + 0.8416) * (se_c * math.sqrt(n)) / 0.02) ** 2)) if se_u else None}
    uniq = ev.drop_duplicates(["player", "event_id", "source", "line"])
    out = {"generated": ts(now()), "rule_file": RULE, "fit_slice": {"dates": "2026-10-23 .. 2026-10-24", "rows": int(len(fit_rows))}, "eval_slice": {"dates": "2026-10-25 .. 2026-10-26"},
           "all_rows_both_sides": stats(ev), "unique_lines": stats(uniq), "by_source_unique": {s: stats(uniq[uniq.source == s]) for s in ("UD", "PP") if (uniq.source == s).any()},
           "baseline": {"UD": "no-vig P(over) from both captured sides' decimal prices", "PP": "0.5"},
           "note": "Both sides of one line carry the same log-loss difference, so the two-sided count double-counts every line; the unique-line figures are the honest ones. n = 800 stands either way (rule 5)."}
    write_json(RECOUNT, out)
    a, u = out["all_rows_both_sides"], out["unique_lines"]
    print(f"[fp-v2 recount] eval rows {a['rows']} (both sides) over {a['player_games']} player-games, DE {a['design_effect']}; unique lines {u['rows']}, sd {u['sd']}, "
          f"SE {u['se_unclustered']} -> clustered {u['se_cluster_robust']}, DE {u['design_effect']}, n for 0.02 nats at 80% power {u['n_for_delta_0.02_power_0.8']}")
    return out


# ---------------- phase B: forward path ----------------
def snapshot():
    """Append tonight's v2 distributions (projections.json fp_v2 block, written by build_model next to the v1 sim) to the v2 log."""
    P = read_json(SITE_DATA / "projections.json", None)
    if not P or not P.get("games"):
        print("[fp-v2] no slate"); return 0
    ev = {}
    for g in P["games"]:
        ev[(g["home"], g["away"])] = g; ev[(g["away"], g["home"])] = g
    t = now(); rows, skipped = [], 0
    for p in P["players"]:
        if p.get("inactive") or "fp_v2" not in p:
            continue
        g = ev.get((p["team"], p["opp"]))
        if not g or not g.get("start"):
            continue
        tip = fp.parse_ts(g["start"])
        if t >= tip:
            skipped += 1; continue
        v = p["fp_v2"]
        rows.append({"date": P["date"], "event_id": g["event_id"], "tip": ts(tip), "player_id": p["id"], "player": p["name"], "team": p["team"], "opp": p["opp"], "pos": p["pos"],
                     "min": p["min"], "gp": v["gp"], "w": v["w"], "sd_factor": v["sd_factor"], "q": v["q_mix"], "fp_mean": v["mean"], "fp_sd": v["sd"], "fp_q": json.dumps(v["q"]), "logged_at": ts(t), "gate": GATE_EPOCH})
    if rows:
        fp.append(LOG, LOG_FIELDS, rows)
    print(f"[fp-v2] appended {len(rows)} rows for {P['date']} ({skipped} players skipped: game already tipped)")
    return len(rows)


def _pretip_log():
    """Every pre-tip v2 log row of the current gate epoch (gate = A1; pre-A1 rows never price a capture), with `_t` = logged_at
    as a UTC timestamp."""
    if not LOG.exists():
        return None
    log = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str})
    if "gate" not in log.columns:
        log["gate"] = PRE_GATE
    log = log[log.gate == GATE_EPOCH]
    log["_t"] = pd.to_datetime(log.logged_at, utc=True)
    return log[log._t < pd.to_datetime(log.tip, utc=True)]


def _row_at_capture(d, captured_at):
    """Amendment 1 pricing timing: the latest log row with logged_at <= captured_at, or None (a capture that precedes every row
    for the player-game is not priced by this run; the next run prices the next capture)."""
    t = captured_at if hasattr(captured_at, "tzinfo") else fp.parse_ts(captured_at)
    d = d[d._t <= pd.Timestamp(t)]
    return None if d.empty else d.sort_values("_t").iloc[-1]


def price(snapshot_path):
    """UD standard Fantasy Points lines from the board copy, priced with v2: raw P(over) from the last pre-tip v2 log row, calibrated
    P(over) through the frozen map, BOTH sides' prices recorded on every row, the book's no-vig P(over) alongside (rule 5)."""
    recal = load_recal()
    snap = json.load(open(snapshot_path, encoding="utf-8"))
    cap = snap.get("captured_at") or ts(now())
    nba = [e for e in snap.get("entries") or [] if (e.get("league") or "").upper() == "NBA"]
    P = read_json(SITE_DATA / "projections.json", None) or {"date": None, "games": [], "players": []}
    t = now(); ok, why = fp.capture_check(cap, P.get("date"), t)
    log = _pretip_log()
    map_fit = (recal or {}).get("fit_date")
    def match_game(e):
        a, b = fp.team_abbr(e.get("team")), fp.team_abbr(e.get("opponent"))
        for g in P["games"]:
            if {fp.team_abbr(g["home"]), fp.team_abbr(g["away"])} == {a, b}:
                return g
        return None
    rows, out, unmatched, tipped, awaiting = [], [], 0, 0, 0
    fp_entries = [e for e in nba if (e.get("market") or "").strip().lower() in fp.FP_MARKETS]
    for e in fp_entries if (ok and log is not None) else []:
        pid = e.get("platform_ids") or {}
        if pid.get("line_type") not in (None, "balanced"):
            continue   # standard main line only
        g = match_game(e)
        if g is None:
            unmatched += 1; continue
        if t >= fp.parse_ts(g["start"]):
            tipped += 1; continue
        try:
            line = float(e["line"])
        except (KeyError, TypeError, ValueError):
            continue
        key = norm_name(e.get("player"))
        d = log[(log.date == P["date"]) & (log.event_id == g["event_id"]) & (log.player.map(norm_name) == key)]
        if d.empty:
            continue
        row = _row_at_capture(d, cap)
        if row is None:
            awaiting += 1; continue   # every v2 row for this player-game was logged after the capture (Amendment 1 timing)
        c = fp.cdf_from_q(json.loads(row.fp_q), line); p_over = 1 - c; p_over_cal = cal(recal, p_over)
        pr = pid.get("prices") or {}
        dec = {s: (pr.get(s) or {}).get("decimal") for s in ("higher", "lower")}; am = {s: (pr.get(s) or {}).get("american") for s in ("higher", "lower")}
        book = novig(dec["higher"], dec["lower"])
        tm = g["home"] if fp.team_abbr(e.get("team")) == fp.team_abbr(g["home"]) else g["away"]; op = g["away"] if tm == g["home"] else g["home"]
        for side in e.get("available_directions") or ["higher", "lower"]:
            m, decs, conv, be = fp.leg_price(e, side)
            p_raw = p_over if side == "higher" else c; p_c = p_over_cal if side == "higher" else 1 - p_over_cal
            rows.append({"captured_at": cap, "date": P["date"], "event_id": g["event_id"], "player": e["player"], "team": tm, "opp": op, "market": e["market"], "line": line, "side": side,
                         "mult": m, "decimal": decs, "american": am.get(side), "convention": conv, "breakeven": round(be, 4) if be else None,
                         "decimal_higher": dec["higher"], "decimal_lower": dec["lower"], "american_higher": am["higher"], "american_lower": am["lower"],
                         "p_book_novig": round(book if side == "higher" else 1 - book, 4) if book is not None else None,
                         "p_raw": round(p_raw, 4), "p_cal": round(p_c, 4), "edge": round(p_c - be, 4) if be else None, "source": "UD", "source_id": e.get("source_id"), "fp_logged_at": row.logged_at, "map_fit": map_fit})
        out.append({"date": P["date"], "event_id": g["event_id"], "player": e["player"], "team": tm, "opp": op, "line": line, "p_raw": round(p_over, 4), "p_cal": round(p_over_cal, 4),
                    "p_book_novig": round(book, 4) if book is not None else None, "decimal_higher": dec["higher"], "decimal_lower": dec["lower"], "fp_mean_v2": float(row.fp_mean), "source_id": e.get("source_id")})
    n_new = 0
    if rows:
        if PRICES.exists() and (pd.read_csv(PRICES, usecols=["captured_at"]).captured_at == cap).any():
            pass   # this capture is already logged
        else:
            fp.append(PRICES, PRICE_FIELDS, rows); n_new = len(rows)
    doc = {"captured_at": cap, "priced_at": ts(t), "slate_date": P.get("date"), "capture_ok": ok, "capture_check": why, "fp_entries": len(fp_entries), "lines_priced": len(out), "new_rows": n_new,
           "unmatched_entries": unmatched, "tipped_entries": tipped, "awaiting_capture_after_snapshot": awaiting, "pair": (load_params() or {}).get("selected"), "map": {"fit_date": recal.get("fit_date"), "n": recal.get("n")} if recal else None,
           "label": LABEL, "rule_file": RULE, "lines": sorted(out, key=lambda r: -(r["p_cal"] - (r["p_book_novig"] or 0.5)))[:400],
           "note": "v2 raw P(over) from the latest v2 log row logged at or before the board capture (Amendment 1 timing); calibrated through the frozen isotonic map (data/fp_v2_recal.json); both sides' prices recorded; the book's no-vig P(over) is the market-test baseline."}
    write_json(PRICING, doc)
    print(f"[fp-v2] board {cap} ({'ok' if ok else 'REFUSED'}: {why}): {len(fp_entries)} FP entries -> {len(out)} lines priced, {n_new} new rows; {unmatched} not on the slate, {tipped} tipped, "
          f"{awaiting} awaiting a capture after the v2 snapshot")
    return doc


def price_pp(path=fp.PP_FILE):
    """PrizePicks standard Fantasy Score lines (the scraper hook's repo copy), priced with v2; baseline 0.5 (rule 5)."""
    recal = load_recal(); t = now()
    p = Path(path)
    if not p.exists() or _pretip_log() is None:
        print("[fp-v2] prizepicks: no copy or no v2 log yet"); return 0
    map_fit = (recal or {}).get("fit_date")
    cp = json.load(open(p, encoding="utf-8")); cap = cp.get("captured_at")
    try:
        if (t - fp.parse_ts(cap)).total_seconds() / 3600 > fp.PP_MAX_AGE_H:
            print("[fp-v2] prizepicks: copy older than the limit; no PP lines today"); return 0
    except Exception:  # noqa: BLE001
        print("[fp-v2] prizepicks: copy carries no capture time"); return 0
    fs = [r for r in cp.get("lines") or [] if str(r.get("odds_type") or "").strip().lower() == "standard" and str(r.get("stat_type") or "").strip().lower() in fp.FP_MARKETS]
    snap = _pretip_log(); snap = snap.assign(_key=snap.player.map(norm_name))
    last = {}; awaiting = 0
    for r in fs:
        try:
            sched = fp.parse_ts(r["scheduled_at"])
        except Exception:  # noqa: BLE001
            continue
        last[(norm_name(r.get("player")), sched.astimezone(ET).date().isoformat())] = (r, sched)
    rows = []
    for (key, date), (r, sched) in last.items():
        lcap = r.get("last_seen_at") or r.get("captured_at") or cap   # the latest capture still showing this line (pirate-bets extract, 2026-10-28)
        try:
            lcap_dt = fp.parse_ts(lcap)
        except Exception:  # noqa: BLE001
            lcap_dt = fp.parse_ts(cap)
        if lcap_dt >= sched:
            continue
        d = snap[(snap.date == date) & (snap._key == key)]
        if d.empty:
            continue
        row = _row_at_capture(d, lcap_dt)
        if row is None:
            awaiting += 1; continue   # Amendment 1 timing: no v2 row logged at or before this capture
        ta = fp.team_abbr(r.get("team"))
        if ta and ta != fp.team_abbr(row.team):
            continue
        try:
            line = float(r["line"])
        except (KeyError, TypeError, ValueError):
            continue
        c = fp.cdf_from_q(json.loads(row.fp_q), line); p_over = 1 - c; p_over_cal = cal(recal, p_over)
        for side, p_raw, p_c in (("higher", p_over, p_over_cal), ("lower", c, 1 - p_over_cal)):
            rows.append({"captured_at": ts(lcap_dt), "date": date, "event_id": row.event_id, "player": r.get("player"), "team": row.team, "opp": row.opp, "market": r.get("stat_type"), "line": line,
                         "side": side, "p_book_novig": 0.5, "p_raw": round(p_raw, 4), "p_cal": round(p_c, 4), "scheduled_at": ts(sched), "game_status": r.get("game_status"), "source": "PP",
                         "source_id": r.get("source_id"), "fp_logged_at": row.logged_at, "map_fit": map_fit})
    new = rows
    if rows and PP_PRICES.exists():
        old = pd.read_csv(PP_PRICES, dtype=str, usecols=["captured_at", "source_id", "side", "player", "date"])
        seen = set(zip(old.captured_at, old.source_id.fillna(""), old.side, old.player, old.date))
        new = [x for x in rows if (x["captured_at"], str(x["source_id"] or ""), x["side"], x["player"], x["date"]) not in seen]
    if new:
        fp.append(PP_PRICES, PP_PRICE_FIELDS, new)
    print(f"[fp-v2] prizepicks: {len(fs)} standard FP rows in the copy -> {len(rows) // 2} lines priced, {len(new)} new rows, {awaiting} awaiting a capture after the v2 snapshot")
    return len(new)


def grade():
    """Rule 5: one row per line (player, game, source, line; the last pre-tip capture; over side graded, a push is void; asserted),
    rule 2d voids (0 minutes, or absent from a box score that exists), calibration verdict at 200 combined lines on calibrated
    P(over), market test at 800 lines (paired log loss vs the book, player-game cluster bootstrap CI), raw PIT as a diagnostic."""
    if not LOG.exists():
        if PERF.exists():
            PERF.unlink()
        print("[fp-v2] no v2 log yet"); return
    recal = load_recal(); params = load_params()
    map_fit = (recal or {}).get("fit_date")
    full = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str})
    if "gate" not in full.columns:
        full["gate"] = PRE_GATE
    pre = full[full.gate != GATE_EPOCH]          # Amendment 1 restart: rows logged before the refit, excluded from both verdicts
    snap = fp.last_pretip(full[full.gate == GATE_EPOCH])
    box = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})
    box_events = set(box.event_id); box = box.set_index(["event_id", "player_id"])
    ud = pd.read_csv(PRICES, dtype={"event_id": str}) if PRICES.exists() else pd.DataFrame(columns=PRICE_FIELDS)
    pp = pd.read_csv(PP_PRICES, dtype={"event_id": str}) if PP_PRICES.exists() else pd.DataFrame(columns=PP_PRICE_FIELDS)
    for df_ in (ud, pp):
        if "map_fit" not in df_.columns:
            df_["map_fit"] = PRE_GATE
    a1_rows = set(zip(full.loc[full.gate == GATE_EPOCH, "date"], full.loc[full.gate == GATE_EPOCH, "event_id"], full.loc[full.gate == GATE_EPOCH, "player"].map(norm_name), full.loc[full.gate == GATE_EPOCH, "logged_at"]))
    def a1_priced(df_):
        return pd.Series([(d, e, norm_name(p), t) in a1_rows for d, e, p, t in zip(df_.date, df_.event_id, df_.player, df_.fp_logged_at)], index=df_.index, dtype=bool)
    ok_ud = (ud.map_fit == map_fit) & a1_priced(ud); ok_pp = (pp.map_fit == map_fit) & a1_priced(pp)
    pre_prices = {"UD": int((~ok_ud).sum()), "PP": int((~ok_pp).sum())}
    ud = ud[ok_ud]; pp = pp[ok_pp]   # only prices through the frozen (refit) map AND from an A1 log row count
    rows, pit = [], []
    for r in snap.itertuples():
        in_box = (r.event_id, r.player_id) in box.index
        if not in_box and r.event_id not in box_events:
            continue   # game not graded yet: pending
        void, why = (0, "")
        if not in_box:
            void, why = 1, "absent from the box score"
        elif float(box.loc[(r.event_id, r.player_id)]["min"]) <= 0:
            void, why = 1, "0 minutes"
        b = box.loc[(r.event_id, r.player_id)] if in_box else None
        actual = None if void else float(fp.fp_of(b)); q = json.loads(r.fp_q)
        base = {"date": r.date, "event_id": r.event_id, "tip": r.tip, "player_id": r.player_id, "player": r.player, "team": r.team, "opp": r.opp, "pos": r.pos, "min_proj": r.min,
                "min_actual": float(b["min"]) if in_box else 0.0, "fp_mean": r.fp_mean, "fp_actual": None if void else round(actual, 1), "pit": None if void else round(fp.cdf_from_q(q, actual), 4),
                "void": void, "void_reason": why, "logged_at": r.logged_at}
        if not void:
            pit.append(base["pit"])
        key = norm_name(r.player)
        lines = []
        for src, df_ in (("UD", ud), ("PP", pp)):
            d = df_[(df_.date == r.date) & (df_.event_id == r.event_id) & (df_.player.map(norm_name) == key) & (df_.side == "higher") & (df_.captured_at < r.tip)]
            if len(d):
                d = d[d.captured_at == d.captured_at.max()]   # the last pre-tip capture
                for x in d.drop_duplicates(["line"]).itertuples():
                    lines.append((src, x))
        if not lines:
            rows.append({**base, "source": "", "market": "", "line": None, "p_raw": None, "p_cal": None, "p_book": None, "hit": None, "ll_v2": None, "ll_book": None, "captured_at": ""})
            continue
        for src, x in lines:
            line = float(x.line)
            hit = None if (void or actual == line) else int(actual > line)
            p_c = float(x.p_cal); p_b = float(x.p_book_novig) if pd.notna(x.p_book_novig) else None
            rows.append({**base, "source": src, "market": x.market, "line": line, "p_raw": float(x.p_raw), "p_cal": p_c, "p_book": p_b, "hit": hit,
                         "ll_v2": round(float(logloss(np.array([p_c]), np.array([hit]))[0]), 5) if hit is not None else None,
                         "ll_book": round(float(logloss(np.array([p_b]), np.array([hit]))[0]), 5) if (hit is not None and p_b is not None) else None, "captured_at": x.captured_at})
    R = pd.DataFrame(rows, columns=GRADE_FIELDS)
    gl = R[(R.source != "") & R.line.notna()]
    dup = gl.duplicated(["player", "event_id", "source", "line"], keep=False)
    if dup.any():
        R.to_csv(GRADE, index=False)
        raise SystemExit(f"[fp-v2 grade] ONE-ROW-PER-LINE VIOLATION: {int(dup.sum())} rows share (player, game, source, line): {gl[dup][['date', 'player', 'source', 'line', 'captured_at']].head(10).to_dict('records')}")
    R.to_csv(GRADE, index=False)
    g = gl[gl.hit.notna()]
    def cal_block(d, gate):
        p_, h_ = d.p_cal.astype(float).values, d.hit.astype(int).values
        bk = fp.buckets(p_, h_) if len(d) else fp.buckets(np.array([]), np.array([]))
        o = {"n": int(len(d)), "hits": int(h_.sum()) if len(d) else 0, "hit_rate": round(float(h_.mean()), 4) if len(d) else None, "mean_p_cal": round(float(p_.mean()), 4) if len(d) else None,
             "mean_p_raw": round(float(d.p_raw.astype(float).mean()), 4) if len(d) else None, "by_decile": bk}
        o["gate" if gate else "decile_read"] = cal_verdict(bk, len(d))
        return o
    def cal_verdict(bk, n_total):
        """Rule 5 as amended (Amendment 1): the decile table is a checkpoint at CHECKPOINT_N lines (no verdict); the calibration
        verdict renders at GATE_N = 800 on the same bars (weighted |gap| <= 0.05 over deciles with n >= MIN_DECILE, none > 0.10)."""
        use = [b for b in bk if b["n"] >= MIN_DECILE]
        stats = {}
        if use:
            wmae = sum(b["n"] * abs(b["gap"]) for b in use) / sum(b["n"] for b in use); worst = max(abs(b["gap"]) for b in use)
            stats = {"weighted_mean_abs_gap": round(wmae, 4), "worst_gap": round(worst, 4)}
        if n_total < CHECKPOINT_N:
            return {"status": "collecting", "graded": int(n_total), "needed": GATE_N, "checkpoint_at": CHECKPOINT_N}
        if n_total < GATE_N:
            return {"status": "checkpoint", "graded": int(n_total), "needed": GATE_N, "checkpoint_at": CHECKPOINT_N, **stats, "note": "decile table reported at 200+ lines as a checkpoint only; no verdict until 800 (Amendment 1)"}
        if not use:
            return {"status": "collecting", "graded": int(n_total), "needed": GATE_N, "note": "no decile with 20+ rows"}
        return {"status": "PASS" if stats["weighted_mean_abs_gap"] <= WMAE_BAR and stats["worst_gap"] <= WORST_BAR else "FAIL", "graded": int(n_total), "needed": GATE_N, **stats}
    def market_block(d, label):
        m = d[d.ll_book.notna()]
        if not len(m):
            return {"label": label, "n": 0, "status": "collecting", "needed": MARKET_N}
        diff = (m.ll_v2.astype(float) - m.ll_book.astype(float)).values
        pg = (m.date + ":" + m.event_id + ":" + m.player_id).values
        ci = cluster_ci(diff, pg)
        status = "collecting" if len(m) < MARKET_N else ("PASS" if ci[1] < 0 else "not shown")
        return {"label": label, "n": int(len(m)), "needed": MARKET_N, "status": status, "mean_logloss_v2": round(float(m.ll_v2.astype(float).mean()), 4), "mean_logloss_book": round(float(m.ll_book.astype(float).mean()), 4),
                "mean_diff_nats": round(float(diff.mean()), 4), "diff_ci_cluster": ci, "baseline": {"UD": "no-vig P(over) from both captured sides", "PP": "0.5"}.get(label, "per source"),
                "note": "negative = v2 beats the book; PASS when the clustered 95% CI excludes 0 in v2's favour at n >= 800"}
    pit_arr = np.array(pit); tab, wm, _, wo, _ = pit_verdict(pit_arr) if len(pit_arr) else ([], None, None, None, False)
    pg = R.drop_duplicates(["date", "event_id", "player_id"])
    doc = {"generated": ts(now()), "rule_file": RULE, "label": LABEL, "pair": (params or {}).get("selected"), "map": {"fit_date": recal.get("fit_date"), "n": recal.get("n")} if recal else None,
           "gate_epoch": GATE_EPOCH, "amendment": (recal or {}).get("amendment"),
           "excluded_pre_amendment": {"log_rows": int(len(pre)), "player_games": int(pre.drop_duplicates(["date", "event_id", "player_id"]).shape[0]), "slates": sorted(set(pre.date)),
                                      "ud_price_rows": pre_prices["UD"], "pp_price_rows": pre_prices["PP"],
                                      "note": "logged or priced before the Amendment 1 refit (2026-10-28): excluded from both verdicts; the forward gate starts with the first slate priced by the refit map"},
           "player_games": int(len(pg)), "void": int(pg.void.sum()), "void_reasons": dict(Counter(pg[pg.void == 1].void_reason)), "pending_in_log": int(len(snap) - len(pg)),
           "lines": {**cal_block(g, True), "scope": "combined UD + PP, one row per line (over side), calibrated P(over): the calibration verdict (rule 5)", "sources": {s: int((g.source == s).sum()) for s in ("UD", "PP")}},
           "by_source": {s: {**cal_block(g[g.source == s], False), "label": {"UD": "Underdog", "PP": "PrizePicks"}[s]} for s in ("UD", "PP")},
           "market_test": {"combined": market_block(g, "UD + PP"), "by_source": {s: market_block(g[g.source == s], s) for s in ("UD", "PP")}},
           "pit": {"n": int(len(pit_arr)), "deciles": tab, "weighted_mean_abs_gap": round(wm, 4) if wm is not None else None, "worst_decile_gap": round(wo, 4) if wo is not None else None,
                   "mean": round(float(pit_arr.mean()), 4) if len(pit_arr) else None, "note": "raw v2 distribution PIT, diagnostic only (rule 5)"},
           "recent": pg.sort_values(["date", "fp_mean"], ascending=[False, False]).head(60)[["date", "player", "team", "opp", "min_proj", "min_actual", "fp_mean", "fp_actual", "pit", "void", "void_reason"]].to_dict("records"),
           "method": ["FP v2 = v1's joint sim with rule 2a (minutes sd x factor, short-night mixture) and 2b (rates blended toward the current season, w = GP/(GP+10)); P(over) through a frozen isotonic map fitted on the v2 replay of the graded v1 lines (2c).",
                      "One row per line: (player, game, source, line), the last pre-tip capture, over side graded; a push is void; the grader asserts uniqueness and fails loudly.",
                      "Rule 2d: a logged player-game with 0 minutes, or absent from a box score that exists, is void and leaves the pending set.",
                      "Calibration verdict at 800 combined lines (weighted |gap| <= 0.05, no decile > 0.10) on calibrated P(over), with a no-verdict checkpoint table at 200 (Amendment 1, 2026-10-28); UD / PP splits displayed. Market test at 800 lines: paired log loss vs the book (UD no-vig, PP 0.5), player-game cluster bootstrap CI must exclude 0 in v2's favour.",
                      "Amendment 1 (2026-10-28): isotonic blocks with n < 20 merged into the nearest neighbour (pooled hit rate), calibrated P(over) clipped to [0.05, 0.95]; a captured line is priced only from a v2 row logged at or before the capture. Rows logged or priced before the refit are marked pre-A1 and excluded; the gate restarted with the first slate priced by the refit map.",
                      "Paper study until both verdicts render. No tuning on these results, ever."]}
    write_json(PERF, doc)
    print(f"[fp-v2] graded {len(pg)} player-games ({int(pg.void.sum())} void {doc['void_reasons']}), {len(g)} lines (UD {doc['lines']['sources']['UD']}, PP {doc['lines']['sources']['PP']}; "
          f"calibration {doc['lines']['gate']['status']}, market {doc['market_test']['combined']['status']}), PIT mean {doc['pit']['mean']}; "
          f"excluded pre-amendment: {len(pre)} log rows, UD {pre_prices['UD']} / PP {pre_prices['PP']} price rows")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "grid", "select", "retro", "fit", "recount", "snapshot", "price", "price_pp", "grade"])
    ap.add_argument("arg", nargs="?", default=None)
    ap.add_argument("--start", default=REPLAY_START); ap.add_argument("--end", default=REPLAY_END); ap.add_argument("--season", type=int, default=REPLAY_SEASON)
    ap.add_argument("--out", default=None); ap.add_argument("--max-days", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "replay":
        replay(a.start, a.end, a.season, Path(a.out) if a.out else REPLAY, a.max_days)
    elif a.cmd == "grid":
        grid(Path(a.out) if a.out else REPLAY)
    elif a.cmd == "price":
        price(a.arg)
    elif a.cmd == "price_pp":
        price_pp(a.arg or fp.PP_FILE)
    else:
        {"select": select, "retro": retro, "fit": fit, "recount": recount, "snapshot": snapshot, "grade": grade}[a.cmd]()
