"""Ladder fix, 2026-11-10 (CEO order). Two shadow tracks next to the live v2 ladder engine; nothing here changes a shipped P.

PART A  live recalibration under backtests/live_recalibration_rule.md (trigger pulled to graded day 20 by the 2026-11-10 amendment)
  python ladder_recal.py evaluate       ONE-TIME, spent after the first run: trigger per stat on graded days 1-20 (band 0.55 <= shipped P
                                        < 0.80, gap = mean P - hit rate, cluster bootstrap by player-game, 2,000 resamples, seed 20261023,
                                        the grader's own code), isotonic refit (ladder_calibrate.fit_map) of hit on p_raw over days 1-20
                                        per qualifying stat -> data/ladder_recal.json (version recal_v1, status shadow)
PART B  minutes uncertainty under backtests/ladder_minutes_sim_rule.md
  python ladder_recal.py minutes-fit    ONE-TIME: empirical minutes-error distribution (data/fp_log.csv projected minutes vs box-score
                                        minutes, 2026-10-23 .. 2026-11-09 only, starter / bench) + the minutes-conditional sigma per stat
                                        (2023-24 walk-forward rows, a measurement, not a fit on live data) + the pre-registered evaluation n
                                        -> data/ladder_minsim.json (version minsim_v1, status shadow)
BOTH
  python ladder_recal.py shadow         after every ladder snapshot: p_recal (Part A map on p_raw) and p_minsim (Part B engine) next to
                                        p_raw and shipped p for every pre-tip rung -> data/ladder_shadow_log.csv (append-only)
  python ladder_recal.py shadow --backfill DATE   the same rows from the committed pre-tip projections.json builds of DATE (git history),
                                        flagged backfill=1; used once for 2026-11-10, the day the shadow went live mid-slate
  python ladder_recal.py grade          after every ladder grade: Part A validation clock (graded days 21-34, decision once at day 34, ship
                                        = write the map into data/ladder_calibration.json) and Part B evaluation (first N graded band rows
                                        from 2026-11-10, read once at N) -> docs/data/ladder_recal.json, merged into performance.json
                                        under diagnostic.recalibration / diagnostic.minutes_sim; failures -> data/failure_log.jsonl
Picks and display keep consuming the shipped P throughout. The graded frame is rebuilt here with the grader's functions and checked
against performance.json so every number comes from the same code path as the page.
"""
import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_log as LL  # noqa: E402  (graded_snapshot, boot_ci, cluster_ids, gap_block, block: the grader's own code)
from common import DATA_DIR, ROOT, SITE_DATA, read_json, write_json  # noqa: E402

RECAL = DATA_DIR / "ladder_recal.json"
MINSIM = DATA_DIR / "ladder_minsim.json"
SHADOW = DATA_DIR / "ladder_shadow_log.csv"
CALIB = DATA_DIR / "ladder_calibration.json"
FAILURES = DATA_DIR / "failure_log.jsonl"
OUT = SITE_DATA / "ladder_recal.json"
PERF = SITE_DATA / "performance.json"
PROJ = SITE_DATA / "projections.json"
BOX = DATA_DIR / "player_games.csv"
V2_ROWS_2024 = DATA_DIR / "v2_rows_2024.csv.gz"
RULE_A = "backtests/live_recalibration_rule.md"
RULE_B = "backtests/ladder_minutes_sim_rule.md"
RECAL_VERSION, MINSIM_VERSION = "recal_v1", "minsim_v1"
FIT_DAYS = 20                       # Part A: days 1-20 (amendment 2026-11-10; the committed rule said 21)
VAL_DAYS = (21, 34)                 # Part A: validation window, the 14 graded days after the fit window (committed rule: 22-35 after a 21-day fit)
BAND = LL.BAND
STATS = list(LL.STAT_ACT)
CLIP = (0.005, 0.995)
MIN_FIT = ("2026-10-23", "2026-11-09")   # Part B: minutes-error fit rows (fp_log starts 2026-10-23; day 20 = 2026-11-09)
EVAL_START = "2026-11-10"                # Part B: graded rows from this date forward only
EVAL_N = 2000                            # Part B: pre-registered n of graded band rows (see minutes-fit's registration block)
GRID = 50                                # Part B: equal-probability quantile grid of the error distribution used in the convolution
STARTER_GAMES = 5                        # starter proxy = started >= half of the last 5 box scores before the date (build_model's starter_rate)
MIN_PROJ_FLOOR = 4.0                     # the engine never publishes a player under ROTATION_FLOOR minutes; the sigma measurement uses the same floor
SHADOW_FIELDS = ["date", "event_id", "tip", "player_id", "player", "team", "opp", "stat", "rung", "p_raw", "p", "p_recal", "p_minsim", "p_minsim_cal", "cal_applied",
                 "min_proj", "group", "recal_version", "minsim_version", "source_updated", "logged_at", "backfill"]


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def stamp(t=None):
    return (t or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


# ---------------- the graded record, the grader's way ----------------
def graded_frame():
    """The rows ladder_log.grade grades (last pre-tip snapshot per key, hit = actual >= rung, void excluded), rebuilt with the
    grader's own functions. Vectorised join instead of the grader's row loop; identical semantics (checked in grade())."""
    log = pd.read_csv(LL.LOG, dtype={"event_id": str, "player_id": str})
    snap = LL.graded_snapshot(log)
    box = pd.read_csv(BOX, dtype={"event_id": str, "player_id": str})
    done = set(box.event_id)
    b = box.drop_duplicates(["event_id", "player_id"], keep="last")[["event_id", "player_id", "min", "tpm", "pts", "ast", "reb"]].rename(
        columns={"min": "minutes", "tpm": "act_tpm", "pts": "act_pts", "ast": "act_ast", "reb": "act_reb"})
    d = snap.merge(b, on=["event_id", "player_id"], how="left")
    d["ingested"] = d.event_id.isin(done)
    d["minutes"] = np.where(d.ingested, d.minutes.fillna(0.0), np.nan)
    d["actual"] = np.select([d.stat == s for s in STATS], [d[f"act_{s}"] for s in STATS], np.nan)
    d.loc[~d.ingested, "actual"] = np.nan
    d["status"] = np.where(~d.ingested, "pending", np.where(d.minutes > 0, "graded", "void"))
    g = d[d.status == "graded"].copy()
    g["hit"] = (g.actual >= g.rung).astype(int)
    g["p"] = g.p.astype(float); g["p_raw"] = g.p_raw.astype(float); g["rung"] = g.rung.astype(int)
    return g.drop(columns=[f"act_{s}" for s in STATS])


def day_index(g):
    days = sorted(g.date.unique())
    return days, {d: i + 1 for i, d in enumerate(days)}


def check_against_performance(g):
    """The page's own numbers must come out of this frame; a mismatch means the join drifted from the grader."""
    P = read_json(PERF, None)
    if not P or P.get("empty"):
        return {"checked": False}
    mine = {"graded": int(len(g)), "player_games": LL.player_games(g), "dates": int(g.date.nunique())}
    theirs = {k: P["log"][k] for k in mine}
    band = g[(g.p >= BAND[0]) & (g.p < BAND[1])]
    mine["band_n"] = int(len(band)); theirs["band_n"] = P["diagnostic"]["all"]["n"]
    return {"checked": True, "match": mine == theirs, "mine": mine, "performance_json": theirs, "generated": P.get("generated")}


# ---------------- bootstrap helpers (the grader's cluster scheme) ----------------
def boot_paired(d, a, b, iters=LL.BOOT_ITERS, seed=LL.BOOT_SEED):
    """95% cluster-bootstrap CI (player-game clusters, percentile) of mean(a - b) over rows of d; a, b = per-row arrays."""
    if len(d) < 2:
        return None
    cid = LL.cluster_ids(d); C = int(cid.max() + 1)
    if C < 2:
        return None
    cn = np.bincount(cid, minlength=C).astype(float)
    cs = np.bincount(cid, weights=np.asarray(a, float) - np.asarray(b, float), minlength=C)
    rng = np.random.default_rng(seed); vals = []
    for start in range(0, iters, LL.BOOT_CHUNK):
        idx = rng.integers(0, C, size=(min(LL.BOOT_CHUNK, iters - start), C))
        vals.append(cs[idx].sum(axis=1) / cn[idx].sum(axis=1))
    v = np.concatenate(vals)
    return [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]


def logloss(p, h):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); h = np.asarray(h, float)
    return -(h * np.log(p) + (1 - h) * np.log(1 - p))


def apply_map(m, p):
    return np.clip(np.interp(np.asarray(p, float), m["x"], m["y"]), *CLIP)


def log_failure(entry):
    with open(FAILURES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at": stamp(), **entry}) + "\n")


# ================= PART A: evaluate (one-time) =================
def evaluate():
    if RECAL.exists():
        raise SystemExit(f"{RECAL.name} exists: the one-time evaluation is SPENT ({read_json(RECAL, {}).get('evaluated_at')}); no second evaluation under this rule")
    from ladder_calibrate import fit_map  # the rule's fit (pool-adjacent-violators), imported, not copied
    g = graded_frame(); chk = check_against_performance(g)
    days, idx = day_index(g)
    if len(days) < FIT_DAYS:
        raise SystemExit(f"only {len(days)} graded days; the evaluation needs day {FIT_DAYS}")
    fit_dates = days[:FIT_DAYS]
    f = g[g.date.isin(fit_dates)].copy()
    band = f[(f.p >= BAND[0]) & (f.p < BAND[1])]
    trig = {"all": LL.gap_block(band), "by_stat": {}}
    for st in STATS:
        b = LL.gap_block(band[band.stat == st])
        ci = b.get("gap_ci")
        b["qualifies"] = bool(b["n"] and b["gap"] is not None and b["gap"] > 0 and ci and ci[0] > 0)
        trig["by_stat"][st] = b
    maps = {}
    for st in STATS:
        if not trig["by_stat"][st]["qualifies"]:
            continue
        rows = f[f.stat == st]
        m = fit_map(rows.p_raw.to_numpy(float), rows.hit.to_numpy(float))
        mapped = apply_map(m, rows.p_raw)
        maps[st] = {**m, "fit_rows": int(len(rows)), "fit_player_games": LL.player_games(rows), "blocks": len(m["x"]),
                    "in_sample": {"logloss_shipped": round(float(logloss(rows.p, rows.hit).mean()), 5), "logloss_raw_engine": round(float(logloss(rows.p_raw, rows.hit).mean()), 5),
                                  "logloss_mapped": round(float(logloss(mapped, rows.hit).mean()), 5), "note": "in-sample, information only; the decision is the days 21-34 validation"}}
    doc = {"version": RECAL_VERSION, "rule": RULE_A, "amendment": "2026-11-10 (evaluation pulled from graded day 21 to day 20 by CEO order after the day-20 diagnostic had been viewed; one-time, spent)",
           "evaluated_at": stamp(), "spent": True, "status": "shadow",
           "fit_window": {"days": [1, FIT_DAYS], "first_date": fit_dates[0], "last_date": fit_dates[-1], "dates": fit_dates, "graded_rows": int(len(f)), "player_games": LL.player_games(f)},
           "band": list(BAND), "bootstrap": {"iters": LL.BOOT_ITERS, "seed": LL.BOOT_SEED, "cluster": "player-game (date, event_id, player_id)", "code": "scripts/ladder_log.py boot_ci"},
           "trigger": trig, "qualifying": [st for st in STATS if trig["by_stat"][st]["qualifies"]], "maps": maps,
           "validation": {"days": list(VAL_DAYS), "status": "pending", "decided_at": None, "decisions": {},
                          "ship_condition": "per stat: log loss of the map applied to p_raw beats the log loss of the currently shipped p on the graded rungs of days 21-34, paired difference (shipped - mapped) > 0 with the 95% player-game cluster-bootstrap CI excluding 0; otherwise the map is dropped for that stat and the result goes to data/failure_log.jsonl. No partial ship, no ship on the fit."},
           "grader_check": chk, "shadow_log": str(SHADOW.relative_to(ROOT)).replace("\\", "/")}
    write_json(RECAL, doc)
    print(f"[recal] days 1-{FIT_DAYS} = {fit_dates[0]}..{fit_dates[-1]}; grader check {'OK' if chk.get('match') else chk}")
    for st in STATS:
        b = trig["by_stat"][st]; print(f"[recal] {st}: band n {b['n']} ({b['player_games']} pg) mean P {b['mean_p']} hit {b['hit_rate']} gap {b['gap']} CI {b['gap_ci']} -> {'REFIT' if b['qualifies'] else 'no'}")
    print(f"[recal] combined gap {trig['all']['gap']} CI {trig['all']['gap_ci']} (reported, never used alone); maps fitted for {list(maps)}; status shadow")
    return doc


# ================= PART B: minutes-fit (one-time) =================
def starter_groups(box, keys):
    """Starter proxy per (date, player_id): started >= half of the last STARTER_GAMES box scores strictly before the date.
    No earlier box score = 'unknown' (pooled distribution). Pre-tip information only."""
    b = box.sort_values("date")[["date", "player_id", "starter"]]
    by = {pid: (grp.date.to_numpy(), grp.starter.to_numpy(float)) for pid, grp in b.groupby("player_id")}
    out = {}
    for date, pid in keys:
        s = by.get(pid)
        if s is None:
            out[(date, pid)] = "unknown"; continue
        dates, st = s; n = int(np.searchsorted(dates, date, side="left"))
        if n == 0:
            out[(date, pid)] = "unknown"; continue
        out[(date, pid)] = "starter" if st[max(0, n - STARTER_GAMES):n].mean() >= 0.5 else "bench"
    return out


def quantile_grid(err, grid=GRID):
    q = (np.arange(grid) + 0.5) / grid
    return [round(float(v), 3) for v in np.quantile(np.asarray(err, float), q)]


def minutes_fit():
    if MINSIM.exists():
        raise SystemExit(f"{MINSIM.name} exists: the fit and the registration are written once ({read_json(MINSIM, {}).get('fitted_at')})")
    f = pd.read_csv(LL.FP_LOG, dtype={"event_id": str, "player_id": str}, usecols=["date", "event_id", "player_id", "tip", "min", "logged_at"])
    f = f[(f.date >= MIN_FIT[0]) & (f.date <= MIN_FIT[1])]
    f = f[pd.to_datetime(f.logged_at, utc=True) < pd.to_datetime(f.tip, utc=True)].sort_values("logged_at").drop_duplicates(["date", "event_id", "player_id"], keep="last")
    box = pd.read_csv(BOX, dtype={"event_id": str, "player_id": str})
    d = f.merge(box.drop_duplicates(["event_id", "player_id"], keep="last")[["event_id", "player_id", "min"]].rename(columns={"min": "act_min"}), on=["event_id", "player_id"], how="inner")
    d = d[(d.act_min > 0) & (d["min"].astype(float) >= MIN_PROJ_FLOOR)].copy()     # played (voids are not graded) and inside the engine's published floor
    d["err"] = d["min"].astype(float) - d.act_min.astype(float)
    grp = starter_groups(box, list(zip(d.date, d.player_id)))
    d["group"] = [grp[(a, b)] for a, b in zip(d.date, d.player_id)]
    groups = {}
    for name, rows in [("starter", d[d.group == "starter"]), ("bench", d[d.group == "bench"]), ("pooled", d)]:
        e = rows.err.to_numpy(float)
        groups[name] = {"n": int(len(e)), "player_games": int(len(rows)), "mean": round(float(e.mean()), 3), "sd": round(float(e.std(ddof=1)), 3),
                        "share_over": round(float((e > 0).mean()), 4), "p10": round(float(np.percentile(e, 10)), 2), "p50": round(float(np.percentile(e, 50)), 2), "p90": round(float(np.percentile(e, 90)), 2),
                        "share_over_5": round(float((e > 5).mean()), 4), "share_over_10": round(float((e > 10).mean()), 4), "grid": quantile_grid(e)}
    # minutes-conditional sigma per stat: the 2023-24 walk-forward residual after scaling the projected mean to the minutes actually
    # played (proj x act_min / proj_min). The shipped sigma is the unconditional residual of the same rows (ladder_v2_rule section 1);
    # convolving the minutes error on top of the unconditional sigma would count the minutes variance twice.
    v = pd.read_csv(V2_ROWS_2024, dtype={"player_id": str, "event_id": str})
    v = v[(v.act_min > 0) & (v.proj_min >= MIN_PROJ_FLOOR)]
    sig = {}
    for s in ("pts", "ast", "reb"):
        scaled = v[f"proj_{s}"] * v.act_min / v.proj_min
        sig[s] = {"sigma_cond": round(float((scaled - v[f"act_{s}"]).std(ddof=1)), 4), "sigma_unconditional_same_rows": round(float((v[f"proj_{s}"] - v[f"act_{s}"]).std(ddof=1)), 4),
                  "shipped_sigma": LL_SIGMA[s]}
    reg = {"registered_at": stamp(), "window_start": EVAL_START, "n": EVAL_N,
           "rows": "graded rungs with shipped P in [0.55, 0.80) dated 2026-11-10 or later that carry a last-pre-tip shadow p_minsim (rows logged live by `shadow`, plus the 2026-11-10 rows backfilled from the committed pre-tip projections.json builds, flagged backfill=1), ordered by date, tip, event_id, player_id, stat, rung; the first n are the evaluation, read once when n is reached",
           "ship_if": ["band log loss of p_minsim < band log loss of the shipped p on those n rows, paired difference (shipped - minsim) with the 95% player-game cluster-bootstrap CI (2,000 resamples, seed 20261023) excluding 0",
                       "|mean p_minsim - hit rate| < |mean shipped p - hit rate| on the same rows (the band gap moves toward 0)"],
           "pooled": "the decision is pooled over the four stats (one engine change, one version); per-stat numbers are reported, never used to ship a subset",
           "power": "about 143 band rungs grade per day (2,854 over days 1-20), so n = 2,000 is about 14 graded days, the length of Part A's validation window; with a per-rung log-loss difference sd near 0.08 and a design effect near 1.7 (rungs per player-game) the 95% half-width at n = 2,000 is about 0.0045 nats, and removing a +0.07 gap on half the band (PTS) is worth about +0.006 nats pooled",
           "freeze": "Part B may not ship before Part A's validation window (graded days 21-34) closes; a read at n before then is recorded and held, never re-read",
           "evaluated_quantity": "p_minsim_cal = the convolved raw probability passed through the probability layer the engine ships at snapshot time (today the 2024-25 isotonic maps adopted under ladder_v2_rule.md; the shadow log records which layer was applied). The sim changes, nothing downstream of it does. The unmapped p_minsim is logged and reported as information only",
           "compares_against": "the shipped p as logged at the time (raw engine through the 2024-25 maps today; if a Part A map ships at day 34 the later rows carry it, so Part B is judged against the then-current engine, never stacked blindly)"}
    doc = {"version": MINSIM_VERSION, "rule": RULE_B, "fitted_at": stamp(), "status": "shadow",
           "fit_window": {"first_date": MIN_FIT[0], "last_date": MIN_FIT[1], "source": "data/fp_log.csv last pre-tip projected minutes per player-game vs data/player_games.csv minutes; players who played; projected >= 4"},
           "error": {"definition": "projected minutes minus actual minutes (positive = the player played less than projected)", "grid_points": GRID, "groups": groups,
                     "starter_proxy": f"started at least half of the last {STARTER_GAMES} box scores before the date (build_model's starter_rate, pre-tip information only); no prior box score = pooled"},
           "sigma": {"source": "data/v2_rows_2024.csv.gz (2023-24 walk-forward of the v2 engine, players who played, projected minutes >= 4)", "rows": int(len(v)), "by_stat": sig},
           "engine": {"pts_ast_reb": "P(X >= k) = mean over the group's error grid e of 1 - Phi((k - 0.5 - mu(e)) / sigma_cond), mu(e) = projected mean x clip(projected minutes - e, 0, 48) / projected minutes",
                      "tpm": "attempts Poisson at the projected 3PA per minute x minutes, makes binomial at make_p, so P(3PM >= k) = mean over e of the Poisson(lambda(e) x make_p) upper tail",
                      "then": "the convolved raw goes through the probability layer the engine ships (adopted isotonic maps), exactly as the fixed-sigma raw does today",
                      "unchanged": "rates, priors, blend, DvP, usage bump, minutes projection, the adopted maps, grading code"},
           "registration": reg, "evaluation": {"status": "collecting", "n_so_far": 0, "read_at": None, "decision": None}}
    write_json(MINSIM, doc)
    for k, v_ in groups.items():
        print(f"[minsim] {k}: n {v_['n']} mean err {v_['mean']} sd {v_['sd']} share proj high {v_['share_over']} (>5: {v_['share_over_5']}, >10: {v_['share_over_10']}) p10/p50/p90 {v_['p10']}/{v_['p50']}/{v_['p90']}")
    print(f"[minsim] sigma_cond {{{', '.join(f'{s}: {x['sigma_cond']} (uncond {x['sigma_unconditional_same_rows']}, shipped {x['shipped_sigma']})' for s, x in sig.items())}}}; registered n = {EVAL_N} from {EVAL_START}")
    return doc


LL_SIGMA = {"pts": 6.5044, "ast": 2.0175, "reb": 2.6527}   # build_model.SIGMA, copied for the provenance line only (build_model is not imported here: its import pulls the odds client)


# ================= shadow engine =================
def poisson_tail(lam, k):
    """P(Poisson(lam) >= k), vectorised over lam."""
    lam = np.asarray(lam, float)
    if k <= 0:
        return np.ones_like(lam)
    terms = np.zeros_like(lam); term = np.exp(-lam)
    for i in range(k):
        terms += term; term = term * lam / (i + 1)
    return np.clip(1 - terms, 0, 1)


def gauss_tail(mu, sigma, k):
    from math import erf, sqrt
    mu = np.asarray(mu, float)
    return 0.5 * (1 - np.vectorize(erf)((k - 0.5 - mu) / (sigma * sqrt(2))))


def minsim_p(ms, stat, k, player, group):
    """Part B probability for one rung from the projection entry (min, means, make_p) and the group's error grid."""
    grid = np.asarray(ms["error"]["groups"].get(group, ms["error"]["groups"]["pooled"])["grid"], float)
    mp = float(player["min"])
    if mp <= 0:
        return None
    m = np.clip(mp - grid, 0, 48)
    if stat == "tpm":
        lam = float(player["tpa"]["mean"]) / mp * m * float(player["make_p"])
        return float(poisson_tail(lam, k).mean())
    mu = float(player[stat]["mean"]) / mp * m
    return float(gauss_tail(mu, ms["sigma"]["by_stat"][stat]["sigma_cond"], k).mean())


def shipped_layer(stat, k, p, cal):
    """The probability layer the engine ships today, applied to a raw probability: build_model.apply_map's step interpolation over the
    adopted map for that stat / rung (identity when none). Part B changes the sim, not this layer ("no other model change")."""
    m = (cal or {}).get(stat, {}).get(str(k))
    if p is None or not m:
        return p
    return float(np.clip(np.interp(p, m["x"], m["y"]), *CLIP))


def shadow_rows(P, recal, ms, groups, logged_at, source_updated, backfill, t_now):
    cal = read_json(CALIB, {}); cal_label = "adopted maps " + ",".join(cal.get("adopted", [])) + (" + live recal " + ",".join(cal["live_recalibration"]) if cal.get("live_recalibration") else "")
    ev = {}
    for g in P["games"]:
        ev[(g["home"], g["away"])] = g; ev[(g["away"], g["home"])] = g
    rows, skipped = [], 0
    maps = (recal or {}).get("maps", {})
    for p in P["players"]:
        if p.get("inactive") or "ladder" not in p:
            continue
        g = ev.get((p["team"], p["opp"]))
        if not g or not g.get("start"):
            continue
        tip = parse_ts(g["start"])
        if t_now >= tip:
            skipped += 1; continue
        grp = groups.get((P["date"], str(p["id"])), "unknown")
        for stat, rungs in p["ladder"].items():
            for k, v in rungs.items():
                pr = float(v["raw"]); pm = minsim_p(ms, stat, int(k), p, grp) if ms else None
                rows.append({"date": P["date"], "event_id": g["event_id"], "tip": tip.strftime("%Y-%m-%dT%H:%M:%SZ"), "player_id": str(p["id"]), "player": p["name"], "team": p["team"], "opp": p["opp"],
                             "stat": stat, "rung": int(k), "p_raw": pr, "p": float(v["p"]),
                             "p_recal": round(float(apply_map(maps[stat], pr)), 4) if stat in maps else "",
                             "p_minsim": "" if pm is None else round(pm, 4), "p_minsim_cal": "" if pm is None else round(shipped_layer(stat, int(k), pm, cal), 4), "cal_applied": cal_label if pm is not None else "",
                             "min_proj": p["min"], "group": grp, "recal_version": recal["version"] if recal else "", "minsim_version": ms["version"] if ms else "",
                             "source_updated": source_updated, "logged_at": logged_at, "backfill": int(backfill)})
    return rows, skipped


def append_shadow(rows):
    new = not SHADOW.exists()
    with open(SHADOW, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHADOW_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def shadow(backfill_date=None):
    recal, ms = read_json(RECAL, None), read_json(MINSIM, None)
    if not recal and not ms:
        print("[shadow] nothing to shadow yet (no data/ladder_recal.json or data/ladder_minsim.json)"); return 0
    box = pd.read_csv(BOX, dtype={"event_id": str, "player_id": str}, usecols=["date", "player_id", "starter"])
    if backfill_date is None:
        P = read_json(PROJ, None)
        if not P or not P.get("games"):
            print("[shadow] no slate"); return 0
        groups = starter_groups(box, [(P["date"], str(p["id"])) for p in P["players"] if not p.get("inactive")])
        t = now(); rows, skipped = shadow_rows(P, recal, ms, groups, stamp(t), P.get("updated"), 0, t)
        append_shadow(rows)
        print(f"[shadow] appended {len(rows)} rows for {P['date']} ({skipped} players skipped: game already tipped)"); return len(rows)
    # backfill: every committed projections.json build dated backfill_date, each applied to the games it preceded; the last pre-tip
    # build per game wins, exactly like the graded snapshot. Rows carry the build's own stamp as logged_at and backfill=1.
    shas = subprocess.run(["git", "log", "--format=%H", "--", "docs/data/projections.json"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    done = set()
    if SHADOW.exists():
        have = pd.read_csv(SHADOW, dtype=str, usecols=["date", "event_id", "backfill"])
        done = set(have[(have.date == backfill_date)].event_id)
    builds = []
    for sha in shas[:200]:
        txt = subprocess.run(["git", "show", f"{sha}:docs/data/projections.json"], capture_output=True, text=True, cwd=ROOT, encoding="utf-8").stdout
        if not txt:
            continue
        P = json.loads(txt)
        if P.get("date") != backfill_date:
            if P.get("date", "") < backfill_date:
                break
            continue
        builds.append((parse_ts(P["updated"]), sha, P))
    builds.sort(key=lambda x: x[0])
    total = 0
    for i, (t_build, sha, P) in enumerate(builds):
        t_next = builds[i + 1][0] if i + 1 < len(builds) else None
        P = {**P, "games": [g for g in P["games"] if g.get("start") and g["event_id"] not in done and parse_ts(g["start"]) > t_build and (t_next is None or parse_ts(g["start"]) <= t_next)]}
        if not P["games"]:
            continue
        groups = starter_groups(box, [(P["date"], str(p["id"])) for p in P["players"] if not p.get("inactive")])
        rows, _ = shadow_rows(P, recal, ms, groups, stamp(t_build), P.get("updated"), 1, t_build)
        append_shadow(rows); total += len(rows)
        print(f"[shadow] backfill {backfill_date} from {sha[:7]} (build {stamp(t_build)}): {len(rows)} rows for {len(P['games'])} games")
    print(f"[shadow] backfill total {total} rows ({len(builds)} builds found, {len(done)} games already present)"); return total


# ================= grade: Part A validation + Part B evaluation =================
def last_shadow():
    if not SHADOW.exists():
        return None
    s = pd.read_csv(SHADOW, dtype={"event_id": str, "player_id": str})
    s = s[pd.to_datetime(s.logged_at, utc=True) < pd.to_datetime(s.tip, utc=True)]
    return s.sort_values("logged_at").drop_duplicates(["date", "event_id", "player_id", "stat", "rung"], keep="last")


def ship_recal(recal, st):
    """Section 3 shipping: the per-stat map replaces that stat's rung maps in data/ladder_calibration.json (applied to p_raw by
    build_model.apply_map); the previous entry is kept under history. Dated."""
    cal = read_json(CALIB, {})
    m = recal["maps"][st]
    cal.setdefault("history", []).append({"stat": st, "replaced_at": stamp(), "previous": cal.get(st), "reason": f"{RULE_A} section 3 ship ({recal['version']})"})
    cal[st] = {str(k): {"x": m["x"], "y": m["y"]} for k in {"tpm": [3, 4, 5, 6, 7], "pts": [10, 15, 20, 25, 30], "ast": [4, 6, 8, 10], "reb": [6, 8, 10, 12]}[st]}
    cal.setdefault("live_recalibration", {})[st] = {"version": recal["version"], "shipped_at": stamp(), "fit": f"isotonic on p_raw, graded days 1-{FIT_DAYS}", "validated": f"graded days {VAL_DAYS[0]}-{VAL_DAYS[1]}"}
    if st not in cal.get("adopted", []):
        cal.setdefault("adopted", []).append(st)
    write_json(CALIB, cal)


def grade():
    recal, ms = read_json(RECAL, None), read_json(MINSIM, None)
    if not recal and not ms:
        print("[recal] nothing to grade yet"); return
    g = graded_frame(); chk = check_against_performance(g)
    days, idx = day_index(g); g["day"] = g.date.map(idx)
    out = {"generated": stamp(), "days_graded": len(days), "last_graded_date": days[-1] if days else None, "grader_check": chk}
    # ---- Part A
    if recal:
        val = recal["validation"]; lo, hi = val["days"]
        w = g[(g.day >= lo) & (g.day <= hi)]
        table = {}
        for st, m in recal["maps"].items():
            r = w[w.stat == st]
            if not len(r):
                table[st] = {"n": 0}; continue
            mapped = apply_map(m, r.p_raw); la, lb = logloss(r.p, r.hit), logloss(mapped, r.hit)
            band = r[(r.p >= BAND[0]) & (r.p < BAND[1])]
            table[st] = {"n": int(len(r)), "player_games": LL.player_games(r), "days_in_window": int(r.date.nunique()),
                         "logloss_shipped": round(float(la.mean()), 5), "logloss_mapped": round(float(lb.mean()), 5), "logloss_raw_engine": round(float(logloss(r.p_raw, r.hit).mean()), 5),
                         "diff_shipped_minus_mapped": round(float((la - lb).mean()), 5), "diff_ci": boot_paired(r, la, lb),
                         "band": {"n": int(len(band)), "gap_shipped": round(float(band.p.mean() - band.hit.mean()), 4) if len(band) else None,
                                  "gap_mapped": round(float(apply_map(m, band.p_raw).mean() - band.hit.mean()), 4) if len(band) else None}}
        complete = len(days) >= hi
        if complete and val["status"] == "pending":
            decisions = {}
            for st, t in table.items():
                ok = bool(t.get("n") and t["diff_shipped_minus_mapped"] > 0 and t["diff_ci"] and t["diff_ci"][0] > 0)
                decisions[st] = {"ship": ok, "numbers": t}
                if ok:
                    ship_recal(recal, st)
                else:
                    log_failure({"track": "Part A live recalibration", "version": recal["version"], "stat": st, "result": "map dropped: validation days 21-34 did not beat the shipped p with a CI excluding 0", "numbers": t})
            val.update({"status": "decided", "decided_at": stamp(), "decisions": decisions, "table": table})
            recal["status"] = "decided: " + ", ".join(f"{st} {'SHIPPED' if d['ship'] else 'DROPPED'}" for st, d in decisions.items())
            write_json(RECAL, recal)
        out["recalibration"] = {"label": "PART A: live recalibration (shadow until the day-34 decision)", "rule": RULE_A, "version": recal["version"], "status": recal["status"], "amendment": recal["amendment"],
                                "evaluated_at": recal["evaluated_at"], "fit_window": recal["fit_window"], "trigger": recal["trigger"], "qualifying": recal["qualifying"],
                                "maps": {st: {k: v for k, v in m.items() if k not in ("x", "y", "n")} | {"blocks": len(m["x"])} for st, m in recal["maps"].items()},
                                "validation": {"days": [lo, hi], "days_graded_in_window": int(w.date.nunique()), "status": val["status"], "decided_at": val["decided_at"],
                                               "clock": (f"day {len(days)} graded; validation reads days {lo}-{hi}; decision at day {hi}" if not complete else f"day {hi} reached; decision recorded {val['decided_at']}"),
                                               "ship_condition": val["ship_condition"], "interim" if not complete else "final": table, "decisions": val.get("decisions", {})},
                                "shadow_rows": int(len(last_shadow())) if SHADOW.exists() else 0,
                                "method": ["Trigger: band gap per stat over graded days 1-20 with the grader's player-game cluster bootstrap; a stat refits only if its own CI excludes 0.",
                                           "Refit: isotonic (pool-adjacent-violators) of hit on the engine's raw p over days 1-20, one map per qualifying stat; logged in shadow as p_recal next to p_raw; picks and display keep the shipped p.",
                                           "Validation: days 21-34 only, games the refit never saw; shipped p vs the map on p_raw, paired log loss, CI excluding 0; the interim table is information, the decision is read once at day 34.",
                                           "Ship = the map written into data/ladder_calibration.json for that stat (previous maps kept under history); drop = data/failure_log.jsonl entry. Never a partial ship, never a ship on the fit."]}
    # ---- Part B
    if ms:
        reg, ev = ms["registration"], ms["evaluation"]
        s = last_shadow()
        rows = pd.DataFrame()
        if s is not None and len(s):
            s = s[s.p_minsim_cal.notna()][["date", "event_id", "player_id", "stat", "rung", "p_minsim", "p_minsim_cal", "p_recal", "min_proj", "group", "backfill"]]
            s["rung"] = s.rung.astype(int)
            e = g[(g.date >= reg["window_start"]) & (g.p >= BAND[0]) & (g.p < BAND[1])].merge(s, on=["date", "event_id", "player_id", "stat", "rung"], how="inner")
            rows = e.sort_values(["date", "tip", "event_id", "player_id", "stat", "rung"]).reset_index(drop=True)
        n_so_far = int(len(rows)); N = reg["n"]
        def compare(r):
            la, lb = logloss(r.p, r.hit), logloss(r.p_minsim_cal, r.hit)
            gs, gm = float(r.p.mean() - r.hit.mean()), float(r.p_minsim_cal.mean() - r.hit.mean())
            return {"n": int(len(r)), "player_games": LL.player_games(r), "hit_rate": round(float(r.hit.mean()), 4), "mean_p_shipped": round(float(r.p.mean()), 4), "mean_p_minsim": round(float(r.p_minsim_cal.mean()), 4),
                    "gap_shipped": round(gs, 4), "gap_minsim": round(gm, 4), "logloss_shipped": round(float(la.mean()), 5), "logloss_minsim": round(float(lb.mean()), 5),
                    "diff_shipped_minus_minsim": round(float((la - lb).mean()), 5), "diff_ci": boot_paired(r, la, lb), "gap_toward_zero": bool(abs(gm) < abs(gs)),
                    "unmapped_information": {"mean_p_minsim_raw": round(float(r.p_minsim.mean()), 4), "logloss_minsim_raw": round(float(logloss(r.p_minsim, r.hit).mean()), 5)}}
        cov = None
        if n_so_far:
            cov = {"rows": n_so_far, "by_stat": {st: compare(rows[rows.stat == st]) if (rows.stat == st).any() else {"n": 0} for st in STATS}, "by_group": {k: compare(r) for k, r in rows.groupby("group")},
                   "backfilled_rows": int(rows.backfill.sum()), "all": compare(rows)}
        freeze_lifted = bool(recal and recal["validation"]["status"] == "decided") if recal else True
        if n_so_far >= N and ev["status"] == "collecting":
            r = rows.head(N); res = compare(r)
            res["by_stat"] = {st: compare(r[r.stat == st]) if (r.stat == st).any() else {"n": 0} for st in STATS}
            passed = bool(res["diff_shipped_minus_minsim"] > 0 and res["diff_ci"] and res["diff_ci"][0] > 0 and res["gap_toward_zero"])
            ev.update({"status": "read", "read_at": stamp(), "n_so_far": n_so_far, "result": res, "decision": "PASS: ship as its own version once the Part A freeze has lifted" if passed else "FAIL: stays in shadow, written to the failure log",
                       "passed": passed, "freeze_lifted_at_read": freeze_lifted, "last_row": {k: str(r.iloc[-1][k]) for k in ("date", "event_id", "player_id", "stat", "rung")}})
            ms["status"] = "read: " + ("PASS (held by the Part A freeze)" if passed and not freeze_lifted else "PASS, ship-ready" if passed else "FAIL")
            if not passed:
                log_failure({"track": "Part B minutes-sim shadow engine", "version": ms["version"], "result": "did not ship: band log loss and/or band gap did not improve on the pre-registered n", "numbers": res})
            write_json(MINSIM, ms)
        elif ev["status"] == "collecting":
            ev["n_so_far"] = n_so_far; write_json(MINSIM, ms)
        out["minutes_sim"] = {"label": "PART B: minutes-uncertainty engine (shadow; pre-registered evaluation)", "rule": RULE_B, "version": ms["version"], "status": ms["status"], "fitted_at": ms["fitted_at"],
                              "fit_window": ms["fit_window"], "error_groups": {k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in ms["error"]["groups"].items()}, "sigma": ms["sigma"],
                              "registration": reg, "evaluation": {**ev, "n_so_far": n_so_far, "clock": f"{n_so_far} of {N} pre-registered band rows graded from {reg['window_start']}" + ("" if ev["status"] == "collecting" else f"; read {ev['read_at']}"),
                                                                 "freeze": "Part A validation " + ("closed" if freeze_lifted else f"open until graded day {VAL_DAYS[1]}; no Part B ship before then")},
                              "interim" if ev["status"] == "collecting" else "coverage_all_rows": cov,
                              "method": ["Shadow engine: each rung's probability integrated over the empirical minutes-error distribution (starter / bench, fit Oct 23 - Nov 9 only) with the minutes-conditional sigma measured on the 2023-24 walk-forward, then through the adopted maps like the shipped raw; rates, priors, blend, maps and grading untouched.",
                                         "Evaluation: graded band rows from 2026-11-10 forward only (the fit saw Nov 3-9); n pre-registered before the window opened, read once at n; interim numbers are information, never the decision.",
                                         "Ship only if band log loss improves with the cluster CI excluding 0 AND the band gap moves toward 0; otherwise the failure log. No ship before Part A's validation window closes."]}
    write_json(OUT, out)
    P = read_json(PERF, None)
    if P and not P.get("empty"):
        P.setdefault("diagnostic", {})
        if "recalibration" in out:
            P["diagnostic"]["recalibration"] = out["recalibration"]
        if "minutes_sim" in out:
            P["diagnostic"]["minutes_sim"] = out["minutes_sim"]
        write_json(PERF, P)
    a = out.get("recalibration", {}).get("validation", {}); b = out.get("minutes_sim", {}).get("evaluation", {})
    print(f"[recal] grader check {'OK' if chk.get('match') else chk}; Part A: {a.get('clock')}; Part B: {b.get('clock')} ({b.get('freeze')})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["evaluate", "minutes-fit", "shadow", "grade"]); ap.add_argument("--backfill")
    a = ap.parse_args()
    {"evaluate": evaluate, "minutes-fit": minutes_fit, "shadow": lambda: shadow(a.backfill), "grade": grade}[a.cmd]()
