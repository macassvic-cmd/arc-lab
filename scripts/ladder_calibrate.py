"""v2 ladder recalibration under backtests/ladder_v2_rule.md (section 3). Fit isotonic maps per stat / rung on the 2023-24
walk-forward of the v2 engine, gate them per stat on the 2024-25 walk-forward, adopt or reject. Also measures the Gaussian
sigmas (section 1) from the 2023-24 rows. Writes
  data/ladder_sigma.json           measured sigmas with provenance (build_model.SIGMA must equal them)
  data/ladder_calibration.json     adopted maps only (what the live engine applies)
  docs/data/ladder_v2_gate.json    the full report for the Backtest / Performance pages
Inputs: data/v2_rows_<season>.csv.gz from `backtest.py --season <s>` run with the v2 engine (2024, 2025; 2026 consistency only).
  python ladder_calibrate.py sigmas      print the measured sigmas
  python ladder_calibrate.py all         fit + gate + write
"""
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_model as bm
from common import DATA_DIR, SITE_DATA, write_json

SEED, ITERS, MIN_CELL, CLIP = 20261019, 1000, 200, (0.005, 0.995)
ROWS = {s: DATA_DIR / f"v2_rows_{s}.csv.gz" for s in (2024, 2025, 2026)}
CURRENT_ROWS = {2025: DATA_DIR / "backtest_rows_2025.csv.gz"}   # the engine before v2, same harness (information only)
STATS = ["tpm", "pts", "ast", "reb"]
ERF = np.vectorize(math.erf, otypes=[float])


def load(season, path=None):
    p = path or ROWS[season]
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"player_id": str, "event_id": str})
    d = d[d.act_min > 0].copy()
    d["pg"] = d.event_id + "|" + d.player_id
    return d


def sigmas(d):
    return {s: round(float((d[f"proj_{s}"] - d[f"act_{s}"]).std(ddof=1)), 4) for s in ("pts", "ast", "reb")}


def gauss_tail(mu, sigma, k):
    return 0.5 * (1 - ERF((k - 0.5 - mu) / (sigma * math.sqrt(2))))


def raw_p(d, stat, k, sigma):
    if stat == "tpm":
        return d[f"p_tpm_{k}"].astype(float).values
    return gauss_tail(d[f"proj_{stat}"].values.astype(float), sigma[stat], k)


def hits(d, stat, k):
    return (d[f"act_{stat}"].values >= k).astype(float)


def pav(x, y):
    """Pool-adjacent-violators: monotone non-decreasing fit of y on x. Returns block (mean x, fitted y, n)."""
    order = np.argsort(x, kind="stable"); x, y = x[order], y[order]
    bx, by, bn = [], [], []
    for xi, yi in zip(x, y):
        bx.append(xi); by.append(yi); bn.append(1)
        while len(by) > 1 and by[-2] > by[-1]:
            n = bn[-2] + bn[-1]
            by[-2] = (by[-2] * bn[-2] + by[-1] * bn[-1]) / n; bx[-2] = (bx[-2] * bn[-2] + bx[-1] * bn[-1]) / n; bn[-2] = n
            by.pop(); bx.pop(); bn.pop()
    return np.array(bx), np.array(by), np.array(bn)


def fit_map(p, h):
    bx, by, bn = pav(p, h)
    return {"x": [round(float(v), 5) for v in bx], "y": [round(float(np.clip(v, *CLIP)), 5) for v in by], "n": [int(v) for v in bn]}


def apply_map(m, p):
    return np.clip(np.interp(p, m["x"], m["y"]), *CLIP)


def logloss(p, h):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(h * np.log(p) + (1 - h) * np.log(1 - p))


def buckets(p, h):
    out = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (p >= lo) & (p < hi) if i < 9 else (p >= lo)
        n = int(m.sum())
        out.append({"lo": lo, "hi": hi, "n": n, "pred": round(float(p[m].mean()), 4) if n else None, "actual": round(float(h[m].mean()), 4) if n else None,
                    "gap": round(float(p[m].mean() - h[m].mean()), 4) if n else None, "graded": n >= MIN_CELL})
    return out


def worst_gap(bk):
    g = [abs(b["gap"]) for b in bk if b["graded"]]
    return round(max(g), 4) if g else None


def boot_diff(loss_raw, loss_cal, pg, iters=ITERS, seed=SEED):
    """Paired bootstrap over player-games of mean(raw) - mean(cal)."""
    codes, uniq = pd.factorize(pg)
    d_sum = np.bincount(codes, weights=loss_raw - loss_cal, minlength=len(uniq)); n_sum = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(iters):
        pick = rng.integers(0, len(uniq), len(uniq)); vals.append(d_sum[pick].sum() / n_sum[pick].sum())
    lo, hi = np.percentile(vals, [2.5, 97.5]); return [round(float(lo), 5), round(float(hi), 5)]


def stat_block(d, stat, maps, sigma):
    """Per stat: pooled rows over its rungs with raw and recalibrated P, losses, buckets."""
    P, C, H, PG, per_rung = [], [], [], [], {}
    for k in bm.RUNGS[stat]:
        p = raw_p(d, stat, k, sigma); h = hits(d, stat, k); c = apply_map(maps[stat][str(k)], p)
        P.append(p); C.append(c); H.append(h); PG.append(d.pg.values)
        per_rung[str(k)] = {"n": int(len(p)), "base_rate": round(float(h.mean()), 4), "mean_p_raw": round(float(p.mean()), 4), "mean_p_cal": round(float(c.mean()), 4),
                            "logloss_raw": round(float(logloss(p, h).mean()), 5), "logloss_cal": round(float(logloss(c, h).mean()), 5),
                            "buckets_raw": buckets(p, h), "buckets_cal": buckets(c, h)}
    P, C, H, PG = map(np.concatenate, (P, C, H, PG))
    lr, lc = logloss(P, H), logloss(C, H)
    bk_raw = [b for k in per_rung for b in per_rung[k]["buckets_raw"]]; bk_cal = [b for k in per_rung for b in per_rung[k]["buckets_cal"]]
    return {"n_rows": int(len(P)), "player_games": int(pd.Series(PG).nunique()), "logloss_raw": round(float(lr.mean()), 5), "logloss_cal": round(float(lc.mean()), 5),
            "diff_raw_minus_cal": round(float((lr - lc).mean()), 5), "diff_ci": boot_diff(lr, lc, PG),
            "worst_gap_raw": worst_gap(bk_raw), "worst_gap_cal": worst_gap(bk_cal), "per_rung": per_rung}


def decide(b):
    a = b["logloss_cal"] < b["logloss_raw"] and b["diff_ci"][0] > 0
    bb = b["worst_gap_raw"] is not None and b["worst_gap_cal"] is not None and b["worst_gap_cal"] < b["worst_gap_raw"]
    return {"a_logloss": bool(a), "b_worst_gap": bool(bb), "adopt": bool(a and bb)}


def current_vs_v2(d25, d25_current, sigma):
    """Information only: the engine before v2 vs the v2 engine on the same 2024-25 harness rows (3PM rungs; PTS/AST/REB means)."""
    if d25_current is None:
        return None
    key = ["event_id", "player_id"]
    m = d25[key + [f"p_tpm_{k}" for k in bm.RUNGS["tpm"]] + ["proj_pts", "proj_ast", "proj_reb", "act_tpm", "act_pts", "act_ast", "act_reb"]].merge(
        d25_current[key + [f"p_tpm_{k}" for k in bm.RUNGS["tpm"]] + ["proj_pts", "proj_ast", "proj_reb"]], on=key, suffixes=("", "_cur"))
    out = {"n": int(len(m)), "tpm": {}}
    for k in bm.RUNGS["tpm"]:
        h = (m.act_tpm >= k).astype(float).values
        out["tpm"][str(k)] = {"logloss_v2": round(float(logloss(m[f"p_tpm_{k}"].values, h).mean()), 5), "logloss_before": round(float(logloss(m[f"p_tpm_{k}_cur"].values, h).mean()), 5)}
    for s in ("pts", "ast", "reb"):
        out[s] = {"mae_v2": round(float((m[f"proj_{s}"] - m[f"act_{s}"]).abs().mean()), 4), "mae_before": round(float((m[f"proj_{s}_cur"] - m[f"act_{s}"]).abs().mean()), 4)}
    return out


def run():
    rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "backtests/ladder_v2_rule.md"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    d24, d25, d26 = load(2024), load(2025), load(2026)
    sig = sigmas(d24)
    write_json(DATA_DIR / "ladder_sigma.json", {"sigma": sig, "n": int(len(d24)), "source": "SD of (projected mean - actual) over the 2023-24 walk-forward rows of the v2 engine (data/v2_rows_2024.csv.gz, players who played)",
                                                "rule": "backtests/ladder_v2_rule.md section 1", "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    if any(abs(bm.SIGMA[s] - sig[s]) > 1e-4 for s in sig):
        raise SystemExit(f"build_model.SIGMA {bm.SIGMA} != measured {sig}: write the measured values into build_model.py (with provenance) and rerun")
    maps = {s: {str(k): fit_map(raw_p(d24, s, k, sig), hits(d24, s, k)) for k in bm.RUNGS[s]} for s in STATS}
    gate = {s: stat_block(d25, s, maps, sig) for s in STATS}
    decisions = {s: decide(gate[s]) for s in STATS}
    adopted = [s for s in STATS if decisions[s]["adopt"]]
    write_json(DATA_DIR / "ladder_calibration.json", {"adopted": adopted, "rule_commit": rule_commit, "fit": "2023-24 walk-forward (v2 engine), isotonic PAV per stat / rung, clipped [0.005, 0.995]",
                                                      "gate": "2024-25 walk-forward, per stat: (a) log loss, paired bootstrap CI above 0; (b) worst calibration gap at n >= 200 shrinks",
                                                      **{s: maps[s] for s in adopted}})
    fit_summary = {s: {k: {"blocks": len(maps[s][k]["x"]), "n": int(sum(maps[s][k]["n"])), "base_rate": round(float(hits(d24, s, int(k)).mean()), 4)} for k in maps[s]} for s in STATS}
    cons = {s: stat_block(d26, s, maps, sig) for s in STATS} if d26 is not None else None
    if cons:
        for s in cons:
            cons[s] = {k: v for k, v in cons[s].items() if k != "per_rung"} | {"per_rung": {r: {kk: vv for kk, vv in b.items() if not kk.startswith("buckets")} for r, b in cons[s]["per_rung"].items()}}
    rep = {"title": "v2 ladder recalibration gate", "rule_file": "backtests/ladder_v2_rule.md", "rule_commit": rule_commit, "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sigma": sig, "sigma_n": int(len(d24)), "rungs": bm.RUNGS, "adopted": adopted, "decisions": decisions,
           "fit_2023_24": {"rows": int(len(d24)), "games": int(d24.event_id.nunique()), "maps": fit_summary},
           "gate_2024_25": {"rows": int(len(d25)), "games": int(d25.event_id.nunique()), **gate},
           "consistency_2025_26": {"rows": int(len(d26)), "games": int(d26.event_id.nunique()), **cons} if cons else {"available": False},
           "before_v2_vs_v2_2024_25": current_vs_v2(d25, load(2025, CURRENT_ROWS[2025]), sig),
           "notes": ["Engine: v2 ladder (3PA = minutes x equal-weight blend of last-5 / last-15 / season 3PA-per-minute, EB-shrunk toward position with K = 120 minutes; PTS / AST / REB Gaussian with the measured sigmas). No Part B parameter anywhere.",
                     "Isotonic maps fit on 2023-24 only; the 2024-25 season read once for this gate; 2025-26 shown as consistency only (it is also the season the live model was tuned on earlier, so it proves nothing).",
                     "A stat whose maps fail either criterion ships raw P(k+); the Performance page says which.",
                     "Walk-forward caveats as on the Backtest page: absences read from the box score, no spreads, no news, play-style positions from the full season."]}
    write_json(SITE_DATA / "ladder_v2_gate.json", rep)
    for s in STATS:
        g = gate[s]; print(f"[gate] {s}: logloss raw {g['logloss_raw']} cal {g['logloss_cal']} diff {g['diff_raw_minus_cal']} CI {g['diff_ci']} | worst gap raw {g['worst_gap_raw']} cal {g['worst_gap_cal']} | {decisions[s]}")
    print(f"[gate] sigma {sig} | adopted {adopted} | rule {rule_commit}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "sigmas":
        print(sigmas(load(2024)))
    else:
        run()
