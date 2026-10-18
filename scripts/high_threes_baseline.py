"""High-threes bar (Part B step 2): the CURRENT Arc Lab model's walk-forward on 2024-25, scored on P(k+) for k = 4, 5, 6 only.
Top-decile capture = share of actual 5+ games whose predicted P(5+) sits in the top 10% of predictions; log loss and calibration
(fixed buckets, n >= 200) for 4+, 5+, 6+; 95% CIs by game-cluster bootstrap. Reads the walk-forward rows the closing-line backtest
produced (data/backtest_rows_<season>.csv.gz, scripts/backtest.py --season); no prices anywhere.
  python high_threes_baseline.py --season 2025 [--rows data/backtest_rows_2025.csv.gz] -> docs/data/high_threes_baseline.json
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, write_json

KS = (4, 5, 6)
BUCKETS = [(0, 0.02, "<2%"), (0.02, 0.05, "2-5%"), (0.05, 0.10, "5-10%"), (0.10, 0.20, "10-20%"), (0.20, 0.35, "20-35%"), (0.35, 1.01, ">=35%")]
MIN_BUCKET = 200
B_ITERS, SEED = 1000, 7


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def capture(p, y, top=0.10):
    """Share of actual positives whose prediction is in the top `top` share of all predictions (ties broken by rank)."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    if y.sum() == 0: return float("nan")
    cut = np.quantile(p, 1 - top)
    return float(y[p >= cut].sum() / y.sum())


def boot(fn, df, cols, iters=B_ITERS, seed=SEED):
    """Game-cluster bootstrap 95% CI of fn(sub) where sub resamples games with replacement."""
    rng = np.random.default_rng(seed); games = df.event_id.unique(); idx = df.groupby("event_id").indices
    vals = []
    for _ in range(iters):
        pick = rng.choice(games, len(games), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        vals.append(fn(df.iloc[rows]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def score(df, pcol_fmt="p_tpm_{k}", label="current model"):
    out = dict(label=label, n=int(len(df)), games=int(df.event_id.nunique()))
    for k in KS:
        p = df[pcol_fmt.format(k=k)].astype(float).values; y = (df.act_tpm >= k).astype(int).values
        sub = df.assign(_p=p, _y=y)
        res = dict(base_rate=round(float(y.mean()), 4), positives=int(y.sum()), logloss=round(logloss(p, y), 4),
                   logloss_ci=boot(lambda s: logloss(s._p, s._y), sub, None), mean_pred=round(float(p.mean()), 4))
        if k == 5:
            res["top_decile_capture"] = round(capture(p, y), 4); res["top_decile_capture_ci"] = boot(lambda s: capture(s._p, s._y), sub, None)
            res["top_decile_n"] = int((p >= np.quantile(p, 0.9)).sum())
        cal = []
        for lo, hi, lab in BUCKETS:
            m = (p >= lo) & (p < hi)
            if m.sum() == 0: cal.append(dict(bucket=lab, n=0)); continue
            pm, hm = float(p[m].mean()), float(y[m].mean()); se = float(np.sqrt(max(hm * (1 - hm), 1e-9) / m.sum()))
            cal.append(dict(bucket=lab, n=int(m.sum()), pred=round(pm, 4), actual=round(hm, 4), gap=round(hm - pm, 4), gap_ci=[round(hm - pm - 1.96 * se, 4), round(hm - pm + 1.96 * se, 4)],
                            within_2pp=bool(abs(hm - pm) <= 0.02) if m.sum() >= MIN_BUCKET else None))
        res["calibration"] = cal
        res["calibration_pass_2pp"] = all(c["within_2pp"] for c in cal if c.get("within_2pp") is not None)
        res["worst_gap_at_n200"] = max((abs(c["gap"]) for c in cal if c["n"] >= MIN_BUCKET), default=None)
        out[f"k{k}"] = res
    return out


def trailing10_poisson(df, rows_all=None):
    """Baseline 2 for the rule: Poisson with mean = the player's trailing-10 (prior games only) threes per game, computed from the
       walk-forward rows' own naive column when present (naive_tpm = trailing average)."""
    lam = df.naive_tpm.astype(float).clip(lower=1e-3).values
    from math import exp, factorial
    out = {}
    for k in KS:
        out[f"p_pois_{k}"] = 1 - np.array([sum(exp(-l) * l ** j / factorial(j) for j in range(k)) for l in lam])
    return pd.DataFrame(out, index=df.index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--rows")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = Path(a.rows) if a.rows else DATA_DIR / f"backtest_rows_{a.season}.csv.gz"
    df = pd.read_csv(rows, dtype={"player_id": str, "event_id": str})
    df = df[(df.act_min > 0) & df.p_tpm_5.notna()].copy()
    cur = score(df)
    pois = trailing10_poisson(df)
    naive = score(pd.concat([df, pois], axis=1), pcol_fmt="p_pois_{k}", label="trailing-10 Poisson")
    bar = max(0.35, round(cur["k5"]["top_decile_capture"] + 0.10, 4))
    rep = dict(title="High-threes bar: current Arc Lab model, 2024-25 walk-forward", season="2024-25" if a.season == 2025 else str(a.season), rows_file=str(rows.name),
               generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), ks=list(KS), buckets=[b[2] for b in BUCKETS], min_bucket=MIN_BUCKET,
               current=cur, trailing10_poisson=naive, capture_bar_for_rule=bar,
               notes=["Rows: every 2024-25 player-game with a walk-forward projection and minutes > 0 (scripts/backtest.py --season 2025, prior games only, no 2024-25 tuning).",
                      "Top-decile capture: actual 5+ games whose predicted P(5+) is in the top 10% of all predictions that season.",
                      "95% CIs: 1,000 game-cluster bootstrap resamples for log loss and capture; normal approximation per calibration bucket.",
                      "Trailing-10 Poisson: Poisson(mean = the player's trailing-10 threes per game from the rows' naive column).",
                      "No prices anywhere in this file."])
    write_json(Path(a.out) if a.out else SITE_DATA / "high_threes_baseline.json", rep)
    for lab, r in (("current", cur), ("trailing-10 Poisson", naive)):
        print(f"[bar] {lab}: n {r['n']} | " + " | ".join(f"{k}+: base {r[f'k{k}']['base_rate']:.3f} logloss {r[f'k{k}']['logloss']:.4f} {r[f'k{k}']['logloss_ci']} worst gap@200 {r[f'k{k}']['worst_gap_at_n200']}" for k in KS))
        print(f"      top-decile capture P(5+): {r['k5']['top_decile_capture']:.3f} {r['k5']['top_decile_capture_ci']} (top decile n {r['k5']['top_decile_n']}, positives {r['k5']['positives']})")
    print(f"[bar] capture bar for the rule = max(0.35, current + 0.10) = {bar}")
