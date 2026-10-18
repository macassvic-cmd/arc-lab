"""Tail program test 1: DraftKings ladder-shape diagnostic (backtests/ladder_shape_diagnostic_rule.md). No game outcomes are used.
Writes docs/data/ladder_shape_diagnostic.json only.
  python ladder_shape_diagnostic.py
"""
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_name, read_json, write_json
from backtest_threes_close import implied

SEED, WINDOW, MIN_GAMES, DISTANCES = 20261018, 15, 10, (1, 2, 3, 4)
SD_F, CORR_F, CORR_P = 0.10, 0.10, 0.20
BOOK = "DraftKings"


def load_box():
    frames = [pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})]
    h = DATA_DIR / "player_games_history.csv"
    if h.exists(): frames.append(pd.read_csv(h, dtype={"event_id": str, "player_id": str}))
    df = pd.concat(frames).drop_duplicates(["event_id", "player_id"])
    df["date"] = pd.to_datetime(df.date); df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0); df["tpm"] = pd.to_numeric(df.tpm, errors="coerce").fillna(0)
    return df.sort_values(["player_id", "date"]).reset_index(drop=True)


def league_prior(df):
    """Method of moments on raw 2023-24 box scores: prior mean VMR and prior strength K (see the rule)."""
    s = df[(df.season == 2024) & (df["min"] > 0)]
    g = s.groupby("player_id").tpm.agg(["mean", "var", "count"]); g = g[(g["count"] >= MIN_GAMES) & (g["mean"] > 0)]
    vmr = g["var"] / g["mean"]; n = g["count"]
    prior_mean = float((vmr * n).sum() / n.sum())
    samp = 2 * vmr ** 2 / (n - 1)
    tau2 = max(float(vmr.var() - samp.mean()), 0.01 * prior_mean ** 2)
    K = float((n * samp).mean() / tau2)
    return dict(prior_mean=round(prior_mean, 4), tau2=round(tau2, 4), K=round(K, 3), players=int(len(g)), season="2023-24 raw box scores (no 2022-23 box scores exist in this repository)")


def shape_scores(df, prior):
    """Shape components for every 2024-25 player-game from the trailing WINDOW prior games with minutes > 0 (>= MIN_GAMES)."""
    p = df[df["min"] > 0].copy()
    out = []
    for pid, g in p.groupby("player_id", sort=False):
        mins = g["min"].values; tpm = g.tpm.values; dates = g.date.values; eids = g.event_id.values; seasons = g.season.values
        for i in range(len(g)):
            if seasons[i] != 2025 or i < MIN_GAMES: continue
            w0 = max(0, i - WINDOW); m = mins[w0:i]; t = tpm[w0:i]; n = len(m)
            cv = float(m.std(ddof=1) / m.mean()) if m.mean() > 0 else np.nan
            vmr = float(t.var(ddof=1) / t.mean()) if t.mean() > 0 else prior["prior_mean"]
            vmr_s = (n * vmr + prior["K"] * prior["prior_mean"]) / (n + prior["K"])
            out.append((eids[i], pid, n, cv, vmr, vmr_s))
    s = pd.DataFrame(out, columns=["event_id", "player_id", "n_window", "min_cv", "vmr_raw", "vmr_shrunk"]).dropna()
    s["z_cv"] = (s.min_cv - s.min_cv.mean()) / s.min_cv.std(ddof=1); s["z_vmr"] = (s.vmr_shrunk - s.vmr_shrunk.mean()) / s.vmr_shrunk.std(ddof=1)
    s["sscore"] = (s.z_cv + s.z_vmr) / 2
    return s


def dk_rungs():
    """DraftKings main line + main price and Over rungs at main + 1..4 per player-game from the Part A closing snapshots."""
    close_dir, alt_dir = DATA_DIR / "odds_close_2025", DATA_DIR / "odds_alt_2025"
    rows = []
    for fn in sorted(alt_dir.glob("4*.json")):
        alt = read_json(fn, None); close = read_json(close_dir / fn.name, None)
        if not alt or not close or close.get("missing"): continue
        main = {}
        for b in close["data"].get("bookmakers", []):
            if b["title"] != BOOK: continue
            for m in b.get("markets", []):
                if m["key"] != "player_threes": continue
                sides = {}
                for o in m["outcomes"]: sides.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = o["price"]
                for (pl, ln), s in sides.items():
                    if ln is None or "Over" not in s or "Under" not in s: continue
                    po, pu = implied(s["Over"]), implied(s["Under"]); cand = (abs(po - pu), float(ln), po)
                    if pl not in main or cand < main[pl]: main[pl] = cand
        for b in alt["data"].get("bookmakers", []):
            if b["title"] != BOOK: continue
            for m in b.get("markets", []):
                if m["key"] != "player_threes_alternate": continue
                for o in m["outcomes"]:
                    if o["name"] != "Over" or o.get("point") is None or o.get("description") not in main: continue
                    _, ml, mp = main[o["description"]]; d = float(o["point"]) - ml
                    if d not in DISTANCES: continue
                    rows.append(dict(event_id=alt["event_id"], player=o["description"], name_key=norm_name(o["description"]), main_line=ml, main_p=mp, d=int(d), line=float(o["point"]), p=implied(o["price"])))
    r = pd.DataFrame(rows)
    r["main_logit"] = np.log(r.main_p / (1 - r.main_p)); r["y"] = np.log(r.p / (1 - r.p))
    return r


def design(r):
    a, b, c = r.main_line.values, r.main_logit.values, r.d.values.astype(float)
    return np.column_stack([np.ones(len(r)), a, b, c, a * b, a * c, b * c, a * b * c])


def corr_ci(x, y, groups, iters=1000, seed=SEED):
    rng = np.random.default_rng(seed); g = np.asarray(groups); ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    vals = []
    for _ in range(iters):
        pick = rng.choice(ug, len(ug), replace=True); rows = np.concatenate([idx[k] for k in pick])
        vals.append(np.corrcoef(x[rows], y[rows])[0, 1])
    return [round(float(v), 4) for v in np.nanpercentile(vals, [2.5, 97.5])]


if __name__ == "__main__":
    rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "backtests/ladder_shape_diagnostic_rule.md"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    box = load_box(); prior = league_prior(box); shape = shape_scores(box, prior)
    pg = box[box.season == 2025][["event_id", "player_id", "player"]].copy(); pg["name_key"] = pg.player.map(norm_name)
    shape = shape.merge(pg[["event_id", "player_id", "name_key"]], on=["event_id", "player_id"], how="left")
    r = dk_rungs().merge(shape[["event_id", "name_key", "sscore", "min_cv", "vmr_shrunk", "n_window"]], on=["event_id", "name_key"], how="inner")
    games = np.sort(r.event_id.unique()); rng = np.random.default_rng(SEED); fit_games = set(rng.choice(games, len(games) // 2, replace=False))
    fit, hold = r[r.event_id.isin(fit_games)], r[~r.event_id.isin(fit_games)]
    X = design(fit); beta, *_ = np.linalg.lstsq(X, fit.y.values, rcond=None)
    r2 = 1 - ((fit.y.values - X @ beta) ** 2).sum() / ((fit.y.values - fit.y.mean()) ** 2).sum()
    res = hold.y.values - design(hold) @ beta
    hold = hold.assign(resid=res)
    sd = float(res.std(ddof=1)); corr = float(np.corrcoef(res, hold.sscore.values)[0, 1]); corr_ci_ = corr_ci(res, hold.sscore.values, hold.event_id.values)
    spear = float(pd.Series(res).rank().corr(pd.Series(hold.sscore.values).rank()))
    pgm = hold.groupby(["event_id", "name_key"]).agg(resid=("resid", "mean"), sscore=("sscore", "first")).reset_index()
    corr_pg = float(np.corrcoef(pgm.resid, pgm.sscore)[0, 1]); corr_pg_ci = corr_ci(pgm.resid.values, pgm.sscore.values, pgm.event_id.values)
    by_d = [dict(d=int(d), n=int(len(g)), resid_sd=round(float(g.resid.std(ddof=1)), 4), corr=round(float(np.corrcoef(g.resid, g.sscore)[0, 1]), 4)) for d, g in hold.groupby("d")]
    comp = dict(corr_min_cv=round(float(np.corrcoef(res, hold.min_cv.values)[0, 1]), 4), corr_vmr=round(float(np.corrcoef(res, hold.vmr_shrunk.values)[0, 1]), 4))
    if sd < SD_F and abs(corr) < CORR_F: outcome, text = "F", "formulaic ladder: shape is unpriced at DraftKings; test 3 proceeds"
    elif corr >= CORR_P: outcome, text = "P", "DraftKings prices shape; shape angles live only via Underdog"
    else: outcome, text = "in-between", "report; run test 3 at low prior"
    rep = dict(title="DK ladder-shape diagnostic (test 1, no outcomes)", rule_file="backtests/ladder_shape_diagnostic_rule.md", rule_commit=rule_commit, generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               outcome=outcome, outcome_text=text, thresholds=dict(resid_sd_F=SD_F, corr_F=CORR_F, corr_P=CORR_P),
               shape=dict(prior=prior, window=WINDOW, min_games=MIN_GAMES, scored_player_games=int(len(shape)), min_cv_mean=round(float(shape.min_cv.mean()), 4), min_cv_sd=round(float(shape.min_cv.std()), 4),
                          vmr_shrunk_mean=round(float(shape.vmr_shrunk.mean()), 4), vmr_shrunk_sd=round(float(shape.vmr_shrunk.std()), 4), corr_components=round(float(shape.min_cv.corr(shape.vmr_shrunk)), 4)),
               ladder=dict(rungs=int(len(r)), player_games=int(r.groupby(["event_id", "name_key"]).ngroups), games=int(len(games)), fit_games=int(len(fit_games)), fit_rungs=int(len(fit)), holdout_rungs=int(len(hold)),
                           coefficients=dict(zip(["intercept", "main_line", "main_logit", "d", "main_line:main_logit", "main_line:d", "main_logit:d", "main_line:main_logit:d"], [round(float(b), 4) for b in beta])),
                           fit_r2=round(float(r2), 4), holdout_resid_sd=round(sd, 4), corr_rung=round(corr, 4), corr_rung_ci=corr_ci_, spearman_rung=round(spear, 4),
                           corr_player_game=round(corr_pg, 4), corr_player_game_ci=corr_pg_ci, by_distance=by_d, corr_components=comp,
                           mean_tail_p_by_d={int(d): round(float(g.p.mean()), 4) for d, g in r.groupby("d")}),
               notes=["No game outcomes anywhere in this diagnostic.", "Shape score from raw box scores only; the league VMR prior by method of moments on 2023-24 (no 2022-23 box scores in the repository).",
                      "Implied tail P is DraftKings' raw Over price (margin included); the regression models the ladder shape DK publishes, not fair odds.",
                      "Games split in half at random (seed 20261018); residuals and correlations are read on the holdout half only."])
    write_json(SITE_DATA / "ladder_shape_diagnostic.json", rep)
    print(f"[shape] prior {prior} | scored player-games {len(shape)} | rungs {len(r)} in {len(games)} games | fit R2 {r2:.3f} | holdout resid SD {sd:.3f} | corr rung {corr:.3f} {corr_ci_} | corr player-game {corr_pg:.3f} {corr_pg_ci} | spearman {spear:.3f} | by d {by_d} | components {comp} | OUTCOME {outcome}: {text}")
