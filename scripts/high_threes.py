"""Dedicated high-threes model (backtests/high_threes_rule.md). P(k+) for k = 4..7 from the full 0-12+ distribution of threes made.
No prices anywhere. Writes only data/high_threes_rows_<season>.csv.gz, docs/data/high_threes_backtest.json and the Backtest page;
never data/paper_log.csv or docs/data/results.json.

  python high_threes.py fit                 # parameters on 2023-24 ONLY -> data/high_threes_params.json
  python high_threes.py run --season 2025   # walk-forward (prior games only) with the frozen parameters -> rows
  python high_threes.py run --season 2026   # consistency check (contaminated season, never a gate)
  python high_threes.py grade               # the committed gate on 2024-25, 2025-26 as a check, baselines side by side -> report JSON
  python high_threes.py all

Model (fixed in the rule before it was built):
  3PA mean  = projected minutes (the current model's walk-forward proj_min, an input) x 3PA/min, where 3PA/min blends last-5,
              last-15 and season-to-date rates (prior games only) and is shrunk (empirical Bayes) toward the position average;
              x multiplicative Poisson-regression terms: pace (team + opponent possessions/48), home, back-to-back, rest days (cap 3),
              usual starters out (starters that night only).
  3PA       ~ negative binomial, dispersion per player group (position x 3PA-rate tercile).
  makes|3PA ~ beta-binomial, 3P% shrunk toward the position rate, one concentration.
  Spread / blowout: no historical spreads in this repository -> absent, reported.
"""
import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json
from high_threes_baseline import KS, BUCKETS, MIN_BUCKET, logloss, capture, boot

FIT_SEASON, GATE_SEASON, CHECK_SEASON = 2024, 2025, 2026
PARAMS = DATA_DIR / "high_threes_params.json"
REPORT = SITE_DATA / "high_threes_backtest.json"
KMAX, AMAX = 12, 40                  # distribution cells 0..12 (13+ folded), attempts 0..40
CAPTURE_BAR = 0.5006                 # from the rule: current model 0.401 + 0.10
POS5 = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C", "G": "G", "F": "F"}
lgamma = np.vectorize(math.lgamma)


# ---------------- data ----------------
def load_games():
    frames = []
    for fn in (DATA_DIR / "player_games.csv", DATA_DIR / "player_games_history.csv"):
        if fn.exists():
            frames.append(pd.read_csv(fn, dtype={"event_id": str, "player_id": str}))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["event_id", "player_id"])
    df["date"] = pd.to_datetime(df.date); df["pos5"] = df.pos.map(POS5).fillna("F")
    for c in ("min", "tpm", "tpa", "fga", "fta", "oreb", "tov"): df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["starter"] = pd.to_numeric(df.starter, errors="coerce").fillna(0).astype(int)
    return df.sort_values(["date", "event_id", "team", "player_id"]).reset_index(drop=True)


def team_games(df):
    """One row per (event, team): date, possessions, pace per 48, rest days, usual-starter absences come later (walk-forward)."""
    g = df.groupby(["event_id", "team", "opp", "date", "season"], as_index=False).agg(fga=("fga", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"), fta=("fta", "sum"), tmin=("min", "sum"))
    g["poss"] = g.fga - g.oreb + g.tov + 0.44 * g.fta
    g["pace48"] = g.poss * 48 / (g.tmin / 5).clip(lower=40)
    g = g.sort_values(["team", "date"])
    g["prev_date"] = g.groupby("team").date.shift(1)
    g["rest"] = (g.date - g.prev_date).dt.days.fillna(4).clip(upper=4)
    return g


def rolling_prior_mean(s, window):
    return s.shift(1).rolling(window, min_periods=1).mean()


def features(df, tg):
    """Walk-forward per player-game features from prior games only. Returns a frame aligned to df."""
    tg = tg.sort_values(["team", "date"]).copy()
    tg["team_pace"] = tg.groupby("team").pace48.transform(lambda s: s.shift(1).ewm(halflife=10, min_periods=3).mean())
    league = tg.sort_values("date").groupby("date").pace48.mean().rolling(30, min_periods=10).mean().shift(1)
    tg["league_pace"] = tg.date.map(league)
    opp = tg[["event_id", "team", "team_pace"]].rename(columns={"team": "opp", "team_pace": "opp_pace"})
    tg = tg.merge(opp, on=["event_id", "opp"], how="left")
    tg["pace_factor"] = ((tg.team_pace.fillna(tg.league_pace) + tg.opp_pace.fillna(tg.league_pace)) / 2 / tg.league_pace).clip(0.85, 1.15).fillna(1.0)
    # usual starters: started >= 50% of the team's last 10 games (prior); absent = usual starter not in tonight's box score
    st = df[["event_id", "team", "date", "player_id", "starter"]].sort_values(["team", "player_id", "date"])
    st["start_share"] = st.groupby(["team", "player_id"]).starter.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    st["n_prev"] = st.groupby(["team", "player_id"]).cumcount()
    usual = st[st.start_share >= 0.5][["team", "player_id", "date", "event_id"]]
    # for each team-game, usual starters = players whose share >= 0.5 as of their most recent appearance before this date
    last_share = st[["team", "player_id", "date", "start_share"]].dropna(subset=["start_share"])
    tg_out = []
    for team, grp in tg.groupby("team"):
        hist = last_share[last_share.team == team].sort_values("date")
        present = df[df.team == team].groupby("event_id").player_id.apply(set)
        for r in grp.itertuples():
            prior = hist[hist.date < r.date]
            if prior.empty: tg_out.append((r.event_id, team, 0)); continue
            latest = prior.groupby("player_id").start_share.last()
            # a usual starter who has not appeared for 21+ days is treated as gone (traded / long absence), like the live roster window
            last_seen = prior.groupby("player_id").date.max()
            usual_ids = [p for p, sh in latest.items() if sh >= 0.5 and (r.date - last_seen[p]).days <= 21]
            tg_out.append((r.event_id, team, sum(1 for p in usual_ids if p not in present.get(r.event_id, set()))))
    so = pd.DataFrame(tg_out, columns=["event_id", "team", "starters_out"])
    tg = tg.merge(so, on=["event_id", "team"], how="left")
    tg["starters_out"] = tg.starters_out.fillna(0).clip(upper=5)
    f = df.merge(tg[["event_id", "team", "pace_factor", "rest", "starters_out"]], on=["event_id", "team"], how="left")
    f["b2b"] = (f.rest <= 1).astype(int); f["rest_c"] = f.rest.clip(upper=3)
    # player rates from prior games: last 5, last 15, season-to-date (prior season as the start when < 5 current games)
    f = f.sort_values(["player_id", "date"]).reset_index(drop=True)
    gp = f.groupby("player_id")
    for w, name in ((5, "r5"), (15, "r15")):
        f[name + "_tpa"] = gp.tpa.transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
        f[name + "_min"] = gp["min"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
    gs = f.groupby(["player_id", "season"])
    f["s_tpa"] = gs.tpa.transform(lambda s: s.shift(1).cumsum()).fillna(0); f["s_min"] = gs["min"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    f["s_n"] = gs.cumcount()
    f["c_tpa"] = gp.tpa.transform(lambda s: s.shift(1).cumsum()).fillna(0); f["c_min"] = gp["min"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    f["c_tpm"] = gp.tpm.transform(lambda s: s.shift(1).cumsum()).fillna(0)
    f["n_prior"] = gp.cumcount()
    # season-to-date rate falls back to the career-to-date rate (previous seasons) under 5 current-season games
    use_s = f.s_n >= 5
    f["rs_tpa"] = np.where(use_s, f.s_tpa, f.c_tpa); f["rs_min"] = np.where(use_s, f.s_min, f.c_min)
    # position averages, prior games only (expanding by date within the league): 3PA per minute and 3P% by pos5
    day = f.groupby(["date", "pos5"], as_index=False).agg(tpa=("tpa", "sum"), mn=("min", "sum"), tpm=("tpm", "sum")).sort_values("date")
    for c in ("tpa", "mn", "tpm"): day["c_" + c] = day.groupby("pos5")[c].transform(lambda s: s.shift(1).rolling(60, min_periods=5).sum())
    day["pos_rate"] = (day.c_tpa / day.c_mn).fillna(0.1); day["pos_pct"] = (day.c_tpm / day.c_tpa).fillna(0.35)
    f = f.merge(day[["date", "pos5", "pos_rate", "pos_pct"]], on=["date", "pos5"], how="left")
    f["pos_rate"] = f.pos_rate.fillna(0.1); f["pos_pct"] = f.pos_pct.fillna(0.35)
    return f


def attach_proj_min(f, season):
    """The current model's walk-forward projected minutes for that season (an input): data/backtest_rows_<season>.csv.gz."""
    fn = DATA_DIR / f"backtest_rows_{season}.csv.gz"
    if not fn.exists():
        raise SystemExit(f"{fn.name} missing: run scripts/backtest.py --season {season} first (projected minutes input)")
    r = pd.read_csv(fn, dtype={"player_id": str, "event_id": str}, usecols=["event_id", "player_id", "proj_min"] + [c for c in pd.read_csv(fn, nrows=0).columns if c.startswith("p_tpm_")] + ["naive_tpm"])
    return f.merge(r, on=["event_id", "player_id"], how="inner")


# ---------------- model pieces ----------------
def blend_rate(f, w5, w15, K):
    r5 = np.where(f.r5_min > 0, f.r5_tpa / f.r5_min.clip(lower=1), f.pos_rate)
    r15 = np.where(f.r15_min > 0, f.r15_tpa / f.r15_min.clip(lower=1), f.pos_rate)
    rs = np.where(f.rs_min > 0, f.rs_tpa / f.rs_min.clip(lower=1), f.pos_rate)
    ws = max(0.0, 1 - w5 - w15)
    r = w5 * r5 + w15 * r15 + ws * rs
    n_eff = np.minimum(f.n_prior.values, 15)
    return (n_eff * r + K * f.pos_rate.values) / (n_eff + K)


def design(f):
    return np.column_stack([np.ones(len(f)), np.log(f.pace_factor.values), f.home.values.astype(float), f.b2b.values.astype(float),
                            f.rest_c.values.astype(float), (f.starters_out.fillna(0).values * f.starter.values).astype(float)])


COEF_NAMES = ["intercept", "log_pace", "home", "back_to_back", "rest_days", "starters_out_x_starter"]


def poisson_irls(y, X, offset, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(offset + X @ b); W = mu
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]); g = X.T @ (y - mu)
        step = np.linalg.solve(H, g); b = b + step
        if np.abs(step).max() < 1e-8: break
    return b


def nb_loglik(y, mu, r):
    return (lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu) + 1e-12)).sum()


def fit_nb_size(y, mu):
    """Golden-section search on log r for the NB size (dispersion) parameter."""
    lo, hi = math.log(0.3), math.log(300.0); gr = (math.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = nb_loglik(y, mu, math.exp(c)), nb_loglik(y, mu, math.exp(d))
    for _ in range(40):
        if fc > fd: hi, d, fd = d, c, fc; c = hi - gr * (hi - lo); fc = nb_loglik(y, mu, math.exp(c))
        else: lo, c, fc = c, d, fd; d = lo + gr * (hi - lo); fd = nb_loglik(y, mu, math.exp(d))
    return math.exp((lo + hi) / 2)


def bb_loglik(k, n, a, b):
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + lgamma(k + a) + lgamma(n - k + b) - lgamma(n + a + b) + lgamma(a + b) - lgamma(a) - lgamma(b)).sum()


def player_pct(f, m):
    return (f.c_tpm.values + m * f.pos_pct.values) / (f.c_tpa.values + m)


def groups_of(f, rate):
    terc = pd.qcut(pd.Series(rate), 3, labels=["lo", "mid", "hi"], duplicates="drop").astype(str).values
    return np.char.add(np.char.add(f.pos5.values.astype(str), "_"), terc)


# ---------------- fit on 2023-24 ----------------
def fit(df, tg):
    f = features(df, tg); f = f[(f.season == FIT_SEASON) & (f["min"] > 0)].copy()
    f = attach_proj_min(f, FIT_SEASON); f = f[f.proj_min > 0].copy()
    print(f"[fit] {len(f)} player-games in {FIT_SEASON} with minutes and a projected-minutes input")
    y = f.tpa.values.astype(float)
    best = None
    for w5 in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for w15 in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            if w5 + w15 > 1: continue
            for K in (2, 4, 6, 8, 12, 16, 24):
                mu = f.proj_min.values * blend_rate(f, w5, w15, K)
                dev = (mu - y * np.log(mu + 1e-9)).sum()            # Poisson deviance up to a constant
                if best is None or dev < best[0]: best = (dev, w5, w15, K)
    _, w5, w15, K = best
    rate = blend_rate(f, w5, w15, K)
    X = design(f); offset = np.log(f.proj_min.values * rate + 1e-9)
    beta = poisson_irls(y, X, offset)
    mu = np.exp(offset + X @ beta)
    grp = groups_of(f, rate); sizes = {}
    for gname in np.unique(grp):
        m = grp == gname; sizes[gname] = round(fit_nb_size(y[m], mu[m]), 4)
    # tercile cut points of the blended rate, frozen for later seasons
    cuts = [float(np.quantile(rate, q)) for q in (1 / 3, 2 / 3)]
    # beta-binomial: shrinkage m and concentration kappa on makes | attempts
    att = f.tpa.values.astype(float); mk = f.tpm.values.astype(float); ok = att > 0
    bestb = None
    for m_s in (10, 20, 40, 80, 160, 320, 640, 1280):
        p = np.clip(player_pct(f, m_s), 0.02, 0.8)
        for kappa in (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120):
            ll = bb_loglik(mk[ok], att[ok], kappa * p[ok], kappa * (1 - p[ok]))
            if bestb is None or ll > bestb[0]: bestb = (ll, m_s, kappa)
    _, m_s, kappa = bestb
    params = dict(fit_season="2023-24", fit_rows=int(len(f)), blend=dict(w5=w5, w15=w15, w_season=round(1 - w5 - w15, 2), K=K), poisson=dict(zip(COEF_NAMES, [round(float(b), 4) for b in beta])),
                  nb_size_by_group=sizes, rate_tercile_cuts=cuts, beta_binomial=dict(m=m_s, kappa=kappa), generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  notes=["Every parameter here was fit on 2023-24 only (grid search on Poisson deviance for the blend, IRLS Poisson regression for the multipliers, golden-section MLE per group for the NB size, grid MLE for the beta-binomial)."])
    write_json(PARAMS, params)
    print(f"[fit] blend w5 {w5} w15 {w15} K {K} | poisson {params['poisson']} | nb sizes {sizes} | bb m {m_s} kappa {kappa}")
    return params


# ---------------- distribution ----------------
def distribution(mu, size, p, kappa):
    """P(makes = k), k = 0..KMAX (13+ folded) for arrays mu (3PA mean), size (NB), p (3P%), kappa."""
    n = len(mu); a = np.arange(AMAX + 1)
    # NB pmf over attempts
    lg = lgamma
    la = (lg(a[None, :] + size[:, None]) - lg(size)[:, None] - lg(a + 1)[None, :] + size[:, None] * np.log(size / (size + mu))[:, None] + a[None, :] * np.log(mu / (size + mu) + 1e-12)[:, None])
    pa = np.exp(la); pa /= pa.sum(axis=1, keepdims=True)
    # beta-binomial pmf of k makes given a attempts, per row
    out = np.zeros((n, KMAX + 1))
    al, be = kappa * p, kappa * (1 - p)
    for ai in range(AMAX + 1):
        k = np.arange(min(ai, KMAX + 30) + 1)
        lk = (lg(ai + 1) - lg(k + 1)[None, :] - lg(ai - k + 1)[None, :] + lg(k[None, :] + al[:, None]) + lg(ai - k[None, :] + be[:, None]) - lg(ai + al + be)[:, None] + lg(al + be)[:, None] - lg(al)[:, None] - lg(be)[:, None])
        pk = np.exp(lk)
        fold = np.zeros((n, KMAX + 1)); w = min(pk.shape[1], KMAX + 1)
        fold[:, :w] = pk[:, :w]
        if pk.shape[1] > KMAX + 1: fold[:, KMAX] += pk[:, KMAX + 1:].sum(axis=1)
        out += pa[:, ai][:, None] * fold
    return out / out.sum(axis=1, keepdims=True)


def run(df, tg, season, params):
    f = features(df, tg); f = f[(f.season == season) & (f["min"] > 0)].copy()
    f = attach_proj_min(f, season); f = f[f.proj_min > 0].copy()
    b = params["blend"]; rate = blend_rate(f, b["w5"], b["w15"], b["K"])
    beta = np.array([params["poisson"][c] for c in COEF_NAMES])
    mu = np.exp(np.log(f.proj_min.values * rate + 1e-9) + design(f) @ beta)
    cuts = params["rate_tercile_cuts"]; terc = np.where(rate < cuts[0], "lo", np.where(rate < cuts[1], "mid", "hi"))
    grp = np.char.add(np.char.add(f.pos5.values.astype(str), "_"), terc)
    size = np.array([params["nb_size_by_group"].get(g, np.median(list(params["nb_size_by_group"].values()))) for g in grp])
    p = np.clip(player_pct(f, params["beta_binomial"]["m"]), 0.02, 0.8)
    dist = distribution(mu, size, p, float(params["beta_binomial"]["kappa"]))
    out = f[["event_id", "date", "season", "team", "opp", "home", "player_id", "player", "pos", "starter", "min", "tpm", "tpa", "proj_min", "naive_tpm", "pace_factor", "rest", "starters_out"] + [c for c in f.columns if c.startswith("p_tpm_")]].copy()
    out["mu_tpa"] = mu; out["p3"] = p; out["nb_size"] = size
    for k in range(KMAX + 1): out[f"d{k}"] = dist[:, k]
    for k in (1, 2, 3, 4, 5, 6, 7, 8): out[f"h_tpm_{k}"] = 1 - dist[:, :k].sum(axis=1)
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    fn = DATA_DIR / f"high_threes_rows_{season}.csv.gz"; out.to_csv(fn, index=False, compression="gzip")
    print(f"[run] {season}: {len(out)} player-games -> {fn.name} | mean 3PA {mu.mean():.2f} vs actual {f.tpa.mean():.2f} | P(5+) mean {out.h_tpm_5.mean():.4f} vs hit {(f.tpm >= 5).mean():.4f}")
    return out


# ---------------- grade ----------------
def pois_cols(df):
    lam = df.naive_tpm.astype(float).clip(lower=1e-3).values
    out = {}
    for k in (4, 5, 6, 7):
        out[f"q_tpm_{k}"] = 1 - np.array([sum(math.exp(-l) * l ** j / math.factorial(j) for j in range(k)) for l in lam])
    return df.assign(**out)


def calib(p, y):
    cal = []
    for lo, hi, lab in BUCKETS:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0: cal.append(dict(bucket=lab, n=0)); continue
        pm, hm = float(p[m].mean()), float(y[m].mean()); se = float(np.sqrt(max(hm * (1 - hm), 1e-9) / m.sum()))
        cal.append(dict(bucket=lab, n=int(m.sum()), pred=round(pm, 4), actual=round(hm, 4), gap=round(hm - pm, 4), gap_ci=[round(hm - pm - 1.96 * se, 4), round(hm - pm + 1.96 * se, 4)],
                        within_2pp=bool(abs(hm - pm) <= 0.02) if m.sum() >= MIN_BUCKET else None))
    return cal


def season_block(rows, label):
    d = pois_cols(rows); d["act_tpm"] = d.tpm; d = d[d.p_tpm_5.notna() & d.h_tpm_5.notna()].copy()
    models = dict(new="h_tpm_{k}", current="p_tpm_{k}", poisson="q_tpm_{k}")
    blk = dict(label=label, n=int(len(d)), games=int(d.event_id.nunique()), models={})
    for name, fmt in models.items():
        mres = {}
        for k in (4, 5, 6, 7):
            p = d[fmt.format(k=k)].astype(float).values; y = (d.act_tpm >= k).astype(int).values; sub = d.assign(_p=p, _y=y)
            r = dict(base_rate=round(float(y.mean()), 4), positives=int(y.sum()), logloss=round(logloss(p, y), 4), logloss_ci=boot(lambda s: logloss(s._p, s._y), sub, None, iters=400),
                     mean_pred=round(float(p.mean()), 4), calibration=calib(p, y))
            r["calibration_pass_2pp"] = all(c["within_2pp"] for c in r["calibration"] if c.get("within_2pp") is not None)
            r["worst_gap_at_n200"] = max((abs(c["gap"]) for c in r["calibration"] if c["n"] >= MIN_BUCKET), default=None)
            if k == 5:
                r["top_decile_capture"] = round(capture(p, y), 4); r["top_decile_capture_ci"] = boot(lambda s: capture(s._p, s._y), sub, None, iters=400)
            mres[f"k{k}"] = r
        blk["models"][name] = mres
    # log-loss differences new - baseline with bootstrap CIs (negative = new is better)
    diffs = {}
    for k in (4, 5):
        y = (d.act_tpm >= k).astype(int).values; pn = d[f"h_tpm_{k}"].values
        for base, fmt in (("current", "p_tpm_{k}"), ("poisson", "q_tpm_{k}")):
            pb = d[fmt.format(k=k)].astype(float).values; sub = d.assign(_pn=pn, _pb=pb, _y=y)
            diffs[f"k{k}_vs_{base}"] = dict(diff=round(logloss(pn, y) - logloss(pb, y), 4), ci=boot(lambda s: logloss(s._pn, s._y) - logloss(s._pb, s._y), sub, None, iters=400))
    blk["logloss_diffs"] = diffs
    return blk


def grade(params, rule_commit):
    seasons = {}
    for season, label in ((GATE_SEASON, "2024-25 (gate)"), (CHECK_SEASON, "2025-26 (consistency check, contaminated)"), (FIT_SEASON, "2023-24 (in-sample fit)")):
        fn = DATA_DIR / f"high_threes_rows_{season}.csv.gz"
        if fn.exists(): seasons[str(season)] = season_block(pd.read_csv(fn, dtype={"event_id": str, "player_id": str}), label)
    g = seasons[str(GATE_SEASON)]; new = g["models"]["new"]; cur = g["models"]["current"]; poi = g["models"]["poisson"]
    a = all(new[f"k{k}"]["logloss"] < cur[f"k{k}"]["logloss"] and new[f"k{k}"]["logloss"] < poi[f"k{k}"]["logloss"] for k in (4, 5))
    b = all(new[f"k{k}"]["calibration_pass_2pp"] for k in (4, 5, 6))
    c = new["k5"]["top_decile_capture"] >= CAPTURE_BAR
    rep = dict(title="Dedicated high-threes model vs the committed rule", rule_file="backtests/high_threes_rule.md", rule_commit=rule_commit,
               generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), params=params, capture_bar=CAPTURE_BAR,
               gate=dict(season="2024-25", a_logloss=dict(pass_=bool(a), detail={k: dict(new=new[f"k{k}"]["logloss"], current=cur[f"k{k}"]["logloss"], poisson=poi[f"k{k}"]["logloss"]) for k in ("4", "5")}, diffs=g["logloss_diffs"]),
                         b_calibration=dict(pass_=bool(b), worst={k: new[f"k{k}"]["worst_gap_at_n200"] for k in ("4", "5", "6")}),
                         c_capture=dict(pass_=bool(c), value=new["k5"]["top_decile_capture"], ci=new["k5"]["top_decile_capture_ci"], bar=CAPTURE_BAR, current=cur["k5"]["top_decile_capture"]),
                         overall_pass=bool(a and b and c)),
               seasons=seasons,
               notes=["Parameters fit on 2023-24 only; 2024-25 is the only gate; 2025-26 is a consistency check (the current model's structure was tuned on it).",
                      "Projected minutes are the current model's walk-forward projection (an input). Spread / blowout input absent: no historical spreads in this repository.",
                      "Starters out = usual starters (started >= 50% of the team's last 10 games, seen within 21 days) missing from the box score; applied to that night's starters only.",
                      "Scored rows: the intersection where the new model, the current model and the trailing-10 Poisson all have a probability and the player logged minutes.",
                      "95% CIs: game-cluster bootstrap (400 resamples here; 1,000 in the step-2 bar file) for log loss, differences and capture; normal approximation per bucket.",
                      "No prices anywhere; Backtest page only."])
    write_json(REPORT, rep)
    print(f"[grade] 2024-25: (a) log loss {a} {rep['gate']['a_logloss']['detail']} | (b) calibration {b} worst {rep['gate']['b_calibration']['worst']} | (c) capture {new['k5']['top_decile_capture']} vs bar {CAPTURE_BAR}: {c} | OVERALL {'PASS' if a and b and c else 'FAIL'}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "run", "grade", "all"]); ap.add_argument("--season", type=int)
    a = ap.parse_args()
    df = load_games(); tg = team_games(df)
    rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "backtests/high_threes_rule.md"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    params = None
    if a.cmd in ("fit", "all"): params = fit(df, tg)
    if a.cmd in ("run", "all"):
        params = params or read_json(PARAMS, None)
        for s in ([a.season] if a.season else [GATE_SEASON, CHECK_SEASON, FIT_SEASON]): run(df, tg, s, params)
    if a.cmd in ("grade", "all"): grade(params or read_json(PARAMS, None), rule_commit)
