"""Tail-program headroom audit (backtests/headroom_audit_rule.md). Report-only, no tuning. Writes docs/data/headroom_audit.json and
nothing else (never paper_log.csv or results.json).
  python headroom_audit.py            # part 1: oracle decomposition on 2023-24 rows; part 2: DK vs current model on 2024-25; decision table
"""
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_name, write_json
from high_threes_baseline import logloss, capture, boot
from backtest_threes_alt_calibration import build_rungs

K = 5
AMAX = 40
lgamma = np.vectorize(math.lgamma, otypes=[float])


def binom_tail(n, p, k=K):
    """P(X >= k), X ~ Binomial(n, p), vectorised over rows."""
    n = np.asarray(n, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = np.zeros(len(n))
    for j in range(k):
        m = n >= j
        lp = lgamma(n[m] + 1) - lgamma(j + 1) - lgamma(n[m] - j + 1) + j * np.log(p[m]) + (n[m] - j) * np.log(1 - p[m])
        out[m] += np.exp(lp)
    return np.clip(1 - out, 0, 1)


def poisson_binom_tail(lam, p, k=K):
    """P(makes >= k) with attempts ~ Poisson(lam), makes | attempts ~ Binomial(attempts, p): sum over attempts 0..AMAX."""
    lam = np.asarray(lam, float); p = np.asarray(p, float); out = np.zeros(len(lam))
    for a in range(AMAX + 1):
        pa = np.exp(-lam + a * np.log(lam + 1e-12) - math.lgamma(a + 1))
        out += pa * binom_tail(np.full(len(lam), a), p, k)
    return out


def level(df, col, label):
    p = df[col].values; y = (df.act_tpm >= K).astype(int).values; sub = df.assign(_p=p, _y=y)
    return dict(label=label, capture=round(capture(p, y), 4), capture_ci=boot(lambda s: capture(s._p, s._y), sub, None), logloss=round(logloss(p, y), 4), mean_p=round(float(p.mean()), 4))


def diff(df, c1, c2):
    y = (df.act_tpm >= K).astype(int).values; sub = df.assign(_a=df[c1].values, _b=df[c2].values, _y=y)
    d = capture(df[c2].values, y) - capture(df[c1].values, y)
    return dict(diff=round(d, 4), ci=boot(lambda s: capture(s._b, s._y) - capture(s._a, s._y), sub, None))


def part1():
    rows = pd.read_csv(DATA_DIR / "backtest_rows_2024.csv.gz", dtype={"player_id": str, "event_id": str})
    if "make_p" not in rows or "proj_tpa" not in rows:
        raise SystemExit("backtest_rows_2024.csv.gz lacks proj_tpa / make_p: re-run scripts/backtest.py --season 2024 (instrumented)")
    d = rows[(rows.act_min > 0) & rows.p_tpm_5.notna() & rows.make_p.notna()].copy()
    pg = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"player_id": str, "event_id": str})
    hist = DATA_DIR / "player_games_history.csv"
    if hist.exists(): pg = pd.concat([pg, pd.read_csv(hist, dtype={"player_id": str, "event_id": str})]).drop_duplicates(["event_id", "player_id"])
    s24 = pg[pg.season == 2024]
    d = d.merge(s24[["event_id", "player_id", "tpa"]].rename(columns={"tpa": "act_tpa"}), on=["event_id", "player_id"], how="left")
    season_pct = s24.groupby("player_id").agg(m=("tpm", "sum"), a=("tpa", "sum"))
    lg = float(s24.tpm.sum() / s24.tpa.sum())
    season_pct["pct"] = np.where(season_pct.a >= 20, season_pct.m / season_pct.a.clip(lower=1), lg)
    d["season_pct"] = d.player_id.map(season_pct.pct).fillna(lg)
    d = d[d.act_tpa.notna()].copy()
    rate = d.proj_tpa.values / d.proj_min.clip(lower=1).values
    d["p_a"] = d.p_tpm_5.values
    d["p_b"] = poisson_binom_tail(d.act_min.values * rate, d.make_p.values)
    d["p_c"] = binom_tail(d.act_tpa.values, d.make_p.values)
    d["p_d"] = binom_tail(d.act_tpa.values, d.season_pct.values)
    levels = dict(a=level(d, "p_a", "current model as-is"), b=level(d, "p_b", "oracle minutes"), c=level(d, "p_c", "oracle minutes + 3PA"), d=level(d, "p_d", "oracle 3PA + season 3P% talent"))
    steps = dict(minutes_error=diff(d, "p_a", "p_b"), attempt_rate_error=diff(d, "p_b", "p_c"), pct_estimation_error=diff(d, "p_c", "p_d"), total_a_to_c=diff(d, "p_a", "p_c"), total_a_to_d=diff(d, "p_a", "p_d"))
    return dict(season="2023-24", n=int(len(d)), games=int(d.event_id.nunique()), positives=int((d.act_tpm >= K).sum()), base_rate=round(float((d.act_tpm >= K).mean()), 4),
                top_decile_n=int(math.ceil(0.1 * len(d))), league_pct_2023_24=round(lg, 4), levels=levels, steps=steps,
                irreducible_note=f"capture at (d) = {levels['d']['capture']:.3f}: with true attempts and true season-long talent, {1 - levels['d']['capture']:.0%} of actual 5+ games still fall outside the top decile (shooting luck).")


def part2():
    alt_dir, close_dir = DATA_DIR / "odds_alt_2025", DATA_DIR / "odds_close_2025"
    df, long, _ = build_rungs(2025, alt_dir, close_dir)
    dk = long[(long.book == "DraftKings") & (long.line == 4.5)][["event_id", "player", "fair_book"]].rename(columns={"fair_book": "dk_p5"})
    pooled = df[df.line == 4.5][["event_id", "player", "fair", "n_books"]].rename(columns={"fair": "pooled_p5"})
    dk = dk.merge(pooled, on=["event_id", "player"], how="left"); dk["name_key"] = dk.player.map(norm_name)
    rows = pd.read_csv(DATA_DIR / "backtest_rows_2025.csv.gz", dtype={"player_id": str, "event_id": str})
    rows = rows[(rows.act_min > 0) & rows.p_tpm_5.notna()].copy(); rows["name_key"] = rows.player.map(norm_name)
    d = rows.merge(dk, on=["event_id", "name_key"], how="inner")
    d["p_dk"] = d.dk_p5.values; d["p_cur"] = d.p_tpm_5.values; d["p_pool"] = d.pooled_p5.fillna(d.dk_p5).values
    y = (d.act_tpm >= K).astype(int).values; sub = d.assign(_dk=d.p_dk.values, _cur=d.p_cur.values, _y=y)
    res = dict(season="2024-25 (gate season, used descriptively: report-only, nothing fit)", n=int(len(d)), games=int(d.event_id.nunique()), players=int(d.player_id.nunique()),
               positives=int(y.sum()), base_rate=round(float(y.mean()), 4), top_decile_n=int(math.ceil(0.1 * len(d))),
               dk=level(d, "p_dk", "DraftKings de-vigged P(5+), own main-line hold"), current=level(d, "p_cur", "current model P(5+)"), pooled_books=level(d, "p_pool", "pooled books (median de-vigged P at 4.5)"),
               dk_minus_current=dict(diff=round(capture(d.p_dk.values, y) - capture(d.p_cur.values, y), 4), ci=boot(lambda s: capture(s._dk, s._y) - capture(s._cur, s._y), sub, None)),
               overlap_top_decile=None)
    cut_dk = np.quantile(d.p_dk.values, 0.9); cut_cur = np.quantile(d.p_cur.values, 0.9)
    top_dk = d.p_dk.values >= cut_dk; top_cur = d.p_cur.values >= cut_cur
    res["overlap_top_decile"] = dict(both=int((top_dk & top_cur).sum()), dk_only=int((top_dk & ~top_cur).sum()), current_only=int((~top_dk & top_cur).sum()),
                                     hit_rate_both=round(float(y[top_dk & top_cur].mean()), 4) if (top_dk & top_cur).any() else None,
                                     hit_rate_dk_only=round(float(y[top_dk & ~top_cur].mean()), 4) if (top_dk & ~top_cur).any() else None,
                                     hit_rate_current_only=round(float(y[~top_dk & top_cur].mean()), 4) if (~top_dk & top_cur).any() else None)
    return res


def decide(p1, p2):
    a, c = p1["levels"]["a"]["capture"], p1["levels"]["c"]["capture"]; gain = c - a
    if gain <= 0.05: oracle = ("physical ceiling", "oracle-3PA capture <= current + 5 pp: kill feature work on ranking, fold back to recalibration")
    elif gain < 0.15: oracle = ("partial headroom", "oracle-3PA gain between +5 and +15 pp: report; no model build without a separate pre-registered rule")
    else: oracle = ("attempts are the lever", "oracle-3PA gain >= +15 pp: ranking headroom exists on the attempts side")
    dd = p2["dk_minus_current"]["diff"]
    if dd >= 0.03: market = ("market has timing / information we lack", "DK >= current + 3 pp on the quoted subset: investigate what (movement timing, not price)")
    elif dd > -0.03: market = ("ranking is commoditized", "DK within +/-3 pp of current: no model beats the market at ranking tails")
    else: market = ("DK ranks tails worse than the current model", "below current - 3 pp: reported as is; the table assigns no action")
    return dict(oracle_gain_pp=round(gain * 100, 1), oracle_verdict=oracle[0], oracle_rule=oracle[1], dk_minus_current_pp=round(dd * 100, 1), market_verdict=market[0], market_rule=market[1],
                model_3_synthesis=("not pre-registered, my reading of the two verdicts together: " +
                                   ("do not build: the market does not rank tails better than the current model either, so the attempts information that would unlock the headroom is not available to anyone pre-game"
                                    if oracle[0] == "attempts are the lever" and market[0] == "ranking is commoditized" else
                                    "do not build" if oracle[0] == "physical ceiling" or market[0] == "ranking is commoditized" else
                                    "only under a new pre-registered rule" if oracle[0] == "partial headroom" else "attempts-side ranking work is justified")))


if __name__ == "__main__":
    rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "backtests/headroom_audit_rule.md"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    p1 = part1(); p2 = part2(); dec = decide(p1, p2)
    rep = dict(title="Tail program: headroom audit (report-only)", rule_file="backtests/headroom_audit_rule.md", rule_commit=rule_commit, generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               oracle=p1, dk_head_to_head=p2, decision=dec,
               notes=["Part 1 rows: 2023-24 current-model walk-forward (prior games only), untouched by any gate; the model is unchanged, only its inputs are replaced by the truth one at a time.",
                      "Oracle 3P% is the player's full-season 2023-24 3P% (talent), never the game's shooting.",
                      "Part 2 uses the 2024-25 gate season descriptively (nothing fit). DK's probability is its own de-vigged Over 4.5 alternate rung at the close (Part A method).",
                      "Top-decile capture: actual 5+ games whose score ranks in the top 10% of the scored rows; 1,000 game-cluster bootstrap resamples for every CI.",
                      "No tuning anywhere. Report and stop."])
    write_json(SITE_DATA / "headroom_audit.json", rep)
    L = p1["levels"]; steps = {k: v["diff"] for k, v in p1["steps"].items()}
    print(f"[audit] 2023-24 n {p1['n']} | capture a {L['a']['capture']} {L['a']['capture_ci']} -> b {L['b']['capture']} -> c {L['c']['capture']} {L['c']['capture_ci']} -> d {L['d']['capture']} {L['d']['capture_ci']} | steps {steps}")
    print(f"[audit] 2024-25 DK subset n {p2['n']} | DK {p2['dk']['capture']} {p2['dk']['capture_ci']} vs current {p2['current']['capture']} {p2['current']['capture_ci']} | diff {p2['dk_minus_current']} | pooled {p2['pooled_books']['capture']}")
    print(f"[audit] decision: {dec}")
