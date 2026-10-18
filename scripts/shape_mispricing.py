"""Tail program test 3: conditional shape mispricing at DraftKings (backtests/shape_mispricing_rule.md). Writes
docs/data/shape_mispricing.json only.
  python shape_mispricing.py
"""
import datetime as dt
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_name, write_json
from backtest_threes_alt_calibration import build_rungs
from ladder_shape_diagnostic import load_box, league_prior, shape_scores

SEED, ITERS, PASS_D = 20261018, 1000, 0.015


def usual_starters_out(box, season):
    """Per (event, team): usual starters (started >= 50% of the team's previous 10 games, seen within 21 days) absent tonight, and
       the set of remaining usual starters who play. Prior games only."""
    b = box[box.season == season].sort_values("date")
    st = b[["event_id", "team", "date", "player_id", "starter"]].copy()
    st["starter"] = pd.to_numeric(st.starter, errors="coerce").fillna(0)
    st = st.sort_values(["team", "player_id", "date"])
    st["share"] = st.groupby(["team", "player_id"]).starter.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    st = st.dropna(subset=["share"])
    present = b.groupby(["event_id", "team"]).player_id.apply(set).to_dict()
    games = b.drop_duplicates(["event_id", "team"])[["event_id", "team", "date"]]
    out = {}
    for team, grp in games.groupby("team"):
        hist = st[st.team == team]
        for r in grp.itertuples():
            prior = hist[hist.date < r.date]
            if prior.empty: continue
            latest = prior.groupby("player_id").share.last(); seen = prior.groupby("player_id").date.max()
            usual = {p for p, s in latest.items() if s >= 0.5 and (r.date - seen[p]).days <= 21}
            here = present.get((r.event_id, team), set())
            absent = usual - here; remaining = usual & here
            out[(r.event_id, team)] = (len(absent), remaining)
    return out


def gap_block(g):
    return dict(n=int(len(g)), hits=int(g.hit.sum()), hit_rate=round(float(g.hit.mean()), 4), devig=round(float(g.fair_book.mean()), 4), gap=round(float(g.hit.mean() - g.fair_book.mean()), 4)) if len(g) else dict(n=0)


def boot_D(df, top, bot, iters=ITERS, seed=SEED):
    """Bootstrap D = gap(top) - gap(bot) resampling player-games with replacement."""
    rng = np.random.default_rng(seed); pg = df.pg.values; ups = np.unique(pg); idx = {k: np.where(pg == k)[0] for k in ups}
    gap = (df.hit - df.fair_book).values; t = top.values; bm = bot.values; vals = []
    for _ in range(iters):
        rows = np.concatenate([idx[k] for k in rng.choice(ups, len(ups), replace=True)])
        g = gap[rows]; tt = t[rows]; bb = bm[rows]
        if tt.sum() == 0 or bb.sum() == 0: continue
        vals.append(g[tt].mean() - g[bb].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5]); return [round(float(lo), 4), round(float(hi), 4)]


def gap_ci(df, mask, iters=ITERS, seed=SEED):
    rng = np.random.default_rng(seed); sub = df[mask]; pg = sub.pg.values; ups = np.unique(pg); idx = {k: np.where(pg == k)[0] for k in ups}
    gap = (sub.hit - sub.fair_book).values; vals = []
    for _ in range(iters):
        rows = np.concatenate([idx[k] for k in rng.choice(ups, len(ups), replace=True)]); vals.append(gap[rows].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5]); return [round(float(lo), 4), round(float(hi), 4)]


def run(season=2025):
    box = load_box(); prior = league_prior(box); shape = shape_scores(box, prior)
    sd_shape = float(shape.sscore.std(ddof=1))
    pg = box[box.season == season][["event_id", "player_id", "player", "team", "starter"]].copy(); pg["name_key"] = pg.player.map(norm_name)
    shape = shape.merge(pg[["event_id", "player_id", "name_key", "team"]], on=["event_id", "player_id"], how="left")
    so = usual_starters_out(box, season)
    adj = []
    for r in shape.itertuples():
        n_abs, remaining = so.get((r.event_id, r.team), (0, set()))
        adj.append(sd_shape if (n_abs > 0 and r.player_id in remaining) else 0.0)
    shape["adj"] = adj; shape["sscore_adj"] = shape.sscore + shape.adj
    df, long, _ = build_rungs(season, DATA_DIR / f"odds_alt_{season}", DATA_DIR / f"odds_close_{season}")
    dk = long[(long.book == "DraftKings") & long.is_tail].copy(); dk["name_key"] = dk.player.map(norm_name)
    u = dk.merge(shape[["event_id", "name_key", "sscore", "sscore_adj", "adj", "min_cv", "vmr_shrunk"]], on=["event_id", "name_key"], how="inner")
    u["pg"] = u.event_id + "|" + u.name_key
    u["kind"] = np.where(u.line >= 4.5, "5+", "2 above main")
    lo_cut, hi_cut = np.percentile(u.sscore_adj, [100 / 3, 200 / 3])
    top, bot = u.sscore_adj >= hi_cut, u.sscore_adj < lo_cut; mid = ~top & ~bot
    D = float((u[top].hit - u[top].fair_book).mean() - (u[bot].hit - u[bot].fair_book).mean()); ci = boot_D(u, top, bot)
    res = dict(season="2024-25" if season == 2025 else str(season), universe=dict(rungs=int(len(u)), player_games=int(u.pg.nunique()), players=int(u.name_key.nunique()), games=int(u.event_id.nunique()), hits=int(u.hit.sum()),
                                                                                  dk_tail_rungs_before_score_join=int(len(dk)), starter_adjusted_rungs=int((u.adj > 0).sum()), tercile_cuts=[round(float(lo_cut), 4), round(float(hi_cut), 4)]),
               top=dict(**gap_block(u[top]), gap_ci=gap_ci(u, top)), middle=dict(**gap_block(u[mid]), gap_ci=gap_ci(u, mid)), bottom=dict(**gap_block(u[bot]), gap_ci=gap_ci(u, bot)),
               D=round(D, 4), D_ci=ci, pass_=bool(D >= PASS_D and ci[0] > 0),
               by_kind={k: dict(top=gap_block(u[top & (u.kind == k)]), bottom=gap_block(u[bot & (u.kind == k)]), D=round(float((u[top & (u.kind == k)].hit - u[top & (u.kind == k)].fair_book).mean() - (u[bot & (u.kind == k)].hit - u[bot & (u.kind == k)].fair_book).mean()), 4)) for k in ("5+", "2 above main")})
    # information only: without the starter-out adjustment
    lo2, hi2 = np.percentile(u.sscore, [100 / 3, 200 / 3]); t2, b2 = u.sscore >= hi2, u.sscore < lo2
    res["without_starter_adjustment"] = dict(D=round(float((u[t2].hit - u[t2].fair_book).mean() - (u[b2].hit - u[b2].fair_book).mean()), 4), D_ci=boot_D(u, t2, b2), top=gap_block(u[t2]), bottom=gap_block(u[b2]))
    # information only: the two components separately
    comp = {}
    for c in ("min_cv", "vmr_shrunk"):
        l3, h3 = np.percentile(u[c], [100 / 3, 200 / 3]); t3, b3 = u[c] >= h3, u[c] < l3
        comp[c] = dict(D=round(float((u[t3].hit - u[t3].fair_book).mean() - (u[b3].hit - u[b3].fair_book).mean()), 4), top=gap_block(u[t3]), bottom=gap_block(u[b3]))
    res["components"] = comp
    return res, prior, sd_shape


if __name__ == "__main__":
    rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "backtests/shape_mispricing_rule.md"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    res, prior, sd_shape = run(2025)
    have_2026 = (DATA_DIR / "odds_alt_2026").exists() and any((DATA_DIR / "odds_alt_2026").glob("4*.json"))
    rep = dict(title="Tail program test 3: conditional shape mispricing at DraftKings", rule_file="backtests/shape_mispricing_rule.md", rule_commit=rule_commit,
               generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), pass_threshold=PASS_D, verdict="PASS" if res["pass_"] else "KILL: seeds A / B / C are dead everywhere, including Underdog",
               gate=res, shape_prior=prior, shape_sd=round(sd_shape, 4),
               consistency_2025_26=dict(available=bool(have_2026), note="no 2025-26 DraftKings closing snapshots exist in the repository (Part A pulled 2024-25 only); pulling main + alternate closes for 1,231 games costs about 24.6K Odds API credits on the shared plan and is not spent under this rule") if not have_2026 else None,
               notes=["Contamination declared in the rule: Part A saw 2024-25 outcomes only as aggregates, Part B never saw this split, test 1 used no outcomes; every feature here uses prior games only.",
                      "Universe: DraftKings tail rungs (line >= 4.5 or >= 2 above the modal main line) from the Part A closing pulls, DK's own de-vigged price (Part A method), outcome = threes made > line, DNP dropped.",
                      "Starter-out: +1 SD of the shape score for every remaining usual starter of a team with a usual starter absent, decided from the pre-game roster rule (usual = started >= 50% of the last 10, seen within 21 days); absence itself is read from the box score, the only historical availability record.",
                      "CIs: 1,000 bootstrap resamples of player-games (clusters), seed 20261018. Report and stop."])
    write_json(SITE_DATA / "shape_mispricing.json", rep)
    g = res
    print(f"[test3] rungs {g['universe']['rungs']} ({g['universe']['player_games']} player-games, {g['universe']['hits']} hits; {g['universe']['starter_adjusted_rungs']} starter-adjusted) | top {g['top']} | bottom {g['bottom']} | D {g['D']} CI {g['D_ci']} | {rep['verdict']}")
    print(f"[test3] by kind {g['by_kind']} | without starter adj {g['without_starter_adjustment']['D']} {g['without_starter_adjustment']['D_ci']} | components { {k: v["D"] for k, v in g["components"].items()} }")
