"""First basket METHOD layer (backtests/first_basket_method_rule.md): P(player, method) = P_model(player first) x P(method | player).

  python first_basket_method.py preship    walk-forward pre-ship check (profiles fit on 2025, 2026 scored) -> docs/data/first_basket_method_preship.json
  python first_basket_method.py project    rebuild profiles (data/first_basket_method_profiles.json), project tonight's logged games ->
                                           docs/data/first_basket_method.json, append data/first_basket_method_log.csv (append-only)
  python first_basket_method.py grade      grade every method projection logged before tip whose label exists -> data/first_basket_method_grades.csv,
                                           docs/data/first_basket_method_perf.json (report-only, no gate; no read before 200 games)
  python first_basket_method.py sanity     print the classifier sanity tables of rule section 1 and the Wembanyama reference

Display and grading only. The who-model (first_basket_model.py) is consumed read-only: its last pre-tip log rows are the P(player)
inputs; nothing of its params, chain, log or grading is touched. Taxonomy and constants are frozen in the rule; the published,
displayed and graded taxonomy is the five book buckets (amendment 2026-10-28); the 8-way split is a diagnostic only.
"""
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json  # noqa: E402
import first_basket_model as fbm  # noqa: E402  (read-only use: load_tables, walk, score_game, last_pretip)
from first_basket import HIST as FB_HIST  # noqa: E402
from first_shot import SEQ, STARTERS  # noqa: E402

RULE = "backtests/first_basket_method_rule.md"
METHODS8 = ["Free throw", "Dunk", "Tip-in", "Layup", "Floater", "Hook", "3-pointer", "Jumper"]
BUCKETS = ["Dunk", "Layup", "3-Point", "Free Throw", "Other"]
TO_BUCKET = {"Dunk": "Dunk", "Layup": "Layup", "3-pointer": "3-Point", "Free throw": "Free Throw", "Tip-in": "Other", "Hook": "Other", "Floater": "Other", "Jumper": "Other"}
RIM, MID = ["Dunk", "Layup", "Tip-in", "Hook"], ["Floater", "Jumper"]
GROUPS = ("G", "F", "C")
PRIOR_N, ATT_CAP, FLOOR, SEED, BOOT = 20, 40, 1e-4, 20261028, 2000      # rule sections 2, 5
THRESH_LEAGUE, THRESH_POS = 0.010, 0.003                                # rule section 6
FIT_SEASON, SCORE_SEASON = 2025, 2026
PROFILES = DATA_DIR / "first_basket_method_profiles.json"
MLOG = DATA_DIR / "first_basket_method_log.csv"
MGRADES = DATA_DIR / "first_basket_method_grades.csv"
OUT = SITE_DATA / "first_basket_method.json"
PERF = SITE_DATA / "first_basket_method_perf.json"
PRESHIP = SITE_DATA / "first_basket_method_preship.json"
LABEL = "Method layer: accuracy framing only. Paper study; no prices, no staking claims."
MLOG_FIELDS = ["date", "event_id", "tip", "who_logged_at", "logged_at", "fit_through", "mode", "n_starters"] + [f"m_{b}" for b in BUCKETS] + [f"po_{b}" for b in BUCKETS] + [f"lg_{b}" for b in BUCKETS]
GRADE_FIELDS = ["date", "event_id", "tip", "away", "home", "scorer", "scorer_started", "actual_8", "actual_5", "is_ft", "mode", "who_logged_at", "logged_at",
                "p_model", "p_pos_only", "p_league", "loss_model", "loss_pos_only", "loss_league"]


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- taxonomy (rule section 1)
def classify8(play_type, is_ft, points):
    t = play_type.lower() if isinstance(play_type, str) else ""
    if (is_ft == 1) or "free throw" in t:
        return "Free throw"
    if "dunk" in t:
        return "Dunk"
    if "tip" in t:
        return "Tip-in"
    if "layup" in t or "finger roll" in t:
        return "Layup"
    if "float" in t:
        return "Floater"
    if "hook" in t:
        return "Hook"
    if points == 3:
        return "3-pointer"
    return "Jumper"


def pos_group(pos):
    g = str(pos).strip()[-1:].upper() if isinstance(pos, str) and pos.strip() else "F"
    return g if g in GROUPS else "F"


def to_bucket5(p8):
    """8-way probability vector (METHODS8 order) -> dict over BUCKETS."""
    out = {b: 0.0 for b in BUCKETS}
    for m, p in zip(METHODS8, p8):
        out[TO_BUCKET[m]] += float(p)
    return out


def fg_only(d5):
    """Form (b): the Free Throw bucket removed, the four remaining buckets renormalised."""
    rest = {b: d5[b] for b in BUCKETS if b != "Free Throw"}
    s = sum(rest.values())
    return {b: v / s for b, v in rest.items()} if s > 0 else {b: 0.25 for b in rest}


def top3(d):
    return [{"bucket": b, "p": round(float(p), 4)} for b, p in sorted(d.items(), key=lambda kv: -kv[1])[:3]]


def rounded(d):
    return {b: round(float(v), 4) for b, v in d.items()}


# ---------------------------------------------------------------- data
def load_history():
    """Clean first-score labels with the 8-way method, the book bucket and the scorer's position group (starters file)."""
    h = pd.read_csv(FB_HIST, dtype={"event_id": str, "scorer_id": str})
    h = h[h.label_clean == 1].copy()
    h["method"] = [classify8(r.play_type, r.is_ft, r.points) for r in h.itertuples()]
    h["bucket"] = h.method.map(TO_BUCKET)
    st = pd.read_csv(STARTERS, dtype={"event_id": str, "player_id": str}, usecols=["event_id", "player_id", "pos"])
    h = h.merge(st.rename(columns={"player_id": "scorer_id"}), on=["event_id", "scorer_id"], how="left")
    h["grp"] = h.pos.map(pos_group)
    return h


def load_attempts():
    """Missed field-goal attempts before the game's first made field goal, by starters (rule section 2)."""
    s = pd.read_csv(SEQ, dtype={"event_id": str, "player_id": str})
    s = s[(s.started == 1) & (s.made == 0) & s.player_id.notna()].copy()
    s["grp"] = s.pos.map(pos_group)
    return s


class Fit:
    """Profiles from one window of history + attempts (rule section 2). All counts are over METHODS8."""

    def __init__(self, h, att, through, stamp=None):
        self.through, self.stamp, self.n_games, self.n_attempts = through, stamp or ts(now()), int(len(h)), int(len(att))
        idx = {m: i for i, m in enumerate(METHODS8)}
        league = np.array([float((h.method == m).sum()) for m in METHODS8])
        self.league8 = league / league.sum() if league.sum() else np.full(8, 1 / 8)
        self.prior, self.rim_share, self.mid_share = {}, {}, {}
        for g in GROUPS:
            hg = h[h.grp == g]
            c = np.array([float((hg.method == m).sum()) for m in METHODS8])
            self.prior[g] = c / c.sum() if c.sum() else self.league8
            rim = np.array([float((hg.method == m).sum()) for m in RIM]); mid = np.array([float((hg.method == m).sum()) for m in MID])
            lr = np.array([float((h.method == m).sum()) for m in RIM]); lm = np.array([float((h.method == m).sum()) for m in MID])
            self.rim_share[g] = rim / rim.sum() if rim.sum() else (lr / lr.sum() if lr.sum() else np.full(4, 0.25))
            self.mid_share[g] = mid / mid.sum() if mid.sum() else (lm / lm.sum() if lm.sum() else np.full(2, 0.5))
        self.own = defaultdict(lambda: np.zeros(8))
        for r in h.itertuples():
            self.own[r.scorer_id][idx[r.method]] += 1
        self.att = defaultdict(lambda: {"rim": 0, "mid": 0, "3": 0})
        for pid, t, n in att.groupby(["player_id", "shot_type"]).size().reset_index().itertuples(index=False):
            self.att[pid][str(t)] += int(n)
        self.players = set(self.own) | set(self.att)

    def counts(self, pid, grp, mode="full"):
        c = PRIOR_N * self.prior[grp]
        comp = {"position_prior_n": PRIOR_N, "position_group": grp, "attempts_n": 0, "attempts_weight": 0.0, "own_makes_n": 0}
        if mode == "position_only":
            return c, comp
        a = self.att.get(pid)
        if a:
            n = sum(a.values()); w = min(n, ATT_CAP) / n
            comp.update({"attempts_n": int(n), "attempts_weight": round(float(min(n, ATT_CAP)), 2)})
            add = np.zeros(8)
            for i, m in enumerate(RIM):
                add[METHODS8.index(m)] += a["rim"] * w * self.rim_share[grp][i]
            for i, m in enumerate(MID):
                add[METHODS8.index(m)] += a["mid"] * w * self.mid_share[grp][i]
            add[METHODS8.index("3-pointer")] += a["3"] * w
            c = c + add
        o = self.own.get(pid)
        if o is not None:
            c = c + o; comp["own_makes_n"] = int(o.sum())
        return c, comp

    def profile8(self, pid, grp, mode="full"):
        c, comp = self.counts(pid, grp, mode)
        return c / c.sum(), comp

    def profile5(self, pid, grp, mode="full"):
        p8, comp = self.profile8(pid, grp, mode)
        return to_bucket5(p8), comp, p8


def league_mix5(h):
    c = {b: float((h.bucket == b).sum()) for b in BUCKETS}
    s = sum(c.values())
    return {b: v / s for b, v in c.items()} if s else {b: 0.2 for b in BUCKETS}


def league_mix5_laplace(h):
    c = {b: float((h.bucket == b).sum()) + 1.0 for b in BUCKETS}
    s = sum(c.values())
    return {b: v / s for b, v in c.items()}


def game_dist(p_players, profiles):
    """p_players: array summing to 1; profiles: list of 5-way dicts. Returns the game's 5-way distribution (form a)."""
    out = {b: 0.0 for b in BUCKETS}
    for p, pr in zip(p_players, profiles):
        for b in BUCKETS:
            out[b] += float(p) * pr[b]
    return out


def loss(d, actual):
    return float(-np.log(max(float(d.get(actual, 0.0)), FLOOR)))


def paired_ci(a, b, seed=SEED, iters=BOOT):
    """95% bootstrap CI of mean(a - b) over games."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return None
    rng = np.random.default_rng(seed); d = a - b
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    v = d[idx].mean(axis=1)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


# ---------------------------------------------------------------- sanity (rule section 1 targets)
def sanity_tables(h):
    fg = h[h.is_ft == 0]
    def mix8(d):
        return {m: round(100 * float((d.method == m).mean()), 1) for m in METHODS8 if m != "Free throw"} if len(d) else {}
    def mix5(d):
        return {b: round(100 * float((d.bucket == b).mean()), 1) for b in BUCKETS} if len(d) else {}
    return {"first_scores": int(len(h)), "fg_first_baskets": int(len(fg)),
            "eight_way_fg_pct": {"league": mix8(fg), "C": mix8(fg[fg.grp == "C"]), "G": mix8(fg[fg.grp == "G"]), "F": mix8(fg[fg.grp == "F"])},
            "book_buckets_all_pct": {"league": mix5(h), "C": mix5(h[h.grp == "C"]), "G": mix5(h[h.grp == "G"]), "F": mix5(h[h.grp == "F"])},
            "targets": {"eight_way_fg_league": {"Dunk": 11.5, "Layup": 25.2, "Tip-in": 1.9, "Hook": 4.2, "Floater": 8.5, "Jumper": 17.8, "3-pointer": 31.0},
                        "book_all_league": {"Dunk": 10.4, "Layup": 22.8, "3-Point": 28.1, "Free Throw": 9.2, "Other": 29.4},
                        "book_all_C": {"Dunk": 21.7, "Layup": 22.1, "3-Point": 14.8, "Other": 33.2}, "book_all_G": {"3-Point": 34.2, "Dunk": 4.3}},
            "starter_coverage_share": round(float(h.scorer_started.mean()), 4), "scorer_position_found_share": round(float(h.pos.notna().mean()), 4)}


def wemby_reference(h):
    """Position + own makes only (prior 20), frozen history, book buckets. Target: Dunk 24.6, Layup 15.3, 3-Point 14.2, FT 11.1, Other 34.9."""
    rows = h[h.scorer.fillna("").str.contains("Wembanyama")]
    if rows.empty:
        return None
    pid = rows.scorer_id.iloc[0]; grp = rows.grp.mode().iloc[0]
    fit = Fit(h, load_attempts().iloc[0:0], through=str(h.date.max()))
    p5, comp, _ = fit.profile5(pid, grp)
    return {"player": rows.scorer.iloc[0], "player_id": pid, "group": grp, "own_makes_n": comp["own_makes_n"], "book_pct": {b: round(100 * v, 1) for b, v in p5.items()},
            "target_pct": {"Dunk": 24.6, "Layup": 15.3, "3-Point": 14.2, "Free Throw": 11.1, "Other": 34.9}}


def sanity():
    h = load_history()
    S = sanity_tables(h); W = wemby_reference(h)
    print("[fb-method] 8-way FG league %:", S["eight_way_fg_pct"]["league"]); print("[fb-method]   C:", S["eight_way_fg_pct"]["C"]); print("[fb-method]   G:", S["eight_way_fg_pct"]["G"])
    print("[fb-method] book buckets all %:", S["book_buckets_all_pct"]["league"]); print("[fb-method]   C:", S["book_buckets_all_pct"]["C"]); print("[fb-method]   G:", S["book_buckets_all_pct"]["G"])
    print(f"[fb-method] starter coverage {S['starter_coverage_share']}, scorer position found {S['scorer_position_found_share']}")
    print("[fb-method] Wembanyama (position + own makes):", W)
    return S, W


# ---------------------------------------------------------------- pre-ship check (rule section 6)
def preship():
    params = read_json(fbm.PARAMS, None)
    if not params:
        print("[fb-method] no frozen who-model params"); return
    K, C = params["K"], params["constants"]
    h, att = load_history(), load_attempts()
    fit = Fit(h[h.season == FIT_SEASON], att[att.season == FIT_SEASON], through=f"season {FIT_SEASON}")
    league = league_mix5(h[h.season == FIT_SEASON])
    games, seq, st = fbm.load_tables()
    records, _ = fbm.walk(games, st, {SCORE_SEASON})
    rec = {G["event_id"]: G for G in records}
    score = h[h.season == SCORE_SEASON]
    rows, missing = [], 0
    for r in score.itertuples():
        G = rec.get(r.event_id)
        if G is None:
            missing += 1; continue
        S = fbm.score_game(G, K, C); p = np.asarray(S["v2"], float); p = p / p.sum()
        full = [fit.profile5(s["player_id"], pos_group(s["pos"]))[0] for s in G["starters"]]
        pos = [fit.profile5(s["player_id"], pos_group(s["pos"]), "position_only")[0] for s in G["starters"]]
        dm, dp = game_dist(p, full), game_dist(p, pos)
        p8m = sum(pp * fit.profile8(s["player_id"], pos_group(s["pos"]))[0] for pp, s in zip(p, G["starters"]))
        rows.append({"event_id": r.event_id, "date": r.date, "actual_5": r.bucket, "actual_8": r.method, "is_ft": int(r.is_ft),
                     "loss_model": loss(dm, r.bucket), "loss_pos": loss(dp, r.bucket), "loss_league": loss(league, r.bucket),
                     "loss_model_8": float(-np.log(max(float(p8m[METHODS8.index(r.method)]), FLOOR))), "loss_league_8": float(-np.log(max(float(fit.league8[METHODS8.index(r.method)]), FLOOR))),
                     **{f"m_{b}": dm[b] for b in BUCKETS}})
    d = pd.DataFrame(rows)
    mm, mp, ml = float(d.loss_model.mean()), float(d.loss_pos.mean()), float(d.loss_league.mean())
    gain_l, gain_p = ml - mm, mp - mm
    ship_full = gain_l >= THRESH_LEAGUE and gain_p >= THRESH_POS
    decision = "SHIP_FULL_BLEND" if ship_full else "SHIP_POSITION_ONLY"
    shares = {"predicted_mean": {b: round(float(d[f"m_{b}"].mean()), 4) for b in BUCKETS}, "actual": {b: round(float((d.actual_5 == b).mean()), 4) for b in BUCKETS}}
    doc = {"title": "First basket METHOD layer: pre-ship check", "rule_file": RULE, "generated": ts(now()), "decision": decision,
           "decision_text": ("full blend ships: beats the league mix by >= 0.010 nats and the position-only ablation by >= 0.003 nats" if ship_full else
                             "position-only profiles ship: the full blend did not clear both thresholds (rule section 6)"),
           "thresholds": {"vs_league_mix_nats": THRESH_LEAGUE, "vs_position_only_nats": THRESH_POS},
           "fit": {"season": FIT_SEASON, "first_scores": fit.n_games, "attempt_rows": fit.n_attempts, "players_with_data": len(fit.players), "league_mix_2025": rounded(league)},
           "score": {"season": SCORE_SEASON, "first_scores_in_history": int(len(score)), "games_scored": int(len(d)), "games_without_who_record": int(missing),
                     "fg_first_baskets": int((d.is_ft == 0).sum()), "free_throw_rows": int((d.is_ft == 1).sum())},
           "log_loss": {"full_blend": round(mm, 4), "position_only": round(mp, 4), "league_mix": round(ml, 4),
                        "gain_vs_league": round(gain_l, 4), "gain_vs_league_ci": paired_ci(d.loss_league, d.loss_model), "passes_league": bool(gain_l >= THRESH_LEAGUE),
                        "gain_vs_position_only": round(gain_p, 4), "gain_vs_position_only_ci": paired_ci(d.loss_pos, d.loss_model), "passes_position_only": bool(gain_p >= THRESH_POS),
                        "diagnostic_8way": {"full_blend": round(float(d.loss_model_8.mean()), 4), "league_mix": round(float(d.loss_league_8.mean()), 4)}},
           "bucket_shares": shares, "who_model": {"params": str(fbm.PARAMS.relative_to(DATA_DIR.parent)), "K": K, "note": "frozen constants, actual ten starters, the gate's walk-forward code, read-only"},
           "sanity": sanity_tables(h), "wembanyama_reference": wemby_reference(h),
           "method": ["Game-level log loss = -log P(actual book bucket) of the form (a) distribution (Free Throw bucket in), floor 1e-4.",
                      "Profiles: 20 position-prior pseudo-counts + min(n, 40) attempt pseudo-counts (missed attempts before the first FG, mapped by position rim / mid shares) + own first scores at face count; fit on the 2025 season only.",
                      "Baselines: the 2025 league bucket mix; who-model x 2025 position priors. 95% CIs: paired bootstrap over games, 2,000 resamples, seed 20261028.",
                      LABEL]}
    write_json(PRESHIP, doc)
    print(f"[fb-method] pre-ship: {len(d)} games scored ({missing} without a who-model record); log loss full {mm:.4f}, position-only {mp:.4f}, league mix {ml:.4f}; "
          f"gain vs league {gain_l:+.4f} (need >= {THRESH_LEAGUE}), vs position-only {gain_p:+.4f} (need >= {THRESH_POS}) -> {decision}")
    return doc


def live_mode():
    """What the pre-ship check said to ship. No verdict file = nothing is projected."""
    P = read_json(PRESHIP, None)
    if not P or P.get("decision") not in ("SHIP_FULL_BLEND", "SHIP_POSITION_ONLY"):
        return None
    return "full" if P["decision"] == "SHIP_FULL_BLEND" else "position_only"


# ---------------------------------------------------------------- live projection (rule section 3)
def project(date=None):
    mode = live_mode()
    if mode is None:
        print("[fb-method] no pre-ship verdict (docs/data/first_basket_method_preship.json): nothing projected"); return
    if not fbm.LOG.exists():
        print("[fb-method] no who-model log"); return
    log = pd.read_csv(fbm.LOG, dtype={"event_id": str, "player_id": str})
    date = date or (str(log.date.max()) if len(log) else None)
    if not date:
        return
    d = fbm.last_pretip(log, date)
    h, att = load_history(), load_attempts()
    hb, ab = h[h.date < date], att[att.date < date]
    fit = Fit(hb, ab, through=str(hb.date.max()) if len(hb) else None)
    P = read_json(SITE_DATA / "projections.json", None) or {}
    season = int(P.get("season") or (hb.season.max() if len(hb) else SCORE_SEASON + 1)) if P.get("date") == date else int(hb.season.max()) if len(hb) else SCORE_SEASON + 1
    std = hb[hb.season == season]
    league = league_mix5_laplace(std)
    t = now(); games = {}; new_rows = []
    old = pd.read_csv(MLOG, dtype={"event_id": str}) if MLOG.exists() else pd.DataFrame(columns=MLOG_FIELDS)
    done = set(zip(old.event_id, old.who_logged_at)) if len(old) else set()
    # position group of a live starter = his latest starters-file position (rule section 1), else the who-log's position
    st_all = pd.read_csv(STARTERS, dtype={"event_id": str, "player_id": str}, usecols=["date", "player_id", "pos"]).sort_values("date")
    grp_of = {pid: pos_group(ps) for pid, ps in zip(st_all.player_id, st_all.pos)}
    grp_live = lambda r: grp_of.get(r.player_id) or pos_group(r.pos)  # noqa: E731
    # data coverage of tonight's starters under the full blend's components, whatever mode ships (reported, rule section 7)
    cov = {"own_makes_and_attempts": 0, "attempts_only": 0, "own_makes_only": 0, "position_only": 0}
    for e, g in d.groupby("event_id"):
        g = g.sort_values("p_first_basket", ascending=False)
        p = g.p_first_basket.to_numpy(float); p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
        prof = []
        for r in g.itertuples():
            p5, comp, _ = fit.profile5(r.player_id, grp_live(r), mode)
            full_comp = fit.counts(r.player_id, grp_live(r), "full")[1]
            comp["data_available"] = {"own_makes_n": full_comp["own_makes_n"], "attempts_n": full_comp["attempts_n"]}
            prof.append((p5, comp))
            cov["own_makes_and_attempts" if full_comp["own_makes_n"] and full_comp["attempts_n"] else "attempts_only" if full_comp["attempts_n"] else "own_makes_only" if full_comp["own_makes_n"] else "position_only"] += 1
        pos_prof = [fit.profile5(r.player_id, grp_live(r), "position_only")[0] for r in g.itertuples()]
        dm, dp = game_dist(p, [x[0] for x in prof]), game_dist(p, pos_prof)
        picks = []
        for i, r in enumerate(list(g.itertuples())[:3]):
            p5, comp = prof[i]
            picks.append({"rank": i + 1, "player": r.player, "player_id": r.player_id, "team": r.team, "pos": r.pos, "p_first_basket": float(r.p_first_basket), "p_renormalised": round(float(p[i]), 4),
                          "method_all": rounded(p5), "method_fg": rounded(fg_only(p5)), "top3_fg": top3(fg_only(p5)), "profile": comp,
                          "profile_source": (f"{comp['position_group']} position prior only (position-only mode shipped by the pre-ship check)" if mode == "position_only" else
                                             f"{comp['position_group']} position prior only" if not (comp["own_makes_n"] or comp["attempts_n"]) else
                                             f"{comp['position_group']} position prior 20 + {comp['attempts_n']} attempts (weight {comp['attempts_weight']}) + {comp['own_makes_n']} own first scores")})
        home = g[g.home == 1].team.iloc[0] if (g.home == 1).any() else None; away = g[g.home == 0].team.iloc[0] if (g.home == 0).any() else None
        games[e] = {"event_id": e, "date": date, "tip": g.tip.iloc[0], "home": home, "away": away, "who_logged_at": g.logged_at.iloc[0], "n_starters": int(len(g)),
                    "game_all": rounded(dm), "game_fg": rounded(fg_only(dm)), "top3_fg": top3(fg_only(dm)), "top3_all": top3(dm),
                    "position_only_all": rounded(dp), "league_mix_all": rounded(league), "picks": picks}
        if (e, g.logged_at.iloc[0]) not in done:
            new_rows.append({"date": date, "event_id": e, "tip": g.tip.iloc[0], "who_logged_at": g.logged_at.iloc[0], "logged_at": ts(t), "fit_through": fit.through, "mode": mode, "n_starters": int(len(g)),
                             **{f"m_{b}": round(dm[b], 5) for b in BUCKETS}, **{f"po_{b}": round(dp[b], 5) for b in BUCKETS}, **{f"lg_{b}": round(league[b], 5) for b in BUCKETS}})
    if new_rows:
        new_file = not MLOG.exists()
        with open(MLOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MLOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(new_rows)
    # profiles file: every player with data in the window plus tonight's starters (rule section 2)
    tonight = {(r.player_id, r.player, grp_live(r)) for r in d.itertuples()}
    names = dict(zip(hb.scorer_id, hb.scorer)); names.update({pid: nm for pid, nm, _ in tonight})
    grp_of.update({pid: gp for pid, _, gp in tonight})
    plist = {}
    for pid in fit.players | {x[0] for x in tonight}:
        gp = grp_of.get(pid, "F")
        p5, comp, p8 = fit.profile5(pid, gp, mode)
        plist[pid] = {"player": names.get(pid), "group": gp, "book": rounded(p5), "book_fg": rounded(fg_only(p5)), "eight_way_diagnostic": {m: round(float(x), 4) for m, x in zip(METHODS8, p8)}, "components": comp}
    write_json(PROFILES, {"generated": ts(t), "rule_file": RULE, "mode": mode, "fit": {"through": fit.through, "first_scores": fit.n_games, "attempt_rows": fit.n_attempts, "constants": {"position_prior_n": PRIOR_N, "attempt_cap": ATT_CAP}},
                          "position_priors_book": {g_: rounded(to_bucket5(fit.prior[g_])) for g_ in GROUPS}, "players": plist})
    write_json(OUT, {"date": date, "generated": ts(t), "rule_file": RULE, "label": LABEL, "mode": mode,
                     "mode_text": "full blend (position prior + attempts + own first scores)" if mode == "full" else "position-only profiles (the pre-ship check did not clear the full blend's thresholds)",
                     "buckets": BUCKETS, "fit": {"through": fit.through, "first_scores": fit.n_games, "attempt_rows": fit.n_attempts, "season_to_date_league_n": int(len(std))},
                     "coverage_tonight": {"starters": int(sum(cov.values())), "data_available_under_full_blend": cov, "shipped": "position-only profiles for every starter" if mode == "position_only" else "full blend"}, "games": games,
                     "note": "P(player, method) = P_model(first) x P(method | player), summed over the ten logged starters; 'fg' = Free Throw bucket removed and renormalised (books settle on the first field goal). Display and grading only; the who-model is untouched."})
    print(f"[fb-method] {date}: {len(games)} games projected ({mode}), {len(new_rows)} new log rows, profiles for {len(plist)} players (fit through {fit.through}, {fit.n_games} first scores); tonight's starters: {cov}")
    return games


# ---------------------------------------------------------------- grading (rule section 5)
def grade():
    if not MLOG.exists():
        print("[fb-method] no method log yet"); return
    m = pd.read_csv(MLOG, dtype={"event_id": str})
    m = m[pd.to_datetime(m.logged_at, utc=True) < pd.to_datetime(m.tip, utc=True)]
    m = m[m.logged_at == m.groupby("event_id").logged_at.transform("max")].drop_duplicates("event_id", keep="last")
    h = load_history().set_index("event_id")
    rows = []
    for r in m.itertuples():
        if r.event_id not in h.index:
            continue
        x = h.loc[r.event_id]
        rec = m.loc[r.Index]   # the bucket columns carry spaces / hyphens, so read them through the frame, not the namedtuple
        dm = {b: float(rec[f"m_{b}"]) for b in BUCKETS}; dp = {b: float(rec[f"po_{b}"]) for b in BUCKETS}; lg = {b: float(rec[f"lg_{b}"]) for b in BUCKETS}
        rows.append({"date": r.date, "event_id": r.event_id, "tip": r.tip, "away": x.away, "home": x.home, "scorer": x.scorer, "scorer_started": int(x.scorer_started), "actual_8": x.method, "actual_5": x.bucket, "is_ft": int(x.is_ft),
                     "mode": r.mode, "who_logged_at": r.who_logged_at, "logged_at": r.logged_at, "p_model": round(dm[x.bucket], 4), "p_pos_only": round(dp[x.bucket], 4), "p_league": round(lg[x.bucket], 4),
                     "loss_model": round(loss(dm, x.bucket), 4), "loss_pos_only": round(loss(dp, x.bucket), 4), "loss_league": round(loss(lg, x.bucket), 4), **{f"_m_{b}": dm[b] for b in BUCKETS}})
    G = pd.DataFrame(rows)
    G[GRADE_FIELDS].to_csv(MGRADES, index=False) if len(G) else pd.DataFrame(columns=GRADE_FIELDS).to_csv(MGRADES, index=False)
    n = int(len(G))
    doc = {"title": "First basket METHOD layer: graded record (report-only, no gate)", "rule_file": RULE, "generated": ts(now()), "label": LABEL, "games": n, "min_games_for_a_read": 200,
           "read": "no read before 200 graded games" if n < 200 else "200+ graded games: the numbers below may be read; still no gate, no verdict",
           "log_loss": {"model": round(float(G.loss_model.mean()), 4) if n else None, "position_only": round(float(G.loss_pos_only.mean()), 4) if n else None, "league_mix": round(float(G.loss_league.mean()), 4) if n else None,
                        "gain_vs_league": round(float((G.loss_league - G.loss_model).mean()), 4) if n else None, "gain_vs_league_ci": paired_ci(G.loss_league, G.loss_model) if n else None,
                        "gain_vs_position_only": round(float((G.loss_pos_only - G.loss_model).mean()), 4) if n else None, "gain_vs_position_only_ci": paired_ci(G.loss_pos_only, G.loss_model) if n else None},
           "bucket_shares": {"predicted_mean": {b: round(float(G[f"_m_{b}"].mean()), 4) for b in BUCKETS} if n else {}, "actual": {b: round(float((G.actual_5 == b).mean()), 4) for b in BUCKETS} if n else {}},
           "actual_8way_diagnostic": {k: int(v) for k, v in G.actual_8.value_counts().items()} if n else {}, "modes": {k: int(v) for k, v in G["mode"].value_counts().items()} if n else {},
           "recent": G.sort_values(["date", "tip"], ascending=False).head(40)[["date", "away", "home", "scorer", "actual_8", "actual_5", "p_model", "p_pos_only", "p_league"]].to_dict("records") if n else [],
           "grades_file": "data/first_basket_method_grades.csv",
           "method": ["One row per graded game: the method projection logged before tip (form (a), Free Throw bucket in) against the classifier's bucket for the game's first score.",
                      "Log loss = -log P(actual bucket), floor 1e-4; baselines logged at the same moment: season-to-date league mix (Laplace +1) and who-model x position-only profiles.",
                      "Report-only: no verdicts or threshold claims before 200 graded games; never a gate. 95% CIs: paired bootstrap over games.", LABEL]}
    write_json(PERF, doc)
    print(f"[fb-method] graded {n} games: log loss model {doc['log_loss']['model']}, position-only {doc['log_loss']['position_only']}, league {doc['log_loss']['league_mix']} ({doc['read']})")
    return doc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    {"sanity": sanity, "preship": preship, "project": project, "grade": grade}[cmd]()
