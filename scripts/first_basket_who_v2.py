"""First basket WHO-model v2: offensive-rebound continuation (backtests/first_basket_who_v2_rule.md).

  python first_basket_who_v2.py preship   fit the continuation tables on 2024-25, re-derive who-v1 on the same fit, score the 2025-26
                                          evaluation set paired per game -> docs/data/first_basket_who_v2_preship.json (Backtest page)
  python first_basket_who_v2.py ship      only after a SHIP verdict: write data/first_basket_who_v2_params.json once (constants from the
                                          check, live_from = now). The snapshot in first_basket_model.py prices with who-v2 from then on.

who-v1 (the live attempt-based model, `first_basket_model.py`) is imported, not copied: its tip model, shares, make rates, frozen K
and constants are reused unchanged. The one addition is a continuation state after a missed attempt (rule section 2).
"""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import first_basket_model as fbm
from common import DATA_DIR, SITE_DATA, read_json, write_json

RULE = "backtests/first_basket_who_v2_rule.md"
PRESHIP = SITE_DATA / "first_basket_who_v2_preship.json"
PARAMS2 = DATA_DIR / "first_basket_who_v2_params.json"
PRIOR = 20                      # rule 2c: continuation shooter prior (the directive's)
POS_FLOOR = 20                  # rule 2d: a position with fewer continuation attempts in the window uses the league mix
FIT_SEASON, EVAL_SEASON = 2025, 2026
BAR1_GAIN, SEED, BOOT = 0.002, 20261029, 1000
P_CONT_EXPECT, P_CONT_FAR = 0.31, 0.05
CLASS = {"4871144": "Alperen Sengun", "4432158": "Evan Mobley"}            # rule 3 BAR 2, pre-specified from the 2024-25 fit data
REPORT_ONLY = {"4066328": "Jarrett Allen", "5104157": "Victor Wembanyama"}  # rule 3 report-only lines (Wembanyama id resolved at run time by name if this is stale)
TYPES = fbm.TYPES


# ---------------------------------------------------------------- continuation tallies (walk-forward, rule 2c)
class ContTallies:
    def __init__(self):
        self.k = defaultdict(int); self.n = defaultdict(int)          # player -> continuation attempts taken / team continuation attempts while he started
        self.pos_k = defaultdict(int); self.pos_n = defaultdict(int)  # position -> the same, pooled

    def update(self, game, starters, cont_rows):
        """cont_rows: the game's continuation attempts (idx > 1, same team as the previous attempt). Regular season + postseason only."""
        if game.phase == "preseason" or not cont_rows:
            return
        for team, shooter in cont_rows:
            for s in starters:
                if s.team != team:
                    continue
                self.n[s.player_id] += 1; self.pos_n[s.pos] += 1
                if s.player_id == shooter:
                    self.k[s.player_id] += 1; self.pos_k[s.pos] += 1

    def features(self, starters):
        out = []
        for s in starters:
            pn = self.pos_n.get(s.pos, 0)
            out.append({"player_id": s.player_id, "k": self.k.get(s.player_id, 0), "n": self.n.get(s.player_id, 0), "base_cont": (self.pos_k.get(s.pos, 0) / pn) if pn else 0.2})
        return out


def cont_rows_by_event(seq):
    """event_id -> [(team, shooter_id)] of its continuation attempts, in order."""
    c = seq[(seq.idx > 1) & (seq.same_team_as_prev == 1)].sort_values(["event_id", "idx"])
    out = defaultdict(list)
    for r in c.itertuples():
        out[r.event_id].append((r.team, r.player_id))
    return out


# ---------------------------------------------------------------- continuation tables (league constants, rule 2d)
def cont_constants(seq, hist, st):
    """hist: the window's history rows (constants() filters label_clean == 1 itself; so do we). Positions from the starters file."""
    h = hist[hist.label_clean == 1]
    s = seq[seq.event_id.isin(h.event_id)]
    later = s[s.idx > 1]; cont = later[later.same_team_as_prev == 1]
    c = cont.merge(st[["event_id", "player_id", "pos"]].rename(columns={"pos": "spos"}), on=["event_id", "player_id"], how="left")
    c = c[c.spos.notna()]                                    # continuation attempts by non-starters are dropped (0.2%)
    league = {t: round(float((c.shot_type == t).mean()), 4) for t in TYPES}
    mix, mix_n = {}, {}
    for pos, g in c.groupby("spos"):
        mix_n[pos] = int(len(g))
        mix[pos] = {t: round(float((g.shot_type == t).mean()), 4) for t in TYPES} if len(g) >= POS_FLOOR else dict(league)
    return {"p_cont": round(float(len(cont) / len(later)), 4), "n_later": int(len(later)), "n_cont": int(len(cont)), "n_cont_starters": int(len(c)),
            "mix_league": league, "mix_pos": mix, "mix_pos_n": mix_n, "pos_floor": POS_FLOOR,
            "M_cont": {t: round(float(c[c.shot_type == t].made.mean()), 4) for t in TYPES}, "M_cont_n": {t: int((c.shot_type == t).sum()) for t in TYPES},
            "make_cont_overall": round(float(c.made.mean()), 4), "n_games": int(len(h)), "prior": PRIOR}


# ---------------------------------------------------------------- scoring (rule 2a-2b)
def chain(home, share, make, c, mc, fh0, pc, eps):
    """Four-state chain. Fresh attempts use (share, make), continuation attempts (c, mc); after a miss the same team continues with
    pc, else the other team gets a fresh attempt. With c = share, mc = make and pc = o this is exactly who-v1's chain.
    Returns (P(first basket) per starter, share of each starter's mass that came through continuation attempts)."""
    n = len(share); fh, fa, ch, ca = fh0, 1 - fh0, 0.0, 0.0
    fresh_h, fresh_a = float((share * make)[home].sum()), float((share * make)[~home].sum())
    cont_h, cont_a = float((c * mc)[home].sum()), float((c * mc)[~home].sum())
    acc = np.zeros(n); acc_cont = np.zeros(n)
    for _ in range(fbm.N_ATTEMPTS):
        fresh = np.where(home, fh, fa) * share * make; cont = np.where(home, ch, ca) * c * mc
        acc += fresh + cont; acc_cont += cont
        miss_h = fh * (1 - fresh_h) + ch * (1 - cont_h); miss_a = fa * (1 - fresh_a) + ca * (1 - cont_a)
        ch, ca = miss_h * pc, miss_a * pc
        fh, fa = miss_a * (1 - pc), miss_h * (1 - pc)
    tot = acc.sum()
    p = acc / tot * (1 - eps) if tot > 0 else np.full(n, (1 - eps) / n)
    return p, acc_cont / np.maximum(acc, 1e-12)


def score(G, F2, K, C, C2):
    """who-v2 P(first basket) per starter, aligned with G['starters']. Reuses who-v1's components (shares, make rates, tip) from
    first_basket_model.score_game and adds the continuation state. Returns who-v1's dict with 'who_v2' and the continuation pieces."""
    S = fbm.score_game(G, K, C)
    st = G["starters"]; n = len(st)
    home = np.array([s["home"] for s in st], dtype=bool)
    share, make = S["share"], S["make"]
    # continuation shooter distribution (2c)
    c_raw = np.array([fbm.shrink(f["k"], f["n"], f["base_cont"], PRIOR) for f in F2])
    c = np.zeros(n)
    for side in (True, False):
        m = home == side; tot = c_raw[m].sum()
        c[m] = c_raw[m] / tot if tot > 0 else (1 / m.sum() if m.sum() else 0)
    # continuation make rate by position (2d)
    Mc = np.array([C2["M_cont"][t] for t in TYPES])
    mc = np.array([float(np.array([C2["mix_pos"].get(s["pos"], C2["mix_league"])[t] for t in TYPES]) @ Mc) for s in st])
    # chain (2a): fresh / continuation masses per team
    fh0 = S["p_tip"] * C["r"] + (1 - S["p_tip"]) * (1 - C["r"])
    S["who_v2"], S["who_v2_cont_share"] = chain(home, share, make, c, mc, fh0, C2["p_cont"], C["eps"])
    S["c"] = c; S["mc"] = mc
    return S


# ---------------------------------------------------------------- walk-forward with both tallies
def walk(games, st, seq, score_seasons, through=None):
    """As first_basket_model.walk, carrying the continuation tallies alongside. Returns [(G, F2)] records and both tallies."""
    st_by_ev = {e: list(g.itertuples()) for e, g in st.groupby("event_id")}
    cont_by_ev = cont_rows_by_event(seq)
    T, T2, records = fbm.Tallies(), ContTallies(), []
    for date, day in games.groupby("date", sort=True):
        if through is not None and date > through:
            break
        rows = list(day.itertuples())
        for g in rows:
            s = st_by_ev.get(g.event_id, [])
            if g.season in score_seasons and g.phase == "regular":
                teams = {x.team for x in s}
                if len(s) == 10 and teams == {g.home, g.away} and sum(x.team == g.home for x in s) == 5:
                    records.append((T.features(g, s, pre_season=g.season), T2.features(s)))
        for g in rows:
            s = st_by_ev.get(g.event_id, [])
            T.update(g, s); T2.update(g, s, cont_by_ev.get(g.event_id, []))
    return records, T, T2


def loss_of(p, G, eps):
    ids = [s["player_id"] for s in G["starters"]]
    if G["scorer"] in ids:
        j = ids.index(G["scorer"]); return -np.log(max(float(p[j]), fbm.P_FLOOR)), j
    return -np.log(max(eps, fbm.P_FLOOR)), None


def rule_commit():
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", RULE], capture_output=True, text=True, cwd=DATA_DIR.parent).stdout.strip() or None
    except Exception:
        return None


def player_line(pid, name, per_game):
    """Over the evaluation games the player started: starts, first FGs, actual rate, mean P under each model."""
    g = [x for x in per_game if pid in x["ids"]]
    if not g:
        return {"player_id": pid, "player": name, "starts": 0}
    hits = sum(x["scorer"] == pid for x in g)
    mean = lambda key: round(float(np.mean([x[key][x["ids"].index(pid)] for x in g])), 4)
    out = {"player_id": pid, "player": name, "starts": len(g), "first_fgs": int(hits), "actual_rate": round(hits / len(g), 4), "mean_p_who_v1": mean("p_v1"), "mean_p_who_v2": mean("p_v2"), "mean_p_makes_only": mean("p_m")}
    out["abs_err_who_v1"] = round(abs(out["mean_p_who_v1"] - out["actual_rate"]), 4); out["abs_err_who_v2"] = round(abs(out["mean_p_who_v2"] - out["actual_rate"]), 4)
    return out


def preship():
    games, seq, st = fbm.load_tables()
    params = read_json(fbm.PARAMS, None)
    if not params:
        raise SystemExit("[who-v2] no frozen who-v1 params")
    K = params["K"]
    fit_hist = games[(games.season == FIT_SEASON) & (games.phase != "preseason")]
    C = fbm.constants(seq, fit_hist)                       # who-v1 re-derived on the same fit (number for number the gate's fit block)
    C2 = cont_constants(seq, fit_hist, st)
    if abs(C2["p_cont"] - P_CONT_EXPECT) > P_CONT_FAR:
        raise SystemExit(f"[who-v2] SANITY STOP: fit-window p_cont {C2['p_cont']} is far from {P_CONT_EXPECT}; nothing scored (rule 3)")
    recs, _, _ = walk(games, st, seq, {EVAL_SEASON}, through=f"{EVAL_SEASON}-07-31")
    per_game = []
    for G, F2 in recs:
        if G["scorer"] is None:
            continue
        S = score(G, F2, K, C, C2)
        p_v1, p_v2, p_m, p_u = S["v2"], S["who_v2"], S["v1"], S["uniform"]
        l1, j = loss_of(p_v1, G, C["eps"]); l2, _ = loss_of(p_v2, G, C["eps"]); lm, _ = loss_of(p_m, G, C["eps"]); lu, _ = loss_of(p_u, G, C["eps"])
        top = lambda p, k: int(j is not None and j in np.argsort(-p)[:k])
        per_game.append({"event_id": G["event_id"], "date": G["date"], "ids": [s["player_id"] for s in G["starters"]], "scorer": G["scorer"], "p_v1": p_v1, "p_v2": p_v2, "p_m": p_m,
                         "l_v1": l1, "l_v2": l2, "l_m": lm, "l_u": lu, "top1_v1": top(p_v1, 1), "top1_v2": top(p_v2, 1), "top3_v1": top(p_v1, 3), "top3_v2": top(p_v2, 3)})
    L1 = np.array([x["l_v1"] for x in per_game]); L2 = np.array([x["l_v2"] for x in per_game]); Lm = np.array([x["l_m"] for x in per_game]); Lu = np.array([x["l_u"] for x in per_game])
    d = L1 - L2                                             # positive = who-v2 better
    rng = np.random.default_rng(SEED); n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    ci = [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)]
    gain = float(d.mean())
    bar1 = bool(gain >= BAR1_GAIN and ci[0] > 0)
    names = {s.player_id: s.player for s in st.drop_duplicates("player_id").itertuples()}
    cls = [player_line(pid, CLASS[pid], per_game) for pid in CLASS]
    e1 = [x["abs_err_who_v1"] for x in cls if x.get("starts")]; e2 = [x["abs_err_who_v2"] for x in cls if x.get("starts")]
    bar2 = bool(e1 and float(np.mean(e2)) < float(np.mean(e1)))
    ro = dict(REPORT_ONLY)
    wemby = [pid for pid, nm in names.items() if nm == "Victor Wembanyama"]
    if wemby:
        ro = {pid: nm for pid, nm in ro.items() if nm != "Victor Wembanyama"}; ro[wemby[0]] = "Victor Wembanyama"
    report_only = [player_line(pid, nm, per_game) for pid, nm in ro.items()]
    ship = bar1 and bar2
    doc = {"title": "First basket WHO-model v2: offensive-rebound continuation, pre-ship check", "rule_file": RULE, "rule_commit": rule_commit(), "generated": fbm.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "fit": {"season": "2024-25", "who_v1_K": K, "who_v1_constants": C, "continuation": C2,
                   "note": "who-v1's K are the frozen params; its constants are re-derived on the 2024-25 window (identical to the gate's fit block). Continuation tables on the same window; shooter tallies walk forward as who-v1's shares do."},
           "eval": {"season": "2025-26", "games": int(n), "set": "gate harness: 2025-26 regular-season games with a gradable first made field goal and ten box-score starters",
                    "log_loss": {"who_v1": round(float(L1.mean()), 5), "who_v2": round(float(L2.mean()), 5), "makes_only": round(float(Lm.mean()), 5), "uniform": round(float(Lu.mean()), 5)},
                    "top1": {"who_v1": round(float(np.mean([x["top1_v1"] for x in per_game])), 4), "who_v2": round(float(np.mean([x["top1_v2"] for x in per_game])), 4)},
                    "top3": {"who_v1": round(float(np.mean([x["top3_v1"] for x in per_game])), 4), "who_v2": round(float(np.mean([x["top3_v2"] for x in per_game])), 4)},
                    "top_pick_changed_games": int(sum(int(np.argmax(x["p_v1"]) != np.argmax(x["p_v2"])) for x in per_game))},
           "bar1": {"gain_nats": round(gain, 5), "threshold": BAR1_GAIN, "ci95_game_bootstrap": ci, "resamples": BOOT, "seed": SEED, "pass": bar1,
                    "text": f"who-v2 beats who-v1 by {gain:.4f} nats/game (needs >= {BAR1_GAIN}), 95% CI {ci} (must exclude 0)"},
           "bar2": {"class": cls, "class_definition": ">= 10 FG first baskets in 2024-25 with putback/tip share >= 10% (first_basket_history play type), pre-specified from fit data",
                    "mean_abs_err_who_v1": round(float(np.mean(e1)), 4) if e1 else None, "mean_abs_err_who_v2": round(float(np.mean(e2)), 4) if e2 else None, "pass": bar2,
                    "text": "class mean |mean P - actual rate per start| must be strictly smaller under who-v2"},
           "report_only": report_only,
           "decision": "SHIP" if ship else "FAIL", "decision_text": ("Both bars pass: who-v2 goes live at the next pre-tip snapshot after `ship` writes data/first_basket_who_v2_params.json; grade rows carry a who-v1 / who-v2 era marker; who-v1's record stands unedited."
                                                                   if ship else "At least one bar failed: who-v1 stays live; these numbers are the record; an override is the CEO's explicit call only. No parameter change, no rerun under this rule.")}
    write_json(PRESHIP, doc)
    e = doc["eval"]
    print(f"[who-v2] fit 2024-25: p_cont {C2['p_cont']} ({C2['n_cont']}/{C2['n_later']}), mix {C2['mix_league']}, M_cont {C2['M_cont']}, make overall {C2['make_cont_overall']}")
    print(f"[who-v2] eval 2025-26 ({n} games): log loss who-v1 {e['log_loss']['who_v1']} who-v2 {e['log_loss']['who_v2']} makes-only {e['log_loss']['makes_only']} uniform {e['log_loss']['uniform']}; top-1 {e['top1']}, top-3 {e['top3']}")
    print(f"[who-v2] BAR 1 {'PASS' if bar1 else 'FAIL'}: {doc['bar1']['text']}")
    print(f"[who-v2] BAR 2 {'PASS' if bar2 else 'FAIL'}: class mean abs err who-v1 {doc['bar2']['mean_abs_err_who_v1']} vs who-v2 {doc['bar2']['mean_abs_err_who_v2']}; {cls}")
    print(f"[who-v2] report-only: {report_only}")
    print(f"[who-v2] decision {doc['decision']}")
    return doc


# ---------------------------------------------------------------- live (rule 4)
def ship():
    """Write the live parameter file once, from the check's fit block, only on a SHIP verdict."""
    d = read_json(PRESHIP, None)
    if not d or d.get("decision") != "SHIP":
        raise SystemExit(f"[who-v2] no SHIP verdict in {PRESHIP.name}: nothing written")
    if PARAMS2.exists():
        print(f"[who-v2] {PARAMS2.name} already written: not touched"); return
    write_json(PARAMS2, {"rule_file": RULE, "rule_commit": d["rule_commit"], "preship_generated": d["generated"], "decision": d["decision"], "live_from": fbm.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "continuation": d["fit"]["continuation"], "who_v1_params": "data/first_basket_params.json (unchanged)",
                         "note": "written once by `first_basket_who_v2.py ship`; the continuation constants are the check's 2024-25 fit, never re-estimated under this rule"})
    print(f"[who-v2] live from {read_json(PARAMS2, {})['live_from']}")


def live():
    """(params2, C2) when who-v2 is live, else None. first_basket_model.snapshot calls this."""
    p = read_json(PARAMS2, None)
    return (p, p["continuation"]) if p and p.get("decision") == "SHIP" else None


def era(logged_at, live_from):
    """Era marker for a grade row (rule 4)."""
    return "who-v2" if live_from and str(logged_at) >= live_from else "who-v1"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preship"
    {"preship": preship, "ship": ship}[cmd]()
