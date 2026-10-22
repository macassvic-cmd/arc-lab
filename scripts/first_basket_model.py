"""First basket v2: attempt-based model, validation gate and live log (backtests/first_basket_v2_rule.md).

  python first_basket_model.py gate       walk-forward 2024-25 (constant search, section 2f) and 2025-26 (grade, section 4) ->
                                          docs/data/first_basket_gate.json + data/first_basket_params.json (frozen constants)
  python first_basket_model.py freeze     write data/first_basket_params.json from the gate's published fit block, as-is (2026-10-21
                                          override: the gate FAILED and stays FAILED; the model ships on the CEO's order, no refit)
  python first_basket_model.py snapshot   append tonight's ten projected starters per untipped game to data/first_basket_log.csv
                                          and write docs/data/first_basket_today.json (tonight's projections for the Performance page)
  python first_basket_model.py grade      grade the last pre-tip snapshot of every logged game -> docs/data/first_basket_performance.json
  python first_basket_model.py today      rebuild docs/data/first_basket_today.json from the log (last pre-tip snapshot per game of the
                                          latest logged date); no new rows

Target: the game's first made field goal (free throws excluded). Inputs are the committed label tables from first_shot.py; no raw
play store is needed here. Every quantity for a game uses only games dated before it (walk-forward); the live model is the same
code with the frozen constants, tallies through yesterday and the current preseason as sample (c).
"""
import csv
import datetime as dt
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json
from first_shot import HIST, PRE, SEQ, STARTERS

LOG = DATA_DIR / "first_basket_log.csv"
PARAMS = DATA_DIR / "first_basket_params.json"
GATE = SITE_DATA / "first_basket_gate.json"
PERF = SITE_DATA / "first_basket_performance.json"
TODAY = SITE_DATA / "first_basket_today.json"
WHY = DATA_DIR / "first_basket_why.jsonl"   # append-only sidecar: the why-lines and tip probability of every snapshot (not log columns)
OVERRIDE = ("Shipped 2026-10-21 by CEO override after the locked gate FAILED (docs/data/first_basket_gate.json, Backtest page). "
            "Constants are the gate's published 2024-25 fit, unchanged; the gate verdict is unchanged.")
RULE = "backtests/first_basket_v2_rule.md"
LOG_FIELDS = ["date", "event_id", "tip", "team", "opp", "home", "player_id", "player", "pos", "p_first_basket", "p_first_attempt", "top_pick", "jumper", "logged_at"]
TYPES = ("rim", "mid", "3")
START_K = {"K_tip": 20, "K_h2h": 5, "K_a": 20, "K_b": 5, "K_c": 2}
GRID = {"K_tip": [5, 10, 20, 40], "K_h2h": [2, 5, 10], "K_a": [5, 10, 20, 40], "K_b": [2, 5, 10, 20], "K_c": [1, 2, 4, 8]}
K_TYPE, N_ATTEMPTS, K_V1, P_FLOOR, SEED, BOOT = 10, 8, 20, 0.005, 20261021, 1000
THIN = 30


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------- tallies (walk-forward state)
class Tallies:
    """Everything the model knows up to a date. Regular-season and postseason games feed (a)/(b), the type mix, the position
    baselines and the v1 makes; preseason games feed only (c) and the jump-ball record (rule 2a: 'labelled games to date')."""

    def __init__(self):
        self.jump = defaultdict(lambda: [0, 0])            # jumper id -> [wins, jumps]
        self.h2h = defaultdict(lambda: [0, 0])             # (jumper A, jumper B) sorted -> [wins of the first, jumps]
        self.team_jumps = defaultdict(lambda: defaultdict(int))   # team -> jumper id -> jumps
        self.n = defaultdict(int); self.f = defaultdict(int)      # player -> games started / games taking the team's first attempt
        self.n_opp = defaultdict(int); self.f_opp = defaultdict(int)   # (player, opp)
        self.n_pre = defaultdict(int); self.f_pre = defaultdict(int)   # (player, preseason season)
        self.types = defaultdict(lambda: {t: 0 for t in TYPES})        # player -> first-attempt type counts
        self.pos_types = defaultdict(lambda: {t: 0 for t in TYPES})    # position -> first-attempt type counts
        self.pos_n = defaultdict(int); self.pos_f = defaultdict(int)    # position -> starter-games / first attempts (base_pos)
        self.g = defaultdict(int); self.pos_g = defaultdict(int)        # v1: player / position -> first made field goals

    def update(self, game, starters):
        """game: a first_shot_history / preseason row; starters: its starter rows (team, player_id, pos)."""
        pre = game.phase == "preseason"
        jh, ja = game.jump_home_id, game.jump_away_id
        if isinstance(jh, str) and isinstance(ja, str) and isinstance(game.jump_gainer_team, str):
            home_won = game.jump_gainer_team == game.home
            self.jump[jh][1] += 1; self.jump[ja][1] += 1; self.jump[jh][0] += home_won; self.jump[ja][0] += (not home_won)
            key = tuple(sorted((jh, ja))); self.h2h[key][1] += 1; self.h2h[key][0] += (home_won if key[0] == jh else not home_won)
            self.team_jumps[game.home][jh] += 1; self.team_jumps[game.away][ja] += 1
        for side in ("home", "away"):
            team = getattr(game, side); fga = getattr(game, f"{side}_fga_id"); ftype = getattr(game, f"{side}_fga_type")
            if not isinstance(fga, str):
                continue   # the team's first attempt is unknown: no share information from this game
            opp = game.away if side == "home" else game.home
            for s in starters:
                if s.team != team:
                    continue
                pid = s.player_id
                if pre:
                    k = (pid, game.season); self.n_pre[k] += 1; self.f_pre[k] += (pid == fga)
                    continue
                self.n[pid] += 1; self.f[pid] += (pid == fga); self.n_opp[(pid, opp)] += 1; self.f_opp[(pid, opp)] += (pid == fga)
                self.pos_n[s.pos] += 1; self.pos_f[s.pos] += (pid == fga)
                if pid == fga and ftype in TYPES:
                    self.types[pid][ftype] += 1; self.pos_types[s.pos][ftype] += 1
                if game.label_clean == 1:
                    self.g[pid] += (pid == game.first_fg_id); self.pos_g[s.pos] += (pid == game.first_fg_id)

    def jumper(self, team, starters):
        """The starter who has jumped most for the team; ties: the C, then the first listed starter."""
        tj = self.team_jumps.get(team, {})
        best = max(starters, key=lambda s: (tj.get(s.player_id, 0), s.pos == "C"), default=None)
        return best.player_id if best is not None else None

    def features(self, game, starters, pre_season):
        """Raw counts per starter and for the tip, frozen at scoring time so constants can be searched without re-walking."""
        home_st = [s for s in starters if s.team == game.home]; away_st = [s for s in starters if s.team == game.away]
        jh, ja = self.jumper(game.home, home_st), self.jumper(game.away, away_st)
        key = tuple(sorted((jh, ja))) if jh and ja else None
        h = self.h2h.get(key, [0, 0]) if key else [0, 0]
        tip = {"home": list(self.jump.get(jh, [0, 0])), "away": list(self.jump.get(ja, [0, 0])), "h2h_home_wins": (h[0] if key and key[0] == jh else h[1] - h[0]) if key else 0, "h2h_n": h[1] if key else 0,
               "jumper_home": jh, "jumper_away": ja}
        rows = []
        for s in home_st + away_st:
            pid, opp = s.player_id, (game.away if s.team == game.home else game.home)
            pn, pf = self.pos_n.get(s.pos, 0), self.pos_f.get(s.pos, 0)
            rows.append({"player_id": pid, "player": s.player, "team": s.team, "pos": s.pos, "home": int(s.team == game.home),
                         "n": self.n.get(pid, 0), "f": self.f.get(pid, 0), "n_opp": self.n_opp.get((pid, opp), 0), "f_opp": self.f_opp.get((pid, opp), 0),
                         "n_pre": self.n_pre.get((pid, pre_season), 0), "f_pre": self.f_pre.get((pid, pre_season), 0),
                         "base_pos": pf / pn if pn else 0.2, "types": dict(self.types.get(pid, {t: 0 for t in TYPES})), "pos_types": dict(self.pos_types.get(s.pos, {t: 0 for t in TYPES})),
                         "g": self.g.get(pid, 0), "base_pos_g": (self.pos_g.get(s.pos, 0) / pn) if pn else 0.2, "jumper": int(pid in (jh, ja))})
        return {"event_id": game.event_id, "date": game.date, "season": game.season, "home": game.home, "away": game.away, "tip": tip, "starters": rows,
                "scorer": game.first_fg_id if game.label_clean == 1 else None, "scorer_started": int(game.first_fg_started) if game.label_clean == 1 and pd.notna(game.first_fg_started) else None}


# ---------------------------------------------------------------- league constants (rule 2d / section 3)
def constants(seq, hist):
    """From the sequence file and the history rows of the chosen seasons (regular season + postseason, gradable games)."""
    h = hist[hist.label_clean == 1]
    s = seq[seq.event_id.isin(h.event_id)]
    first = s[s.idx == 1].merge(h[["event_id", "jump_gainer_team"]], on="event_id")
    first = first[first.jump_gainer_team.notna()]
    hk = h[h.jump_gainer_team.notna()]
    later = s[s.idx > 1]
    return {"r": round(float((first.team == first.jump_gainer_team).mean()), 4), "o": round(float(later.same_team_as_prev.mean()), 4),
            "M": {t: round(float(s[s.shot_type == t].made.mean()), 4) for t in TYPES},
            "eps": round(float((h.first_fg_started.fillna(0) == 0).mean()), 4), "r1": round(float((hk.first_fg_team == hk.jump_gainer_team).mean()), 4),
            "n_games": int(len(h)), "n_attempts": int(len(s))}


# ---------------------------------------------------------------- scoring
def shrink(f, n, prior, k):
    return (f + k * prior) / (n + k)


def tip_prob(tip, K):
    wa = shrink(tip["home"][0], tip["home"][1], 0.5, K["K_tip"]) if tip["home"][1] else 0.5
    wb = shrink(tip["away"][0], tip["away"][1], 0.5, K["K_tip"]) if tip["away"][1] else 0.5
    p0 = wa * (1 - wb) / (wa * (1 - wb) + wb * (1 - wa))
    return shrink(tip["h2h_home_wins"], tip["h2h_n"], p0, K["K_h2h"])


def score_game(G, K, C):
    """P(first basket) and P(first attempt) per starter for v2, plus v1 and uniform. Returns dict of arrays aligned with G['starters']."""
    st = G["starters"]; n = len(st)
    home = np.array([s["home"] for s in st], dtype=bool)
    p_tip = tip_prob(G["tip"], K)
    # 2b attempt shares
    sa = np.array([shrink(s["f"], s["n"], s["base_pos"], K["K_a"]) for s in st])
    sb = np.array([shrink(s["f_opp"], s["n_opp"], sa[i], K["K_b"]) for i, s in enumerate(st)])
    sc = np.array([shrink(s["f_pre"], s["n_pre"], sb[i], K["K_c"]) for i, s in enumerate(st)])
    share = np.zeros(n)
    for side in (True, False):
        m = home == side; tot = sc[m].sum()
        share[m] = sc[m] / tot if tot > 0 else (1 / m.sum() if m.sum() else 0)
    # 2c make rates
    M = np.array([C["M"][t] for t in TYPES])
    make = np.zeros(n)
    for i, s in enumerate(st):
        own = np.array([s["types"][t] for t in TYPES], dtype=float); pos = np.array([s["pos_types"][t] for t in TYPES], dtype=float)
        pos_mix = pos / pos.sum() if pos.sum() else np.array([0.4, 0.3, 0.3])
        mix = (own + K_TYPE * pos_mix) / (own.sum() + K_TYPE)
        make[i] = float(mix @ M)
    # 2d chain
    r, o, eps = C["r"], C["o"], C["eps"]
    ph = p_tip * r + (1 - p_tip) * (1 - r); pa = 1 - ph
    p_attempt = np.where(home, ph, pa) * share
    acc = np.zeros(n)
    mh = float((share * make)[home].sum()); ma = float((share * make)[~home].sum())
    for _ in range(N_ATTEMPTS):
        acc += np.where(home, ph, pa) * share * make
        miss_h, miss_a = ph * (1 - mh), pa * (1 - ma)
        ph, pa = miss_h * o + miss_a * (1 - o), miss_a * o + miss_h * (1 - o)
    p_v2 = acc / acc.sum() * (1 - eps) if acc.sum() > 0 else np.full(n, (1 - eps) / n)
    # section 3: v1 makes-only
    r1 = C["r1"]; th = p_tip * r1 + (1 - p_tip) * (1 - r1)
    g = np.array([shrink(s["g"], s["n"], s["base_pos_g"], K_V1) for s in st]); gsh = np.zeros(n)
    for side in (True, False):
        m = home == side; tot = g[m].sum()
        gsh[m] = g[m] / tot if tot > 0 else (1 / m.sum() if m.sum() else 0)
    p_v1 = np.where(home, th, 1 - th) * gsh * (1 - eps)
    return {"v2": p_v2, "v1": p_v1, "uniform": np.full(n, (1 - eps) / n), "p_attempt": p_attempt, "p_tip": p_tip, "share": share, "make": make}


def game_losses(G, P, eps):
    """Per-model log loss for the actual scorer, plus top-1 / top-3 hits (bench scorer = eps for every model, a miss for ranks)."""
    ids = [s["player_id"] for s in G["starters"]]
    out = {}
    for name in ("v2", "v1", "uniform"):
        p = P[name]
        if G["scorer"] in ids:
            j = ids.index(G["scorer"]); loss = -np.log(max(float(p[j]), P_FLOOR))
            order = np.argsort(-p); top1 = int(order[0] == j); top3 = int(j in order[:3])
        else:
            loss = -np.log(max(eps, P_FLOOR)); top1 = top3 = 0
        out[name] = (loss, top1, top3)
    return out


# ---------------------------------------------------------------- walk-forward
def load_tables():
    hist = pd.read_csv(HIST, dtype={"event_id": str, "jump_home_id": str, "jump_away_id": str, "jump_gainer_id": str, "home_fga_id": str, "away_fga_id": str, "first_fga_id": str, "first_fg_id": str})
    pre = pd.read_csv(PRE, dtype=hist.dtypes.to_dict()) if PRE.exists() else hist.iloc[0:0]
    seq = pd.read_csv(SEQ, dtype={"event_id": str, "player_id": str})
    st = pd.read_csv(STARTERS, dtype={"event_id": str, "player_id": str})
    games = pd.concat([hist, pre]).sort_values(["date", "event_id"]).reset_index(drop=True)
    return games, seq, st


def walk(games, starters_df, score_seasons, through=None):
    """Walk every labelled game in date order; return feature records for the regular-season games of score_seasons (built
    before their date's games are tallied) and the final tallies. Games of a date are all scored before any is tallied."""
    st_by_ev = {e: list(g.itertuples()) for e, g in starters_df.groupby("event_id")}
    T, records = Tallies(), []
    for date, day in games.groupby("date", sort=True):
        if through is not None and date > through:
            break
        rows = list(day.itertuples())
        for g in rows:
            st = st_by_ev.get(g.event_id, [])
            if g.season in score_seasons and g.phase == "regular":
                teams = {s.team for s in st}
                if len(st) == 10 and teams == {g.home, g.away} and sum(s.team == g.home for s in st) == 5:
                    records.append(T.features(g, st, pre_season=g.season))
        for g in rows:
            T.update(g, st_by_ev.get(g.event_id, []))
    return records, T


def evaluate(records, K, C):
    L = {m: [] for m in ("v2", "v1", "uniform")}
    for G in records:
        if G["scorer"] is None:
            continue
        P = score_game(G, K, C)
        for m, v in game_losses(G, P, C["eps"]).items():
            L[m].append(v)
    return {m: np.array(v) for m, v in L.items()}


def summary(arr):
    return {"n": int(len(arr)), "log_loss": round(float(arr[:, 0].mean()), 5), "top1": round(float(arr[:, 1].mean()), 4), "top3": round(float(arr[:, 2].mean()), 4)}


def paired_ci(a, b):
    """95% bootstrap CI of mean(b - a) over games (a = v2 loss, b = comparator loss); positive = v2 better."""
    rng = np.random.default_rng(SEED); d = b - a; n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    return [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)]


def search(records, C):
    K = dict(START_K); best = evaluate(records, K, C)["v2"][:, 0].mean(); trail = []
    for p in range(2):
        for name, values in GRID.items():
            for v in values:
                trial = {**K, name: v}
                ll = evaluate(records, trial, C)["v2"][:, 0].mean()
                trail.append({"pass": p + 1, "param": name, "value": v, "log_loss": round(float(ll), 5)})
                if ll < best - 1e-9:
                    best, K = ll, trial
    return K, float(best), trail


def gate():
    games, seq, st = load_tables()
    cov = read_json(SITE_DATA / "first_shot_coverage.json", {})
    fit_hist = games[(games.season == 2025) & (games.phase != "preseason")]
    C = constants(seq, fit_hist)
    rec_fit, _ = walk(games, st, {2025}, through="2025-06-30")
    K, fit_ll, trail = search(rec_fit, C)
    fit = {m: summary(v) for m, v in evaluate(rec_fit, K, C).items()}
    rec_grade, _ = walk(games, st, {2026})
    L = evaluate(rec_grade, K, C)
    grade = {m: summary(v) for m, v in L.items()}
    ci_u, ci_1 = paired_ci(L["v2"][:, 0], L["uniform"][:, 0]), paired_ci(L["v2"][:, 0], L["v1"][:, 0])
    beats_u = grade["v2"]["log_loss"] < grade["uniform"]["log_loss"] and ci_u[0] > 0
    beats_1 = grade["v2"]["log_loss"] < grade["v1"]["log_loss"] and ci_1[0] > 0
    ship = bool(beats_u and beats_1)
    rule_commit = None
    try:
        import subprocess
        rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", RULE], capture_output=True, text=True, cwd=DATA_DIR.parent).stdout.strip() or None
    except Exception:
        pass
    doc = {"title": "First basket v2: attempt-based model gate", "rule_file": RULE, "rule_commit": rule_commit, "generated": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "target": "first made field goal of the game (free throws excluded)",
           "fit": {"season": "2024-25", "games_scored": len([r for r in rec_fit if r["scorer"]]), "constants": C, "K": K, "search_trail": trail, "models": fit},
           "grade": {"season": "2025-26", "games_scored": len([r for r in rec_grade if r["scorer"]]), "models": grade,
                     "v2_minus_uniform_ci": ci_u, "v2_minus_v1_ci": ci_1, "beats_uniform": bool(beats_u), "beats_v1": bool(beats_1)},
           "decision": "SHIP" if ship else "FAIL",
           "decision_text": ("v2 beats both the uniform 1/10 model and the v1 makes-only model on 2025-26 log loss with paired-bootstrap 95% CIs excluding 0: the live log and Performance section go live."
                             if ship else "v2 did not beat both comparators with the CI excluding 0: no live log, no page section, no retry (rule section 4)."),
           "coverage": cov.get("blocks", [])}
    write_json(GATE, doc)
    if ship:
        live_hist = games[(games.season.isin([2025, 2026])) & (games.phase != "preseason")]
        write_json(PARAMS, {"rule_file": RULE, "rule_commit": rule_commit, "frozen": doc["generated"], "K": K, "constants": constants(seq, live_hist),
                            "constants_note": "chain terms re-estimated on the two complete seasons (2024-25 and 2025-26, regular season + postseason) for the live model, frozen for 2026-27; the gate used 2024-25 only"})
    print(f"[first-basket] fit 2024-25: K={K}, v2 {fit['v2']} v1 {fit['v1']} uniform {fit['uniform']}")
    print(f"[first-basket] grade 2025-26: v2 {grade['v2']} v1 {grade['v1']} uniform {grade['uniform']}; CI vs uniform {ci_u}, vs v1 {ci_1} -> {doc['decision']}")
    return doc


# ---------------------------------------------------------------- live
def freeze():
    """CEO override (2026-10-21): write the frozen-parameter file from the fit block already published in the gate JSON, number
    for number. No refit, no constant change, no edit to the gate JSON. Refuses to overwrite an existing file."""
    if PARAMS.exists():
        print(f"[first-basket] {PARAMS.name} already frozen: not touched"); return
    g = read_json(GATE, None)
    if not g or g.get("decision") is None:
        raise SystemExit("[first-basket] no gate JSON to freeze from")
    write_json(PARAMS, {"rule_file": RULE, "rule_commit": g["rule_commit"], "gate_decision": g["decision"], "gate_generated": g["generated"],
                        "frozen": now().strftime("%Y-%m-%dT%H:%M:%SZ"), "K": g["fit"]["K"], "constants": g["fit"]["constants"],
                        "constants_note": "copied as-is from docs/data/first_basket_gate.json fit block (2024-25 chain terms); frozen for 2026-27",
                        "override": OVERRIDE})
    print(f"[first-basket] frozen K={g['fit']['K']} constants={g['fit']['constants']} (gate decision {g['decision']})")


def projected_starters(P, box):
    """Per team on tonight's slate: the five active players with the highest start share over their last five games, projected
    minutes breaking ties (rule section 5)."""
    recent = box.sort_values("date", ascending=False).groupby("player_id").head(5).groupby("player_id").starter.mean()
    out = defaultdict(list)
    for p in P["players"]:
        if p.get("inactive"):
            continue
        out[p["team"]].append((float(recent.get(p["id"], 0.0)), float(p.get("min") or 0), p))
    return {team: [x[2] for x in sorted(lst, key=lambda x: (-x[0], -x[1]))[:5]] for team, lst in out.items()}


class Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def team_first_shooters(games, st, seasons):
    """team -> {player_id: (name, games in which he started and took the team's first attempt)} over the given seasons."""
    started = set(zip(st.event_id, st.player_id))
    out = defaultdict(lambda: defaultdict(lambda: [None, 0])); team_games = defaultdict(int)
    for g in games[games.season.isin(seasons)].itertuples():
        for side in ("home", "away"):
            team, pid, name = getattr(g, side), getattr(g, f"{side}_fga_id"), getattr(g, f"{side}_fga_player")
            if isinstance(pid, str):
                team_games[team] += 1
                if (g.event_id, pid) in started:
                    out[team][pid][0] = name; out[team][pid][1] += 1
    return {t: {pid: (nm, c, c / team_games[t]) for pid, (nm, c) in d.items()} for t, d in out.items()}


TEAM_NAMES = {"ATL": "Hawks", "BKN": "Nets", "BOS": "Celtics", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
              "DET": "Pistons", "GS": "Warriors", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers", "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies",
              "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves", "NO": "Pelicans", "NOP": "Pelicans", "NY": "Knicks", "NYK": "Knicks", "OKC": "Thunder",
              "ORL": "Magic", "PHI": "76ers", "PHX": "Suns", "PHO": "Suns", "POR": "Trail Blazers", "SA": "Spurs", "SAS": "Spurs", "SAC": "Kings",
              "TOR": "Raptors", "UTAH": "Jazz", "UTA": "Jazz", "WSH": "Wizards", "WAS": "Wizards"}


def tname(abbr):
    return TEAM_NAMES.get(abbr, abbr)


def poss(abbr):
    nm = tname(abbr)
    return nm + ("'" if nm.endswith("s") else "'s")


def join_names(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def player_team_counts(games, st, pre_season):
    """(player, team) -> [starter games with a known team first attempt, first attempts taken], split into regular+postseason
    (all labelled seasons) and the current preseason. Same counting as Tallies.update, kept per team for attribution."""
    st_by_ev = {e: list(zip(g.team, g.player_id)) for e, g in st.groupby("event_id")}
    reg, pre = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for g in games.itertuples():
        is_pre = g.phase == "preseason"
        if is_pre and g.season != pre_season:
            continue
        for side in ("home", "away"):
            team, fga = getattr(g, side), getattr(g, f"{side}_fga_id")
            if not isinstance(fga, str):
                continue
            for tm, pid in st_by_ev.get(g.event_id, []):
                if tm == team:
                    d = (pre if is_pre else reg)[(pid, team)]; d[0] += 1; d[1] += (pid == fga)
    return reg, pre


def why_lines(G, S, K, top, shooters, tenure):
    """2-3 lines on what moved the top pick's number, each computed from the components score_game used: tip (2a), attempt-share
    samples (2b), a departed first-shot taker whose share redistributed (2b normalisation), an extreme shot mix (2c). Nothing is
    asserted that the counts do not show; with no standout driver the fallback says so."""
    st = G["starters"]; s = st[top]; team = s["team"]; home = bool(s["home"]); opp = G["away"] if home else G["home"]
    names = {x["player_id"]: x["player"] for x in st}
    tip_l, role_l, mix_l = None, None, None
    # tip
    tip = G["tip"]; p_team = float(S["p_tip"]) if home else 1 - float(S["p_tip"])
    my_j, opp_j = (tip["jumper_home"], tip["jumper_away"]) if home else (tip["jumper_away"], tip["jumper_home"])
    n_h2h = tip["h2h_n"]; w_h2h = tip["h2h_home_wins"] if home else n_h2h - tip["h2h_home_wins"]
    rec = tip["home" if home else "away"]
    if p_team >= 0.58:
        if n_h2h >= 3 and w_h2h / n_h2h >= 0.75:
            tip_l = f"{names.get(my_j, 'Jumper')} {w_h2h}-{n_h2h - w_h2h} vs {names.get(opp_j, 'the opposing jumper')} on tips; {tname(team)} first possession ~{p_team:.0%}."
        elif rec[1]:
            tip_l = f"{tname(team)} first possession ~{p_team:.0%} ({names.get(my_j, 'jumper')} {rec[0]}-{rec[1] - rec[0]} on tips in the sample)."
    # attempt share: raw counts split by team tenure (the % printed is always the ratio of the counts beside it); the model's
    # shrunk share is labelled separately
    reg, pre = tenure; pid = s["player_id"]
    here = reg.get((pid, team), [0, 0]); others = {t: v for (p_, t), v in reg.items() if p_ == pid and t != team and v[0]}
    n_oth, f_oth = sum(v[0] for v in others.values()), sum(v[1] for v in others.values())
    pre_here = pre.get((pid, team), [0, 0]); pre_others = {t: v for (p_, t), v in pre.items() if p_ == pid and t != team and v[0]}
    sa = shrink(s["f"], s["n"], s["base_pos"], K["K_a"]); sb = shrink(s["f_opp"], s["n_opp"], sa, K["K_b"])
    share = float(S["share"][top]); movers = []
    if here[0]:
        base = f"Took ~{here[1] / here[0]:.0%} of the {poss(team)} first shots as a starter since 2024-25 ({here[1]} of {here[0]})"
        if n_oth:
            base += f"; {f_oth} of {n_oth} for the {join_names(tname(t) for t in others)} before that"
    elif n_oth:
        base = f"Took ~{f_oth / n_oth:.0%} of his teams' first shots as a starter since 2024-25 ({f_oth} of {n_oth}, for the {join_names(tname(t) for t in others)}; no regular-season games for the {tname(team)} yet)"
    else:
        base = f"No regular-season starts in the sample ({s['pos']} baseline {s['base_pos']:.0%})"
    if s["n_opp"] and abs(sb - sa) >= 0.02:
        movers.append(f"{s['f_opp']} of {s['n_opp']} vs the {tname(opp)}" + (" (all teams)" if n_oth else ""))
    if pre_here[0]:
        movers.append(f"{pre_here[1]} of {pre_here[0]} for the {tname(team)} this preseason")
    for t, v in pre_others.items():
        movers.append(f"{v[1]} of {v[0]} for the {tname(t)} this preseason")
    share_l = base + ("; " + "; ".join(movers) if movers else "") + f". Model share after shrinkage: ~{share:.0%} of the {poss(team)} first shots tonight."
    # role change: a first-shot taker of this team (started, since last season) who is not among tonight's five
    tonight = {x["player_id"] for x in st if x["team"] == team}
    gone = sorted([(sh, c, nm) for pid, (nm, c, sh) in shooters.get(team, {}).items() if pid not in tonight and c >= 5 and sh >= 0.10], reverse=True)[:2]
    if gone:
        role_l = "; ".join(f"{nm} took {sh:.0%} of the {poss(team)} first shots since last season ({c}) and is not in tonight's projected five" for sh, c, nm in gone) + ": that share redistributes pro-rata."
    # shot mix, only when extreme and on his own attempts
    own = np.array([s["types"][t] for t in TYPES], dtype=float); pos = np.array([s["pos_types"][t] for t in TYPES], dtype=float)
    pos_mix = pos / pos.sum() if pos.sum() else np.array([0.4, 0.3, 0.3])
    mix = (own + K_TYPE * pos_mix) / (own.sum() + K_TYPE); make = float(S["make"][top]); n_own = int(own.sum())
    if n_own >= 5 and mix[0] >= 0.6:
        mix_l = f"Rim-heavy first shots ({own[0]:.0f} of {n_own} at the rim): makes more of them, make rate ~{make:.0%}."
    elif n_own >= 5 and mix[2] >= 0.5:
        mix_l = f"Three-heavy first shots ({own[2]:.0f} of {n_own} from three): boom or bust, make rate ~{make:.0%}."
    if not (tip_l or role_l or mix_l or movers):
        return [f"No standout driver: highest base attempt share on his team ({base})."]
    return [x for x in (tip_l, share_l, role_l, mix_l) if x][:3]


def snapshot():
    params = read_json(PARAMS, None)
    if not params:
        print("[first-basket] no frozen params (gate not passed): nothing logged"); return 0
    P = read_json(SITE_DATA / "projections.json", None)
    if not P or not P.get("games"):
        print("[first-basket] no slate"); return 0
    box = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str}, usecols=["date", "player_id", "starter"])
    games, seq, st = load_tables()
    _, T = walk(games, st, set())
    pre_season = int(P.get("season") or games.season.max())
    starters = projected_starters(P, box)
    shooters = team_first_shooters(games, st, {pre_season - 1, pre_season}); tenure = player_team_counts(games, st, pre_season)
    K, C = params["K"], params["constants"]
    t, ts = now(), now().strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, today, skipped = [], [], []
    for g in P["games"]:
        tip = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
        tip_s = tip.strftime("%Y-%m-%dT%H:%M:%SZ")
        if t >= tip:
            skipped.append({"event_id": g["event_id"], "away": g["away"], "home": g["home"], "tip": tip_s, "reason": "already tipped when this run started (earlier rows, if any, stand)"}); continue
        hs, as_ = starters.get(g["home"], []), starters.get(g["away"], [])
        if len(hs) != 5 or len(as_) != 5:
            short = [tm for tm, lst in ((g["home"], hs), (g["away"], as_)) if len(lst) != 5]
            skipped.append({"event_id": g["event_id"], "away": g["away"], "home": g["home"], "tip": tip_s, "reason": f"fewer than five projected starters for {', '.join(short)}: not guessed"})
            print(f"[first-basket] {g['away']}@{g['home']}: {skipped[-1]['reason']}, skipped"); continue
        strs = [Row(team=p["team"], player_id=p["id"], player=p["name"], pos=p["pos"]) for p in hs + as_]
        game = Row(event_id=g["event_id"], date=P["date"], season=pre_season, home=g["home"], away=g["away"], label_clean=0, first_fg_id=None, first_fg_started=None)
        G = T.features(game, strs, pre_season=pre_season)
        S = score_game(G, K, C)
        top = int(np.argmax(S["v2"]))
        game_rows = []
        for i, s in enumerate(G["starters"]):
            game_rows.append({"date": P["date"], "event_id": g["event_id"], "tip": tip_s, "team": s["team"], "opp": g["away"] if s["home"] else g["home"], "home": s["home"],
                              "player_id": s["player_id"], "player": s["player"], "pos": s["pos"], "p_first_basket": round(float(S["v2"][i]), 4), "p_first_attempt": round(float(S["p_attempt"][i]), 4),
                              "top_pick": int(i == top), "jumper": s["jumper"], "logged_at": ts})
        rows += game_rows
        today.append({"event_id": g["event_id"], "tip": tip_s, "away": g["away"], "home": g["home"], "p_tip_home": round(float(S["p_tip"]), 3),
                      "top_pick": game_rows[top]["player"], "top_team": game_rows[top]["team"], "top_p": game_rows[top]["p_first_basket"], "why": why_lines(G, S, K, top, shooters, tenure),
                      "starters": [{k: r[k] for k in ("team", "home", "player", "pos", "p_first_basket", "p_first_attempt", "top_pick", "jumper")} for r in game_rows]})
    new_file = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    with open(WHY, "a", encoding="utf-8") as f:
        for g in today:
            f.write(json.dumps({"date": P["date"], "event_id": g["event_id"], "logged_at": ts, "why": g["why"], "p_tip_home": g["p_tip_home"]}) + "\n")
    build_today(P["date"], slate=P, skipped=skipped)
    print(f"[first-basket] appended {len(rows)} rows for {P['date']} ({len(today)} games logged, {len(skipped)} skipped)")
    for x in skipped:
        print(f"[first-basket]   skipped {x['away']}@{x['home']} ({x['tip']}): {x['reason']}")
    return len(rows)


def last_pretip(log, date=None):
    """Per (date, game): the rows of the last snapshot logged before tip (rule 5). The log is never edited; this only reads it."""
    d = log if date is None else log[log.date == date]
    d = d[pd.to_datetime(d.logged_at, utc=True) < pd.to_datetime(d.tip, utc=True)]
    return d[d.logged_at == d.groupby(["date", "event_id"]).logged_at.transform("max")]


def labels_by_event():
    if not HIST.exists():
        return {}
    h = pd.read_csv(HIST, dtype={"event_id": str, "first_fg_id": str})
    return {r.event_id: r for r in h.itertuples()}


def game_result(h, g):
    """h: history row or None; g: the game's snapshot rows sorted by P desc. None until the label exists."""
    if h is None:
        return None
    if h.label_clean != 1:
        return {"void": True, "reason": h.reason if isinstance(h.reason, str) else "no clean label"}
    ids = list(g.player_id); j = ids.index(h.first_fg_id) if h.first_fg_id in ids else None
    return {"void": False, "scorer": h.first_fg_player, "scorer_id": h.first_fg_id, "scorer_logged": int(j is not None), "scorer_p": round(float(g.p_first_basket.iloc[j]), 4) if j is not None else None,
            "hit": int(j == 0), "top3": int(j is not None and j < 3)}


def picks_for(d, hist, t):
    """One entry per game in d (last pre-tip rows): matchup, top candidate, tipped flag, result when labelled."""
    out = []
    for e, g in d.groupby("event_id"):
        g = g.sort_values(["top_pick", "p_first_basket"], ascending=[False, False]); top = g.iloc[0]
        tip = g.tip.iloc[0]; tipped = t >= dt.datetime.fromisoformat(tip.replace("Z", "+00:00"))
        res = game_result(hist.get(e), g)
        out.append({"date": g.date.iloc[0], "event_id": e, "tip": tip, "home": g[g.home == 1].team.iloc[0], "away": g[g.home == 0].team.iloc[0], "logged_at": g.logged_at.iloc[0],
                    "top_pick": top.player, "top_team": top.team, "top_p": round(float(top.p_first_basket), 4), "tipped": bool(tipped),
                    "status": "pending" if res is None else ("void" if res["void"] else "graded"), "result": res, "rows": g})
    return sorted(out, key=lambda x: (x["tip"], x["event_id"]))


def build_today(date=None, slate=None, skipped=None):
    """docs/data/first_basket_today.json from the log: every game logged on `date` (default: the latest logged date) with its last
    pre-tip snapshot. Games are never removed within the day; tipped games stay, with the actual first scorer once labelled."""
    params = read_json(PARAMS, None)
    if not params or not LOG.exists():
        return
    log = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str})
    date = date or (str(log.date.max()) if len(log) else None)
    if not date:
        return
    d = last_pretip(log, date)
    why = {}
    if WHY.exists():
        for line in open(WHY, encoding="utf-8"):
            e = json.loads(line)
            if e["date"] == date:
                why[(e["event_id"], e["logged_at"])] = e
    prev = read_json(TODAY, {}) or {}
    prev_games = {g["event_id"]: g for g in prev.get("games", [])} if prev.get("date") == date else {}
    hist = labels_by_event(); t = now(); games = []
    for pk in picks_for(d, hist, t):
        g = pk.pop("rows").sort_values("p_first_basket", ascending=False)
        w = why.get((pk["event_id"], pk["logged_at"]))
        if w is None and pk["event_id"] in prev_games and prev_games[pk["event_id"]].get("why"):
            pg = prev_games[pk["event_id"]]; w = {"why": pg["why"], "p_tip_home": pg.get("p_tip_home")}
            with open(WHY, "a", encoding="utf-8") as f:   # carry the explanation into the sidecar so later rebuilds keep it
                f.write(json.dumps({"date": date, "event_id": pk["event_id"], "logged_at": pk["logged_at"], **w}) + "\n")
        pk.update({"p_tip_home": (w or {}).get("p_tip_home"), "why": (w or {}).get("why") or [], "snapshots": int(log[(log.date == date) & (log.event_id == pk["event_id"])].logged_at.nunique()),
                   "starters": [{"team": r.team, "home": int(r.home), "player": r.player, "pos": r.pos, "p_first_basket": float(r.p_first_basket), "p_first_attempt": float(r.p_first_attempt),
                                 "top_pick": int(r.top_pick), "jumper": int(r.jumper)} for r in g.itertuples()]})
        games.append(pk)
    logged = {g["event_id"] for g in games}; skipped = list(skipped or [])
    if slate and slate.get("date") == date:
        seen = {x["event_id"] for x in skipped}
        for sg in slate["games"]:
            if sg["event_id"] not in logged and sg["event_id"] not in seen:
                skipped.append({"event_id": sg["event_id"], "away": sg["away"], "home": sg["home"], "tip": sg["start"], "reason": "not in the log"})
        skipped = [x for x in skipped if x["event_id"] not in logged]
    elif not slate:
        skipped = [x for x in prev.get("skipped", []) if x["event_id"] not in logged]
    write_json(TODAY, {"date": date, "generated": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "slate_updated": (slate or prev).get("updated") if slate else prev.get("slate_updated"),
                       "source": "data/first_basket_log.csv: every game logged on this date, its last pre-tip snapshot; the tab empties only when the date rolls over",
                       "games": games, "skipped": skipped, "tipped": sum(g["tipped"] for g in games), "graded": sum(g["status"] == "graded" for g in games),
                       "label": "Hit rate measures accuracy, not profit. Edge requires comparing to prices.", "override": OVERRIDE, "params_frozen": params.get("frozen"), "K": params["K"], "constants": params["constants"]})
    print(f"[first-basket] today {date}: {len(games)} games from the log ({sum(g['tipped'] for g in games)} tipped, {sum(g['status'] == 'graded' for g in games)} graded), {len(skipped)} skipped")
    return games


def bucket_table(d):
    out = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (d.p >= lo) & (d.p < hi) if i < 9 else (d.p >= lo)
        g = d[m]; n = int(len(g))
        out.append({"lo": lo, "hi": hi, "n": n, "pred": round(float(g.p.mean()), 4) if n else None, "actual": round(float(g.hit.mean()), 4) if n else None, "thin": n < THIN})
    return out


def grade():
    params = read_json(PARAMS, None)
    if not LOG.exists() or not params:
        if PERF.exists():
            PERF.unlink()
        print("[first-basket] no log yet"); return
    log = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str})
    snap = log[pd.to_datetime(log.logged_at, utc=True) < pd.to_datetime(log.tip, utc=True)].sort_values("logged_at").drop_duplicates(["date", "event_id", "player_id"], keep="last").copy()
    hist = pd.read_csv(HIST, dtype={"event_id": str, "first_fg_id": str}) if HIST.exists() else pd.DataFrame(columns=["event_id", "first_fg_id", "label_clean", "first_fg_player", "first_fg_started"])
    h = hist.set_index("event_id")
    st = pd.read_csv(STARTERS, dtype={"event_id": str, "player_id": str}) if STARTERS.exists() else pd.DataFrame(columns=["event_id", "player_id"])
    started = set(zip(st.event_id, st.player_id))
    eps = params["constants"]["eps"]
    snap["status"] = ["graded" if e in h.index and h.loc[e].label_clean == 1 else ("void" if e in h.index else "pending") for e in snap.event_id]
    g = snap[snap.status == "graded"].copy()
    g["scorer"] = [h.loc[e].first_fg_id for e in g.event_id]
    g["scorer_name"] = [h.loc[e].first_fg_player for e in g.event_id]
    g["hit"] = (g.player_id == g.scorer).astype(int); g["p"] = g.p_first_basket.astype(float)
    g["started"] = [int((e, p) in started) for e, p in zip(g.event_id, g.player_id)]
    per_game = []
    for e, d in g.groupby("event_id"):
        d = d.sort_values("p", ascending=False); top = d.iloc[0]; ids = list(d.player_id)
        scorer = d.scorer.iloc[0]; in_ten = scorer in ids
        j = ids.index(scorer) if in_ten else None
        per_game.append({"date": d.date.iloc[0], "event_id": e, "home": d[d.home == 1].team.iloc[0] if (d.home == 1).any() else None, "away": d[d.home == 0].team.iloc[0] if (d.home == 0).any() else None,
                         "top_pick": top.player, "top_p": round(float(top.p), 3), "hit": int(top.player_id == scorer), "top3": int(in_ten and j < 3),
                         "scorer": d.scorer_name.iloc[0], "scorer_logged": int(in_ten),
                         "loss_v2": -np.log(max(float(d.p.iloc[j]), P_FLOOR)) if in_ten else -np.log(max(eps, P_FLOOR)), "loss_uniform": -np.log((1 - eps) / 10) if in_ten else -np.log(max(eps, P_FLOOR))})
    pg = pd.DataFrame(per_game)
    n = int(len(pg)); hits = int(pg.hit.sum()) if n else 0
    latest = str(log.date.max()); hist_by = labels_by_event(); t = now()
    picks = [{k: v for k, v in pk.items() if k != "rows"} for pk in picks_for(last_pretip(log, latest), hist_by, t)]
    doc = {"header": "Every pick graded against real results. Nothing filtered, nothing hidden.", "generated": now().strftime("%Y-%m-%dT%H:%M:%SZ"), "rule_file": RULE,
           "label": "Hit rate measures accuracy, not profit. Edge requires comparing to prices.",
           "games": n, "overall": {"n": n, "hits": hits, "misses": n - hits, "hit_rate": round(hits / n, 4) if n else None, "record": f"{hits}-{n - hits}", "mean_p": round(float(pg.top_p.mean()), 4) if n else None},
           "top3_rate": round(float(pg.top3.mean()), 4) if n else None, "log_loss_v2": round(float(pg.loss_v2.mean()), 4) if n else None, "log_loss_uniform": round(float(pg.loss_uniform.mean()), 4) if n else None,
           "scorer_among_logged_share": round(float(pg.scorer_logged.mean()), 4) if n else None, "starter_match_rate": round(float(g.started.mean()), 4) if len(g) else None,
           "buckets": bucket_table(g) if len(g) else [], "rows_in_file": int(len(log)), "keys": int(len(snap)), "pending_games": int(snap[snap.status == "pending"].event_id.nunique()), "void_games": int(snap[snap.status == "void"].event_id.nunique()),
           "recent": pg.sort_values(["date", "top_p"], ascending=[False, False]).head(60).to_dict("records") if n else [],
           "today": {"date": latest, "picks": picks, "pending": sum(p["status"] == "pending" for p in picks), "graded": sum(p["status"] == "graded" for p in picks), "void": sum(p["status"] == "void" for p in picks)},
           "method": ["Ten projected starters per game logged before tip; the graded snapshot is the last row before tip. Append-only log (data/first_basket_log.csv).",
                      "Hit = the player scored the game's first made field goal (free throws excluded). A game without a clean label is void (shown, not hidden).",
                      f"Calibration cells with n < {THIN} are flagged thin. No tuning on these results, ever; constants frozen from the gate's 2024-25 fit ({RULE}).",
                      OVERRIDE]}
    write_json(PERF, doc)
    print(f"[first-basket] graded {n} games ({doc['overall']['record']}), {doc['pending_games']} pending, {doc['void_games']} void")
    build_today()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    {"gate": gate, "freeze": freeze, "snapshot": snapshot, "grade": grade, "today": build_today}[cmd]()
