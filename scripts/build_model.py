"""Build defense-vs-position tables and player projections for today's slate.

Outputs (read by the site):
  docs/data/dvp.json          team x position allowed rates, factors, ranks
  docs/data/projections.json  per-player minutes, means and full stat distributions
"""
import datetime as dt
import math

import numpy as np
import pandas as pd

from common import DATA_DIR, NBA_TEAMS, POSITIONS, SITE_DATA, norm_name, norm_pos, read_json, write_json
from odds import OddsBlaze
from positions import position_map
from health import STATE as HEALTH_STATE, notify

N_SIM = 8000
STATS = ["tpm", "tpa", "reb", "ast", "pts"]
PRIOR_SEASON_DVP_W = 0.35   # rosters turn over, so last season counts less for defense
PRIOR_SEASON_RATE_W = 0.5   # and a bit less for player rates
K_MIN_DVP = 600             # position-minutes of league average blended into each team's DvP
PCT_PRIOR_ATT = 150         # 3PA of league-average 3P% blended into each shooter's %
MIN_HALF_LIFE = 4           # games
PRIOR_TAIL_DAYS = 14        # last two weeks of the prior regular season: rest/tank games, excluded from minutes
BLEND_GAMES = 5             # under this many current-season games, minutes blend with last season
RATE_HALF_LIFE = 12         # games
TEAM_MINUTES = 240
MAX_MIN = 39  # 95th percentile of regular starters' minutes in regulation games, 2025-26 backfill (40.0 with OT)
# Blowout risk: NBA final margins run ~N(spread, 13.5). In a 20+ point game regular starters sit
# roughly the last 8 minutes. Base minutes already include the blowouts a pick'em produces, so
# only the EXTRA blowout chance a lopsided spread adds is shaved off.
BLOWOUT_MARGIN = 20
MARGIN_SD = 13.5
BLOWOUT_STARTER_LOSS = 8.0
RNG = np.random.default_rng(7)


def load_games(apply_positions=True):
    """Game logs. pos is ESPN's generic G/F/C mapped to SG/SF/C, then replaced per player by the
    play-style assignment in data/positions.csv (see positions.py) when apply_positions is on."""
    df = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"player_id": str, "event_id": str})
    df["date"] = pd.to_datetime(df["date"])
    df["pos"] = df["pos"].map(norm_pos)
    df = df[df.team.isin(NBA_TEAMS) & df.opp.isin(NBA_TEAMS)]
    if apply_positions:
        pmap = position_map()
        if pmap:
            df["pos"] = df["player_id"].map(pmap).fillna(df["pos"])
    return df


# ---------------- defense vs position ----------------
def build_dvp(df, cur):
    d = df[df.season >= cur - 1].copy()
    d["w"] = np.where(d.season == cur, 1.0, PRIOR_SEASON_DVP_W)
    for s in STATS + ["min"]:
        d[f"w_{s}"] = d[s] * d.w
    # Positions with no minutes (ESPN box scores are mostly generic G/F, so PF can be empty) are left out
    # rather than producing NaN, which would be written into dvp.json and break the site's JSON parse.
    lg = {p: {s: d.loc[d.pos == p, f"w_{s}"].sum() / d.loc[d.pos == p, "w_min"].sum() for s in STATS}
          for p in POSITIONS if d.loc[d.pos == p, "w_min"].sum() > 0}
    agg = d.groupby(["opp", "pos"])[[f"w_{s}" for s in STATS] + ["w_min"]].sum()

    shown = df[df.season == cur]
    gp = shown.groupby("opp").event_id.nunique()
    per_game = shown.groupby(["opp", "pos"])[STATS].sum()

    teams = {}
    for (team, pos), row in agg.iterrows():
        if pos not in lg:
            continue
        for s in STATS:
            rate = (row[f"w_{s}"] + K_MIN_DVP * lg[pos][s]) / (row["w_min"] + K_MIN_DVP)
            pg = per_game.loc[(team, pos), s] / gp[team] if (team, pos) in per_game.index else None
            teams.setdefault(team, {}).setdefault(pos, {})[s] = {
                "f": round(rate / lg[pos][s], 3), "pg": None if pg is None else round(float(pg), 2)}
    for pos in POSITIONS:  # rank 1 = gives up the most relative to league
        for s in STATS:
            order = sorted((t for t in teams if pos in teams[t]), key=lambda t: -teams[t][pos][s]["f"])
            for i, t in enumerate(order, 1):
                teams[t][pos][s]["rk"] = i
    return teams, lg


def dvp_factor(dvp, opp, pos, stat):
    return dvp.get(opp, {}).get(pos, {}).get(stat, {}).get("f", 1.0)


# ---------------- player profiles ----------------
def minutes_baseline(g, cur, prior_cutoff):
    """(base_min, sd_min, label, n_current). g sorted newest first.
    - n >= BLEND_GAMES current-season games: last 10 of them, half-life MIN_HALF_LIFE ("recent").
    - fewer: (n * recent + (5 - n) * prior_avg) / 5, where prior_avg is last season's average with
      its final PRIOR_TAIL_DAYS excluded ("blend n/5", or "prior" when n = 0).
    - no usable prior season (rookie, or only played in the tail): whatever games exist, weighted."""
    cur_g = g[g.season == cur].head(10)
    n = len(cur_g)
    prior = g[(g.season == cur - 1) & (g.date <= prior_cutoff)] if prior_cutoff is not None else g.iloc[0:0]

    def weighted(x):
        w = 0.5 ** (np.arange(len(x)) / MIN_HALF_LIFE)
        return float((x["min"].values * w).sum() / w.sum())

    def sd(x):
        return float(np.clip(x["min"].std(ddof=0) if len(x) > 2 else 6, 3, 8))

    if n >= BLEND_GAMES or len(prior) == 0:
        use = cur_g if n else g.head(10)
        return weighted(use), sd(use), "recent" if n else "fallback", n
    prior_avg = float(prior["min"].mean())
    if n == 0:
        return prior_avg, sd(prior), "prior", 0
    base = (n * weighted(cur_g) + (BLEND_GAMES - n) * prior_avg) / BLEND_GAMES
    return base, max(sd(cur_g), sd(prior)), f"blend {n}/{BLEND_GAMES}", n


def profile(g, cur, lg_pct, prior_cutoff=None):
    g = g.sort_values("date", ascending=False)
    k = np.arange(len(g))
    season_w = np.where(g.season.values == cur, 1.0, PRIOR_SEASON_RATE_W)
    w_rate = 0.5 ** (k / RATE_HALF_LIFE) * season_w
    recent = g.head(10)
    base_min, sd_min, baseline, n_cur = minutes_baseline(g, cur, prior_cutoff)

    m = g["min"].values
    rates = {s: float((g[s].values * w_rate).sum() / (m * w_rate).sum()) for s in STATS}
    non3 = (g["pts"].values - 3 * g["tpm"].values)
    rates["non3"] = float((non3 * w_rate).sum() / (m * w_rate).sum())
    tpm_sum, tpa_sum = (g.tpm.values * season_w).sum(), (g.tpa.values * season_w).sum()
    pct = (tpm_sum + PCT_PRIOR_ATT * lg_pct) / (tpa_sum + PCT_PRIOR_ATT)
    last10 = recent.iloc[::-1]
    return {
        "base_min": base_min, "sd_min": sd_min, "baseline": baseline, "n_cur": n_cur,
        "rates": rates, "pct": float(pct), "season_pct_att": int(tpa_sum),
        "starter_rate": float(g.head(5).starter.mean()), "last_team": g.iloc[0].team,
        "last_season": int(g.iloc[0].season),
        "last10": {s: last10[s].astype(int).tolist() for s in ["tpm", "reb", "ast", "pts"]}
                  | {"min": last10["min"].round().astype(int).tolist(),
                     "pra": (last10.pts + last10.reb + last10.ast).astype(int).tolist()},
    }


def news_status(news, slate_date):
    """Latest classified tag per player from the last ~20 hours of news."""
    cutoff = dt.datetime.fromisoformat(slate_date) - dt.timedelta(hours=8)
    status = {}
    for item in sorted(news, key=lambda x: x["received"]):
        ts = dt.datetime.fromisoformat(item["received"].replace("Z", "+00:00")).replace(tzinfo=None)
        if ts >= cutoff and item.get("player"):
            status[norm_name(item["player"])] = item["tag"]
    return status


def espn_status(raw):
    s = (raw or "").lower()
    if s.startswith("out"):
        return "OUT"
    if s.startswith("doubt"):
        return "DOUBTFUL"
    if s.startswith(("quest", "day-to-day")):
        return "QUESTIONABLE"
    return None


def scale_to(players, total):
    """Scale minutes so the group sums to `total`, respecting the per-player cap."""
    for _ in range(10):
        capped = [p for p in players if p["min"] >= MAX_MIN]
        free = [p for p in players if p["min"] < MAX_MIN]
        have = sum(p["min"] for p in free)
        if have <= 0:
            return
        scale = (total - MAX_MIN * len(capped)) / have
        for p in free:
            p["min"] = min(MAX_MIN, p["min"] * scale)
        if abs(sum(p["min"] for p in players) - total) < 0.5:
            return


ROTATION_FLOOR = 4  # projected minutes under this: not in the rotation, dropped from the output
# Vacated minutes (OUT / DOUBTFUL players). Shares are of VACATED minutes (the absent player's
# projected minutes). Backtested on 1,235 absence team-games of the 2025-26 backfill (per-player MAE of
# rotation teammates' minutes; no redistribution = 5.34):
#   spread by top-absorber probability:   20% 5.22  30% 5.25  40% 5.34  67% 5.74   <- shipped, share 0.20
#   proportional to role:                 20% 5.25  30% 5.26  40% 5.31  67% 5.57
#   concentrate 40/27 on predicted #1/#2: 20% 5.25  30% 5.34  40% 5.50  67% 6.07
# Observed: only ~26% of vacated minutes land on rotation teammates (trailing avg >= 6 who played);
# the rest goes to deep bench / call-ups, so ROTATION_SHARE is far below Muse's 67%.
REDISTRIBUTION = "spread"      # "spread" (b) or "concentrate" (a)
ROTATION_SHARE = 0.20          # (b): share of vacated minutes spread over the rotation by absorber probability
TOP1_SHARE, TOP2_SHARE = 0.40, 0.27  # (a): shares of vacated minutes to the predicted #1 and #2 absorbers
SAME_POS_WEIGHT = 2.5          # role-similarity prior: same position counts this much more
HISTORY_MIN_GAMES = 3          # past games this player missed (same team) before history outranks role similarity
# Usage bump when regular starters sit, measured on the 2025-26 backfill (1,390 team-games with a regular
# starter absent; remaining starters' per-36 rates vs their own baseline, minutes-weighted, 95% CI):
#   1 absent:  FGA +4.9% [3.3, 6.5]  pts +5.8% [3.6, 8.2]  3PA +3.2% [0.7, 6.1]  ast +3.9% [0.8, 7.6]  reb +1.4% [-0.9, 3.9]
#   2+ absent: FGA +13.1%            pts +15.4%            3PA +6.7%             ast +16.9%            reb +6.0%
# Modeled as a per-absent-starter multiplier on the rates, counted up to USAGE_MAX_ABSENT.
USAGE_BUMP = {"tpa": 0.035, "pts": 0.055, "ast": 0.045, "reb": 0.015}
USAGE_MAX_ABSENT = 2


def trim_to(players, total):
    """Depth-chart normalization. Per-game averages of a full roster sum well past 240 (they ignore
    DNPs, and preseason rosters carry 21 players with minutes from last season). Real teams don't
    shave everyone evenly: the bottom of the depth chart loses its minutes first. So walk the roster
    from the biggest role down, keep each player's average until the 240 run out, and zero the rest.
    If the roster is short of 240 (injuries), scale the remaining players up instead."""
    running = 0.0
    for p in sorted(players, key=lambda p: -p["min"]):
        p["min"] = max(0.0, min(p["min"], MAX_MIN, total - running))
        running += p["min"]
    if running < total - 0.5:
        scale_to(players, total)


def role_probs(out, active):
    """Role-similarity prior: bigger roles absorb more, same position SAME_POS_WEIGHT times more."""
    w = {p["id"]: p["base_min"] * (SAME_POS_WEIGHT if p["pos"] == out["pos"] else 1.0) for p in active}
    z = sum(w.values()) or 1.0
    return {k: v / z for k, v in w.items()}


class AbsorberHistory:
    """Who soaked up minutes in past games this player missed (same team). Built from the game logs
    with each player's trailing-10 average before each game, so a gain is actual minus trailing."""

    def __init__(self, df, before):
        d = df[df.date < before].sort_values("date")
        d = d.assign(trail=d.groupby(["player_id", "team"])["min"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean()))
        self.games = {t: [(e, g.date.iloc[0], dict(zip(g.player_id, g["min"] - g.trail))) for e, g in grp.groupby("event_id")]
                      for t, grp in d.groupby("team")}
        self.span = {(pid, t): (g.date.min(), g.date.max()) for (pid, t), g in d.groupby(["player_id", "team"])}

    def missed(self, out, team):
        span = self.span.get((out["id"], team))
        if not span:
            return []
        return [gains for e, date, gains in self.games.get(team, []) if span[0] <= date <= span[1] and out["id"] not in gains]

    def probs(self, out, team, active):
        """{player id: P(top absorber)} from HISTORY_MIN_GAMES+ past absences, else None."""
        ids = {p["id"] for p in active}
        tops = []
        for gains in self.missed(out, team):
            g = {k: v for k, v in gains.items() if k in ids and v == v}
            if g:
                tops.append(max(g, key=g.get))
        if len(tops) < HISTORY_MIN_GAMES:
            return None
        return {i: tops.count(i) / len(tops) for i in ids}


def give(active, vacated, shares):
    """Add vacated * share to each player, respecting MAX_MIN; what doesn't fit spills to the others by share."""
    pool = {p["id"]: p for p in active}
    pending = {k: vacated * v for k, v in shares.items() if v > 0 and k in pool}
    for _ in range(5):
        if sum(pending.values()) < 0.3:
            break
        spill = 0.0
        for k, add in list(pending.items()):
            room = max(MAX_MIN - pool[k]["min"], 0.0)
            pool[k]["min"] += min(add, room)
            spill += max(0.0, add - room)
        open_ = [p for p in active if p["min"] < MAX_MIN - 0.1]
        if spill < 0.3 or not open_:
            break
        z = sum(shares.get(p["id"], 0) or 1e-9 for p in open_)
        pending = {p["id"]: spill * (shares.get(p["id"], 0) or 1e-9) / z for p in open_}


def allocate_minutes(active, outs, team=None, history=None):
    """1) Normalize the full roster to 240 from the top of the depth chart down (trim_to).
    2) Redistribute each ruled-out player's minutes per REDISTRIBUTION (see the constants above)."""
    everyone = active + outs
    for p in outs:
        p["min"] = p["base_min"]
    trim_to(everyone, TEAM_MINUTES)
    for p in active:
        p["pre_min"] = p["min"]
    for o in outs:
        if o["min"] < 0.5 or not active:
            continue
        probs = (history.probs(o, team, active) if history is not None else None) or role_probs(o, active)
        if REDISTRIBUTION == "concentrate":
            rank = sorted(active, key=lambda p: -probs.get(p["id"], 0))
            shares = {rank[0]["id"]: TOP1_SHARE}
            if len(rank) > 1:
                shares[rank[1]["id"]] = TOP2_SHARE
        else:
            shares = {k: ROTATION_SHARE * v for k, v in probs.items()}
        give(active, o["min"], shares)


def blowout_shave(spread):
    """Expected extra minutes a full-time starter loses to garbage time at this spread."""
    if spread is None:
        return 0.0

    def p_blowout(s):
        return 1 - 0.5 * (1 + math.erf((BLOWOUT_MARGIN - abs(s)) / (MARGIN_SD * math.sqrt(2))))

    return BLOWOUT_STARTER_LOSS * max(p_blowout(spread) - p_blowout(0), 0.0)


def apply_blowout(players, own_spread):
    """Shave starters' minutes in a lopsided game (both sides sit starters) and hand them to the
    bench. own_spread is from this team's view (negative = favored). Returns minutes shaved."""
    shave = blowout_shave(own_spread)
    if shave < 0.3:
        return 0.0
    starters = [p for p in players if p["starter_rate"] >= 0.5 and p["min"] >= 20]
    bench = [p for p in players if p not in starters and p["min"] < MAX_MIN]
    if not starters or not bench:
        return 0.0
    freed = 0.0
    for p in starters:
        cut = shave * min(p["min"] / 34.0, 1.0)
        p["min"] -= cut
        p["sd_min"] += cut / 2
        p["flags"].append(f"Blowout risk ({own_spread:+.1f})")
        freed += cut
    tw = sum(p["min"] for p in bench)
    for p in bench:
        p["min"] += freed * p["min"] / tw
    return shave


def surname(name):
    parts = [w for w in name.split() if w.rstrip(".").lower() not in ("jr", "sr", "ii", "iii", "iv")]
    return parts[-1] if parts else name


def pmf(x, top):
    counts = np.bincount(np.clip(x, 0, top), minlength=top + 1) / len(x)
    arr = [round(float(v), 4) for v in counts]
    while arr and arr[-1] == 0:
        arr.pop()
    return arr


def simulate(p, opp, dvp, absent_starters=0):
    pos, r = p["pos"], p["rates"]
    mins = np.clip(RNG.normal(p["min"], p["sd_min"], N_SIM), 0, 48)
    f = {s: dvp_factor(dvp, opp, pos, s) for s in STATS}
    n_abs = min(absent_starters, USAGE_MAX_ABSENT)
    u = {s: 1 + USAGE_BUMP[s] * n_abs for s in USAGE_BUMP}
    tpa = RNG.poisson(mins * r["tpa"] * f["tpa"] * u["tpa"])
    make_p = np.clip(p["pct"] * f["tpm"] / max(f["tpa"], 0.5), 0.12, 0.6)
    tpm = RNG.binomial(tpa, make_p)
    reb = RNG.poisson(mins * r["reb"] * f["reb"] * u["reb"])
    ast = RNG.poisson(mins * r["ast"] * f["ast"] * u["ast"])
    pts = 3 * tpm + 2 * RNG.poisson(np.maximum(mins * r["non3"] * f["pts"] * u["pts"], 0) / 2)
    pra = pts + reb + ast
    out = {}
    for name, arr, top in [("tpm", tpm, 12), ("tpa", tpa, 25), ("reb", reb, 30),
                           ("ast", ast, 25), ("pts", pts, 70), ("pra", pra, 90)]:
        out[name] = {"mean": round(float(arr.mean()), 2), "pmf": pmf(arr, top)}
    out["make_p"] = round(float(make_p), 3)
    return out


STARTER_SHARE = 0.8       # started this share of their games last season (20+ games) = a regular starter
NEWS_DAYS = 7             # a news item within this many days of the slate confirms an ESPN inactive tag


def unconfirmed_starters(df, cur, slate, news, out_players):
    """Regular starters from last season whose team plays tonight but who are (a) tagged inactive by ESPN
    with no news item about them in the last NEWS_DAYS, or (b) on no ESPN roster at all (only judged when the
    slate carries every roster). Posts one Discord message per run, once per player per slate date, state in
    data/health_state.json. They stay out of the projections; this just stops it being silent."""
    prior = df[df.season == cur - 1].sort_values("date")
    a = prior.groupby("player_id").agg(gp=("event_id", "nunique"), starts=("starter", "sum"), name=("player", "last"), team=("team", "last"))
    regs = a[(a.starts >= STARTER_SHARE * a.gp) & (a.gp >= 20)]
    tonight = {g["home"] for g in slate["games"]} | {g["away"] for g in slate["games"]}
    rosters = slate.get("rosters") or {}
    on_roster = {str(x["id"]): t for t, r in rosters.items() for x in r}
    full_rosters = len(rosters) >= 25
    since = dt.datetime.fromisoformat(slate["date"]) - dt.timedelta(days=NEWS_DAYS)
    newsed = {norm_name(n["player"]) for n in news if n.get("player") and dt.datetime.fromisoformat(n["received"].replace("Z", "")) >= since}
    inactive = {p["id"]: p for p in out_players if p.get("inactive")}
    found = []
    for pid, r in regs.iterrows():
        team = on_roster.get(pid, None if full_rosters else r.team)
        if team is None:
            if r.team in tonight:
                found.append({"id": pid, "name": r["name"], "team": r.team, "reason": "not on any ESPN roster"})
            continue
        if team in tonight and pid in inactive and norm_name(r["name"]) not in newsed:
            found.append({"id": pid, "name": r["name"], "team": team, "reason": f"ESPN tag {inactive[pid]['status']}, no news in {NEWS_DAYS} days"})
    if found:
        state = read_json(HEALTH_STATE, {})
        done = state.get("unconfirmed", {})
        fresh = [f for f in found if done.get(f["id"]) != slate["date"]]
        if fresh:
            notify("\U0001f50d Arc Lab: regular starters inactive without news tonight: " +
                   "; ".join(f"{f['name']} ({f['team']}, {f['reason']})" for f in fresh))
            for f in fresh:
                done[f["id"]] = slate["date"]
            write_json(HEALTH_STATE, {**state, "unconfirmed": done})
    return found


def season_of(date_str):
    """ESPN season year for a slate date: the 2026-27 season is 2027, and it starts in the fall."""
    d = dt.date.fromisoformat(date_str)
    return d.year + 1 if d.month >= 8 else d.year


def main():
    df = load_games()
    slate = read_json(DATA_DIR / "slate.json", {"date": dt.date.today().isoformat(), "games": [], "rosters": {}})
    cur = season_of(slate["date"])
    if cur - 1 not in set(df.season):  # data doesn't reach last season: fall back to whatever is newest
        cur = int(df.season.max())
    prior_dates = df.loc[df.season == cur - 1, "date"]
    prior_cutoff = (prior_dates.max() - pd.Timedelta(days=PRIOR_TAIL_DAYS)) if len(prior_dates) else None
    news = read_json(SITE_DATA / "news.json", [])
    dvp, lg = build_dvp(df, cur)
    tpm_lg = df[df.season >= cur - 1]  # two seasons, so opening night has a real league 3P%
    lg_pct = float(tpm_lg.tpm.sum() / max(tpm_lg.tpa.sum(), 1))
    nstat = news_status(news, slate["date"])
    played = df.groupby("team").date.max()
    history = AbsorberHistory(df, pd.Timestamp(slate["date"]))
    ob = OddsBlaze()
    spreads = ob.spreads() if ob.enabled and slate["games"] else {}
    if slate["games"] and not spreads:
        print("[model] no spreads (" + ("OddsBlaze returned nothing" if ob.enabled else "ODDSBLAZE_API_KEY not set") + "), blowout risk skipped")

    by_id = {pid: g for pid, g in df[df.season >= cur - 1].groupby("player_id")}
    pos_map = position_map()
    out_players, games_out = [], []
    for game in slate["games"]:
        mk = spreads.get((game["away"], game["home"]))
        games_out.append({**game, "spread": mk["home_spread"] if mk else None, "total": mk["total"] if mk else None,
                          "odds_book": mk["book"] if mk else None})
        for team, opp, home in [(game["home"], game["away"], 1), (game["away"], game["home"], 0)]:
            roster = slate.get("rosters", {}).get(team)
            if roster is None:  # fallback: whoever played for the team most recently
                recent = df[df.team == team].sort_values("date").tail(60)
                roster = [{"id": i, "name": n, "pos": p, "injury": None}
                          for i, n, p in recent[["player_id", "player", "pos"]].drop_duplicates("player_id").values]
            team_players, outs = [], []
            for ath in roster:
                g = by_id.get(str(ath["id"]))
                if g is None or len(g) < 3:
                    continue
                prof = profile(g, cur, lg_pct, prior_cutoff)
                status = nstat.get(norm_name(ath["name"])) or espn_status(ath.get("injury"))
                entry = {"id": str(ath["id"]), "name": ath["name"], "team": team, "opp": opp, "home": home,
                         "pos": pos_map.get(str(ath["id"])) or (norm_pos(ath.get("pos")) if ath.get("pos") else g.iloc[-1].pos),
                         "status": status, "flags": [], **prof}
                if status in ("OUT", "DOUBTFUL"):
                    outs.append(entry)
                    continue
                if prof["base_min"] < 6:
                    continue
                entry["min"] = prof["base_min"]
                if status == "QUESTIONABLE":
                    entry["sd_min"] += 4
                    entry["flags"].append("Q")
                if status == "MIN RESTRICTION":
                    entry["min"] = min(entry["min"], 24)
                    entry["flags"].append("Minutes limit")
                if prof["last_team"] != team:
                    entry["flags"].append(f"Offseason move ({prof['last_team']}→{team})" if prof["last_season"] < cur else "New team")
                if prof["baseline"] == "prior":
                    entry["flags"].append("Last season minutes")
                elif prof["baseline"].startswith("blend"):
                    entry["flags"].append(f"Minutes {prof['baseline']}")
                team_players.append(entry)
            allocate_minutes(team_players, outs, team, history)
            team_players = [p for p in team_players if p["min"] >= ROTATION_FLOOR]
            apply_blowout(team_players, (mk["home_spread"] if home else -mk["home_spread"]) if mk else None)
            absent_starters = sum(1 for o in outs if o["starter_rate"] >= 0.5)
            yesterday = pd.Timestamp(slate["date"]) - pd.Timedelta(days=1)
            b2b = team in played and played[team] == yesterday
            for p in team_players:
                if b2b:
                    p["flags"].append("B2B")
                gain = p["min"] - p["pre_min"]
                if outs and gain >= 2:
                    p["flags"].append(f"+{gain:.0f} min ({', '.join(surname(o['name']) for o in outs)} out)")
                if absent_starters:
                    p["flags"].append(f"Usage +{100 * USAGE_BUMP['pts'] * min(absent_starters, USAGE_MAX_ABSENT):.0f}% pts")
                sim = simulate(p, opp, dvp, absent_starters)
                out_players.append({
                    "id": p["id"], "name": p["name"], "team": team, "opp": opp, "home": home,
                    "pos": p["pos"], "min": round(p["min"], 1), "base_min": round(p["pre_min"], 1),
                    "pct": round(p["pct"], 3), "flags": p["flags"], "last10": p["last10"], "baseline": p["baseline"],
                    "dvp_rank": {s: dvp.get(opp, {}).get(p["pos"], {}).get(s, {}).get("rk") for s in STATS},
                    **sim,
                })
            for o in outs:
                out_players.append({"id": o["id"], "name": o["name"], "team": team, "opp": opp,
                                    "pos": o["pos"], "status": o["status"], "inactive": True})

    alerts = unconfirmed_starters(df, cur, slate, news, out_players)
    write_json(SITE_DATA / "dvp.json", {"season": cur, "positions": POSITIONS, "league": lg, "teams": dvp,
                                        "updated": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"})
    write_json(SITE_DATA / "projections.json", {
        "date": slate["date"], "season": cur, "games": games_out, "league_3p_pct": round(lg_pct, 3),
        "players": out_players, "alerts": alerts,
        "updated": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"})
    print(f"[model] {len([p for p in out_players if not p.get('inactive')])} players projected, {len(alerts)} unconfirmed starters")


if __name__ == "__main__":
    main()
