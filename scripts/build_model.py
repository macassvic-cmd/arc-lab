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

N_SIM = 8000
STATS = ["tpm", "tpa", "reb", "ast", "pts"]
PRIOR_SEASON_DVP_W = 0.35   # rosters turn over, so last season counts less for defense
PRIOR_SEASON_RATE_W = 0.5   # and a bit less for player rates
K_MIN_DVP = 600             # position-minutes of league average blended into each team's DvP
PCT_PRIOR_ATT = 150         # 3PA of league-average 3P% blended into each shooter's %
MIN_HALF_LIFE = 4           # games
RATE_HALF_LIFE = 12         # games
TEAM_MINUTES = 240
MAX_MIN = 40
# Blowout risk: NBA final margins run ~N(spread, 13.5). In a 20+ point game regular starters sit
# roughly the last 8 minutes. Base minutes already include the blowouts a pick'em produces, so
# only the EXTRA blowout chance a lopsided spread adds is shaved off.
BLOWOUT_MARGIN = 20
MARGIN_SD = 13.5
BLOWOUT_STARTER_LOSS = 8.0
RNG = np.random.default_rng(7)


def load_games():
    df = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"player_id": str, "event_id": str})
    df["date"] = pd.to_datetime(df["date"])
    df["pos"] = df["pos"].map(norm_pos)
    return df[df.team.isin(NBA_TEAMS) & df.opp.isin(NBA_TEAMS)]


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
def profile(g, cur, lg_pct):
    g = g.sort_values("date", ascending=False)
    k = np.arange(len(g))
    season_w = np.where(g.season.values == cur, 1.0, PRIOR_SEASON_RATE_W)
    w_rate = 0.5 ** (k / RATE_HALF_LIFE) * season_w
    recent = g.head(10)
    w_min = 0.5 ** (np.arange(len(recent)) / MIN_HALF_LIFE)

    m = g["min"].values
    rates = {s: float((g[s].values * w_rate).sum() / (m * w_rate).sum()) for s in STATS}
    non3 = (g["pts"].values - 3 * g["tpm"].values)
    rates["non3"] = float((non3 * w_rate).sum() / (m * w_rate).sum())
    tpm_sum, tpa_sum = (g.tpm.values * season_w).sum(), (g.tpa.values * season_w).sum()
    pct = (tpm_sum + PCT_PRIOR_ATT * lg_pct) / (tpa_sum + PCT_PRIOR_ATT)
    last10 = recent.iloc[::-1]
    return {
        "base_min": float((recent["min"].values * w_min).sum() / w_min.sum()),
        "sd_min": float(np.clip(recent["min"].std(ddof=0) if len(recent) > 2 else 6, 3, 8)),
        "rates": rates, "pct": float(pct), "season_pct_att": int(tpa_sum),
        "starter_rate": float(g.head(5).starter.mean()), "last_team": g.iloc[0].team,
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


def allocate_minutes(active, outs):
    """1) Normalize the full roster to 240 (season averages ignore DNPs, so they overcount).
    2) Hand ruled-out players' minutes to teammates, weighted toward the same position."""
    everyone = active + outs
    for p in outs:
        p["min"] = p["base_min"]
    scale_to(everyone, TEAM_MINUTES)
    for p in active:
        p["pre_min"] = p["min"]
    freed = sum(o["min"] for o in outs)
    out_pos = {o["pos"] for o in outs}
    for _ in range(5):
        open_ = [p for p in active if p["min"] < MAX_MIN]
        if freed < 0.5 or not open_:
            break
        w = {id(p): p["min"] * (2.5 if p["pos"] in out_pos else 1.0) for p in open_}
        tw = sum(w.values())
        left = 0.0
        for p in open_:
            add = freed * w[id(p)] / tw
            room = MAX_MIN - p["min"]
            p["min"] += min(add, room)
            left += max(0.0, add - room)
        freed = left


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


def pmf(x, top):
    counts = np.bincount(np.clip(x, 0, top), minlength=top + 1) / len(x)
    arr = [round(float(v), 4) for v in counts]
    while arr and arr[-1] == 0:
        arr.pop()
    return arr


def simulate(p, opp, dvp):
    pos, r = p["pos"], p["rates"]
    mins = np.clip(RNG.normal(p["min"], p["sd_min"], N_SIM), 0, 48)
    f = {s: dvp_factor(dvp, opp, pos, s) for s in STATS}
    tpa = RNG.poisson(mins * r["tpa"] * f["tpa"])
    make_p = np.clip(p["pct"] * f["tpm"] / max(f["tpa"], 0.5), 0.12, 0.6)
    tpm = RNG.binomial(tpa, make_p)
    reb = RNG.poisson(mins * r["reb"] * f["reb"])
    ast = RNG.poisson(mins * r["ast"] * f["ast"])
    pts = 3 * tpm + 2 * RNG.poisson(np.maximum(mins * r["non3"] * f["pts"], 0) / 2)
    pra = pts + reb + ast
    out = {}
    for name, arr, top in [("tpm", tpm, 12), ("tpa", tpa, 25), ("reb", reb, 30),
                           ("ast", ast, 25), ("pts", pts, 70), ("pra", pra, 90)]:
        out[name] = {"mean": round(float(arr.mean()), 2), "pmf": pmf(arr, top)}
    out["make_p"] = round(float(make_p), 3)
    return out


def main():
    df = load_games()
    cur = int(df.season.max())
    slate = read_json(DATA_DIR / "slate.json", {"date": dt.date.today().isoformat(), "games": [], "rosters": {}})
    news = read_json(SITE_DATA / "news.json", [])
    dvp, lg = build_dvp(df, cur)
    tpm_lg = df[df.season == cur]
    lg_pct = float(tpm_lg.tpm.sum() / max(tpm_lg.tpa.sum(), 1))
    nstat = news_status(news, slate["date"])
    played = df.groupby("team").date.max()
    ob = OddsBlaze()
    spreads = ob.spreads() if ob.enabled and slate["games"] else {}
    if slate["games"] and not spreads:
        print("[model] no spreads (" + ("OddsBlaze returned nothing" if ob.enabled else "ODDSBLAZE_API_KEY not set") + "), blowout risk skipped")

    by_id = {pid: g for pid, g in df[df.season >= cur - 1].groupby("player_id")}
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
                prof = profile(g, cur, lg_pct)
                status = nstat.get(norm_name(ath["name"])) or espn_status(ath.get("injury"))
                entry = {"id": str(ath["id"]), "name": ath["name"], "team": team, "opp": opp, "home": home,
                         "pos": norm_pos(ath.get("pos")) if ath.get("pos") else g.iloc[-1].pos,
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
                    entry["flags"].append("New team")
                team_players.append(entry)
            allocate_minutes(team_players, outs)
            apply_blowout(team_players, (mk["home_spread"] if home else -mk["home_spread"]) if mk else None)
            yesterday = pd.Timestamp(slate["date"]) - pd.Timedelta(days=1)
            b2b = team in played and played[team] == yesterday
            for p in team_players:
                if b2b:
                    p["flags"].append("B2B")
                gain = p["min"] - p["pre_min"]
                if outs and gain >= 2:
                    p["flags"].append(f"+{gain:.0f} min ({', '.join(o['name'].split()[-1] for o in outs)} out)")
                sim = simulate(p, opp, dvp)
                out_players.append({
                    "id": p["id"], "name": p["name"], "team": team, "opp": opp, "home": home,
                    "pos": p["pos"], "min": round(p["min"], 1), "base_min": round(p["pre_min"], 1),
                    "pct": round(p["pct"], 3), "flags": p["flags"], "last10": p["last10"],
                    "dvp_rank": {s: dvp.get(opp, {}).get(p["pos"], {}).get(s, {}).get("rk") for s in STATS},
                    **sim,
                })
            for o in outs:
                out_players.append({"id": o["id"], "name": o["name"], "team": team, "opp": opp,
                                    "pos": o["pos"], "status": o["status"], "inactive": True})

    write_json(SITE_DATA / "dvp.json", {"season": cur, "positions": POSITIONS, "league": lg, "teams": dvp,
                                        "updated": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"})
    write_json(SITE_DATA / "projections.json", {
        "date": slate["date"], "season": cur, "games": games_out, "league_3p_pct": round(lg_pct, 3),
        "players": out_players, "updated": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"})
    print(f"[model] {len([p for p in out_players if not p.get('inactive')])} players projected")


if __name__ == "__main__":
    main()
