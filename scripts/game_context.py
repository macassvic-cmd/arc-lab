"""v3 data foundation, item 3: one context row per game -> data/game_context.csv. No credits, no modelling.

  python game_context.py                 all seasons in the box-score files
  python game_context.py --seasons 2025

Columns: event_id, date, season, home, away; per side (home_/away_): rest_days, b2b, travel_miles, tz_shift_hours, prev_city;
city, state, neutral, altitude (DEN or UTAH home); officials (crew, "|"-joined), crew_pace, crew_fouls (mean of each referee's
trailing-30 games before this date; refs with no history skipped), crew_refs_with_history; home_pace10 / away_pace10 (the team's
trailing-10 pace before this date); game pace and fouls (this game, for the trailing tables); vegas_total, vegas_spread,
vegas_provider (ESPN pickcenter when the summary carries it: null for every past game, filled going forward on the live slate).
Pace = possessions per 48 minutes, possessions = FGA - OREB + TOV + 0.44 FTA per team from the box score, averaged over both teams.
Environment only: the Vegas number is never a prop price.
"""
import argparse
import datetime as dt
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json
from pbp_store import RAW

REF_TRAIL, TEAM_TRAIL = 30, 10
ALTITUDE_HOME = {"DEN", "UTAH"}
# Arena city, latitude, longitude, IANA zone (ESPN abbreviations).
ARENAS = {
    "ATL": ("Atlanta", 33.7573, -84.3963, "America/New_York"), "BOS": ("Boston", 42.3662, -71.0621, "America/New_York"),
    "BKN": ("Brooklyn", 40.6826, -73.9754, "America/New_York"), "CHA": ("Charlotte", 35.2251, -80.8392, "America/New_York"),
    "CHI": ("Chicago", 41.8807, -87.6742, "America/Chicago"), "CLE": ("Cleveland", 41.4965, -81.6882, "America/New_York"),
    "DAL": ("Dallas", 32.7905, -96.8103, "America/Chicago"), "DEN": ("Denver", 39.7487, -105.0077, "America/Denver"),
    "DET": ("Detroit", 42.3410, -83.0552, "America/Detroit"), "GS": ("San Francisco", 37.7680, -122.3877, "America/Los_Angeles"),
    "HOU": ("Houston", 29.7508, -95.3621, "America/Chicago"), "IND": ("Indianapolis", 39.7640, -86.1555, "America/Indiana/Indianapolis"),
    "LAC": ("Inglewood", 33.9450, -118.3417, "America/Los_Angeles"), "LAL": ("Los Angeles", 34.0430, -118.2673, "America/Los_Angeles"),
    "MEM": ("Memphis", 35.1382, -90.0506, "America/Chicago"), "MIA": ("Miami", 25.7814, -80.1870, "America/New_York"),
    "MIL": ("Milwaukee", 43.0451, -87.9174, "America/Chicago"), "MIN": ("Minneapolis", 44.9795, -93.2760, "America/Chicago"),
    "NO": ("New Orleans", 29.9490, -90.0821, "America/Chicago"), "NY": ("New York", 40.7505, -73.9934, "America/New_York"),
    "OKC": ("Oklahoma City", 35.4634, -97.5151, "America/Chicago"), "ORL": ("Orlando", 28.5392, -81.3839, "America/New_York"),
    "PHI": ("Philadelphia", 39.9012, -75.1720, "America/New_York"), "PHX": ("Phoenix", 33.4457, -112.0712, "America/Phoenix"),
    "POR": ("Portland", 45.5316, -122.6668, "America/Los_Angeles"), "SA": ("San Antonio", 29.4270, -98.4375, "America/Chicago"),
    "SAC": ("Sacramento", 38.5802, -121.4997, "America/Los_Angeles"), "TOR": ("Toronto", 43.6435, -79.3791, "America/Toronto"),
    "UTAH": ("Salt Lake City", 40.7683, -111.9011, "America/Denver"), "WSH": ("Washington", 38.8981, -77.0209, "America/New_York"),
}
# Cities a team has called home inside these seasons (LAC: Crypto.com Arena through 2023-24, Intuit Dome from 2024-25).
HOME_CITIES = {t: {a[0]} for t, a in ARENAS.items()} | {"LAC": {"Inglewood", "Los Angeles"}}
# Cities ESPN names for neutral-site regular-season games (city -> lat, lon, zone).
CITIES = {a[0]: a[1:] for a in ARENAS.values()} | {
    "Mexico City": (19.4061, -99.0931, "America/Mexico_City"), "Paris": (48.8386, 2.3782, "Europe/Paris"), "Las Vegas": (36.0909, -115.1784, "America/Los_Angeles"),
    "Seattle": (47.6221, -122.3540, "America/Los_Angeles"), "Austin": (30.2672, -97.7431, "America/Chicago"), "Abu Dhabi": (24.4539, 54.3773, "Asia/Dubai"),
    "Tokyo": (35.6762, 139.6503, "Asia/Tokyo"), "Berlin": (52.5200, 13.4050, "Europe/Berlin"), "London": (51.5074, -0.1278, "Europe/London"),
    "Honolulu": (21.3069, -157.8583, "Pacific/Honolulu"), "Edmonton": (53.5461, -113.4938, "America/Edmonton"), "Vancouver": (49.2827, -123.1207, "America/Vancouver"),
    "San Diego": (32.7157, -117.1611, "America/Los_Angeles"), "Macau": (22.1987, 113.5439, "Asia/Macau"), "Melbourne": (-37.8136, 144.9631, "Australia/Melbourne"),
}


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def utc_offset(zone, date):
    return ZoneInfo(zone).utcoffset(dt.datetime(date.year, date.month, date.day, 20)).total_seconds() / 3600


def load_box():
    frames = []
    for fn in ("player_games.csv", "player_games_history.csv"):
        p = DATA_DIR / fn
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"event_id": str, "player_id": str}))
    b = pd.concat(frames).drop_duplicates(["event_id", "player_id"])
    for c in ("min", "fga", "fta", "oreb", "tov", "pf"):
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0)
    b["date"] = pd.to_datetime(b.date)
    return b


def meta_for(event_id):
    p = RAW / f"{event_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    periods = max((pl.get("period") or 0) for pl in d["plays"]) if d["plays"] else 4
    return {"meta": d["meta"], "periods": max(periods, 4)}


def build(seasons=None):
    box = load_box()
    if seasons:
        box = box[box.season.isin(seasons)]
    tg = box.groupby(["event_id", "team"]).agg(date=("date", "first"), season=("season", "first"), opp=("opp", "first"), home=("home", "first"), fga=("fga", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"), pf=("pf", "sum"), mins=("min", "sum")).reset_index()
    tg["poss"] = tg.fga - tg.oreb + tg.tov + 0.44 * tg.fta
    games = tg[tg.home == 1][["event_id", "date", "season", "team", "opp"]].rename(columns={"team": "home", "opp": "away"}).sort_values(["date", "event_id"]).reset_index(drop=True)
    metas = {e: meta_for(e) for e in games.event_id}
    # game-level pace / fouls (minutes from the play-by-play periods when stored, else 48)
    gp = {}
    for e, g in tg.groupby("event_id"):
        m = metas.get(e)
        glen = 48 + 5 * (m["periods"] - 4) if m else 48
        gp[e] = (float(g.poss.mean() * 48 / glen), float(g.pf.sum()), glen)
    games["pace"] = games.event_id.map(lambda e: round(gp[e][0], 2)); games["fouls"] = games.event_id.map(lambda e: gp[e][1]); games["game_minutes"] = games.event_id.map(lambda e: gp[e][2])
    # per-team schedule (rest, travel, trailing pace)
    sched = tg.sort_values(["team", "date"])
    prev_date = sched.groupby("team").date.shift(1)
    sched["rest_days"] = (sched.date - prev_date).dt.days
    sched["pace_game"] = sched.event_id.map(lambda e: gp[e][0])
    sched["pace10"] = sched.groupby("team").pace_game.transform(lambda x: x.shift(1).rolling(TEAM_TRAIL, min_periods=3).mean())
    sched["prev_event"] = sched.groupby("team").event_id.shift(1)
    srow = sched.set_index(["event_id", "team"])
    cov = defaultdict(int)

    def city_of(e):
        m = metas.get(e)
        c = ((m or {}).get("meta") or {}).get("venue", {}).get("city") if m else None
        home_abbr = games.loc[games.event_id == e, "home"]
        if c and c in CITIES:
            return c, CITIES[c], bool(c not in HOME_CITIES.get(home_abbr.iloc[0], set())) if len(home_abbr) else False
        if len(home_abbr) and home_abbr.iloc[0] in ARENAS:
            a = ARENAS[home_abbr.iloc[0]]
            return a[0], a[1:], False
        return None, None, False

    cities = {e: city_of(e) for e in games.event_id}
    # referee trailing tables
    ref_games = defaultdict(list)   # name -> [(date, pace, fouls)]
    for r in games.itertuples():
        m = metas.get(r.event_id)
        for name in (m["meta"].get("officials") if m else []) or []:
            ref_games[name].append((r.date, gp[r.event_id][0], gp[r.event_id][1]))
    for k in ref_games:
        ref_games[k].sort()
    rows = []
    for r in games.itertuples():
        m = metas.get(r.event_id)
        meta = (m or {}).get("meta") or {}
        city, coord, neutral = cities[r.event_id]
        row = dict(event_id=r.event_id, date=r.date.date().isoformat(), season=int(r.season), home=r.home, away=r.away, city=city, state=meta.get("venue", {}).get("state"), neutral=int(bool(neutral or meta.get("neutral"))),
                   altitude=int(r.home in ALTITUDE_HOME and not neutral), game_minutes=r.game_minutes, pace=r.pace, fouls=r.fouls)
        for side, team in (("home", r.home), ("away", r.away)):
            s = srow.loc[(r.event_id, team)]
            row[f"{side}_rest_days"] = None if pd.isna(s.rest_days) else int(s.rest_days)
            row[f"{side}_b2b"] = int(s.rest_days == 1) if not pd.isna(s.rest_days) else 0
            row[f"{side}_pace10"] = None if pd.isna(s.pace10) else round(float(s.pace10), 2)
            pc, pcoord = (cities.get(s.prev_event, (None, None, None))[:2] if isinstance(s.prev_event, str) else (None, None))
            row[f"{side}_prev_city"] = pc
            if pc and pcoord and coord:
                row[f"{side}_travel_miles"] = round(haversine(pcoord, coord), 0)
                row[f"{side}_tz_shift_hours"] = round(utc_offset(coord[2], r.date) - utc_offset(pcoord[2], r.date), 1)
            else:
                row[f"{side}_travel_miles"] = None; row[f"{side}_tz_shift_hours"] = None
        offs = meta.get("officials") or []
        row["officials"] = "|".join(offs) if offs else None
        paces, fouls = [], []
        for name in offs:
            hist = [(p, f) for d, p, f in ref_games[name] if d < r.date][-REF_TRAIL:]
            if hist:
                paces.append(np.mean([h[0] for h in hist])); fouls.append(np.mean([h[1] for h in hist]))
        row["crew_pace"] = round(float(np.mean(paces)), 2) if paces else None
        row["crew_fouls"] = round(float(np.mean(fouls)), 2) if fouls else None
        row["crew_refs_with_history"] = len(paces)
        pk = [p for p in (meta.get("pickcenter") or []) if p.get("total") is not None]
        row["vegas_total"] = pk[0]["total"] if pk else None
        row["vegas_spread"] = pk[0].get("spread") if pk else None
        row["vegas_provider"] = pk[0].get("provider") if pk else None
        cov["games"] += 1; cov["with_file"] += int(m is not None); cov["with_officials"] += int(bool(offs)); cov["with_venue_city"] += int(bool(meta.get("venue", {}).get("city")))
        cov["with_known_city"] += int(coord is not None); cov["with_vegas_total"] += int(bool(pk)); cov["with_travel"] += int(row["home_travel_miles"] is not None and row["away_travel_miles"] is not None)
        cov["with_crew_history"] += int(bool(paces)); cov["neutral"] += row["neutral"]; cov["altitude"] += row["altitude"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "game_context.csv", index=False)
    n = cov["games"] or 1
    block = dict(games=cov["games"], with_play_file=cov["with_file"], with_officials=cov["with_officials"], share_officials=round(cov["with_officials"] / n, 4),
                 with_venue_city=cov["with_venue_city"], share_venue_city=round(cov["with_venue_city"] / n, 4), with_known_city_coordinates=cov["with_known_city"],
                 with_travel_both_sides=cov["with_travel"], share_travel=round(cov["with_travel"] / n, 4), with_crew_history=cov["with_crew_history"],
                 with_vegas_total=cov["with_vegas_total"], share_vegas_total=round(cov["with_vegas_total"] / n, 4), neutral_site_games=cov["neutral"], altitude_games=cov["altitude"],
                 by_season=[dict(season=f"{s - 1}-{str(s)[2:]}", games=int(len(g)), with_officials=int(g.officials.notna().sum()), with_vegas_total=int(g.vegas_total.notna().sum()),
                                 with_travel=int(g.home_travel_miles.notna().sum()), mean_pace=round(float(g.pace.mean()), 2), b2b_team_games=int(g.home_b2b.sum() + g.away_b2b.sum())) for s, g in out.groupby("season")],
                 note=f"vegas_total comes from the ESPN summary's pickcenter block, which ESPN keeps only for recent games: present for {cov['with_vegas_total']} of {cov['games']} games here (none before 2025-26). Older seasons stay null without credits; the live slate run can fill it from the pre-game summary going forward. Context only, never a prop price.")
    rep = read_json(SITE_DATA / "v3_coverage.json", {})
    rep["game_context"] = block
    write_json(SITE_DATA / "v3_coverage.json", rep)
    return out, block


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int)
    a = ap.parse_args()
    import time; t0 = time.time()
    out, block = build(a.seasons)
    print(f"[game_context] {len(out)} games -> data/game_context.csv | {block} | {time.time() - t0:.0f}s")
