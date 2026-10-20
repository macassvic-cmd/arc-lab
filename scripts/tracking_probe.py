"""v3 item 2 capture check: can stats.nba.com / cdn.nba.com tracking and role dashboards be read from this environment?
Tries one request per endpoint family with a short timeout and writes the result as the `tracking` block of
docs/data/v3_coverage.json (numbers only). No credits, no data stored.
  python tracking_probe.py [--timeout 20]
"""
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SITE_DATA, read_json, write_json

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
     "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
     "x-nba-stats-origin": "stats", "x-nba-stats-token": "true"}
COMMON = "College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&Height=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=PerGame&PlayerExperience=&PlayerOrTeam=Player&PlayerPosition=&Season=2024-25&SeasonSegment=&SeasonType=Regular%20Season&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight="
ENDPOINTS = [
    ("touches / possession (leaguedashptstats Possessions)", f"https://stats.nba.com/stats/leaguedashptstats?{COMMON}&PtMeasureType=Possessions"),
    ("passing / potential assists (leaguedashptstats Passing)", f"https://stats.nba.com/stats/leaguedashptstats?{COMMON}&PtMeasureType=Passing"),
    ("drives (leaguedashptstats Drives)", f"https://stats.nba.com/stats/leaguedashptstats?{COMMON}&PtMeasureType=Drives"),
    ("catch-and-shoot (leaguedashptstats CatchShoot)", f"https://stats.nba.com/stats/leaguedashptstats?{COMMON}&PtMeasureType=CatchShoot"),
    ("rebound chances (leaguedashptstats Rebounding)", f"https://stats.nba.com/stats/leaguedashptstats?{COMMON}&PtMeasureType=Rebounding"),
    ("shot locations, corner vs above-break (leaguedashplayershotlocations)", f"https://stats.nba.com/stats/leaguedashplayershotlocations?{COMMON}&DistanceRange=By%20Zone&MeasureType=Base&PaceAdjust=N&PlusMinus=N&Rank=N&ShotClockRange=&GameSegment="),
    ("defender distance (leaguedashplayerptshot)", f"https://stats.nba.com/stats/leaguedashplayerptshot?{COMMON}&CloseDefDistRange=&ShotClockRange=&ShotDistRange=&TouchTimeRange=&DribbleRange=&GeneralRange=Overall&AheadBehind=&ClutchTime=&PointDiff="),
    ("box outs / contested rebounds (leaguehustlestatsplayer)", f"https://stats.nba.com/stats/leaguehustlestatsplayer?{COMMON}&PaceAdjust=N&PlusMinus=N&Rank=N"),
    ("play types, PnR handler (synergyplaytypes)", "https://stats.nba.com/stats/synergyplaytypes?LeagueID=00&PerMode=PerGame&PlayerOrTeam=P&SeasonType=Regular%20Season&SeasonYear=2024-25&PlayType=PRBallHandler&TypeGrouping=offensive"),
    ("per-game tracking box (boxscoreplayertrackv3)", "https://stats.nba.com/stats/boxscoreplayertrackv3?GameID=0022400197&LeagueID=00&endPeriod=0&endRange=28800&rangeType=0&startPeriod=0&startRange=0"),
    ("cdn.nba.com live play-by-play", "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022400197.json"),
]


def probe(timeout):
    out = []
    for name, url in ENDPOINTS:
        t = time.time()
        try:
            r = requests.get(url, headers=H, timeout=timeout)
            ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json")
            rows = None
            if ok:
                j = r.json(); rs = j.get("resultSets") or ([j["resultSet"]] if "resultSet" in j else [])
                rows = int(len(rs[0]["rowSet"])) if rs else None
            out.append({"endpoint": name, "status": r.status_code, "json": bool(ok), "rows": rows, "seconds": round(time.time() - t, 1)})
        except requests.RequestException as e:
            out.append({"endpoint": name, "status": None, "json": False, "rows": None, "seconds": round(time.time() - t, 1), "error": type(e).__name__})
        time.sleep(0.5)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--timeout", type=float, default=20); a = ap.parse_args()
    res = probe(a.timeout)
    ok = sum(1 for r in res if r["json"])
    env = "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local"
    block = {"environment": env, "attempted": len(res), "responded_with_json": ok, "timeout_seconds": a.timeout, "endpoints": res,
             "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "verdict": ("usable" if ok == len(res) else "partly usable" if ok else "unusable: no tracking or role endpoint answered; families (c), (d), (e) have no capture path from this environment")}
    cov = read_json(SITE_DATA / "v3_coverage.json", {})
    cov.setdefault("tracking", {})[env] = block
    write_json(SITE_DATA / "v3_coverage.json", cov)
    print(f"[tracking] {env}: {ok}/{len(res)} endpoints answered with JSON -> {block['verdict']}")
    for r in res:
        print(f"  {r['endpoint'][:60]:60s} status {r['status']} {r['seconds']}s {r.get('error', '')}")
