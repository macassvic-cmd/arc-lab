"""Play-by-play store: ESPN summary `plays` for every regular-season game, compacted to data/pbp_raw/<event_id>.json.gz (a local
cache like data/odds_*; never served). Zero Odds API credits. Shared by first_basket.py (v2) and the v3 stint / lineup / context
builders.

  python pbp_store.py --seasons 2025 2026 2024      backfill (skips games already stored)
  python pbp_store.py                               games of the last 3 days from the live box-score file
"""
import argparse
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
RAW = DATA_DIR / "pbp_raw"
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (arc-lab research)"


def events(seasons=None, days=None):
    """(event_id, date, season) for stored box-score games (live file + history file)."""
    frames = []
    for fn in ("player_games.csv", "player_games_history.csv"):
        p = DATA_DIR / fn
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"event_id": str}, usecols=["event_id", "date", "season"]))
    d = pd.concat(frames).drop_duplicates("event_id")
    if seasons:
        d = d[d.season.isin([int(s) for s in seasons])]
    if days:
        d = d[pd.to_datetime(d.date) >= pd.Timestamp.today().normalize() - pd.Timedelta(days=days)]
    return d.sort_values("date").reset_index(drop=True)


def get(url, tries=3):
    for i in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return None


def clock_secs(s):
    try:
        if ":" in s:
            m, sec = s.split(":"); return int(m) * 60 + float(sec)
        return float(s)
    except (ValueError, AttributeError):
        return None


def compact(summary):
    comp = summary.get("header", {}).get("competitions", [{}])[0]
    teams = {c.get("homeAway"): {"id": c["team"]["id"], "abbr": c["team"].get("abbreviation"), "score": c.get("score")} for c in comp.get("competitors", [])}
    gi = summary.get("gameInfo", {})
    meta = {"date": comp.get("date"), "neutral": comp.get("neutralSite"), "teams": teams,
            "officials": [o.get("displayName") for o in gi.get("officials", [])],
            "venue": {k: (gi.get("venue") or {}).get(k) for k in ("id", "fullName")} | {"city": ((gi.get("venue") or {}).get("address") or {}).get("city"), "state": ((gi.get("venue") or {}).get("address") or {}).get("state")},
            "attendance": gi.get("attendance"),
            "pickcenter": [{"provider": (p.get("provider") or {}).get("name"), "spread": p.get("spread"), "total": p.get("overUnder"), "details": p.get("details")} for p in summary.get("pickcenter", [])]}
    plays = []
    for p in summary.get("plays", []):
        plays.append({"seq": int(p.get("sequenceNumber") or 0), "period": (p.get("period") or {}).get("number"), "clock": (p.get("clock") or {}).get("displayValue"),
                      "secs": clock_secs((p.get("clock") or {}).get("displayValue")), "type_id": (p.get("type") or {}).get("id"), "type": (p.get("type") or {}).get("text"),
                      "text": p.get("text"), "team": (p.get("team") or {}).get("id"), "scoring": bool(p.get("scoringPlay")), "value": p.get("scoreValue"),
                      "att": p.get("pointsAttempted"), "shooting": bool(p.get("shootingPlay")), "home": p.get("homeScore"), "away": p.get("awayScore"),
                      "ath": [(a.get("athlete") or {}).get("id") for a in p.get("participants", [])], "x": (p.get("coordinate") or {}).get("x"), "y": (p.get("coordinate") or {}).get("y"),
                      "wall": p.get("wallclock")})
    starters = {}
    for block in summary.get("boxscore", {}).get("players", []):
        tid = block["team"]["id"]
        for stat in block.get("statistics", []):
            starters[tid] = [a["athlete"]["id"] for a in stat.get("athletes", []) if a.get("starter")]
            break
    return {"meta": meta, "starters": starters, "plays": plays}


def store(event_id, date, season):
    out = RAW / f"{event_id}.json.gz"
    if out.exists():
        return "cached"
    summ = get(f"{BASE}/summary?event={event_id}")
    if summ is None:
        return "fetch_failed"
    doc = {"event_id": str(event_id), "date": date, "season": int(season), "fetched": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **compact(summ)}
    doc["status"] = "ok" if doc["plays"] else "no_plays"
    RAW.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    return doc["status"]


def load(event_id):
    p = RAW / f"{event_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", help="ESPN season years, e.g. 2025 = 2024-25")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.4)
    a = ap.parse_args()
    ev = events(a.seasons) if a.seasons else events(days=a.days)
    counts, t0 = {}, time.time()
    for i, r in enumerate(ev.itertuples(), 1):
        st = store(r.event_id, r.date, r.season)
        counts[st] = counts.get(st, 0) + 1
        if st != "cached":
            time.sleep(a.sleep)
        if i % 100 == 0 or i == len(ev):
            print(f"[pbp] {i}/{len(ev)} {r.date} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"[pbp] done {counts}")
