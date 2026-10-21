"""Play-by-play store: ESPN summary `plays` for every regular-season game, compacted to data/pbp_raw/<event_id>.json.gz (a local
cache like data/odds_*; never served). Zero Odds API credits. Shared by first_basket.py / first_shot.py and the v3 stint /
lineup / context builders. Postseason and preseason games (first_basket_v2_rule.md, section 1) are enumerated from the ESPN
scoreboard with `seasontype` 3 / 1 day by day and stored with their box-score roster, since they are outside the box-score files.

  python pbp_store.py --seasons 2025 2026 2024      backfill (skips games already stored)
  python pbp_store.py                               games of the last 3 days from the live box-score file
  python pbp_store.py --phase postseason --seasons 2025 2026     play-in + playoffs of 2024-25 and 2025-26
  python pbp_store.py --phase preseason --seasons 2025 2026 2027  the 2024, 2025 and 2026 preseasons
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


def get(url, tries=3, params=None):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    return None


# Day spans scanned on the ESPN scoreboard for games outside the regular season (season = ESPN season year, e.g. 2026 = 2025-26).
PHASE_SPANS = {"postseason": (3, lambda y: (dt.date(y, 4, 10), dt.date(y, 6, 30))), "preseason": (1, lambda y: (dt.date(y - 1, 9, 25), dt.date(y - 1, 10, 25)))}
EXTRA = DATA_DIR / "pbp_extra_events.csv"   # every postseason / preseason game found, committed so Actions can label without rescanning


def extra_events(phase, seasons, sleep=0.15):
    """(event_id, date, season, phase, home, away) for completed NBA-vs-NBA games of the phase; appended to data/pbp_extra_events.csv."""
    from common import NBA_TEAMS
    seasontype, span = PHASE_SPANS[phase]
    known = pd.read_csv(EXTRA, dtype={"event_id": str}) if EXTRA.exists() else pd.DataFrame(columns=["event_id", "date", "season", "phase", "home", "away"])
    rows = []
    for season in seasons:
        day, end = span(int(season))
        while day <= end:
            data = get(f"{BASE}/scoreboard", params={"dates": day.strftime("%Y%m%d"), "seasontype": seasontype}) or {}
            for e in data.get("events", []):
                comp = e["competitions"][0]
                tm = {c["homeAway"]: (c.get("team") or {}).get("abbreviation") for c in comp["competitors"]}
                if (e.get("season") or {}).get("type") != seasontype or not comp["status"]["type"].get("completed") or not all(a in NBA_TEAMS for a in tm.values()):
                    continue   # not this phase, not final, or an exhibition against a non-NBA club
                rows.append({"event_id": e["id"], "date": day.isoformat(), "season": int(season), "phase": phase, "home": tm["home"], "away": tm["away"]})
            day += dt.timedelta(days=1); time.sleep(sleep)
    found = pd.DataFrame(rows, columns=known.columns)
    allx = pd.concat([known, found]).drop_duplicates("event_id").sort_values(["date", "event_id"])
    allx.to_csv(EXTRA, index=False)
    print(f"[pbp] {phase} {list(seasons)}: {len(found)} games on the scoreboard, {len(allx)} in {EXTRA.name}")
    return found


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
    starters, roster = {}, {}
    for block in summary.get("boxscore", {}).get("players", []):
        tid = block["team"]["id"]
        for stat in block.get("statistics", []):
            starters[tid] = [a["athlete"]["id"] for a in stat.get("athletes", []) if a.get("starter")]
            labels = [l.upper() for l in stat.get("labels", [])]
            roster[tid] = [{"id": (a.get("athlete") or {}).get("id"), "name": (a.get("athlete") or {}).get("displayName"), "pos": ((a.get("athlete") or {}).get("position") or {}).get("abbreviation"),
                            "starter": int(bool(a.get("starter"))), "min": dict(zip(labels, a.get("stats") or [])).get("MIN"), "dnp": bool(a.get("didNotPlay"))}
                           for a in stat.get("athletes", []) if (a.get("athlete") or {}).get("id")]
            break
    return {"meta": meta, "starters": starters, "roster": roster, "plays": plays}


def store(event_id, date, season, phase="regular"):
    out = RAW / f"{event_id}.json.gz"
    if out.exists():
        return "cached"
    summ = get(f"{BASE}/summary?event={event_id}")
    if summ is None:
        return "fetch_failed"
    doc = {"event_id": str(event_id), "date": date, "season": int(season), "phase": phase, "fetched": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **compact(summ)}
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
    ap.add_argument("--phase", choices=list(PHASE_SPANS), help="postseason / preseason games of --seasons (scoreboard scan) instead of regular-season box-score games")
    a = ap.parse_args()
    if a.phase:
        ev = extra_events(a.phase, a.seasons or [])
    else:
        ev = events(a.seasons) if a.seasons else events(days=a.days)
        ev["phase"] = "regular"
    counts, t0 = {}, time.time()
    for i, r in enumerate(ev.itertuples(), 1):
        st = store(r.event_id, r.date, r.season, r.phase)
        counts[st] = counts.get(st, 0) + 1
        if st != "cached":
            time.sleep(a.sleep)
        if i % 100 == 0 or i == len(ev):
            print(f"[pbp] {i}/{len(ev)} {r.date} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"[pbp] done {counts}")
