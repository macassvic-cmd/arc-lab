"""Pull NBA box scores from ESPN's public JSON endpoints.

Incremental (default): last 3 days through yesterday, plus today's slate.
Backfill:  python fetch_games.py --start 2025-10-21 --end 2026-04-12
"""
import argparse
import csv
import datetime as dt
import time

import requests

from common import DATA_DIR, norm_pos, read_json, write_json

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
PLAYER_CSV = DATA_DIR / "player_games.csv"
SEEN = DATA_DIR / "seen_events.json"
SLATE = DATA_DIR / "slate.json"
FIELDS = ["event_id", "date", "season", "team", "opp", "home", "player_id", "player", "pos",
          "starter", "min", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "reb",
          "ast", "stl", "blk", "tov", "pf", "pts"]
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (arc-lab research)"


def get(url, tries=3):
    for i in range(tries):
        try:
            r = S.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    print(f"[fail] {url}")
    return None


def et_today():
    return (dt.datetime.utcnow() - dt.timedelta(hours=5)).date()


def scoreboard(day):
    data = get(f"{BASE}/scoreboard?dates={day:%Y%m%d}&limit=50") or {}
    return data.get("events", [])


def split(v):
    try:
        a, b = v.split("-")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def num(v):
    try:
        return float(str(v).split(":")[0])
    except ValueError:
        return 0.0


def parse_box(event_id, date, season, summary):
    rows = []
    blocks = summary.get("boxscore", {}).get("players", [])
    if len(blocks) != 2:
        return rows
    comp = summary.get("header", {}).get("competitions", [{}])[0]
    home_ids = {c["team"]["id"] for c in comp.get("competitors", []) if c.get("homeAway") == "home"}
    abbrs = [b["team"]["abbreviation"] for b in blocks]
    for i, block in enumerate(blocks):
        team, opp = abbrs[i], abbrs[1 - i]
        home = int(block["team"]["id"] in home_ids)
        for stat in block.get("statistics", []):
            labels = [l.upper() for l in stat.get("labels", [])]
            for a in stat.get("athletes", []):
                vals = a.get("stats") or []
                if a.get("didNotPlay") or len(vals) != len(labels):
                    continue
                d = dict(zip(labels, vals))
                mins = num(d.get("MIN", 0))
                if mins <= 0:
                    continue
                fgm, fga = split(d.get("FG"))
                tpm, tpa = split(d.get("3PT"))
                ftm, fta = split(d.get("FT"))
                ath = a.get("athlete", {})
                rows.append({
                    "event_id": event_id, "date": date, "season": season, "team": team, "opp": opp,
                    "home": home, "player_id": ath.get("id"), "player": ath.get("displayName"),
                    "pos": norm_pos(ath.get("position", {}).get("abbreviation")),
                    "starter": int(bool(a.get("starter"))), "min": mins,
                    "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                    "oreb": num(d.get("OREB", 0)), "dreb": num(d.get("DREB", 0)),
                    "reb": num(d.get("REB", 0)), "ast": num(d.get("AST", 0)),
                    "stl": num(d.get("STL", 0)), "blk": num(d.get("BLK", 0)),
                    "tov": num(d.get("TO", 0)), "pf": num(d.get("PF", 0)), "pts": num(d.get("PTS", 0)),
                })
    return rows


def ingest(start, end):
    seen = set(read_json(SEEN, []))
    new_file = not PLAYER_CSV.exists()
    added = 0
    with open(PLAYER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        day = start
        while day <= end:
            for ev in scoreboard(day):
                eid = ev["id"]
                season = ev.get("season", {})
                comp = ev["competitions"][0]
                if eid in seen or season.get("type") != 2 or not comp["status"]["type"].get("completed"):
                    continue  # regular season, completed, not yet stored
                summ = get(f"{BASE}/summary?event={eid}")
                rows = parse_box(eid, day.isoformat(), season.get("year"), summ or {})
                if rows:
                    w.writerows(rows)
                    seen.add(eid)
                    added += 1
                time.sleep(0.4)
            day += dt.timedelta(days=1)
    write_json(SEEN, sorted(seen))
    print(f"[ingest] {added} new games")


def roster(team_id):
    data = get(f"{BASE}/teams/{team_id}/roster") or {}
    items = data.get("athletes", [])
    if items and "items" in items[0]:  # grouped format
        items = [a for grp in items for a in grp.get("items", [])]
    out = []
    for a in items:
        inj = a.get("injuries") or []
        out.append({"id": a.get("id"), "name": a.get("displayName"),
                    "pos": (a.get("position") or {}).get("abbreviation"),
                    "injury": inj[0].get("status") if inj else None})
    return out


def save_slate(day):
    games, rosters = [], {}
    for ev in scoreboard(day):
        if ev.get("season", {}).get("type") != 2:
            continue
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c["team"] for c in comp["competitors"]}
        games.append({"event_id": ev["id"], "start": ev.get("date"),
                      "home": teams["home"]["abbreviation"], "away": teams["away"]["abbreviation"]})
        for t in teams.values():
            rosters[t["abbreviation"]] = roster(t["id"])
            time.sleep(0.3)
    write_json(SLATE, {"date": day.isoformat(), "games": games, "rosters": rosters})
    print(f"[slate] {day}: {len(games)} games")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    today = et_today()
    if args.start:
        ingest(dt.date.fromisoformat(args.start),
               dt.date.fromisoformat(args.end) if args.end else today - dt.timedelta(days=1))
    else:
        ingest(today - dt.timedelta(days=3), today - dt.timedelta(days=1))
    save_slate(today)
