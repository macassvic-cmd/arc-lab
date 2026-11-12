"""Sportsbook stat-line capture for the FP rebuild (CEO order 2026-11-12): The Odds API, NBA player props, archived raw with
timestamps. Inventory only: nothing here prices, grades or compares a line to an outcome.

Source: The Odds API event-odds endpoint, one call per game (player props are not on the bulk /odds endpoint):
  GET /v4/sports/basketball_nba/events                      free listing (also returns the usage headers)
  GET /v4/sports/basketball_nba/events/{id}/odds?regions=us&markets=<the seven below>&oddsFormat=american
Markets: player_points, player_rebounds, player_assists, player_points_rebounds_assists, player_steals, player_blocks,
player_turnovers. Region us only, every US book the API returns. ODDS_API_KEY comes from the environment exactly as
first_basket_odds.py reads it; it is never printed, logged or written (the stored record holds response bodies and usage
headers only, never a request URL).

Rules built in:
  - only games that have NOT tipped are requested (commence_time in the future and, when the slate file knows the game, its
    tip in the future); a game in progress or finished is never called
  - shared pool floor: remaining credits are read from the usage headers before the first paid call and after every call;
    under POOL_FLOOR (20,000) the pull stops and reports
  - daily cap: DAILY_CAP (300) credits across all pulls of the UTC day; a pull whose projected cost (markets x regions x games)
    would cross it is not started
  - storage: the RAW responses go ONLY to the private arc-lab-data checkout named by ARC_LAB_DATA_DIR, gzipped, one file per
    day (oddsapi_props_YYYY-MM-DD.jsonl.gz, one JSON object per pull). The script refuses to write raw data inside this
    repository. Only the credit ledger (data/oddsapi_props_credits.csv) and derived coverage counts
    (data/oddsapi_props_coverage.json) are written here.

  python oddsapi_props.py pull --label test|morning|pretip|board [--push]   one pull of tonight's untipped games
  python oddsapi_props.py coverage [--file PATH] [--pull N]                 coverage inventory of an archived pull (default: latest)
  python oddsapi_props.py budget                                           free: usage headers + today's ledger, no paid call
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, ROOT, norm_name, read_json, write_json  # noqa: E402
from teams import team_abbr  # noqa: E402

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKETS = ["player_points", "player_rebounds", "player_assists", "player_points_rebounds_assists", "player_steals", "player_blocks", "player_turnovers"]
STAT = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_points_rebounds_assists": "PRA",
        "player_steals": "STL", "player_blocks": "BLK", "player_turnovers": "TO"}
REGIONS = "us"
POOL_FLOOR = 20000
DAILY_CAP = 300
ET = ZoneInfo("America/New_York")
LEDGER = DATA_DIR / "oddsapi_props_credits.csv"
COVERAGE = DATA_DIR / "oddsapi_props_coverage.json"
BOARD = DATA_DIR / "board_latest.json"
SLATE = DATA_DIR / "slate.json"
LEDGER_FIELDS = ["pulled_at", "label", "slate_date", "games_requested", "games_skipped_tipped", "credits", "projected", "remaining_before", "remaining_after", "used_after", "status"]
USAGE = ("x-requests-remaining", "x-requests-used", "x-requests-last")


def now():
    return dt.datetime.now(dt.timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def key():
    k = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not k:
        sys.exit("[oddsapi-props] ODDS_API_KEY not set: no pull")
    return k


def get(path, params, k):
    try:
        r = requests.get(f"{BASE}/{path}".rstrip("/"), params={**params, "apiKey": k}, timeout=30)
    except requests.RequestException as e:
        # a requests error message carries the full URL, key included: never let it reach a log or a traceback
        sys.exit(f"[oddsapi-props] request for /{path} failed ({type(e).__name__}); no further calls")
    return r, {h: r.headers.get(h) for h in USAGE}


def num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def data_dir():
    d = (os.environ.get("ARC_LAB_DATA_DIR") or "").strip()
    if not d:
        sys.exit("[oddsapi-props] ARC_LAB_DATA_DIR not set: raw responses go only to a checkout of the private arc-lab-data repo. No pull.")
    p = Path(d).resolve()
    if not (p / ".git").exists():
        sys.exit(f"[oddsapi-props] {p} is not a git checkout (expected the private arc-lab-data repo). No pull.")
    if p == ROOT.resolve() or ROOT.resolve() in p.parents:
        sys.exit("[oddsapi-props] ARC_LAB_DATA_DIR points inside the public arc-lab repo: refused, nothing raw is written here.")
    return p


def ledger_today(day):
    if not LEDGER.exists():
        return 0
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return sum(num(r["credits"]) or 0 for r in csv.DictReader(f) if r["pulled_at"][:10] == day)


def ledger_add(row):
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in LEDGER_FIELDS})


def untipped(listing, t):
    """(events to request, number skipped as tipped) for tonight's slate: the ET date of the earliest future game."""
    slate = read_json(SLATE, {}) or {}
    tips = {}
    for g in slate.get("games") or []:
        tips[(team_abbr(g.get("away")), team_abbr(g.get("home")))] = g.get("start")
    fut = [e for e in listing if parse(e["commence_time"]) > t]
    if not fut:
        return [], len(listing), None
    day = slate.get("date") or min(parse(e["commence_time"]) for e in fut).astimezone(ET).date().isoformat()
    todo, skipped = [], 0
    for e in listing:
        if parse(e["commence_time"]).astimezone(ET).date().isoformat() != day:
            continue
        tip = tips.get((team_abbr(e.get("away_team")), team_abbr(e.get("home_team"))))
        if parse(e["commence_time"]) <= t or (tip and parse(tip) <= t):
            skipped += 1
            continue
        todo.append(e)
    return todo, skipped, day


def budget():
    k = key()
    r, u = get("events", {}, k)
    t = now()
    print(f"[oddsapi-props] usage headers: remaining {u['x-requests-remaining']}, used {u['x-requests-used']}; ledger today {ledger_today(iso(t)[:10])} of {DAILY_CAP}; pool floor {POOL_FLOOR}")
    return r, u


def pull(label, push=False):
    k = key(); out_dir = data_dir(); t = now(); day_utc = iso(t)[:10]
    r, u = get("events", {}, k)
    if r.status_code != 200:
        sys.exit(f"[oddsapi-props] events listing HTTP {r.status_code}: no pull")
    listing = r.json(); remaining = num(u["x-requests-remaining"])
    todo, skipped, slate_date = untipped(listing, t)
    projected = len(MARKETS) * len(todo)
    spent = ledger_today(day_utc)
    base = {"pulled_at": iso(t), "label": label, "slate_date": slate_date or "", "games_skipped_tipped": skipped, "projected": projected, "remaining_before": remaining}
    if remaining is None or remaining < POOL_FLOOR:
        ledger_add({**base, "games_requested": 0, "credits": 0, "status": "stopped_pool_floor"})
        sys.exit(f"[oddsapi-props] STOP: shared pool at {remaining} credits, under the {POOL_FLOOR} floor. Nothing pulled.")
    if not todo:
        print(f"[oddsapi-props] no untipped games for {slate_date} ({skipped} tipped): nothing pulled"); return None
    if spent + projected > DAILY_CAP:
        ledger_add({**base, "games_requested": 0, "credits": 0, "status": "stopped_daily_cap"})
        sys.exit(f"[oddsapi-props] STOP: today's ledger {spent} + projected {projected} would cross the {DAILY_CAP} daily cap. Nothing pulled.")
    rec = {"captured_at": iso(t), "label": label, "slate_date": slate_date, "regions": REGIONS, "markets": MARKETS, "events_listing": listing, "event_odds": []}
    credits, status = 0, "ok"
    for e in todo:
        if parse(e["commence_time"]) <= now():
            continue                              # tipped while the pull was running: never requested
        r, u = get(f"events/{e['id']}/odds", {"regions": REGIONS, "markets": ",".join(MARKETS), "oddsFormat": "american"}, k)
        cost = num(u["x-requests-last"]) or 0; credits += cost; remaining = num(u["x-requests-remaining"]) if u["x-requests-remaining"] is not None else remaining
        try:
            body = r.json()
        except ValueError:
            body = {"non_json_body": r.text[:2000]}
        rec["event_odds"].append({"event_id": e["id"], "requested_at": iso(now()), "http_status": r.status_code, "usage": u, "body": body})
        if remaining is not None and remaining < POOL_FLOOR:
            status = "stopped_pool_floor_mid_pull"; break
        if spent + credits > DAILY_CAP:
            status = "stopped_daily_cap_mid_pull"; break
        time.sleep(0.25)
    rec["cost"] = {"credits": credits, "projected": projected, "remaining_after": remaining}
    path = out_dir / f"oddsapi_props_{slate_date}.jsonl.gz"
    with gzip.open(path, "ab") as f:              # one gzip member per pull; the file reads as one JSON object per line
        f.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))
    ledger_add({**base, "games_requested": len(rec["event_odds"]), "credits": credits, "remaining_after": remaining, "used_after": (rec["event_odds"][-1]["usage"]["x-requests-used"] if rec["event_odds"] else ""), "status": status})
    print(f"[oddsapi-props] {label}: {len(rec['event_odds'])} games requested ({skipped} tipped, skipped), cost {credits} credits (projected {projected}), remaining {remaining}, today {spent + credits} of {DAILY_CAP}; archived {path.name} ({path.stat().st_size} bytes); status {status}")
    cov = coverage(path, -1)
    if push:
        for cmd in (["git", "add", path.name], ["git", "commit", "-q", "-m", f"oddsapi props {slate_date} {label} {iso(t)}"], ["git", "push", "-q"]):
            subprocess.run(cmd, cwd=out_dir, check=True)
        print("[oddsapi-props] raw archive pushed to arc-lab-data")
    if status != "ok":
        sys.exit(f"[oddsapi-props] STOP: {status}")
    return cov


def read_pulls(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def coverage(path=None, which=-1):
    """Inventory of one archived pull against the Underdog FP players on the repo's board copy. Counts only: no price is
    compared with a line and no outcome is read."""
    if path is None:
        files = sorted(data_dir().glob("oddsapi_props_*.jsonl.gz"))
        if not files:
            sys.exit("[oddsapi-props] no archived pull yet")
        path = files[-1]
    rec = read_pulls(path)[which]
    slate = rec["slate_date"]
    book_players = {}                              # (name key, frozenset(event teams)) -> {stat: {book,...}}
    books_by_market = {s: set() for s in STAT.values()}
    names = {}
    for ev in rec["event_odds"]:
        b = ev.get("body") or {}
        teams = frozenset(x for x in (team_abbr(b.get("away_team")), team_abbr(b.get("home_team"))) if x)
        for bm in b.get("bookmakers") or []:
            for m in bm.get("markets") or []:
                s = STAT.get(m.get("key"))
                if not s:
                    continue
                for o in m.get("outcomes") or []:
                    if o.get("point") is None or not o.get("description"):
                        continue
                    k = (norm_name(o["description"]), teams); names[k] = o["description"]
                    book_players.setdefault(k, {}).setdefault(s, set()).add(bm["key"]); books_by_market[s].add(bm["key"])
    board = read_json(BOARD, {}) or {}
    ud = {}
    for e in board.get("entries") or []:
        if (e.get("market") or "").strip().lower() != "fantasy points" or not e.get("scheduled_at"):
            continue
        if ((e.get("platform_ids") or {}).get("line_type") or "balanced") != "balanced":
            continue
        if parse(e["scheduled_at"]).astimezone(ET).date().isoformat() != slate:
            continue
        ud[(norm_name(e["player"]), team_abbr(e.get("team")))] = e["player"]
    def find(nk, team):
        hits = [k for k in book_players if k[0] == nk and (not team or team in k[1] or not k[1])]
        return hits[0] if hits else None
    matched, ud_only = {}, []
    for (nk, team), name in sorted(ud.items(), key=lambda kv: kv[1]):
        k = find(nk, team)
        if k:
            matched[(nk, team)] = book_players[k]
        else:
            ud_only.append(f"{name} ({team})")
    used = {find(nk, team) for (nk, team) in matched}
    def n_with(stats, same_book=False):
        c = 0
        for st in matched.values():
            if not all(s in st for s in stats):
                continue
            c += 1 if not same_book else int(bool(set.intersection(*[st[s] for s in stats])))
        return c
    doc = {"captured_at": rec["captured_at"], "label": rec["label"], "slate_date": slate, "board_captured_at": board.get("captured_at"),
           "games_requested": len(rec["event_odds"]), "games_with_any_book": sum(1 for ev in rec["event_odds"] if (ev.get("body") or {}).get("bookmakers")),
           "credits": (rec.get("cost") or {}).get("credits"),
           "books_by_market": {s: sorted(v) for s, v in books_by_market.items()},
           "ud_fp_players": len(ud), "ud_fp_players_with_any_book_line": len(matched),
           "pts_reb_ast_any_books": n_with(("PTS", "REB", "AST")), "pts_reb_ast_one_book": n_with(("PTS", "REB", "AST"), same_book=True),
           "by_stat": {s: n_with((s,)) for s in STAT.values()},
           "all_six_any_books": n_with(("PTS", "REB", "AST", "STL", "BLK", "TO")), "all_six_one_book": n_with(("PTS", "REB", "AST", "STL", "BLK", "TO"), same_book=True),
           "book_players": len(book_players),
           "ud_fp_players_with_no_book_match": ud_only,
           "book_players_with_no_ud_fp_line": sorted(f"{names[k]} ({'/'.join(sorted(k[1]))})" for k in book_players if k not in used),
           "note": "Inventory only. A stat counts when at least one US book returned a line with a point for the player. Name matching is by normalized name within the game's two teams; the two unmatched lists exist so a spelling mismatch is not read as missing coverage."}
    hist = read_json(COVERAGE, {"pulls": []}) or {"pulls": []}
    hist["pulls"] = [p for p in hist["pulls"] if not (p["captured_at"] == doc["captured_at"] and p["label"] == doc["label"])] + [doc]
    write_json(COVERAGE, hist)
    show = {k: v for k, v in doc.items() if k not in ("ud_fp_players_with_no_book_match", "book_players_with_no_ud_fp_line", "note")}
    print(json.dumps(show, indent=1))
    print(f"UD FP players with no book match ({len(ud_only)}): {ud_only}")
    print(f"book players with no UD FP line: {len(doc['book_players_with_no_ud_fp_line'])} (full list in {COVERAGE.name})")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "coverage", "budget"])
    ap.add_argument("--label", default="test", choices=["test", "morning", "pretip", "board"])
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--file"); ap.add_argument("--pull", type=int, default=-1)
    a = ap.parse_args()
    if a.cmd == "pull":
        pull(a.label, push=a.push)
    elif a.cmd == "coverage":
        coverage(Path(a.file) if a.file else None, a.pull)
    else:
        budget()
