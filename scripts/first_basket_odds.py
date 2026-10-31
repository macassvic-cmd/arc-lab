"""First basket: market odds as a DISPLAY layer next to the model's P (the Performance record stays accuracy-only).

Source, chosen openly: The Odds API, market `player_first_basket` ("First Basket Scorer", Yes/No, US books), fetched one event
at a time via /v4/sports/basketball_nba/events/{id}/odds with regions=us. Checked 2026-10-22: DraftKings, FanDuel, BetMGM,
Caesars (williamhill_us), BetRivers, Fanatics and MyBookie returned prices; cost 1 credit per game per pull (x-requests-last).
The /events listing is free. ODDS_API_KEY comes from the environment (GitHub secret / the user's env) and is never printed or
written. Nothing is scraped from any book's site.

  python first_basket_odds.py pull   for every game on tonight's slate that has NOT tipped: pull the market, append every book's
                                     Yes price to data/first_basket_odds.csv (timestamped, append-only) and the credit cost to
                                     data/first_basket_odds_credits.csv, then rebuild docs/data/first_basket_today.json
  python first_basket_odds.py pull <event_id> ...   the same for those untipped games only (the confirmed-lineup refresh, 2026-10-31:
                                     one credit for the repriced game, not the slate)

A price seen after tip is worthless as a record: tipped games are never requested. The tab shows, per starter, the last pre-tip
DraftKings and FanDuel prices, their raw implied probability (vig included, nothing removed) and the gap = model P minus the
lower implied of the two.
"""
import csv
import datetime as dt
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, NBA_TEAMS, SITE_DATA, read_json
from first_basket_model import TEAM_NAMES, now

ODDS = DATA_DIR / "first_basket_odds.csv"
CREDITS = DATA_DIR / "first_basket_odds_credits.csv"
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKET, REGIONS = "player_first_basket", "us"
SHOW_BOOKS = ("draftkings", "fanduel")
ODDS_FIELDS = ["pulled_at", "date", "event_id", "odds_event_id", "tip", "home", "away", "book", "player", "price", "implied", "market_last_update"]
CREDIT_FIELDS = ["pulled_at", "date", "requests", "credits", "remaining", "games_pulled", "games_skipped_tipped", "games_unmatched"]
NICK_TO_ABBR = {nick: abbr for abbr, nick in TEAM_NAMES.items() if abbr in NBA_TEAMS}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w ]", "", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)).strip()


def abbr_for(full_name):
    for nick, abbr in NICK_TO_ABBR.items():
        if full_name.endswith(nick):
            return abbr
    return None


def implied(american):
    a = float(american)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def get(path, params, key):
    r = requests.get(f"{BASE}/{path}", params={**params, "apiKey": key}, timeout=30)
    return r


def append(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def pull(events=None):
    """events: restrict the pull to these slate event_ids (still untipped only); None = every untipped game."""
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        print("[first-basket-odds] ODDS_API_KEY not set: no pull"); return 0
    P = read_json(SITE_DATA / "projections.json", None)
    if not P or not P.get("games"):
        print("[first-basket-odds] no slate"); return 0
    t = now(); ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    untipped, tipped = [], 0
    for g in P["games"]:
        if t < dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00")):
            untipped.append(g)
        else:
            tipped += 1
    if events is not None:
        want = {str(e) for e in events}; untipped = [g for g in untipped if str(g["event_id"]) in want]
        if not untipped:
            print(f"[first-basket-odds] none of {sorted(want)} is an untipped slate game: no pull, no credits"); return 0
    if not untipped:
        print(f"[first-basket-odds] every game has tipped ({tipped}): no pull, no credits"); return 0
    ev = get("events", {"dateFormat": "iso"}, key)   # free (0 credits)
    if ev.status_code != 200:
        print(f"[first-basket-odds] events listing HTTP {ev.status_code}: no pull"); return 0
    by_teams = {(abbr_for(e["home_team"]), abbr_for(e["away_team"])): e for e in ev.json()}
    rows, reqs, credits, remaining, unmatched, pulled = [], 0, 0, None, [], 0
    for g in untipped:
        e = by_teams.get((g["home"], g["away"]))
        if not e:
            unmatched.append(f"{g['away']}@{g['home']}"); continue
        r = get(f"events/{e['id']}/odds", {"regions": REGIONS, "markets": MARKET, "oddsFormat": "american", "dateFormat": "iso"}, key)
        reqs += 1; credits += int(r.headers.get("x-requests-last") or 0); remaining = r.headers.get("x-requests-remaining", remaining)
        if r.status_code != 200:
            print(f"[first-basket-odds] {g['away']}@{g['home']}: HTTP {r.status_code}"); continue
        n_before = len(rows)
        for b in r.json().get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") != MARKET:
                    continue
                for o in m.get("outcomes", []):
                    if o.get("name") != "Yes" or not o.get("description"):
                        continue
                    rows.append({"pulled_at": ts, "date": P["date"], "event_id": g["event_id"], "odds_event_id": e["id"], "tip": g["start"], "home": g["home"], "away": g["away"],
                                 "book": b["key"], "player": o["description"], "price": int(o["price"]), "implied": round(implied(o["price"]), 4), "market_last_update": m.get("last_update")})
        pulled += len(rows) > n_before
        time.sleep(0.3)
    if rows:
        append(ODDS, ODDS_FIELDS, rows)
    append(CREDITS, CREDIT_FIELDS, [{"pulled_at": ts, "date": P["date"], "requests": reqs, "credits": credits, "remaining": remaining, "games_pulled": pulled,
                                     "games_skipped_tipped": tipped, "games_unmatched": len(unmatched)}])
    print(f"[first-basket-odds] {P['date']}: {pulled} games priced, {len(rows)} book prices, {tipped} tipped games skipped, {len(unmatched)} unmatched {unmatched or ''}; "
          f"{reqs} requests = {credits} credits, {remaining} remaining")
    try:
        from first_basket_model import build_today
        build_today(P["date"], slate=P)
    except Exception as ex:   # the odds layer must never break the model's own files
        print(f"[first-basket-odds] today rebuild failed: {ex}")
    return len(rows)


def attach(games, date):
    """For build_today: add the last pre-tip DraftKings / FanDuel Yes price, raw implied and gap to every starter of every game."""
    if not ODDS.exists() or not games:
        return
    O = pd.read_csv(ODDS, dtype={"event_id": str, "odds_event_id": str})
    O = O[O.date == date]
    for g in games:
        d = O[(O.event_id == g["event_id"]) & (O.pulled_at < g["tip"])]   # same ISO format: string order is time order
        if d.empty:
            continue
        last = d.sort_values("pulled_at").groupby(["book", "player"]).tail(1)
        idx = {(r.book, norm(r.player)): r for r in last.itertuples()}
        g["odds"] = {"pulled_at": str(last.pulled_at.max()), "books": sorted(last.book.unique()), "shown": list(SHOW_BOOKS), "market": MARKET,
                     "note": "raw implied from the American Yes price, vig included; gap = model P minus the lower implied of the books shown; display only"}
        for s in g["starters"]:
            q = {}
            for book in SHOW_BOOKS:
                r = idx.get((book, norm(s["player"])))
                if r is not None:
                    q[book] = {"price": int(r.price), "implied": float(r.implied)}
            if q:
                best = min(v["implied"] for v in q.values())
                s["odds"] = q; s["implied_best"] = round(best, 4); s["gap"] = round(float(s["p_first_basket"]) - best, 4)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
    {"pull": pull}[cmd](events=sys.argv[2:] or None)
