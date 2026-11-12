"""Sam Method, Part 1: the daily capture (CEO order 2026-11-12, optimizer spec v5 + Amendment A1). PAPER / SHADOW ONLY.

One pass per slate, everything inside the same few minutes:
  Kalshi      public API, no key: every open market of series KXNBA3PT / KXNBAPTS / KXNBAREB / KXNBAAST on today's slate
              (ticker, event, player, strike, yes/no bid and ask, ask sizes, volume). Games tipping before 14:00 PT are
              captured and flagged early_tip.
  Sportsbook  The Odds API per-event endpoint, eight markets (points / rebounds / assists / threes and their alternates),
              region us. Reference only, never a venue. The credit cost goes to the existing spend ledger
              data/oddsapi_props_credits.csv (label sam_capture).
  Model       the ladder model's P for every player-stat rung, from docs/data/projections.json. It must be TODAY'S slate
              (addition C): a snapshot dated another day is never used; the slate is skipped and logged instead.

Schedule (addition A): three cron triggers; the script itself runs only inside 13:00-13:25 America/Los_Angeles and only once per
slate (lock = the slate's capture file already exists). A manual run (--manual, workflow_dispatch) bypasses the time guard and
is stamped off_schedule=true; off-schedule slates count toward no gate (addition B).

Storage (standing decision 2026-11-12): the capture, raw API responses included, is one gzipped JSON per slate in the PRIVATE
arc-lab-data checkout named by ARC_LAB_DATA_DIR: sam_method/capture/YYYY-MM-DD.json.gz. Nothing raw is written to this
repository; --dry-run-dir writes the capture to a directory outside both repositories and skips the sportsbook pull. Request
URLs are never logged (the Odds API key rides in them); a failed request reports its type only.

  python sam_capture.py [--manual] [--dry-run-dir DIR]    -> prints the capture path (or the reason it skipped), exit 0
"""
import argparse
import datetime as dt
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, ROOT, SITE_DATA, norm_name, read_json  # noqa: E402
from teams import team_abbr  # noqa: E402
import oddsapi_props as op  # noqa: E402  (the shared credit ledger, pool floor and daily cap live there: one number, one place)

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"KXNBA3PT": "tpm", "KXNBAPTS": "pts", "KXNBAREB": "reb", "KXNBAAST": "ast"}
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ODDS_MARKETS = ["player_points", "player_points_alternate", "player_rebounds", "player_rebounds_alternate",
                "player_assists", "player_assists_alternate", "player_threes", "player_threes_alternate"]
ODDS_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "tpm"}
BOOK_ORDER = ("fanduel", "draftkings")
PT, ET = ZoneInfo("America/Los_Angeles"), ZoneInfo("America/New_York")
WINDOW = (dt.time(13, 0), dt.time(13, 25))
EARLY_TIP = dt.time(14, 0)
POOL_FLOOR, DAILY_CAP = op.POOL_FLOOR, op.DAILY_CAP
RUNS = DATA_DIR / "sam_method" / "capture_log.csv"
RUN_FIELDS = ["slate_date", "captured_at", "off_schedule", "status", "kalshi_markets", "kalshi_by_series", "games", "early_tip_games",
              "odds_credits", "odds_games", "model_updated", "model_players", "note"]
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def now():
    return dt.datetime.now(dt.timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def fetch(url, params=None, what="request"):
    """GET with the URL kept out of every message (the Odds API key is a query parameter)."""
    try:
        r = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"{what} failed ({type(e).__name__})") from None
    return r


def log_run(row):
    import csv
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    new = not RUNS.exists()
    with open(RUNS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in RUN_FIELDS})


def event_date(event_ticker):
    """KXNBAPTS-26NOV12DALMIN -> ('2026-11-12', 'DAL', 'MIN') (away, home as Kalshi writes them)."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})([A-Z]{6})$", event_ticker or "")
    if not m:
        return None, None, None
    return f"20{m.group(1)}-{MONTHS[m.group(2)]:02d}-{m.group(3)}", m.group(4)[:3], m.group(4)[3:]


def kalshi_markets(series):
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = fetch(f"{KALSHI}/markets", params, f"Kalshi {series}")
        if r.status_code != 200:
            raise RuntimeError(f"Kalshi {series} HTTP {r.status_code}")
        d = r.json(); out += d.get("markets") or []; cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out


def kalshi_market(ticker):
    r = fetch(f"{KALSHI}/markets/{ticker}", None, "Kalshi market")
    return (r.json() or {}).get("market") if r.status_code == 200 else None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def flatten(m, stat, tips):
    """One Kalshi market -> the derived row the optimizer reads (prices in dollars, sizes in contracts)."""
    date, away, home = event_date(m.get("event_ticker"))
    seg = (m.get("ticker") or "").split("-")
    team = seg[2][:3] if len(seg) >= 4 and seg[2][:3] in (away, home) else None
    title = m.get("title") or m.get("yes_sub_title") or ""
    player = title.split(":")[0].strip()
    strike = fnum(m.get("floor_strike"))
    x = int(strike + 0.5) if strike is not None else None            # floor_strike 9.5 = "10 or more"
    a, h = team_abbr(away), team_abbr(home)
    tip = tips.get((a, h))
    ya, yb, na, nb = (fnum(m.get(k)) for k in ("yes_ask_dollars", "yes_bid_dollars", "no_ask_dollars", "no_bid_dollars"))
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series": m.get("ticker", "").split("-")[0], "stat": stat, "slate_date": date,
            "away": a, "home": h, "game": f"{a}@{h}", "tip": tip, "player": player, "player_key": norm_name(player), "team": team_abbr(team), "x": x,
            "yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na,
            "yes_ask_size": fnum(m.get("yes_ask_size_fp")), "no_ask_size": fnum(m.get("yes_bid_size_fp")),   # a resting Yes bid IS the No offer
            "volume": fnum(m.get("volume_fp")), "open_interest": fnum(m.get("open_interest_fp")), "updated_time": m.get("updated_time")}


def odds_reference(events_today, key):
    """Raw per-event responses + usage, and the spend-ledger row. Never called for a tipped game."""
    out, credits, remaining, used = [], 0, None, None
    for e in events_today:
        if parse(e["commence_time"]) <= now():
            continue
        r = fetch(f"{ODDS_BASE}/events/{e['id']}/odds", {"regions": "us", "markets": ",".join(ODDS_MARKETS), "oddsFormat": "american", "apiKey": key}, "Odds API event odds")
        u = {h: r.headers.get(h) for h in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
        credits += int(float(u["x-requests-last"] or 0)); remaining = u["x-requests-remaining"] or remaining; used = u["x-requests-used"] or used
        try:
            body = r.json()
        except ValueError:
            body = {"non_json_body": r.text[:1000]}
        out.append({"event_id": e["id"], "requested_at": iso(now()), "http_status": r.status_code, "usage": u, "body": body})
        if remaining is not None and float(remaining) < POOL_FLOOR:
            break
        time.sleep(0.25)
    return out, credits, remaining, used


def morning_minutes(slate_date, snapshot_updated):
    """{player id: projected minutes} from the slate's MORNING build = the first committed docs/data/projections.json dated the
    slate (git history), plus that build's stamp. (None, None) when no such build can be read: the guard then fails closed.
    When the capture snapshot is itself the first build of the day, it is its own morning build (move 0)."""
    try:
        shas = subprocess.run(["git", "log", "--format=%H", "--since", f"{slate_date}T00:00:00Z", "--", "docs/data/projections.json"],
                              capture_output=True, text=True, cwd=ROOT, timeout=60).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None, None
    first = None
    for sha in reversed(shas):                                   # oldest first
        txt = subprocess.run(["git", "show", f"{sha}:docs/data/projections.json"], capture_output=True, text=True, cwd=ROOT, encoding="utf-8").stdout
        try:
            P = json.loads(txt)
        except ValueError:
            continue
        if P.get("date") == slate_date:
            first = P; break
    if first is None:
        cur = read_json(SITE_DATA / "projections.json", {}) or {}
        first = cur if cur.get("date") == slate_date and cur.get("updated") == snapshot_updated and not shas else None
    if first is None:
        return None, None
    return {str(p.get("id")): p.get("min") for p in first.get("players") or [] if not p.get("inactive")}, first.get("updated")


def reference_index(event_odds):
    """(player key, stat, X) -> {'yes': (american, book), 'no': (american, book)}: FanDuel first, else DraftKings, else the best
    available price, the source book named per rung. A book rung matches a Kalshi 'X or more' only at the same threshold
    (Over X-0.5); nothing is interpolated."""
    seen = {}
    for ev in event_odds:
        for bm in (ev.get("body") or {}).get("bookmakers") or []:
            for mk in bm.get("markets") or []:
                stat = ODDS_STAT.get((mk.get("key") or "").replace("_alternate", ""))
                if not stat:
                    continue
                for o in mk.get("outcomes") or []:
                    if o.get("point") is None or not o.get("description") or o.get("price") is None:
                        continue
                    x = o["point"] + 0.5
                    if abs(x - round(x)) > 1e-9:
                        continue
                    side = "yes" if (o.get("name") or "").lower() == "over" else "no" if (o.get("name") or "").lower() == "under" else None
                    if side:
                        seen.setdefault((norm_name(o["description"]), stat, int(round(x)), side), []).append((bm["key"], o["price"]))
    ref = {}
    for (pk, stat, x, side), lst in seen.items():
        pick = next((p for b in BOOK_ORDER for p in lst if p[0] == b), None) or max(lst, key=lambda p: p[1])
        ref.setdefault((pk, stat, x), {})[side] = {"price": pick[1], "book": pick[0]}
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", action="store_true", help="bypass the 13:00-13:25 PT guard; the capture is stamped off_schedule")
    ap.add_argument("--dry-run-dir", help="write the capture here (outside both repos) and skip the sportsbook pull")
    a = ap.parse_args()
    t = now(); t_pt = t.astimezone(PT)
    if not a.manual and not (WINDOW[0] <= t_pt.time() <= WINDOW[1]):
        print(f"[sam-capture] {t_pt.strftime('%H:%M')} PT is outside the 13:00-13:25 PT window: skipped (scheduled trigger, no capture)"); return None
    if a.dry_run_dir:
        out_dir = Path(a.dry_run_dir).resolve()
        if out_dir == ROOT.resolve() or ROOT.resolve() in out_dir.parents:
            sys.exit("[sam-capture] the dry-run directory is inside the public repo: refused")
        global RUNS
        RUNS = out_dir / "capture_log.csv"            # a dry run leaves no trace in this repository
    else:
        d = (os.environ.get("ARC_LAB_DATA_DIR") or "").strip()
        if not d or not (Path(d) / ".git").exists():
            sys.exit("[sam-capture] ARC_LAB_DATA_DIR is not a checkout of the private arc-lab-data repo: no capture (raw responses are stored nowhere else)")
        out_dir = Path(d).resolve()
        if out_dir == ROOT.resolve() or ROOT.resolve() in out_dir.parents:
            sys.exit("[sam-capture] ARC_LAB_DATA_DIR points inside the public repo: refused")
    slate = read_json(DATA_DIR / "slate.json", {}) or {}
    tips = {(team_abbr(g["away"]), team_abbr(g["home"])): g.get("start") for g in slate.get("games") or []}
    slate_date = t.astimezone(ET).date().isoformat()
    path = out_dir / "sam_method" / "capture" / f"{slate_date}.json.gz"
    base = {"slate_date": slate_date, "captured_at": iso(t), "off_schedule": int(bool(a.manual))}
    if path.exists():
        print(f"[sam-capture] {slate_date} already captured (lock): {path.name} exists, nothing done"); return None
    # ---- Kalshi
    raw, rows = {}, []
    for series, stat in SERIES.items():
        ms = kalshi_markets(series); raw[series] = ms
        rows += [r for r in (flatten(m, stat, tips) for m in ms) if r["slate_date"] == slate_date]
    by_series = {s: sum(1 for r in rows if r["series"] == s) for s in SERIES}
    if not rows:
        log_run({**base, "status": "skipped_no_kalshi_markets", "kalshi_markets": 0, "kalshi_by_series": json.dumps(by_series), "note": "zero Kalshi markets for the slate: a skipped slate, not an error"})
        print(f"[sam-capture] {slate_date}: zero Kalshi markets, slate skipped and logged"); return None
    games = sorted({r["game"] for r in rows})
    early = sorted({r["game"] for r in rows if r["tip"] and parse(r["tip"]).astimezone(PT).time() < EARLY_TIP and parse(r["tip"]).astimezone(PT).date() == t_pt.date()})
    for r in rows:
        r["early_tip"] = int(r["game"] in early)
        r["tipped_at_capture"] = int(bool(r["tip"]) and parse(r["tip"]) <= t)
    # ---- model snapshot: today's slate or nothing (addition C)
    proj = read_json(SITE_DATA / "projections.json", {}) or {}
    if proj.get("date") != slate_date:
        log_run({**base, "status": "skipped_stale_projection", "kalshi_markets": len(rows), "kalshi_by_series": json.dumps(by_series), "games": len(games),
                 "model_updated": proj.get("updated", ""), "note": f"projection snapshot is dated {proj.get('date')}, not {slate_date}: never used"})
        print(f"[sam-capture] {slate_date}: the projection snapshot is dated {proj.get('date')}: slate skipped and logged (a previous day's snapshot is never used)"); return None
    morning, morning_updated = morning_minutes(slate_date, proj.get("updated"))
    model = [{"player_id": p.get("id"), "player": p["name"], "player_key": norm_name(p["name"]), "team": p["team"], "opp": p.get("opp"), "min": p.get("min"),
              "min_morning": (morning or {}).get(str(p.get("id"))), "ladder": p.get("ladder")}
             for p in proj.get("players") or [] if not p.get("inactive") and p.get("ladder")]
    # ---- sportsbook reference (skipped on a dry run: its raw responses have no private home there)
    odds = {"status": "not_pulled_dry_run" if a.dry_run_dir else "not_pulled", "event_odds": [], "credits": 0}
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not a.dry_run_dir and key:
        r = fetch(f"{ODDS_BASE}/events", {"apiKey": key}, "Odds API events")
        remaining = r.headers.get("x-requests-remaining")
        listing = r.json() if r.status_code == 200 else []
        today = [e for e in listing if parse(e["commence_time"]).astimezone(ET).date().isoformat() == slate_date]
        spent, projected = op.ledger_today(iso(t)[:10]), len(ODDS_MARKETS) * len(today)
        if remaining is None or float(remaining) < POOL_FLOOR:
            odds["status"] = f"stopped_pool_floor ({remaining})"
        elif spent + projected > DAILY_CAP:
            odds["status"] = f"stopped_daily_cap ({spent} spent today + {projected} projected > {DAILY_CAP})"
        else:
            ev, credits, rem, used = odds_reference(today, key)
            odds = {"status": "ok", "events_listing": listing, "event_odds": ev, "credits": credits, "remaining_after": rem}
            op.ledger_add({"pulled_at": iso(t), "label": "sam_capture", "slate_date": slate_date, "games_requested": len(ev), "games_skipped_tipped": len(today) - len(ev),
                           "credits": credits, "projected": len(ODDS_MARKETS) * len(today), "remaining_before": remaining, "remaining_after": rem, "used_after": used, "status": "ok"})
    elif not a.dry_run_dir:
        odds["status"] = "no_key"
    ref = reference_index(odds["event_odds"])
    for r in rows:
        r["reference"] = ref.get((r["player_key"], r["stat"], r["x"]))
    cap = {"slate_date": slate_date, "captured_at": iso(t), "captured_at_pt": t_pt.strftime("%Y-%m-%d %H:%M:%S %Z"), "off_schedule": bool(a.manual), "dry_run": bool(a.dry_run_dir),
           "games": games, "early_tip_games": early, "kalshi": {"by_series": by_series, "markets": rows, "raw": raw},
           "sportsbook": odds, "model": {"source": "docs/data/projections.json", "date": proj.get("date"), "updated": proj.get("updated"), "morning_build": morning_updated, "players": model}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(cap, f, separators=(",", ":"))
    log_run({**base, "status": "captured_dry_run" if a.dry_run_dir else "captured", "kalshi_markets": len(rows), "kalshi_by_series": json.dumps(by_series), "games": len(games),
             "early_tip_games": len(early), "odds_credits": odds.get("credits", 0), "odds_games": len(odds.get("event_odds") or []), "model_updated": proj.get("updated", ""),
             "model_players": len(model), "note": odds["status"]})
    print(f"[sam-capture] {slate_date} captured at {cap['captured_at_pt']} (off_schedule={cap['off_schedule']}): {len(rows)} Kalshi markets {by_series}, {len(games)} games "
          f"({len(early)} early tip), sportsbook {odds['status']} cost {odds.get('credits', 0)} credits, model {len(model)} players (snapshot {proj.get('updated')})")
    print(f"CAPTURE_PATH={path}")
    return path


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"[sam-capture] {e}")
