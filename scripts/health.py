"""Lines freshness watchdog.

Runs from .github/workflows/health.yml every 15 minutes in the evening (and on demand). During
tonight's game window it checks docs/data/lines.json and posts to Discord (DISCORD_WEBHOOK_URL)
when the file is missing, has no `updated` stamp, or is older than STALE_MINUTES.

Throttling is the same as mlb-fantasy's stale_lines_watchdog: alert once on the TRANSITION into
stale, once more on recovery, never every 15 minutes for the same outage. The state lives in
data/health_state.json, committed by the workflow. Outside the game window the state is cleared
silently so each night starts clean.

Game window: from 90 minutes before tonight's first tip to 3 hours after the last tip, taken from
data/slate.json. No games on the slate means no window and no alerts.

  python health.py            # check once, print the verdict, exit 0
  python health.py --force    # ignore the window (manual test; still respects the throttle)
"""
import argparse
import datetime as dt
import os
import sys

import requests

from common import DATA_DIR, SITE_DATA, read_json, write_json

LINES = SITE_DATA / "lines.json"
SLATE = DATA_DIR / "slate.json"
STATE = DATA_DIR / "health_state.json"
STALE_MINUTES = 45
WINDOW_BEFORE = dt.timedelta(minutes=90)
WINDOW_AFTER = dt.timedelta(hours=3)


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def parse_ts(value):
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def game_window(slate, now):
    """(start, end) of tonight's window, or None when the slate is empty or unparseable."""
    tips = [parse_ts(g.get("start")) for g in slate.get("games") or []]
    tips = [t for t in tips if t]
    if not tips:
        return None
    return min(tips) - WINDOW_BEFORE, max(tips) + WINDOW_AFTER


def lines_status(now, path=LINES):
    """(stale: bool, detail: str, age_minutes: float|None)"""
    doc = read_json(path, None)
    if doc is None:
        return True, "lines.json is missing", None
    updated = parse_ts(doc.get("updated"))
    if updated is None:
        return True, "lines.json has no `updated` timestamp", None
    age = (now - updated).total_seconds() / 60
    n = len(doc.get("lines") or [])
    detail = f"{n} lines, updated {age:.0f} min ago ({updated:%Y-%m-%d %H:%M}Z)"
    return age > STALE_MINUTES, detail, age


def notify(text, webhook=None):
    webhook = webhook or os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        print(f"[health] (no DISCORD_WEBHOOK_URL) would post: {text}")
        return False
    r = requests.post(webhook, json={"content": text}, timeout=20)
    print(f"[health] discord {r.status_code}: {text}")
    return r.ok


def check(now=None, force=False, state_path=STATE, lines_path=LINES, slate_path=SLATE, notify_fn=notify):
    """One watchdog pass. Returns the action taken: "alert" | "recovered" | "quiet" | "outside"."""
    now = now or utcnow()
    state = read_json(state_path, {})
    slate = read_json(slate_path, {})
    window = game_window(slate, now)
    in_window = force or (window is not None and window[0] <= now <= window[1])

    if not in_window:
        if state.get("already_alerted"):
            write_json(state_path, {"already_alerted": False, "cleared": now.isoformat()})
        print(f"[health] outside game window ({'no games on slate' if window is None else f'{window[0]:%H:%M}-{window[1]:%H:%M}Z'}), nothing to do")
        return "outside"

    stale, detail, age = lines_status(now, lines_path)
    if stale and not state.get("already_alerted"):
        notify_fn(f"⚠️ Arc Lab lines stale: {detail}. Is the Underdog scraper (pirate-bets) running with ARC_LAB_DIR set?")
        write_json(state_path, {"already_alerted": True, "since": now.isoformat(), "detail": detail})
        return "alert"
    if not stale and state.get("already_alerted"):
        notify_fn(f"✅ Arc Lab lines fresh again: {detail}.")
        write_json(state_path, {"already_alerted": False, "recovered": now.isoformat()})
        return "recovered"
    print(f"[health] {'STALE (already alerted)' if stale else 'fresh'}: {detail}")
    return "quiet"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles choke on the alert emoji
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="check even outside the game window")
    args = ap.parse_args()
    check(force=args.force)
