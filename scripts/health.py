"""Lines freshness watchdog, projections-stale alarm and the morning self-heal.

Runs from .github/workflows/health.yml every 15 minutes all day (and on demand). Three jobs, all throttled through
data/health_state.json (committed by the workflow):

1. Lines (inside the game window only): docs/data/lines.json missing, without an `updated` stamp, or older than STALE_MINUTES
   posts to Discord (DISCORD_WEBHOOK_URL). Same throttle as mlb-fantasy's stale_lines_watchdog: alert once on the TRANSITION
   into stale, once more on recovery, never every 15 minutes for the same outage. Outside the window the lines state is
   cleared silently so each night starts clean. Game window: 90 minutes before tonight's first tip to 3 hours after the last,
   from data/slate.json; no games on the slate means no window and no lines alerts.
2. Projections (every pass): docs/data/projections.json older than PROJ_STALE_HOURS posts once (and once on recovery).
   Added 2026-10-28 after the morning update-data cron never fired and nothing said so.
3. Self-heal (every pass): on a game day, after SELF_HEAL_AFTER (ET), if projections.json's date is not today's ET date, dispatch
   the update-data workflow (`gh workflow run update-data.yml`, GH_TOKEN with actions: write) and say so on Discord. At most one
   self-dispatch per SELF_HEAL_MIN_GAP. Game day = ESPN's scoreboard lists an NBA game today (free; unreachable = assume yes).
4. Pre-tip self-heal (every pass, added 2026-10-28 evening): on a game day, inside the PRETIP_WINDOW before the day's earliest tip
   (data/slate.json), if no update-data run has refreshed the build since that window opened (projections.json `updated`), dispatch
   update-data and say so on Discord. Same SELF_HEAL_MIN_GAP throttle as job 3 (shared state key), skipped while an update-data run
   is already queued or running. The pre-tip snapshots (ladders, FP v1 + v2, first basket) cannot be recovered after tip; the
   evening cron has fired 1.8-6 h late all month. The PC's one-off dispatch timer is a backup, not the primary.

5. Flip-check (every pass, added 2026-10-30): a self-heal dispatch (job 3 or 4) is not trusted on its run's conclusion. The state
   records `heal_pending` (kind, dispatch time, expected slate date = today's ET date on the game day already detected, the gh
   output). On later passes the PUBLISHED docs/data/projections.json is read: flipped to the expected date (pre-tip kind: and
   built after the dispatch, and docs/data/first_basket_today.json carries the date) -> cleared, no message; still stale after
   FLIP_GRACE -> update-data is re-dispatched ONCE with a Discord alarm that names this exact failure ("dispatched ... reported
   success but the slate did not flip ... re-dispatched"); still stale FLIP_GRACE after that -> the alarm repeats on every pass,
   no third dispatch, until it flips or the day's first tip passes. While a heal is pending, jobs 3 and 4 do not dispatch.
   Cause (2026-10-30, run 41108710651): a dispatched run concluded SUCCESS but its data commit died on a rebase conflict.

  python health.py            # one pass, print the verdict, exit 0
  python health.py --force    # ignore the game window for the lines check (manual test; still respects the throttles)
  HEALTH_DRY_RUN=1            # never dispatch or post; print what would happen (local tests)
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import requests

from common import DATA_DIR, SITE_DATA, read_json, write_json

LINES = SITE_DATA / "lines.json"
PROJ = SITE_DATA / "projections.json"
SLATE = DATA_DIR / "slate.json"
STATE = DATA_DIR / "health_state.json"
STALE_MINUTES = 45
WINDOW_BEFORE = dt.timedelta(minutes=90)
WINDOW_AFTER = dt.timedelta(hours=3)
PROJ_STALE_HOURS = 14
SELF_HEAL_AFTER = dt.time(10, 30)                  # ET; the morning cron is 14:07Z = 10:07 ET
SELF_HEAL_MIN_GAP = dt.timedelta(hours=1)
PRETIP_WINDOW = dt.timedelta(minutes=90)             # job 4: a build must land inside this window before the earliest tip
FLIP_GRACE = dt.timedelta(minutes=45)                # job 5: a dispatched run gets this long to flip the published slate
FB_TODAY = SITE_DATA / "first_basket_today.json"
ET = ZoneInfo("America/New_York")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
WORKFLOW = "update-data.yml"
REPO = os.environ.get("GITHUB_REPOSITORY") or "macassvic-cmd/arc-lab"
DRY_RUN = os.environ.get("HEALTH_DRY_RUN", "").strip() not in ("", "0")


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


def projections_status(now, path=PROJ):
    """(stale: bool, slate_date: str|None, age_hours: float|None, detail: str) for docs/data/projections.json."""
    doc = read_json(path, None)
    if doc is None:
        return True, None, None, "projections.json is missing"
    updated = parse_ts(doc.get("updated"))
    if updated is None:
        return True, doc.get("date"), None, "projections.json has no `updated` timestamp"
    age = (now - updated).total_seconds() / 3600
    n = len(doc.get("players") or [])
    return age > PROJ_STALE_HOURS, doc.get("date"), age, f"slate {doc.get('date')}, {n} players, built {age:.1f} h ago ({updated:%Y-%m-%d %H:%M}Z)"


def is_game_day(day, session=requests):
    """True when ESPN's scoreboard lists an NBA game on `day` (ET date). None when ESPN cannot be reached."""
    try:
        r = session.get(ESPN_SCOREBOARD, params={"dates": day.strftime("%Y%m%d"), "limit": 50}, timeout=20)
        r.raise_for_status()
        events = r.json().get("events") or []
        return any((e.get("competitions") or [{}])[0].get("competitors") for e in events)
    except Exception as e:  # noqa: BLE001
        print(f"[health] ESPN scoreboard unreachable ({e}); assuming a game day")
        return None


def dispatch_update(repo=REPO, workflow=WORKFLOW):
    """`gh workflow run` with the job's GH_TOKEN. Returns (ok, detail)."""
    if DRY_RUN:
        return True, f"dry run: would dispatch {workflow} on {repo}"
    if not os.environ.get("GH_TOKEN") and not os.environ.get("GITHUB_TOKEN"):
        return False, "no GH_TOKEN in the environment"
    try:
        r = subprocess.run(["gh", "workflow", "run", workflow, "--repo", repo], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"gh failed to run: {e}"
    return r.returncode == 0, (r.stdout + r.stderr).strip() or f"exit {r.returncode}"


def update_running(repo=REPO, workflow=WORKFLOW):
    """True when an update-data run is queued or in progress (so the pre-tip heal does not pile a second one on). None = unknown."""
    if DRY_RUN or not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        return None
    try:
        r = subprocess.run(["gh", "api", f"repos/{repo}/actions/workflows/{workflow}/runs", "-q", "[.workflow_runs[] | select(.status == \"queued\" or .status == \"in_progress\" or .status == \"waiting\")] | length",
                            "-f", "per_page=10"], capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip() or 0) > 0 if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def run_conclusion(detail, repo=REPO):
    """'status conclusion' of the dispatched run named in the gh output (its URL), or None when unknown (dry run, no token, no URL)."""
    m = re.search(r"/actions/runs/(\d+)", str(detail or ""))
    if DRY_RUN or not m or not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        return None
    try:
        r = subprocess.run(["gh", "run", "view", m.group(1), "--repo", repo, "--json", "status,conclusion", "-q", r'"\(.status) \(.conclusion)"'], capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        status, _, conclusion = r.stdout.strip().partition(" ")
        return conclusion if status == "completed" else f"status {status}"
    except (OSError, subprocess.TimeoutExpired):
        return None


def mark_pending(state, now, kind, expected, how):
    """Job 5 state: the dispatch to verify. One pending heal at a time (jobs 3 and 4 do not dispatch while it is set)."""
    state["heal_pending"] = {"kind": kind, "dispatched_at": now.isoformat(), "expected_date": expected, "dispatch_detail": how, "redispatched_at": None, "redispatch_detail": None, "alarms": 0}


def slate_flipped(pending, proj_path, fb_path):
    """(flipped, what the published files show). Morning kind: projections.json `date` == expected. Pre-tip kind: that, and the
    build is newer than the dispatch, and first_basket_today.json carries the date (the pre-tip products must have landed)."""
    doc = read_json(proj_path, None) or {}
    seen = doc.get("date")
    if seen != pending["expected_date"]:
        return False, f"projections still {seen}"
    if pending["kind"] == "pre-tip":
        built = parse_ts(doc.get("updated")); fb = (read_json(fb_path, None) or {}).get("date")
        if built is None or built < parse_ts(pending["dispatched_at"]):
            return False, f"projections {seen} but built {doc.get('updated')}, before the dispatch"
        if fb != pending["expected_date"]:
            return False, f"projections {seen} but first_basket_today still {fb}"
    return True, f"projections {seen}"


def flip_check(now, state, proj_path, fb_path, slate_path, notify_fn, dispatch_fn=dispatch_update, status_fn=run_conclusion):
    """Job 5. Mutates state. Returns "none" | "flipped" | "waiting" | "redispatched" | "redispatch_failed" | "alarmed" | "expired"."""
    pend = state.get("heal_pending")
    if not pend:
        return "none"
    ok, seen = slate_flipped(pend, proj_path, fb_path)
    if ok:
        state.pop("heal_pending"); state["heal_last_flipped"] = {**pend, "flipped_at": now.isoformat()}
        print(f"[health] flip-check: {pend['kind']} dispatch at {pend['dispatched_at']} flipped the slate to {pend['expected_date']}; cleared")
        return "flipped"
    slate = read_json(slate_path, {}) or {}
    tips = [t for t in (parse_ts(g.get("start")) for g in slate.get("games") or []) if t]
    if tips and slate.get("date") == pend["expected_date"] and now >= min(tips):
        state.pop("heal_pending"); state["heal_last_expired"] = {**pend, "expired_at": now.isoformat(), "seen": seen}
        print(f"[health] flip-check: first tip {min(tips):%H:%M}Z passed without the slate flipping ({seen}); no further dispatch, cleared")
        return "expired"
    d0 = parse_ts(pend["dispatched_at"]); d0_local = d0.astimezone(ET)
    if pend.get("redispatched_at") is None:
        if now - d0 < FLIP_GRACE:
            print(f"[health] flip-check: {pend['kind']} dispatch at {d0:%H:%M}Z, {seen}; {FLIP_GRACE - (now - d0)} of grace left")
            return "waiting"
        concl = status_fn(pend.get("dispatch_detail")) or "success"
        ok2, how2 = dispatch_fn()
        pend.update({"redispatched_at": now.isoformat(), "redispatch_detail": how2, "redispatch_ok": ok2, "alarms": pend.get("alarms", 0) + 1})
        state.update({"self_dispatch_at": now.isoformat(), "self_dispatch_ok": ok2, "self_dispatch_detail": how2, "self_dispatch_kind": f"{pend['kind']} re-dispatch"})
        if ok2:
            notify_fn(f"🚨 Arc Lab self-heal did not flip: update-data dispatched at {d0_local:%H:%M} ET ({d0:%H:%M}Z, {pend['kind']}) reported {concl} but the slate did not flip "
                      f"({seen}, expected {pend['expected_date']}); re-dispatched ({how2}). Check that run's Commit step.")
            return "redispatched"
        notify_fn(f"🚨 Arc Lab self-heal did not flip: update-data dispatched at {d0_local:%H:%M} ET reported {concl} but the slate did not flip ({seen}); re-dispatch FAILED: {how2}. Run it by hand now.")
        return "redispatch_failed"
    r0 = parse_ts(pend["redispatched_at"])
    if now - r0 < FLIP_GRACE:
        print(f"[health] flip-check: re-dispatched at {r0:%H:%M}Z, {seen}; {FLIP_GRACE - (now - r0)} of grace left")
        return "waiting"
    pend["alarms"] = pend.get("alarms", 0) + 1
    notify_fn(f"🚨 Arc Lab self-heal still not flipped: update-data dispatched at {d0_local:%H:%M} ET and re-dispatched at {r0.astimezone(ET):%H:%M} ET, and the slate has still not flipped "
              f"({seen}, expected {pend['expected_date']}). No third dispatch: run it by hand and read both runs' Commit steps.")
    return "alarmed"


def notify(text, webhook=None):
    webhook = webhook or os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if DRY_RUN or not webhook:
        print(f"[health] ({'dry run' if DRY_RUN else 'no DISCORD_WEBHOOK_URL'}) would post: {text}")
        return False
    r = requests.post(webhook, json={"content": text}, timeout=20)
    print(f"[health] discord {r.status_code}: {text}")
    return r.ok


def check_lines(now, state, force, lines_path, slate_path, notify_fn):
    """Job 1. Mutates state. Returns "alert" | "recovered" | "quiet" | "outside"."""
    slate = read_json(slate_path, {})
    window = game_window(slate, now)
    in_window = force or (window is not None and window[0] <= now <= window[1])
    if not in_window:
        if state.get("already_alerted"):
            state.update({"already_alerted": False, "cleared": now.isoformat()})
        print(f"[health] outside game window ({'no games on slate' if window is None else f'{window[0]:%H:%M}-{window[1]:%H:%M}Z'}), lines check skipped")
        return "outside"
    stale, detail, _ = lines_status(now, lines_path)
    if stale and not state.get("already_alerted"):
        notify_fn(f"⚠️ Arc Lab lines stale: {detail}. Is the Underdog scraper (pirate-bets) running with ARC_LAB_DIR set?")
        state.update({"already_alerted": True, "since": now.isoformat(), "detail": detail})
        return "alert"
    if not stale and state.get("already_alerted"):
        notify_fn(f"✅ Arc Lab lines fresh again: {detail}.")
        state.update({"already_alerted": False, "recovered": now.isoformat()})
        return "recovered"
    print(f"[health] lines {'STALE (already alerted)' if stale else 'fresh'}: {detail}")
    return "quiet"


def check_projections(now, state, proj_path, notify_fn):
    """Job 2. Mutates state. Returns "alert" | "recovered" | "quiet"."""
    stale, _, _, detail = projections_status(now, proj_path)
    if stale and not state.get("proj_alerted"):
        notify_fn(f"⚠️ Arc Lab projections stale (> {PROJ_STALE_HOURS} h): {detail}. Did the update-data cron fire? The self-heal dispatches it after 10:30 ET on game days.")
        state.update({"proj_alerted": True, "proj_since": now.isoformat(), "proj_detail": detail})
        return "alert"
    if not stale and state.get("proj_alerted"):
        notify_fn(f"✅ Arc Lab projections fresh again: {detail}.")
        state.update({"proj_alerted": False, "proj_recovered": now.isoformat()})
        return "recovered"
    print(f"[health] projections {'STALE (already alerted)' if stale else 'fresh'}: {detail}")
    return "quiet"


def self_heal(now, state, proj_path, notify_fn, game_day_fn=is_game_day, dispatch_fn=dispatch_update):
    """Job 3. Mutates state. Returns "dispatched" | "dispatch_failed" | "throttled" | "not_needed" | "too_early" | "off_day"."""
    local = now.astimezone(ET)
    _, slate_date, age, detail = projections_status(now, proj_path)
    today = local.date().isoformat()
    if slate_date == today:
        return "not_needed"
    if local.time() < SELF_HEAL_AFTER:
        print(f"[health] self-heal: projections show {slate_date}, not {today}, but it is {local:%H:%M} ET (< {SELF_HEAL_AFTER:%H:%M}); waiting for the cron")
        return "too_early"
    if game_day_fn(local.date()) is False:
        print(f"[health] self-heal: {today} is not an NBA game day; nothing to build")
        return "off_day"
    if state.get("heal_pending"):
        print(f"[health] self-heal: a {state['heal_pending']['kind']} dispatch from {state['heal_pending']['dispatched_at']} is pending verification (job 5 owns the re-dispatch)")
        return "flip_pending"
    last = parse_ts(state.get("self_dispatch_at"))
    if last and now - last < SELF_HEAL_MIN_GAP:
        print(f"[health] self-heal: projections still show {slate_date}; last self-dispatch {last:%H:%M}Z, within {SELF_HEAL_MIN_GAP}; not again yet")
        return "throttled"
    ok, how = dispatch_fn()
    state.update({"self_dispatch_at": now.isoformat(), "self_dispatch_ok": ok, "self_dispatch_detail": how, "self_dispatch_for": today, "self_dispatch_kind": "morning"})
    if ok:
        mark_pending(state, now, "morning", today, how)
        notify_fn(f"🔁 Arc Lab self-heal: projections.json shows {slate_date} at {local:%H:%M} ET on game day {today} ({detail}). Dispatched update-data ({how}).")
        return "dispatched"
    notify_fn(f"❌ Arc Lab self-heal could not dispatch update-data for {today}: {how}. Run it by hand (Actions → Update data and projections → Run workflow).")
    return "dispatch_failed"


def pretip_heal(now, state, proj_path, slate_path, notify_fn, dispatch_fn=dispatch_update, running_fn=update_running):
    """Job 4. Mutates state. Returns "dispatched" | "dispatch_failed" | "throttled" | "running" | "not_needed" | "outside_window" | "no_slate" | "wrong_day"."""
    local = now.astimezone(ET); today = local.date().isoformat()
    slate = read_json(slate_path, {}) or {}
    tips = [parse_ts(g.get("start")) for g in slate.get("games") or []]
    tips = [t for t in tips if t]
    if not tips:
        return "no_slate"
    if slate.get("date") != today:
        return "wrong_day"   # the slate is not today's (the morning heal, job 3, owns that case)
    first = min(tips); window_open = first - PRETIP_WINDOW
    if now < window_open or now >= first:
        return "outside_window"
    _, slate_date, _, detail = projections_status(now, proj_path)
    doc = read_json(proj_path, None) or {}
    built = parse_ts(doc.get("updated"))
    if slate_date == today and built is not None and built >= window_open:
        print(f"[health] pre-tip: build at {built:%H:%M}Z is inside the {PRETIP_WINDOW} window before the {first:%H:%M}Z tip; nothing to do")
        return "not_needed"
    if state.get("heal_pending"):
        print(f"[health] pre-tip: a {state['heal_pending']['kind']} dispatch from {state['heal_pending']['dispatched_at']} is pending verification (job 5 owns the re-dispatch)")
        return "flip_pending"
    last = parse_ts(state.get("self_dispatch_at"))
    if last and now - last < SELF_HEAL_MIN_GAP:
        print(f"[health] pre-tip: no build since {window_open:%H:%M}Z (first tip {first:%H:%M}Z) but a self-dispatch at {last:%H:%M}Z is within {SELF_HEAL_MIN_GAP}; not again yet")
        return "throttled"
    if running_fn() is True:
        print(f"[health] pre-tip: no build since {window_open:%H:%M}Z but an update-data run is already queued or running; waiting for it")
        return "running"
    ok, how = dispatch_fn()
    state.update({"self_dispatch_at": now.isoformat(), "self_dispatch_ok": ok, "self_dispatch_detail": how, "self_dispatch_for": today, "self_dispatch_kind": "pre-tip"})
    if ok:
        mark_pending(state, now, "pre-tip", today, how)
        notify_fn(f"🔁 Arc Lab pre-tip self-heal: first tip {first:%H:%M}Z, no update-data build since {window_open:%H:%M}Z ({detail}). Dispatched update-data at {local:%H:%M} ET ({how}).")
        return "dispatched"
    notify_fn(f"❌ Arc Lab pre-tip self-heal could not dispatch update-data before the {first:%H:%M}Z tip: {how}. Run it by hand now (Actions → Update data and projections → Run workflow).")
    return "dispatch_failed"


def check(now=None, force=False, state_path=STATE, lines_path=LINES, slate_path=SLATE, proj_path=PROJ, notify_fn=notify, game_day_fn=is_game_day, dispatch_fn=dispatch_update, running_fn=update_running,
          fb_path=FB_TODAY, status_fn=run_conclusion):
    """One watchdog pass. Returns {"lines", "projections", "flip", "self_heal", "pretip"}; the state file is rewritten once."""
    now = now or utcnow()
    state = read_json(state_path, {}) or {}
    before = json.dumps(state, sort_keys=True)
    out = {"lines": check_lines(now, state, force, lines_path, slate_path, notify_fn),
           "projections": check_projections(now, state, proj_path, notify_fn),
           "flip": flip_check(now, state, proj_path, fb_path, slate_path, notify_fn, dispatch_fn, status_fn)}
    out["self_heal"] = self_heal(now, state, proj_path, notify_fn, game_day_fn, dispatch_fn)
    out["pretip"] = pretip_heal(now, state, proj_path, slate_path, notify_fn, dispatch_fn, running_fn)
    if json.dumps(state, sort_keys=True) != before:
        write_json(state_path, state)
    print(f"[health] {out}")
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles choke on the alert emoji
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="check lines even outside the game window")
    args = ap.parse_args()
    check(force=args.force)
