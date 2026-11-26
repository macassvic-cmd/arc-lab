"""First-basket confirmed-lineup WATCHER (backtests/first_basket_confirmed_lineup_rule.md section 2).

Runs on the CEO's PC from Windows Task Scheduler every 2 minutes (GitHub Actions cron is not the runner: its start delay in this
repo is 4-7 hours). One invocation = one tick:

  fast path (standard library only, a fraction of a second): build the watch list from the SCHEDULE ITSELF (ESPN's scoreboard for
      the slate date, cached a few minutes; every slate game is on the list by construction) and see whether any game has a
      check due: T-22, T-15, T-8 and T-3 minutes before the scheduled tip. Nothing due -> exit.
  slow path (a check is due): fetch + rebase this clone onto main, read NBA.com's daily lineups once, and for every due game:
      Confirmed stamp present (five listed with a position on both teams, all stamped Confirmed) -> append a CONFIRMED row: ten
      log rows priced by the existing snapshot machinery with the frozen params (first_basket_lineups.reprice, unchanged), and
      the why-sidecar line carrying lineup_status "confirmed" (that line is what marks the snapshot as a confirmed row for
      grading). A game whose last confirmed row already has the same ten gets no second row. A row is never written at or
      after the scheduled tip.
  then: commit the appended rows and PUSH them itself: fetch + rebase + retry, 3 attempts. A push that does not land = the
      process exits non-zero and posts to Discord; the local commit stays and the next tick tries again first thing.

Every check is recorded with its target time and the time NBA.com actually answered: locally at once
(<state dir>/checks.jsonl), and in data/first_basket_watch_checks.csv with the next push (and once more after the slate's last
tip). A target the watcher never reached is recorded as missed; a check more than 3 minutes late is flagged missed.

Discord: no webhook is stored on this PC. Posts go through the fb-watch-notify workflow (dispatched with gh under the same
GitHub login the pushes use), which holds the repo's Discord secret. A local webhook (env DISCORD_WEBHOOK_URL or
<state dir>/discord_webhook.txt) is used instead when present. No URL and no token is ever printed or logged.

Never touched here: model params, the grading code, any derived JSON (the page picks the confirmed row up at the next build).

  pythonw first_basket_watch.py                 one tick (what the scheduled task runs)
  python  first_basket_watch.py --dry-run [--now 2026-11-26T00:40:00Z]
                                                  the same tick with no git, no repo writes, no posts; rows go to <state>/dryrun
"""
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("FBW_STATE") or (Path.home() / "arc-lab-watcher-state"))
GH_REPO = "macassvic-cmd/arc-lab"
REMOTE = os.environ.get("FBW_REMOTE") or "origin"          # tests point this at a remote that does not exist
TARGETS = (22, 15, 8, 3)                                    # minutes before the scheduled tip
LATE_SECONDS = 180                                          # a check more than 3 minutes late counts as missed
SCHEDULE_TTL = 600                                          # seconds the cached schedule is trusted
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}&limit=50"
CHECKS_CSV = REPO / "data" / "first_basket_watch_checks.csv"
CHECK_FIELDS = ["date", "event_id", "matchup", "tip", "target_min", "target_at", "actual_at", "late_s", "missed", "result", "row_logged_at", "note"]
PUSH_PATHS = ["data/first_basket_log.csv", "data/first_basket_why.jsonl", "data/nbacom_player_ids.csv", "data/first_basket_watch_checks.csv"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}   # a credential prompt must fail, not hang
DRY = False


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    s = str(s).replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def slate_date(now):
    """The slate date is the Eastern calendar date, as everywhere in the pipeline."""
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:  # noqa: BLE001  (no tz database: Eastern standard time)
        return (now - dt.timedelta(hours=5)).date().isoformat()


def log(msg):
    line = f"{ts(utcnow())} {msg}"
    try:
        print(line)
    except Exception:  # noqa: BLE001  (pythonw has no console)
        pass
    if not DRY:
        try:
            STATE.mkdir(parents=True, exist_ok=True)
            with open(STATE / "watch.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------- the watch list: the schedule itself
def fetch_schedule(date):
    req = urllib.request.Request(ESPN.format(d=date.replace("-", "")), headers={"User-Agent": "Mozilla/5.0 (arc-lab research)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        doc = json.loads(r.read().decode("utf-8"))
    games = []
    for e in doc.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): (c.get("team") or {}).get("abbreviation") for c in comp.get("competitors") or []}
        st = ((e.get("status") or {}).get("type") or {})
        if not teams.get("home") or not teams.get("away") or not e.get("date"):
            continue
        games.append({"event_id": str(e["id"]), "tip": ts(parse_ts(e["date"])), "home": teams["home"], "away": teams["away"], "state": st.get("state"), "status": st.get("name")})
    return games


def schedule(date, now, cache=True):
    """[{event_id, tip, home, away, state, status}] for the slate date. ESPN first (cached SCHEDULE_TTL seconds); when ESPN cannot
    be reached, the last cached copy, then the repo's data/slate.json if it is for this date. Postponed / cancelled games are dropped."""
    path = STATE / f"schedule_{date}.json"
    cached = None
    if cache and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            cached = None
    games, src = None, None
    if cached and (now - parse_ts(cached["fetched_at"])).total_seconds() < SCHEDULE_TTL and now >= parse_ts(cached["fetched_at"]):
        games, src = cached["games"], "cache"
    else:
        try:
            games, src = fetch_schedule(date), "espn"
            if cache:
                STATE.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"fetched_at": ts(utcnow()), "games": games}), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            if cached:
                games, src = cached["games"], "stale cache (ESPN unreachable)"
            else:
                try:
                    sl = json.loads((REPO / "data" / "slate.json").read_text(encoding="utf-8"))
                    if sl.get("date") == date:
                        games = [{"event_id": str(g["event_id"]), "tip": ts(parse_ts(g["start"])), "home": g["home"], "away": g["away"], "state": "pre", "status": ""} for g in sl.get("games") or []]
                        src = "data/slate.json (ESPN unreachable)"
                except Exception:  # noqa: BLE001
                    games = None
            if games is None:
                log(f"schedule for {date} unavailable ({type(e).__name__}); nothing can be watched this tick")
                return [], "unavailable"
    return [g for g in games if (g.get("status") or "") not in ("STATUS_POSTPONED", "STATUS_CANCELED")], src


# ---------------------------------------------------------------- local check record
def read_checks():
    p = STATE / "checks.jsonl"
    out = []
    if p.exists():
        for line in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def append_checks(rows):
    if DRY or not rows:
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "checks.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def synced_count():
    try:
        return int((STATE / "synced_count.txt").read_text().strip())
    except (OSError, ValueError):
        return 0


def due_checks(games, done, now):
    """{event_id: [target minutes due and not yet recorded, oldest first]} for games that have not tipped."""
    out = {}
    for g in games:
        tip = parse_ts(g["tip"])
        if now >= tip:
            continue
        d = [m for m in TARGETS if now >= tip - dt.timedelta(minutes=m) and (g["event_id"], m) not in done]
        if d:
            out[g["event_id"]] = d
    return out


# ---------------------------------------------------------------- git + notify
def run(cmd, timeout=120):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=GIT_ENV, creationflags=NO_WINDOW)
    return r.returncode, re.sub(r"//[^/@\s]+@", "//***@", (r.stdout + r.stderr).strip())   # never carry credentials from a URL into a log


def git(*args, timeout=120):
    return run(["git", *args], timeout)


def sync_from_main():
    """fetch + rebase this clone onto main. A stranded local commit (an earlier push that did not land) is kept and rebased."""
    code, out = git("fetch", REMOTE, "main")
    if code != 0:
        return False, f"fetch failed: {out[-200:]}"
    code, out = git("rebase", f"{REMOTE}/main")
    if code != 0:
        git("rebase", "--abort")
        return False, f"rebase failed: {out[-200:]}"
    return True, "ok"


def ahead():
    code, out = git("rev-list", "--count", f"{REMOTE}/main..HEAD")
    return int(out) if code == 0 and out.isdigit() else 0


def push():
    """Push HEAD to main: 3 attempts, each failure followed by fetch + rebase. Returns (landed, detail)."""
    detail = ""
    for attempt in (1, 2, 3):
        code, out = git("push", REMOTE, "HEAD:main")
        if code == 0:
            return True, f"landed on attempt {attempt}"
        detail = out[-200:].replace("\n", " ")
        log(f"push attempt {attempt} failed: {detail}")
        time.sleep(2 * attempt)
        ok, why = sync_from_main()
        if not ok:
            detail = f"{detail}; then {why}"
    return False, detail


def notify(text):
    """Post to the Arc Lab Discord channel. Never logs a URL. Returns how it went."""
    if DRY:
        log(f"(dry run) would post: {text}"); return "dry"
    hook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    f = STATE / "discord_webhook.txt"
    if not hook and f.exists():
        hook = f.read_text(encoding="utf-8").strip()
    if hook:
        try:
            req = urllib.request.Request(hook, data=json.dumps({"content": text[:1900]}).encode(), headers={"Content-Type": "application/json", "User-Agent": "arc-lab-watcher"})
            urllib.request.urlopen(req, timeout=20).read()
            log(f"posted (local webhook): {text}"); return "webhook"
        except Exception as e:  # noqa: BLE001
            log(f"local webhook post failed ({type(e).__name__}); trying the workflow")
    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    try:
        code, out = run([gh, "workflow", "run", "fb-watch-notify.yml", "--repo", GH_REPO, "-f", f"message={text[:1900]}"], 60)
        log(f"{'posted via the fb-watch-notify workflow' if code == 0 else 'NOT POSTED (workflow dispatch failed: ' + out[-160:] + ')'}: {text}")
        return "workflow" if code == 0 else "failed"
    except Exception as e:  # noqa: BLE001
        log(f"NOT POSTED ({type(e).__name__}): {text}"); return "failed"


def sync_checks_into_repo():
    """Append the check rows not yet in data/first_basket_watch_checks.csv. Returns how many were added."""
    rows = read_checks(); n0 = synced_count(); new = rows[n0:]
    if not new:
        return 0
    fresh = not CHECKS_CSV.exists()
    with open(CHECKS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
        if fresh:
            w.writeheader()
        w.writerows([{k: r.get(k, "") for k in CHECK_FIELDS} for r in new])
    (STATE / "synced_count.txt").write_text(str(len(rows)))
    return len(new)


def commit_and_push(message):
    """Commit the append-only files and push. Returns (landed, detail); nothing to commit and nothing stranded = (True, 'nothing')."""
    git("add", "--", *[p for p in PUSH_PATHS if (REPO / p).exists()])
    code, _ = git("diff", "--cached", "--quiet")
    if code != 0:
        code, out = git("commit", "-q", "-m", message)
        if code != 0:
            return False, f"commit failed: {out[-200:]}"
    if not ahead():
        return True, "nothing"
    landed, detail = push()
    marker = STATE / "unpushed"
    if landed:
        marker.unlink(missing_ok=True)
        with open(STATE / "pushes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"landed_at": ts(utcnow()), "message": message, "detail": detail}) + "\n")
    else:
        marker.write_text(ts(utcnow()))
    return landed, detail


# ---------------------------------------------------------------- one tick
def tick(now=None, dry=False):
    global DRY
    DRY = dry
    simulated = now is not None
    now = now or utcnow()
    date = slate_date(now)
    games, src = schedule(date, now, cache=not simulated)
    done = {(c["event_id"], int(c["target_min"])) for c in read_checks() if c.get("date") == date} if not simulated else set()
    due = due_checks(games, done, now)
    stranded = (STATE / "unpushed").exists() and not dry
    tips = [parse_ts(g["tip"]) for g in games]
    flush = (not dry and not due and len(read_checks()) > synced_count() and (not tips or now > max(tips)))
    if not due and not stranded and not flush:
        hb = STATE / "heartbeat.txt"
        if not dry and (not hb.exists() or time.time() - hb.stat().st_mtime > 3600):
            STATE.mkdir(parents=True, exist_ok=True); hb.write_text(ts(now))
            nxt = sorted(t for t in tips if t > now)
            log(f"idle: {date}, {len(games)} games on the schedule ({src})" + (f", next tip {nxt[0]:%H:%M}Z" if nxt else ", no tip ahead"))
        elif dry:
            log(f"(dry run) {date}: {len(games)} games on the schedule ({src}); no check due at {ts(now)}")
        return 0

    lock = STATE / "lock"
    if not dry:
        STATE.mkdir(parents=True, exist_ok=True)
        if lock.exists() and time.time() - lock.stat().st_mtime < 600:
            log("another tick is still running; this one stands down"); return 0
        lock.write_text(ts(utcnow()))
    try:
        return _work(now, date, games, src, due, dry, simulated)
    finally:
        if not dry:
            lock.unlink(missing_ok=True)


def _work(now, date, games, src, due, dry, simulated):
    exit_code = 0
    if not dry:
        ok, why = sync_from_main()
        if not ok:
            log(f"could not sync with main ({why}); checking anyway on the local copy")
    records, appended, posts, alarms = [], [], [], []
    if due:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pandas as pd
        import first_basket_lineups as fbl
        import first_basket_model as fbm
        xwalk = fbl.XWALK
        if dry:                                        # a dry run writes nothing in the repo: rows go to <state>/dryrun
            dd = STATE / "dryrun"; dd.mkdir(parents=True, exist_ok=True)
            for name in ("LOG", "WHY"):
                dst = dd / getattr(fbm, name).name; shutil.copyfile(getattr(fbm, name), dst); setattr(fbm, name, dst)
            xwalk = dd / fbl.XWALK.name; shutil.copyfile(fbl.XWALK, xwalk)
        try:
            doc = fbl.nbacom_daily(date); err = "" if doc is not None else "NBA.com has not published the daily lineups file (404)"
        except Exception as e:  # noqa: BLE001
            doc, err = None, f"NBA.com unreachable ({type(e).__name__})"
        actual = now if simulated else utcnow()
        listed = fbl.nbacom_games(doc) if doc else {}
        logdf = pd.read_csv(fbm.LOG, dtype={"event_id": str, "player_id": str})
        conf_keys = fbm.confirmed_keys(fbm.WHY)
        uni = xw = ctx = None
        for g in games:
            e = g["event_id"]
            if e not in due:
                continue
            tip = parse_ts(g["tip"]); label = f"{g['away']}@{g['home']}"
            base = {"date": date, "event_id": e, "matchup": label, "tip": g["tip"]}
            for m in due[e][:-1]:                      # targets the watcher never reached: recorded, not read
                records.append({**base, "target_min": m, "target_at": ts(tip - dt.timedelta(minutes=m)), "actual_at": "", "late_s": "", "missed": 1, "result": "missed", "row_logged_at": "",
                                "note": "no tick ran between this target and the next one"})
            m = due[e][-1]; target = tip - dt.timedelta(minutes=m); late = int((actual - target).total_seconds())
            rec = {**base, "target_min": m, "target_at": ts(target), "actual_at": ts(actual), "late_s": late, "missed": int(late > LATE_SECONDS), "row_logged_at": "", "note": ""}
            ng = listed.get((g["away"], g["home"]))
            if not doc:
                rec.update(result="source_error", note=err)
            elif ng is None:
                rec.update(result="not_listed", note=f"no NBA.com game matches {label} (listed: {', '.join(f'{a}@{h}' for a, h in listed) or 'none'})")
            elif not all(t["confirmed"] for t in ng["teams"].values()):
                rec.update(result="not_confirmed", note="; ".join(f"{t}: {len(x['starters'])} with a position, {'/'.join(x['statuses']) or '-'}" for t, x in ng["teams"].items()))
            else:
                try:
                    uni = uni or fbl.Universe(); xw = xw or fbl.Crosswalk(xwalk)
                    conf = fbl.resolve_game(ng, uni, xw, actual)
                    ten = {x["id"] for t in conf for x in conf[t]}
                    mine = logdf[(logdf.event_id == e) & (logdf.logged_at < logdf.tip)]
                    stamps = sorted(t for t in mine.logged_at.unique() if (e, t) in conf_keys)
                    if stamps and set(mine[mine.logged_at == stamps[-1]].player_id) == ten:
                        rec.update(result="confirmed_already_logged", row_logged_at=stamps[-1], note="the last confirmed row has the same ten")
                    else:
                        t_row = actual if simulated else utcnow()
                        if t_row >= tip:
                            rec.update(result="after_tip_no_row", note="confirmed, but the scheduled tip had passed: no row written")
                        else:
                            prev = fbm.graded_rows(mine, conf=conf_keys) if len(mine) else mine
                            diff = fbl.lineup_diff(prev, conf) if len(prev) and set(prev.team) == set(conf) else {}
                            before = prev.sort_values("p_first_basket", ascending=False).player.iloc[0] if len(prev) else None
                            ctx = ctx or fbl.Ctx()
                            stamp_src = sorted({p["stamp"] for t in ng["teams"].values() for p in t["starters"] if p.get("stamp")})
                            rows, _, _ = fbl.reprice(ctx, {**g, "date": date}, conf, t_row, diff,
                                                     {"nbacom_game_id": ng["game_id"], "nbacom_stamp": stamp_src[-1] if stamp_src else None, "watcher": "first_basket_watch", "target_min": m})
                            top3 = sorted(rows, key=lambda r: -r["p_first_basket"])[:3]
                            rec.update(result="confirmed_row_appended", row_logged_at=ts(t_row), note=(fbl.diff_text(diff) or "confirmed ten = the previous snapshot's ten") + f"; top pick {top3[0]['player']}")
                            appended.append(label)
                            if diff or (before and before != top3[0]["player"]):       # section 7: the picks changed
                                posts.append(f"🔁 Arc Lab first basket, {label} (tip {tip:%H:%M}Z): lineups CONFIRMED with changes ({fbl.diff_text(diff) or 'same ten, new order'}). "
                                             f"Corrected picks: " + ", ".join(f"{r['player']} {100 * r['p_first_basket']:.1f}%" for r in top3) + (f" (top pick was {before})" if before and before != top3[0]['player'] else "") + ".")
                except fbl.Unresolved as ex:
                    rec.update(result="unresolved", note=f"confirmed but a starter could not be matched: {ex}")
                    alarms.append(f"🚨 Arc Lab first-basket watcher: {label} is confirmed on NBA.com but was NOT repriced ({ex}). The projected row stands; add the pair to data/nbacom_player_ids.csv.")
                except Exception as ex:  # noqa: BLE001
                    rec.update(result="error", note=f"{type(ex).__name__}: {str(ex)[:160]}")
                    alarms.append(f"🚨 Arc Lab first-basket watcher: {label} check failed ({type(ex).__name__}). The projected row stands unless a later check succeeds.")
            records.append(rec)
            log(f"{label} T-{m}: target {rec['target_at']}, actual {rec['actual_at']} ({late:+d} s){' MISSED (late)' if rec['missed'] else ''} -> {rec['result']}" + (f" [{rec['note'][:140]}]" if rec["note"] else ""))
        append_checks(records)
    if dry:
        log(f"(dry run) {len(records)} check(s), {len(appended)} confirmed row(s) written to {STATE / 'dryrun'} only; nothing committed, pushed or posted")
        for p in posts + alarms:
            log(f"(dry run) would post: {p}")
        return 0
    n = sync_checks_into_repo() if (appended or not due) else 0
    if appended or n or ahead():
        landed, detail = commit_and_push(f"fb-watch: {'confirmed ' + ', '.join(appended) if appended else 'check log'} ({ts(utcnow())})")
        if landed:
            log(f"push: {detail}" + (f"; confirmed rows on main for {', '.join(appended)}" if appended else ""))
        else:
            exit_code = 1
            log(f"PUSH DID NOT LAND after 3 attempts: {detail}")
            notify(f"🚨 Arc Lab first-basket watcher: push did NOT land after 3 attempts ({ts(utcnow())}). Confirmed rows waiting on this PC: {', '.join(appended) or 'earlier ticks'}. "
                   f"They do not count until they are on main; the next tick retries. Last error: {detail[:300]}")
    for p in posts + alarms:
        notify(p)
    return exit_code


if __name__ == "__main__":
    args = sys.argv[1:]
    at = parse_ts(args[args.index("--now") + 1]) if "--now" in args else None
    try:
        code = tick(now=at, dry="--dry-run" in args)
    except Exception as e:  # noqa: BLE001
        log(f"tick crashed: {type(e).__name__}: {str(e)[:300]}")
        try:
            if "--dry-run" not in args:
                (STATE / "lock").unlink(missing_ok=True)
                notify(f"🚨 Arc Lab first-basket watcher crashed ({type(e).__name__}) at {ts(utcnow())}. Read {STATE / 'watch.log'} on the PC.")
        finally:
            code = 2
    sys.exit(code)
