"""Arc Lab redesign, Builds 1 and 3 (CEO queue 2026-11-10): forward logger + integrity audit + anchored gap price in shadow.
Paper only. Nothing live changes: FP v2's logs, prices and grades are not read for pricing and never written; ladders, first basket
and grading code untouched. Rule: backtests/fp_redesign_forward_rule.md (pre-registered before the clock's first slate was graded).

  python fp_forward.py capture              every health pass (push-triggered on each board copy): append every Underdog standard
                                            Fantasy Points line of the board copy (data/board_latest.json) with the copy's timestamps
                                            and NBA.com's lineup status for the player -> data/fp_forward_ud.csv; append every new
                                            PrizePicks Fantasy Score history row of data/pp_fantasy_lines.json -> data/fp_forward_pp.csv
  python fp_forward.py backfill --since D   the same rows from every committed board / PP copy since date D (git history; the hook's
                                            own capture stamps), flagged backfill=1 and no lineup status; used once at build time
  python fp_forward.py price                per UD line (slate date, player, team): one priced record when the line's priced UD
                                            capture and the slate's PP MORNING pull are both on file -> data/fp_forward_log.csv (append-only)
  python fp_forward.py grade                morning grade pass: graded / void records (new rows, originals stay), the integrity audit,
                                            the Stage-3 gate clock (n = 750 from 2026-11-10; futility look at 375), the regime-decay
                                            monitor and the descriptive money ledger -> docs/data/fp_forward.json
  python fp_forward.py power                the registration's power report from the pre-clock lines (dates before 2026-11-10)
"""
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, ROOT, SITE_DATA, norm_name, read_json, write_json  # noqa: E402
from odds import team_abbr  # noqa: E402

RULE = "backtests/fp_redesign_forward_rule.md"
UD = DATA_DIR / "fp_forward_ud.csv"
PP = DATA_DIR / "fp_forward_pp.csv"
LOG = DATA_DIR / "fp_forward_log.csv"
OUT = SITE_DATA / "fp_forward.json"
OUTAGES = DATA_DIR / "fp_forward_outages.json"      # verified scraper downtime, hand-entered (rule addendum 2026-11-11)
OUTAGE_LOG = DATA_DIR / "fp_forward_outage_log.csv" # the outage-tagged lines, logged separately
BOARD = DATA_DIR / "board_latest.json"
PP_COPY = DATA_DIR / "pp_fantasy_lines.json"
BOX = DATA_DIR / "player_games.csv"
FPV2_LOG = DATA_DIR / "fp_v2_log.csv"       # read-only: the zero-weight engine deviation column (never priced)
DVP = SITE_DATA / "dvp.json"                # read-only: the d_match shadow column's inputs
PROJ = SITE_DATA / "projections.json"
NBACOM = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{d}.json"
NBA_TO_ESPN = {"UTA": "UTAH", "NOP": "NO", "SAS": "SA", "GSW": "GS", "NYK": "NY", "WAS": "WSH"}
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001
    ET = dt.timezone(dt.timedelta(hours=-5))
FP_MARKETS = {"fantasy points", "fantasy score", "fantasy pts"}
WEIGHTS = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "tov": -1.0}
CLOCK_START = "2026-11-10"
MORNING_WINDOW = (10, 17)              # UTC hours: the LAST PP copy captured inside [10:00Z, 17:00Z) on the slate date = the morning pull (DS-1: median ~9 h pre-tip;
                                       # set on pre-clock data: the last copy pairs 95-100% of UD lines at a median lead of 7-11 h, the earliest copy only 20-40%)
GATE_N, FUTILITY_N, GAIN_BAR = 750, 375, 0.003
BETA, P_CLIP = 0.034, (0.05, 0.95)     # linear, capped at 0.034 per FP point (the cap is the registered slope; see the rule)
SELECT_P = 0.55                        # money ledger: a pick is the side with P >= 0.55 (|gap - offset| >= 1.47 FP)
LEDGER_N, LEDGER_FLOOR = 400, 0.52
GATE_CALIBRATION = "one-sided"         # CEO decision #1, 2026-11-10: over-confidence only
CAL_TOL, CAL_MIN = 0.05, 50
MONITOR_WINDOW = 150
UD_FIELDS = ["captured_at", "source_updated_at", "board_sha1", "date", "scheduled_at", "player", "player_key", "team", "opp", "line", "line_type", "decimal_higher", "decimal_lower",
             "american_higher", "american_lower", "mult_higher", "mult_lower", "source_id", "lineup_status", "lineup_checked_at", "backfill", "logged_at"]
PP_FIELDS = ["captured_at", "last_seen_at", "copy_captured_at", "date", "scheduled_at", "player", "player_key", "team", "opp", "line", "source_id", "backfill", "logged_at"]
LOG_FIELDS = ["record_at", "kind", "date", "player", "player_key", "team", "opp", "event_id", "tip", "scheduled_at",
              "ud_line", "ud_captured_at", "ud_source_id", "ud_line_last", "ud_last_captured_at", "ud_decimal_higher", "ud_decimal_lower", "ud_novig_over",
              "pp_morning_line", "pp_morning_captured_at", "pp_morning_copy_at", "pp_pretip_line", "pp_pretip_captured_at",
              "gap_morning", "gap_missing", "gap_pretip", "offset", "offset_n", "beta", "gap_term", "p_gap", "p_flat",
              "dev_engine", "fp_v2_mean", "fp_v2_logged_at", "engine_weight", "pos", "d_match", "dvp_pts", "dvp_reb", "dvp_ast",
              "lineup_status_last", "lineup_checked_at", "clock", "backfill",
              "fp_actual", "min_actual", "hit_over", "push", "void", "void_reason", "ll_gap", "ll_flat", "pick_side", "pick_hit"]


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def fmt(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return fmt(now())


def parse_ts(s):
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def slate_date(scheduled_at):
    return parse_ts(scheduled_at).astimezone(ET).date().isoformat()


def append(path, fields, rows):
    if not rows:
        return 0
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def read(path, **kw):
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw) if path.exists() else pd.DataFrame(columns=kw.get("usecols") or [])


# ---------------------------------------------------------------- NBA.com lineup status (as of the capture)
def nbacom_status(date):
    """{(player_key, ESPN team): status} from NBA.com's daily lineups for the slate date: starter:Confirmed / starter:Expected /
    bench:Active / bench:Inactive. None when the file is not published or unreachable (logged as no_file)."""
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0 (arc-lab research)", "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "Accept": "application/json"})
    try:
        r = s.get(NBACOM.format(d=date.replace("-", "")), timeout=20)
        if r.status_code != 200:
            return None
        doc = r.json()
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for g in doc.get("games") or []:
        for side in ("awayTeam", "homeTeam"):
            t = g.get(side) or {}; abbr = NBA_TO_ESPN.get(t.get("teamAbbreviation"), t.get("teamAbbreviation"))
            for p in t.get("players") or []:
                name = p.get("playerName") or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                if (p.get("position") or "").strip():
                    st = "starter:" + str(p.get("lineupStatus") or "?")
                else:
                    st = "bench:" + str(p.get("rosterStatus") or p.get("lineupStatus") or "?")
                out[(norm_name(name), abbr)] = st
    return out


# ---------------------------------------------------------------- capture
def ud_rows(board, lineup, checked_at, backfill=0):
    cap, upd, sha = board.get("captured_at"), board.get("source_updated_at"), board.get("board_sha1")
    rows = []
    for e in board.get("entries") or []:
        if (e.get("league") or "").upper() != "NBA" or (e.get("market") or "").strip().lower() not in FP_MARKETS:
            continue
        pid = e.get("platform_ids") or {}
        try:
            line = float(e["line"])
        except (KeyError, TypeError, ValueError):
            continue
        if not e.get("scheduled_at"):
            continue
        team, opp = team_abbr(e.get("team")) or e.get("team"), team_abbr(e.get("opponent")) or e.get("opponent")
        pr, mu = pid.get("prices") or {}, pid.get("payout_multipliers") or {}
        key = norm_name(e.get("player"))
        st = "not_fetched" if lineup is None else lineup.get((key, team), "absent")
        if backfill:
            st = "not_captured_backfill"
        rows.append({"captured_at": cap, "source_updated_at": upd, "board_sha1": sha, "date": slate_date(e["scheduled_at"]), "scheduled_at": e["scheduled_at"], "player": e.get("player"), "player_key": key,
                     "team": team, "opp": opp, "line": line, "line_type": pid.get("line_type") or "", "decimal_higher": (pr.get("higher") or {}).get("decimal", ""), "decimal_lower": (pr.get("lower") or {}).get("decimal", ""),
                     "american_higher": (pr.get("higher") or {}).get("american", ""), "american_lower": (pr.get("lower") or {}).get("american", ""), "mult_higher": mu.get("higher", ""), "mult_lower": mu.get("lower", ""),
                     "source_id": e.get("source_id"), "lineup_status": st, "lineup_checked_at": checked_at or "", "backfill": backfill, "logged_at": stamp()})
    return rows


def pp_rows(copy, backfill=0):
    cap = copy.get("captured_at"); rows = []
    for l in copy.get("lines") or []:
        if str(l.get("odds_type") or "").lower() != "standard" or str(l.get("stat_type") or "").lower() not in FP_MARKETS:
            continue
        try:
            line = float(l["line"])
        except (KeyError, TypeError, ValueError):
            continue
        if not l.get("scheduled_at") or not l.get("captured_at"):
            continue
        rows.append({"captured_at": fmt(parse_ts(l["captured_at"])), "last_seen_at": fmt(parse_ts(l.get("last_seen_at") or l["captured_at"])), "copy_captured_at": cap, "date": slate_date(l["scheduled_at"]),
                     "scheduled_at": fmt(parse_ts(l["scheduled_at"])), "player": l.get("player"), "player_key": norm_name(l.get("player")), "team": team_abbr(l.get("team")) or l.get("team"), "opp": team_abbr(l.get("opponent")) or l.get("opponent"),
                     "line": line, "source_id": str(l.get("source_id") or ""), "backfill": backfill, "logged_at": stamp()})
    return rows


def dedupe_pp(rows):
    """A PP history row is new when its (source_id, captured_at, line) is not on file; last_seen_at is refreshed by a later row."""
    have = read(PP, usecols=["source_id", "captured_at", "line", "last_seen_at"]) if PP.exists() else None
    seen = set(zip(have.source_id, have.captured_at, have.line.astype(float).round(2))) if have is not None and len(have) else set()
    last = dict(zip(zip(have.source_id, have.captured_at), have.last_seen_at)) if have is not None and len(have) else {}
    out = []
    for r in rows:
        k = (r["source_id"], r["captured_at"], round(float(r["line"]), 2))
        if k not in seen or (last.get((r["source_id"], r["captured_at"]), "") < r["last_seen_at"]):
            out.append(r); seen.add(k); last[(r["source_id"], r["captured_at"])] = r["last_seen_at"]
    return out


def capture():
    board = read_json(BOARD, None)
    n_ud = n_pp = 0
    if board and board.get("entries"):
        if UD.exists() and (read(UD, usecols=["board_sha1"]).board_sha1 == board.get("board_sha1")).any():
            print(f"[fp-forward] board {board.get('captured_at')} already captured")
        else:
            dates = sorted({slate_date(e["scheduled_at"]) for e in board["entries"] if e.get("scheduled_at") and (e.get("market") or "").strip().lower() in FP_MARKETS and (e.get("league") or "").upper() == "NBA"})
            lineup, checked = {}, stamp()
            for d in dates:
                l = nbacom_status(d)
                if l is None:
                    lineup = None; break
                lineup.update(l)
            n_ud = append(UD, UD_FIELDS, ud_rows(board, lineup, checked))
    copy = read_json(PP_COPY, None)
    if copy and copy.get("lines"):
        n_pp = append(PP, PP_FIELDS, dedupe_pp(pp_rows(copy)))
    print(f"[fp-forward] captured {n_ud} UD FP line rows (board {board.get('captured_at') if board else None}), {n_pp} new PP history rows (copy {copy.get('captured_at') if copy else None})")
    return n_ud, n_pp


def git_versions(path, since):
    out = subprocess.run(["git", "log", "--format=%H %cI", f"--since={since}T00:00:00Z", "--", path], capture_output=True, text=True, cwd=ROOT).stdout.split("\n")
    return [(l.split()[0], l.split()[1]) for l in out if l.strip()][::-1]


def git_json(sha, path):
    txt = subprocess.run(["git", "show", f"{sha}:{path}"], capture_output=True, text=True, cwd=ROOT, encoding="utf-8").stdout
    try:
        return json.loads(txt) if txt else None
    except json.JSONDecodeError:
        return None


def backfill(since):
    have = set(read(UD, usecols=["board_sha1"]).board_sha1) if UD.exists() else set()
    n_ud = n_pp = 0; boards = git_versions("data/board_latest.json", since)
    for sha, _ in boards:
        b = git_json(sha, "data/board_latest.json")
        if not b or b.get("board_sha1") in have:
            continue
        have.add(b.get("board_sha1")); n_ud += append(UD, UD_FIELDS, ud_rows(b, None, None, backfill=1))
    pps = git_versions("data/pp_fantasy_lines.json", since)
    for sha, _ in pps:
        c = git_json(sha, "data/pp_fantasy_lines.json")
        if not c:
            continue
        n_pp += append(PP, PP_FIELDS, dedupe_pp(pp_rows(c, backfill=1)))
    print(f"[fp-forward] backfill since {since}: {len(boards)} board versions -> {n_ud} UD rows; {len(pps)} PP versions -> {n_pp} PP history rows")


# ---------------------------------------------------------------- price (one record per UD line)
def morning_copy_at(pp, date):
    """The slate's PP morning pull = the LAST copy captured on the slate date inside MORNING_WINDOW (UTC) that carries rows for the
    slate. None = no morning pull for that slate (the gap arm has no priced line that day; the late pull is never substituted)."""
    c = pp[pp.date == date].copy_captured_at
    c = c[c.str.startswith(date) & (c.str[11:13].astype(int) >= MORNING_WINDOW[0]) & (c.str[11:13].astype(int) < MORNING_WINDOW[1])]
    return c.max() if len(c) else None


def pp_line_at(rows, t):
    """The PP line as of time t: the latest history row captured at or before t (None if the line was not yet posted)."""
    r = rows[rows.captured_at <= t]
    if not len(r):
        return None
    x = r.sort_values(["captured_at", "logged_at"]).iloc[-1]
    return float(x.line), x.captured_at


def novig(dh, dl):
    try:
        a, b = 1 / float(dh), 1 / float(dl)
        return round(a / (a + b), 4)
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def price():
    ud, pp = read(UD), read(PP)
    if not len(ud):
        print("[fp-forward] no UD captures yet"); return 0
    log = read(LOG)
    priced = set(zip(log[log.kind == "priced"].date, log[log.kind == "priced"].player_key, log[log.kind == "priced"].team)) if len(log) else set()
    fpl = pd.read_csv(FPV2_LOG, dtype={"event_id": str, "player_id": str}, usecols=["date", "player", "team", "fp_mean", "logged_at", "tip"]) if FPV2_LOG.exists() else None
    if fpl is not None:
        fpl["player_key"] = fpl.player.map(norm_name); fpl = fpl[pd.to_datetime(fpl.logged_at, utc=True) < pd.to_datetime(fpl.tip, utc=True)]
    dvp = read_json(DVP, {}).get("teams", {}); proj = read_json(PROJ, {})
    pos_of = {(p["date"] if "date" in p else proj.get("date"), norm_name(p["name"]), p["team"]): p.get("pos") for p in proj.get("players") or [] if not p.get("inactive")}
    ud = ud[ud.line_type.isin(["", "balanced"])].copy(); ud["line"] = ud.line.astype(float)
    t_now = stamp(); rows = []
    # expanding offset: mean gap over PRIOR slate dates' priced records (non-missing gaps), never the same slate
    hist = log[(log.kind == "priced") & (log.gap_missing == "0")][["date", "gap_morning"]].copy() if len(log) else pd.DataFrame(columns=["date", "gap_morning"])
    hist["gap_morning"] = pd.to_numeric(hist.gap_morning, errors="coerce")
    for (date, key, team), g in ud.sort_values("captured_at").groupby(["date", "player_key", "team"]):
        if (date, key, team) in priced:
            continue
        sched = g.scheduled_at.iloc[0]; tip = fmt(parse_ts(sched))
        g = g[g.captured_at < tip]            # pre-tip captures only
        if not len(g):
            continue
        m_at = morning_copy_at(pp, date)
        if m_at is None and t_now < f"{date}T{MORNING_WINDOW[1]:02d}:00:00Z":
            continue                           # the morning window is still open: wait
        after = g[g.captured_at >= (m_at or "")]
        u = (after if len(after) else g).iloc[0]; last = g.iloc[-1]
        if t_now < tip and not len(after):
            continue                           # no capture after the morning pull yet; the next pass prices it
        pm = pp[(pp.date == date) & (pp.player_key == key) & (pp.team == team)]
        pp_m = pp_line_at(pm, m_at) if m_at else None
        pp_t = pp_line_at(pm[pm.captured_at < tip], tip) if len(pm) else None
        gap_missing = pp_m is None
        gap = 0.0 if gap_missing else round(pp_m[0] - u.line, 2)
        prior = hist[hist.date < date]
        offset = round(float(prior.gap_morning.mean()), 4) if len(prior) else 0.0
        term = 0.0 if gap_missing else BETA * (gap - offset)
        p_gap = round(float(np.clip(0.5 + term, *P_CLIP)), 4)
        dev = ""; v2m = ""; v2t = ""
        if fpl is not None:
            f = fpl[(fpl.date == date) & (fpl.player_key == key) & (fpl.team == team) & (fpl.logged_at <= u.captured_at)]
            if len(f):
                x = f.sort_values("logged_at").iloc[-1]; v2m, v2t = float(x.fp_mean), x.logged_at; dev = round(v2m - u.line, 2)
        pos = pos_of.get((date, key, team), "")
        dv = dvp.get(u.opp, {}).get(pos, {}) if pos else {}
        f3 = [dv.get(s, {}).get("f") for s in ("pts", "reb", "ast")]
        dmatch = round(float(np.mean([x for x in f3 if x is not None])), 4) if all(x is not None for x in f3) else ""
        rows.append({"record_at": t_now, "kind": "priced", "date": date, "player": u.player, "player_key": key, "team": team, "opp": u.opp, "event_id": "", "tip": tip, "scheduled_at": sched,
                     "ud_line": u.line, "ud_captured_at": u.captured_at, "ud_source_id": u.source_id, "ud_line_last": last.line, "ud_last_captured_at": last.captured_at,
                     "ud_decimal_higher": u.decimal_higher, "ud_decimal_lower": u.decimal_lower, "ud_novig_over": novig(u.decimal_higher, u.decimal_lower),
                     "pp_morning_line": "" if gap_missing else pp_m[0], "pp_morning_captured_at": "" if gap_missing else pp_m[1], "pp_morning_copy_at": m_at or "",
                     "pp_pretip_line": "" if pp_t is None else pp_t[0], "pp_pretip_captured_at": "" if pp_t is None else pp_t[1],
                     "gap_morning": gap, "gap_missing": int(gap_missing), "gap_pretip": "" if pp_t is None else round(pp_t[0] - last.line, 2), "offset": offset, "offset_n": int(len(prior)), "beta": BETA, "gap_term": round(term, 4),
                     "p_gap": p_gap, "p_flat": 0.5, "dev_engine": dev, "fp_v2_mean": v2m, "fp_v2_logged_at": v2t, "engine_weight": 0, "pos": pos, "d_match": dmatch,
                     "dvp_pts": f3[0] if f3[0] is not None else "", "dvp_reb": f3[1] if f3[1] is not None else "", "dvp_ast": f3[2] if f3[2] is not None else "",
                     "lineup_status_last": last.lineup_status, "lineup_checked_at": last.lineup_checked_at, "clock": int(date >= CLOCK_START), "backfill": int(str(u.backfill) == "1"),
                     "fp_actual": "", "min_actual": "", "hit_over": "", "push": "", "void": "", "void_reason": "", "ll_gap": "", "ll_flat": "", "pick_side": "", "pick_hit": ""})
        if not gap_missing:
            hist = pd.concat([hist, pd.DataFrame([{"date": date, "gap_morning": gap}])], ignore_index=True)
    n = append(LOG, LOG_FIELDS, rows)
    print(f"[fp-forward] priced {n} new UD lines ({sum(r['gap_missing'] for r in rows)} without a PP morning pair -> gap 0, flagged)")
    tag_outages()
    return n


def outage_for(date, outages):
    """The verified downtime entry overlapping the slate's morning window (10:00-17:00 UTC on the slate date), or None."""
    lo, hi = f"{date}T{MORNING_WINDOW[0]:02d}:00:00Z", f"{date}T{MORNING_WINDOW[1]:02d}:00:00Z"
    for o in outages:
        if o["start"] < hi and o["end"] > lo:
            return o
    return None


def tag_outages():
    """Rule addendum 2026-11-11: a priced clock line with no PP pair, on a slate with NO morning pull while a verified downtime
    window (data/fp_forward_outages.json) overlaps the slate's morning window, gets a NEW record kind = outage (void, final):
    never graded, out of the gate, the audit shares, the monitor and the ledger. The priced original stays; the same row is
    appended to the separate outage log. Ungraded lines only: a line already graded or void is left as it stands."""
    outages = read_json(OUTAGES, {}).get("outages") or []
    log = read(LOG)
    if not outages or not len(log):
        return 0
    pp = read(PP)
    closed = log[log.kind.isin(["graded", "void", "outage"])]
    closed = set(zip(closed.date, closed.player_key, closed.team))
    t_now = stamp(); rows = []; pull = {}
    for r in log[(log.kind == "priced") & (log.clock == "1") & (log.gap_missing == "1")].itertuples():
        if (r.date, r.player_key, r.team) in closed:
            continue
        if r.date not in pull:
            pull[r.date] = morning_copy_at(pp, r.date) if len(pp) else None
        o = outage_for(r.date, outages)
        if pull[r.date] is not None or o is None:
            continue
        rec = {f: getattr(r, f) for f in LOG_FIELDS}
        rec.update({"record_at": t_now, "kind": "outage", "void": 1, "void_reason": f"outage:{o['id']}"})
        rows.append(rec)
    append(LOG, LOG_FIELDS, rows); append(OUTAGE_LOG, LOG_FIELDS, rows)
    if rows:
        print(f"[fp-forward] tagged {len(rows)} lines as outage ({sorted({r['void_reason'] for r in rows})}); excluded from the morning-pair share and the graded results")
    return len(rows)


# ---------------------------------------------------------------- grade + audit + gate clock
def box_fp(b):
    return sum(WEIGHTS[k] * b[k].astype(float) for k in WEIGHTS)


def logloss(p, h):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); h = np.asarray(h, float)
    return -(h * np.log(p) + (1 - h) * np.log(1 - p))


def twoway_se(x, c1, c2):
    """Two-way cluster-robust SE of the mean of x (Cameron-Gelbach-Miller): V = V_c1 + V_c2 - V_(c1 x c2)."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    def v(c):
        s = pd.Series(x).groupby(pd.factorize(c)[0]).sum().to_numpy(); return float((s ** 2).sum()) / n ** 2
    c12 = pd.factorize(pd.Series(c1).astype(str) + "|" + pd.Series(c2).astype(str))[0]
    return float(np.sqrt(max(v(np.asarray(c1)) + v(np.asarray(c2)) - v(c12), 0.0)))


def calibration_one_sided(d):
    """Over-confidence only (GATE_CALIBRATION one-sided): by confidence bucket of the priced side, realized directional hit rate must be
    at least the claimed probability minus CAL_TOL in every bucket with n >= CAL_MIN, and the n-weighted mean of (claimed - realized)+
    must be <= CAL_TOL. Under-confidence never fails."""
    p = d.p_gap.astype(float); side_p = np.where(p >= 0.5, p, 1 - p); hit = np.where(p >= 0.5, d.hit_over.astype(float), 1 - d.hit_over.astype(float))
    edges = [0.5, 0.55, 0.6, 0.65, 0.7, 1.01]; rows = []; worst = 0.0; wsum = 0.0; n_all = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (side_p >= lo) & (side_p < hi); n = int(m.sum())
        if not n:
            continue
        claimed, real = float(side_p[m].mean()), float(hit[m].mean()); over = max(claimed - real, 0.0)
        rows.append({"lo": lo, "hi": hi, "n": n, "claimed": round(claimed, 4), "realized": round(real, 4), "over_confidence": round(over, 4), "graded": n >= CAL_MIN})
        if n >= CAL_MIN:
            worst = max(worst, over)
        wsum += over * n; n_all += n
    wmean = wsum / n_all if n_all else 0.0
    return {"mode": GATE_CALIBRATION, "buckets": rows, "worst_over_confidence": round(worst, 4), "weighted_over_confidence": round(wmean, 4), "pass": bool(worst <= CAL_TOL and wmean <= CAL_TOL)}


def grade():
    log = read(LOG)
    if not len(log):
        print("[fp-forward] nothing priced yet"); return
    box = pd.read_csv(BOX, dtype={"event_id": str, "player_id": str})
    box["player_key"] = box.player.map(norm_name); box["fp"] = box_fp(box)
    done_events = box.groupby(["date", "team"]).event_id.first().to_dict()
    tag_outages(); log = read(LOG)
    closed = log[log.kind.isin(["graded", "void", "outage"])]            # an outage record is final: the line is never graded
    graded_keys = set(zip(closed.date, closed.player_key, closed.team))
    rows = []
    for r in log[log.kind == "priced"].itertuples():
        k = (r.date, r.player_key, r.team)
        if k in graded_keys or (r.date, r.team) not in done_events:
            continue
        ev = done_events[(r.date, r.team)]
        b = box[(box.event_id == ev) & (box.player_key == r.player_key)]
        rec = {f: getattr(r, f) for f in LOG_FIELDS}; rec.update({"record_at": stamp(), "event_id": ev})
        if not len(b) or float(b.iloc[0]["min"]) <= 0:
            rec.update({"kind": "void", "void": 1, "void_reason": "absent from the box score" if not len(b) else "0 minutes", "min_actual": 0.0 if not len(b) else float(b.iloc[0]["min"])})
        else:
            fpa = round(float(b.iloc[0].fp), 1); line = float(r.ud_line)
            if abs(fpa - line) < 1e-9:
                rec.update({"kind": "void", "void": 1, "void_reason": "push", "fp_actual": fpa, "min_actual": float(b.iloc[0]["min"])})
            else:
                hit = int(fpa > line); p = float(r.p_gap)
                side = "higher" if p >= SELECT_P else "lower" if p <= 1 - SELECT_P else ""
                rec.update({"kind": "graded", "void": 0, "fp_actual": fpa, "min_actual": float(b.iloc[0]["min"]), "hit_over": hit, "push": 0,
                            "ll_gap": round(float(logloss([p], [hit])[0]), 5), "ll_flat": round(float(logloss([0.5], [hit])[0]), 5), "pick_side": side, "pick_hit": "" if not side else int(hit if side == "higher" else 1 - hit)})
        rows.append(rec); graded_keys.add(k)
    append(LOG, LOG_FIELDS, rows)
    log = read(LOG)
    report(log, box)
    print(f"[fp-forward] {len(rows)} new graded/void records; log {len(log)} records")


def latest_records(log):
    """One record per line: the latest graded/void record if any, else the priced one (corrections = later records win)."""
    rank = log.kind.map({"priced": 0, "void": 1, "graded": 1, "correction": 2, "outage": 3}).fillna(0)
    d = log.assign(_r=rank).sort_values(["_r", "record_at"]).drop_duplicates(["date", "player_key", "team"], keep="last")
    return d.drop(columns="_r")


def audit(ud, log, box):
    """Integrity audit, slate by slate (first 5 clock slates reported in full, all slates summarised): UD FP lines captured vs the
    lines the board history carries; PP morning pull present; grade reconciliation (every graded line's hit recomputed from the box)."""
    out = []
    boards = {}
    for sha, _ in git_versions("data/board_latest.json", CLOCK_START):
        b = git_json(sha, "data/board_latest.json")
        if not b:
            continue
        for e in b.get("entries") or []:
            if (e.get("league") or "").upper() == "NBA" and (e.get("market") or "").strip().lower() in FP_MARKETS and e.get("scheduled_at"):
                boards.setdefault(slate_date(e["scheduled_at"]), set()).add((norm_name(e.get("player")), team_abbr(e.get("team")) or e.get("team")))
    lat = latest_records(log); lat_clock = lat[lat.clock == "1"]
    pp = read(PP)
    for date in sorted(d for d in set(boards) | set(lat_clock.date) if d >= CLOCK_START):
        seen = boards.get(date, set()); got = set(zip(ud[ud.date == date].player_key, ud[ud.date == date].team))
        rec = lat_clock[lat_clock.date == date]; n_out = int((rec.kind == "outage").sum()); rec = rec[rec.kind != "outage"]; g = rec[rec.kind == "graded"]
        recon_ok = 0
        for r in g.itertuples():
            b = box[(box.event_id == r.event_id) & (box.player_key == r.player_key)]
            if len(b) and int(float(b.iloc[0].fp) > float(r.ud_line)) == int(float(r.hit_over)):
                recon_ok += 1
        out.append({"date": date, "board_lines": len(seen), "captured": len(seen & got), "capture_share": round(len(seen & got) / len(seen), 4) if seen else None,
                    "pp_morning_pull": morning_copy_at(pp, date) if len(pp) else None, "priced": int(len(rec)), "gap_missing": int((rec.gap_missing == "1").sum()), "outage_lines": n_out,
                    "graded": int(len(g)), "void": int((rec.kind == "void").sum()), "reconciled": recon_ok, "reconciliation": round(recon_ok / len(g), 4) if len(g) else None,
                    "lineup_status_known": int((~rec.lineup_status_last.isin(["", "not_fetched", "not_captured_backfill"])).sum())})
    slates = out[:5]
    pp_all = sum(1 for s in out if s["pp_morning_pull"]) / len(out) if out else None      # as it stood before the 2026-11-11 addendum
    for s in out:
        s["outage_slate"] = bool(s["outage_lines"] and not s["pp_morning_pull"])            # no morning pull under a verified outage
    kept = [s for s in out if not s["outage_slate"]]
    pp_ok = sum(1 for s in kept if s["pp_morning_pull"]) / len(kept) if kept else None
    return {"slates": out, "first_five": slates, "bars": {"capture_share": 0.95, "pp_morning_share": 0.90, "reconciliation": 1.0},
            "pp_morning_share_so_far": round(pp_ok, 4) if pp_ok is not None else None, "pp_morning_share_before_outage_exclusion": round(pp_all, 4) if pp_all is not None else None,
            "outage_slates": [s["date"] for s in out if s["outage_slate"]], "gap_arm_alive_before_outage_exclusion": bool(pp_all is None or pp_all >= 0.90), "capture_share_so_far": round(float(np.mean([s["capture_share"] for s in out if s["capture_share"] is not None])), 4) if any(s["capture_share"] is not None for s in out) else None,
            "reconciliation_so_far": round(float(np.mean([s["reconciliation"] for s in out if s["reconciliation"] is not None])), 4) if any(s["reconciliation"] is not None for s in out) else None,
            "gap_arm_alive": bool(pp_ok is None or pp_ok >= 0.90)}


def gate_block(g):
    """The Stage-3 read on the first GATE_N graded clock lines (ordered by tip, then player): paired gain vs flat 0.5 with the
    two-way (player + date) clustered CI, the one-sided calibration prong, the futility look at FUTILITY_N."""
    g = g.sort_values(["tip", "player_key"]).reset_index(drop=True)
    diff = g.ll_flat.astype(float) - g.ll_gap.astype(float)
    def read_at(n):
        d = diff[:n]; se = twoway_se(d, g.player_key[:n], g.date[:n]); m = float(d.mean())
        return {"n": int(n), "gain_nats": round(m, 5), "se_twoway": round(se, 5), "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}
    n = len(g)
    fut = read_at(FUTILITY_N) if n >= FUTILITY_N else None
    final = None
    if n >= GATE_N:
        r = read_at(GATE_N); cal = calibration_one_sided(g[:GATE_N])
        passed = bool(r["gain_nats"] >= GAIN_BAR and r["ci95"][0] > 0 and cal["pass"])
        final = {**r, "gain_bar": GAIN_BAR, "calibration": cal, "pass": passed, "verdict": "PASS: an edge over a coin-flip price, not profit" if passed else "FAIL (no re-bar, no extension)"}
    interim = read_at(n) if n >= 30 else None
    return {"n_graded": int(n), "gate_n": GATE_N, "futility_n": FUTILITY_N, "gain_bar": GAIN_BAR, "calibration_mode": GATE_CALIBRATION,
            "interim": interim, "futility": {**fut, "stop": bool(fut["gain_nats"] <= 0), "rule": "point estimate <= 0 at 375 stops the arm; the stop can only stop, never pass"} if fut else None,
            "final": final, "status": ("FAIL" if final and not final["pass"] else "PASS" if final else "STOPPED at the futility look" if fut and fut["gain_nats"] <= 0 else f"collecting ({n} of {GATE_N})")}


def monitor(g):
    """Regime-decay monitor: rolling slope of (hit_over - 0.5) on (gap_morning - offset) over the last MONITOR_WINDOW graded lines
    with a PP pair, reported per slate date (the slope the price assumes is BETA = 0.034 per FP point)."""
    g = g[g.gap_missing == "0"].sort_values(["tip", "player_key"]).reset_index(drop=True)
    x = g.gap_morning.astype(float) - g.offset.astype(float); y = g.hit_over.astype(float) - 0.5
    out = []
    for date, idx in g.groupby("date").groups.items():
        end = max(idx) + 1; lo = max(0, end - MONITOR_WINDOW)
        xs, ys = x[lo:end], y[lo:end]
        if len(xs) < 30 or xs.var() == 0:
            continue
        slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
        resid = ys - ys.mean() - slope * (xs - xs.mean()); se = float(np.sqrt((resid ** 2).sum() / (len(xs) - 2) / ((xs - xs.mean()) ** 2).sum()))
        out.append({"date": date, "window_n": int(len(xs)), "slope_per_fp": round(slope, 4), "se": round(se, 4), "beta_assumed": BETA, "mean_abs_gap": round(float(xs.abs().mean()), 3)})
    return out


def ledger(g):
    """Descriptive money ledger: picks = sides with P >= SELECT_P, 1 unit each at Underdog's standard 2-leg payout (3x, break-even
    57.7%; the ~55-58% band in the registration). Never a gate input; < 52% hit rate at 400 picks = no stakes even on a PASS."""
    p = g[g.pick_side != ""].sort_values(["tip", "player_key"])
    n = int(len(p)); hits = int(p.pick_hit.astype(float).sum()) if n else 0
    return {"picks": n, "hits": hits, "hit_rate": round(hits / n, 4) if n else None, "breakeven_band": [0.55, 0.58], "ledger_n": LEDGER_N, "floor": LEDGER_FLOOR,
            "no_stakes_rule": f"selected-pick hit rate < {LEDGER_FLOOR:.0%} at {LEDGER_N} picks = no stakes even on a gate PASS",
            "verdict_at_n": (None if n < LEDGER_N else ("NO STAKES" if hits / n < LEDGER_FLOOR else "above the floor (descriptive; a PASS certifies an edge over a coin flip, not profit)")),
            "units_at_2leg_3x": round(hits * 2.0 - (n - hits) * 1.0, 1) if n else 0.0, "by_slate": [{"date": d, "picks": int(len(s)), "hits": int(s.pick_hit.astype(float).sum())} for d, s in p.groupby("date")]}


def report(log, box):
    ud = read(UD); lat = latest_records(log)
    clock_all = lat[lat.clock == "1"]; outage = clock_all[clock_all.kind == "outage"]
    clock = clock_all[clock_all.kind != "outage"]; g = clock[clock.kind == "graded"]             # outage lines: tagged, logged separately, out of every read
    pre = lat[(lat.clock == "0") & (lat.kind == "graded")]
    doc = {"title": "FP redesign forward test (Builds 1 and 3, paper only)", "rule": RULE, "generated": stamp(), "clock_start": CLOCK_START, "beta": BETA, "gate_calibration": GATE_CALIBRATION,
           "label": "Shadow price only: nothing here is staked, shown as a pick or fed to any live model. A PASS certifies an edge over a coin-flip price, not profit.",
           "capture": {"ud_rows": int(len(ud)), "ud_captures": int(ud.board_sha1.nunique()) if len(ud) else 0, "ud_lines": int(ud.drop_duplicates(["date", "player_key", "team"]).shape[0]) if len(ud) else 0,
                       "first_capture": ud.captured_at.min() if len(ud) else None, "last_capture": ud.captured_at.max() if len(ud) else None, "pp_rows": int(len(read(PP)))},
           "log": {"records": int(len(log)), "lines": int(len(lat)), "clock_lines": int(len(clock)), "clock_graded": int(len(g)), "clock_void": int((clock.kind == "void").sum()), "pre_clock_graded": int(len(pre)),
                   "gap_missing_share_clock": round(float((clock.gap_missing == "1").mean()), 4) if len(clock) else None,
                   "outage_lines": int(len(outage)), "outage_by_slate": {d: int(n) for d, n in outage.groupby("date").size().items()}, "outage_ids": sorted(set(outage.void_reason)),
                   "clock_lines_before_outage_exclusion": int(len(clock_all)),
                   "gap_missing_share_clock_before_outage_exclusion": round(float((clock_all.gap_missing == "1").mean()), 4) if len(clock_all) else None},
           "audit": audit(ud, log, box), "gate": gate_block(g) if len(g) else {"n_graded": 0, "status": f"collecting (0 of {GATE_N})"}, "monitor": monitor(g) if len(g) else [],
           "ledger": ledger(g) if len(g) else None,
           "recent": [{k: r[k] for k in ("date", "player", "team", "opp", "ud_line", "ud_captured_at", "pp_morning_line", "pp_morning_copy_at", "gap_morning", "offset", "p_gap", "kind", "fp_actual", "hit_over", "lineup_status_last")}
                      for r in clock.sort_values(["date", "tip"], ascending=False).head(80).to_dict("records")],
           "method": ["Every Underdog standard Fantasy Points line is captured from every committed board copy with the copy's own timestamps (the hook commits about every 30 minutes); NBA.com's lineup status for the player is stamped on live captures.",
                      "Priced feature = PrizePicks Fantasy Score line as of the slate's MORNING pull (the last copy inside 10:00-17:00Z on the slate date, median lead about 9 h) minus the UD line at its first capture on or after that pull. The PP pre-tip line is a secondary column, never priced. No PP pair = gap 0, flagged.",
                      "Addendum 2026-11-11: lines priced without a morning pair because of verified scraper downtime (data/fp_forward_outages.json) are tagged outage, logged separately (data/fp_forward_outage_log.csv) and left out of the morning-pair share and of every graded read.",
                      "P = 0.5 + 0.034 x (gap - offset), offset = mean gap over prior slates only, clipped to [0.05, 0.95]; the engine deviation (FP v2 mean - UD line) and the matchup column are logged at zero weight and never priced.",
                      f"Gate: n = {GATE_N} graded clock lines, paired gain vs flat 0.5 >= +{GAIN_BAR} nats with the two-way (player + date) clustered CI excluding 0, and the one-sided calibration prong; futility stop at {FUTILITY_N} if the point estimate is <= 0. Not shown at {GATE_N} = FAIL."]}
    write_json(OUT, doc)
    return doc


# ---------------------------------------------------------------- power report (registration)
def power():
    log = read(LOG); lat = latest_records(log)
    d = lat[(lat.clock == "0") & (lat.kind == "graded") & (lat.gap_missing == "0")].copy()
    if not len(d):
        print("[fp-forward] no pre-clock graded lines"); return None
    diff = (d.ll_flat.astype(float) - d.ll_gap.astype(float)).to_numpy()
    sd = float(np.std(diff, ddof=1)); n = len(diff)
    se_tw = twoway_se(diff, d.player_key, d.date); se_iid = sd / np.sqrt(n); deff = (se_tw / se_iid) ** 2 if se_iid else 1.0
    hw750 = 1.96 * sd * np.sqrt(deff) / np.sqrt(GATE_N)
    out = {"source": "pre-clock lines (slates before 2026-11-10) rebuilt from the committed board and PrizePicks copies with the same logger, graded from the box scores",
           "n_lines": int(n), "dates": [str(d.date.min()), str(d.date.max())], "mean_gain_nats": round(float(diff.mean()), 5), "sd_per_line": round(sd, 5),
           "se_twoway": round(se_tw, 5), "se_iid": round(float(se_iid), 5), "design_effect": round(float(deff), 3),
           "ci_half_width_at_750": round(float(hw750), 5), "min_observed_gain_to_pass_at_750": round(float(max(hw750, GAIN_BAR)), 5),
           "n_for_gain_0003_80pct_power": int(np.ceil(((1.96 + 0.84) * sd * np.sqrt(deff) / GAIN_BAR) ** 2)),
           "hit_rate_over": round(float(d.hit_over.astype(float).mean()), 4), "mean_abs_gap_fp": round(float((d.gap_morning.astype(float) - d.offset.astype(float)).abs().mean()), 3),
           "slope_check_per_fp": None}
    x = d.gap_morning.astype(float) - d.offset.astype(float); y = d.hit_over.astype(float) - 0.5
    if x.var() > 0:
        out["slope_check_per_fp"] = round(float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()), 4)
    print(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["capture", "backfill", "price", "grade", "power", "outage"]); ap.add_argument("--since", default=CLOCK_START)
    a = ap.parse_args()
    {"capture": capture, "backfill": lambda: backfill(a.since), "price": price, "grade": grade, "power": power, "outage": tag_outages}[a.cmd]()
