"""First-basket confirmed-lineup refresh (first-basket product only, 2026-10-31).

Runs inside health.py's existing cadence (job 6, every 15 minutes; no new cron). For every logged game of tonight's slate
(docs/data/first_basket_today.json) that tips within WINDOW and has no confirmed-lineup check yet, the ESPN summary is polled
until ESPN lists a starting five per team (starter truth: the same box-score `starter` field first_shot.py grades from) or
tip passes.

  identical to the game's last pre-tip snapshot -> the check is logged (data/first_basket_lineups.csv); no snapshot, nothing else.
  different                                     -> single-game reprice through the existing snapshot machinery, unchanged
                                                  (first_basket_model: Tallies.features + score_game with the frozen params), fed the
                                                  confirmed ten. A scratched player has no row and no mass; the remaining shares
                                                  renormalise inside score_game's own per-team normalisation; a replacement is priced
                                                  off his own tallies (the position baseline where he has none); the tip pair is
                                                  Tallies.jumper over the confirmed five. A NEW snapshot is appended to
                                                  data/first_basket_log.csv (never an edit); the why sidecar line carries
                                                  lineup_status="confirmed" and the diff; the METHOD layer re-projects from the log
                                                  (first_basket_method.project, unchanged rules); that game's first-basket odds are
                                                  re-pulled (first_basket_odds.pull, that event only); docs/data/first_basket_today.json
                                                  is rebuilt. Grading convention unchanged: the last pre-tip snapshot grades, so the
                                                  confirmed snapshot becomes the graded one when it lands.
  after tip                                     -> nothing: no poll, no row, no snapshot.

Record: data/first_basket_lineups.csv, append-only, one row per check (pending / identical / changed; a changed game has one row
per team with a diff: out = projected starters ESPN did not list, in = listed starters the snapshot did not have).
Published: docs/data/first_basket_lineups.json (per game: status, confirmed_at, diff, the refreshed snapshot's logged_at) for the
page's "lineups confirmed" marker. Strict JSON through common.write_json like every other published file.

  python first_basket_lineups.py            one pass (what health.py runs)
  python first_basket_lineups.py overlay    rebuild docs/data/first_basket_lineups.json from the checks file (tonight's date)

Untouched: FP v2, ladders, model params (data/first_basket_params.json), grading code.
"""
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_pos, read_json, write_json  # noqa: E402
import first_basket_model as fbm  # noqa: E402  (read-only use of the snapshot machinery: load_tables, walk, score_game, why_lines, build_today)

CHECKS = DATA_DIR / "first_basket_lineups.csv"
OUT = SITE_DATA / "first_basket_lineups.json"
WINDOW = dt.timedelta(minutes=45)
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
CHECK_FIELDS = ["date", "event_id", "tip", "checked_at", "status", "lineup_status", "team", "out", "in", "projected_logged_at", "refreshed_logged_at", "top_before", "top_after", "note"]
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (arc-lab research)"


def ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def clean(x):
    return None if x is None or (isinstance(x, float) and x != x) else x


def espn_summary(event_id):
    r = S.get(ESPN_SUMMARY, params={"event": event_id}, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- 1. watch: ESPN's starting five per team
def confirmed_lineups(summary):
    """{team abbr: [five {id, name, pos}]} once ESPN lists exactly five starters for both teams, else None.
    The box-score players block (the `starter` field first_shot.py / fetch_games.py read) first; ESPN's pre-game `rosters`
    block as the fallback. Before lineups are posted the summary's boxscore carries only `teams`: not confirmed."""
    out = {}
    for b in (summary.get("boxscore") or {}).get("players") or []:
        sts = b.get("statistics") or []
        ath = (sts[0].get("athletes") or []) if sts else []
        five = [a for a in ath if a.get("starter") and (a.get("athlete") or {}).get("id")]
        out[(b.get("team") or {}).get("abbreviation")] = [{"id": str(a["athlete"]["id"]), "name": a["athlete"].get("displayName"),
                                                           "pos": (a["athlete"].get("position") or {}).get("abbreviation")} for a in five]
    if not out:
        for t in summary.get("rosters") or []:
            five = [x for x in t.get("roster") or [] if x.get("starter") and (x.get("athlete") or {}).get("id")]
            out[(t.get("team") or {}).get("abbreviation")] = [{"id": str(x["athlete"]["id"]), "name": x["athlete"].get("displayName"),
                                                              "pos": (x.get("position") or {}).get("abbreviation") if isinstance(x.get("position"), dict) else x.get("position")} for x in five]
    if len(out) != 2 or any(len(v) != 5 for v in out.values()):
        return None
    return out


def checks_done(date):
    """event_id -> final status (identical / changed) already logged for the date; those games are not polled again."""
    if not CHECKS.exists():
        return {}
    c = pd.read_csv(CHECKS, dtype={"event_id": str})
    c = c[(c.date == date) & (c.status != "pending")]
    return dict(zip(c.event_id, c.status))


def log_checks(rows):
    new = not CHECKS.exists()
    with open(CHECKS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
        if new:
            w.writeheader()
        w.writerows([{k: r.get(k, "") for k in CHECK_FIELDS} for r in rows])


# ---------------------------------------------------------------- 2. diff: confirmed ten vs the last snapshot's ten
def lineup_diff(snapshot, confirmed):
    """snapshot: the game's last pre-tip log rows; confirmed: {team: [five]}. {team: {out: [names], in: [names]}}; empty = identical."""
    names = dict(zip(snapshot.player_id, snapshot.player))
    d = {}
    for team, g in snapshot.groupby("team"):
        proj = set(g.player_id); conf = {x["id"]: x for x in confirmed.get(team, [])}
        out = sorted(names[p] for p in proj - set(conf)); inn = sorted(conf[p]["name"] or p for p in set(conf) - proj)
        if out or inn:
            d[team] = {"out": out, "in": inn}
    return d


def diff_text(diff):
    return "; ".join(f"{t}: out {', '.join(d['out']) or '-'}, in {', '.join(d['in']) or '-'}" for t, d in diff.items())


# ---------------------------------------------------------------- 3. reprice: the existing snapshot machinery, fed the confirmed ten
class Ctx:
    """Tonight's shared inputs, built once per pass: the walk-forward tallies through every labelled game (what `snapshot` uses),
    the frozen params, tonight's projections and slate rosters for names / positions."""

    def __init__(self):
        self.params = read_json(fbm.PARAMS, None)
        if not self.params:
            raise RuntimeError("no frozen first-basket params")
        self.P = read_json(SITE_DATA / "projections.json", None) or {}
        self.slate = read_json(DATA_DIR / "slate.json", {}) or {}
        self.games, self.seq, self.st = fbm.load_tables()
        _, self.T = fbm.walk(self.games, self.st, set())
        self.season = int(self.P.get("season") or self.games.season.max())
        self.shooters = fbm.team_first_shooters(self.games, self.st, {self.season - 1, self.season})
        self.tenure = fbm.player_team_counts(self.games, self.st, self.season)
        self.players = {str(p["id"]): p for p in self.P.get("players") or []}
        self.roster = {str(x["id"]): x for lst in (self.slate.get("rosters") or {}).values() for x in lst}
        latest = self.st.sort_values("date").drop_duplicates("player_id", keep="last")
        self.last_pos = dict(zip(latest.player_id, latest.pos))

    def identity(self, x):
        """Name and position of a confirmed starter: tonight's projections, then the slate roster, then his latest starters-file
        position, then ESPN's generic position. His attempt history enters by id through Tallies.features (own history)."""
        pid = x["id"]; p = self.players.get(pid) or self.roster.get(pid) or {}
        return p.get("name") or x["name"] or pid, norm_pos(p.get("pos") or self.last_pos.get(pid) or x["pos"])


def reprice(ctx, game, confirmed, now, diff):
    """Single-game fast path. game: the first_basket_today entry (event_id, tip, home, away) plus date. Appends the new snapshot
    (ten rows, logged_at = now) and its why line; returns (rows, S, G)."""
    date = game["date"]; K, C = ctx.params["K"], ctx.params["constants"]
    strs = []
    for team in (game["home"], game["away"]):
        for x in confirmed[team]:
            name, pos = ctx.identity(x)
            strs.append(fbm.Row(team=team, player_id=x["id"], player=name, pos=pos))
    g = fbm.Row(event_id=game["event_id"], date=date, season=ctx.season, home=game["home"], away=game["away"], label_clean=0, first_fg_id=None, first_fg_started=None)
    G = ctx.T.features(g, strs, pre_season=ctx.season)
    S = fbm.score_game(G, K, C)
    top = int(np.argmax(S["v2"])); stamp = ts(now)
    rows = [{"date": date, "event_id": game["event_id"], "tip": game["tip"], "team": s["team"], "opp": game["away"] if s["home"] else game["home"], "home": s["home"],
             "player_id": s["player_id"], "player": s["player"], "pos": s["pos"], "p_first_basket": round(float(S["v2"][i]), 4), "p_first_attempt": round(float(S["p_attempt"][i]), 4),
             "top_pick": int(i == top), "jumper": s["jumper"], "logged_at": stamp} for i, s in enumerate(G["starters"])]
    why = fbm.why_lines(G, S, K, top, ctx.shooters, ctx.tenure)
    with open(fbm.LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fbm.LOG_FIELDS)
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)
    with open(fbm.WHY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": date, "event_id": game["event_id"], "logged_at": stamp, "why": why, "p_tip_home": round(float(S["p_tip"]), 3),
                            "lineup_status": "confirmed", "confirmed_at": stamp, "lineup_diff": diff}) + "\n")
    return rows, S, G


def pull_odds(events):
    from first_basket_odds import pull   # display layer; needs ODDS_API_KEY in the environment (never printed, never written)
    return pull(events=events)


def regenerate(ctx, date, refreshed, odds_fn=pull_odds):
    """After the refreshed snapshots are in the log: METHOD layer from the log (unchanged rules), that game's odds, today's JSON."""
    try:
        import first_basket_method as fbmth
        fbmth.project(date)
    except Exception as e:  # noqa: BLE001
        print(f"[fb-lineups] method layer re-projection failed: {e!r}")
    if odds_fn is not None:
        try:
            odds_fn(refreshed)
        except Exception as e:  # noqa: BLE001
            print(f"[fb-lineups] odds re-pull failed: {e!r}")
    fbm.build_today(date, slate=ctx.P if ctx.P.get("date") == date else None)


# ---------------------------------------------------------------- 4. publish: the page's marker
def publish_overlay(date, now=None):
    c = pd.read_csv(CHECKS, dtype={"event_id": str}) if CHECKS.exists() else pd.DataFrame(columns=CHECK_FIELDS)
    c = c[c.date == date] if len(c) else c
    games = {}
    for e, g in c.groupby("event_id"):
        fin = g[g.status != "pending"]
        if fin.empty:
            last = g.iloc[-1]
            games[e] = {"status": "pending", "polls": int(len(g)), "last_checked_at": last.checked_at, "note": clean(last.note)}
            continue
        st = fin.status.iloc[0]; r0 = fin.iloc[0]
        games[e] = {"status": st, "lineup_status": "confirmed", "confirmed_at": r0.checked_at, "polls": int(len(g)), "projected_logged_at": clean(r0.projected_logged_at),
                    "refreshed_logged_at": clean(r0.refreshed_logged_at) if st == "changed" else None, "top_before": clean(r0.top_before), "top_after": clean(r0.top_after),
                    "diff": {r.team: {"out": [x for x in str(clean(r.out) or "").split("; ") if x], "in": [x for x in str(clean(r.get("in")) or "").split("; ") if x]}
                             for _, r in fin.iterrows() if st == "changed" and isinstance(r.team, str) and r.team}}
    write_json(OUT, {"date": date, "generated": ts(now or fbm.now()), "window_minutes": int(WINDOW.total_seconds() // 60), "games": games, "checks": int(len(c)),
                     "source": "data/first_basket_lineups.csv (health workflow, every 15 minutes inside 45 minutes of tip): ESPN's listed starting five per team vs the game's last pre-tip snapshot",
                     "note": "changed = a new snapshot was appended to data/first_basket_log.csv from the confirmed ten (its logged_at is refreshed_logged_at); identical = prices stand; "
                             "pending = ESPN had not listed a starting five at the last poll. The last pre-tip snapshot grades, as before."})
    return games


# ---------------------------------------------------------------- one pass
def watch(now=None, fetch_fn=espn_summary, odds_fn=pull_odds):
    now = now or fbm.now()
    FB = read_json(fbm.TODAY, None)
    if not FB or not FB.get("games"):
        print("[fb-lineups] no logged slate (docs/data/first_basket_today.json)"); return {"status": "no_slate"}
    date = FB["date"]; done = checks_done(date)
    due = [g for g in FB["games"] if g["event_id"] not in done and now < parse_ts(g["tip"]) <= now + WINDOW]
    if not due:
        nxt = sorted(parse_ts(g["tip"]) for g in FB["games"] if parse_ts(g["tip"]) > now)
        print(f"[fb-lineups] {date}: nothing due at {now:%H:%M}Z ({len(done)} games checked" + (f", next tip {nxt[0]:%H:%M}Z" if nxt else ", no tip ahead") + ")")
        return {"status": "idle", "date": date, "checked": len(done)}
    if not fbm.LOG.exists():
        print("[fb-lineups] no first-basket log"); return {"status": "no_log"}
    log = pd.read_csv(fbm.LOG, dtype={"event_id": str, "player_id": str})
    snap = fbm.last_pretip(log, date)
    ctx = None; checks, refreshed, results, details = [], [], {}, {}
    for g in due:
        e = g["event_id"]; base = {"date": date, "event_id": e, "tip": g["tip"], "checked_at": ts(now)}; label = f"{g['away']}@{g['home']}"
        try:
            summ = fetch_fn(e)
        except Exception as ex:  # noqa: BLE001
            checks.append({**base, "status": "pending", "lineup_status": "unknown", "note": f"ESPN summary unreachable: {str(ex)[:100]}"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: ESPN unreachable ({ex!r}); again next pass"); continue
        conf = confirmed_lineups(summ)
        if conf is None:
            checks.append({**base, "status": "pending", "lineup_status": "projected", "note": "ESPN has not listed a starting five per team yet"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: lineups not listed yet (tip {parse_ts(g['tip']) - now} away); again next pass"); continue
        rows = snap[snap.event_id == e]
        if rows.empty:
            checks.append({**base, "status": "pending", "lineup_status": "confirmed", "note": "no pre-tip snapshot in the log for this game: nothing to compare"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: confirmed but no snapshot in the log; skipped"); continue
        if set(conf) != set(rows.team):
            checks.append({**base, "status": "pending", "lineup_status": "confirmed", "note": f"ESPN teams {sorted(conf)} do not match the log's {sorted(set(rows.team))}"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: team mismatch {sorted(conf)} vs {sorted(set(rows.team))}; skipped"); continue
        diff = lineup_diff(rows, conf); before = rows.sort_values("p_first_basket", ascending=False).player.iloc[0]; proj_at = rows.logged_at.iloc[0]
        if not diff:
            checks.append({**base, "status": "identical", "lineup_status": "confirmed", "projected_logged_at": proj_at, "top_before": before, "top_after": before,
                           "note": "confirmed ten = projected ten; prices stand, no snapshot"}); results[e] = "identical"
            print(f"[fb-lineups] {label}: lineups confirmed, identical to the {proj_at} snapshot; no snapshot, no change"); continue
        ctx = ctx or Ctx()
        new_rows, S_, G = reprice(ctx, {**g, "date": date}, conf, now, diff)
        after = max(new_rows, key=lambda r: r["p_first_basket"])["player"]
        for team, d in diff.items():
            checks.append({**base, "status": "changed", "lineup_status": "confirmed", "team": team, "out": "; ".join(d["out"]), "in": "; ".join(d["in"]), "projected_logged_at": proj_at,
                           "refreshed_logged_at": ts(now), "top_before": before, "top_after": after, "note": "repriced from the confirmed ten; new snapshot appended"})
        refreshed.append(e); results[e] = "changed"
        details[e] = {"matchup": label, "diff": diff, "diff_text": diff_text(diff), "top_before": before, "top_after": after, "logged_at": ts(now),
                      "top_p_after": max(r["p_first_basket"] for r in new_rows)}
        print(f"[fb-lineups] {label}: lineups confirmed with changes ({diff_text(diff)}); repriced, top pick {before} -> {after}, snapshot {ts(now)} appended")
    if checks:
        log_checks(checks)
    if refreshed:
        regenerate(ctx, date, refreshed, odds_fn)
    publish_overlay(date, now)
    return {"status": "checked", "date": date, "due": [g["event_id"] for g in due], "results": results, "refreshed": refreshed, "details": details}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "watch"
    if cmd == "overlay":
        FB = read_json(fbm.TODAY, None) or {}
        publish_overlay(FB.get("date") or dt.date.today().isoformat())
    else:
        watch()
