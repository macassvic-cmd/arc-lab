"""First-basket confirmed-lineup refresh (first-basket product only; 2026-10-31, lineup source switched to NBA.com 2026-11-02).

Runs inside health.py's existing cadence (job 6; no new cron, no timer). For every logged game of tonight's slate
(docs/data/first_basket_today.json) that tips within WINDOW (150 minutes) and has no final lineup check yet, NBA.com's daily
lineups file for the slate date (stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json) is read. A team's starting five
is the five players NBA.com lists with a position; it counts as CONFIRMED only when every one of them carries
lineupStatus "Confirmed" (the file says "Expected" for hours before the stamp lands). A listed five without the stamp is not a
confirmation: no snapshot, no reprice, read again next pass.

ESPN is no longer consulted by the watch (its summary carries no starters before tip; job 6 proved that live on 2026-10-31).
ESPN stays the starter truth for GRADING only: once a confirmed game is labelled, `audit` (run at the start of every pass,
report-only) compares the confirmed ten with the starters file and logs any mismatch to data/first_basket_lineup_audit.csv;
the existing late-scratch void governs the grade. No at-tip fallback: not confirmed by tip -> the projected snapshot stands.

Game matching: NBA.com game ids are not ESPN event ids; a game is matched to the slate by the (away, home) team pair after the
six abbreviation differences (UTA/NOP/SAS/GSW/NYK/WAS -> UTAH/NO/SA/GS/NY/WSH); an unmatched game is skipped and logged.
Player ids: data/nbacom_player_ids.csv is the persisted crosswalk NBA.com personId -> the pipeline's ESPN player id
(seeded by `seed`, appended at runtime). A starter missing from it falls back to a normalised-name match inside that team's
roster universe (tonight's slate roster + projections + every player the starters file last saw on that team) and the new
pair is appended. Still unresolved, or ambiguous -> the game's refresh is skipped ENTIRELY (status "unresolved", alarm);
never a guess, never a partial reprice. NBA.com's listed position is recorded in the why sidecar only; the snapshot row keeps
the pipeline's position, so method-layer grouping is unchanged.

  identical to the game's last pre-tip snapshot -> the check is logged (data/first_basket_lineups.csv); no snapshot, nothing else.
  different                                     -> single-game reprice through the existing snapshot machinery, unchanged
                                                  (first_basket_model: Tallies.features + score_game with the frozen params), fed the
                                                  confirmed ten. A scratched player has no row and no mass; the remaining shares
                                                  renormalise inside score_game's own per-team normalisation; a replacement is priced
                                                  off his own tallies (the position baseline where he has none); the tip pair is
                                                  Tallies.jumper over the confirmed five. A NEW snapshot is appended to
                                                  data/first_basket_log.csv (never an edit); the why sidecar line carries
                                                  lineup_status="confirmed", lineup_source="nbacom", the diff, the confirmed ids and
                                                  NBA.com's positions; the METHOD layer re-projects from the log (unchanged rules);
                                                  that game's first-basket odds are re-pulled (that event only);
                                                  docs/data/first_basket_today.json is rebuilt. Grading convention unchanged: the last
                                                  pre-tip snapshot grades, so the confirmed snapshot becomes the graded one.
  after tip                                     -> nothing: no read, no row, no snapshot.

Record: data/first_basket_lineups.csv, append-only, one row per check (pending / identical / changed / unresolved; a changed game
has one row per team with a diff: out = projected starters not in the confirmed five, in = confirmed starters the snapshot did not
have). Published: docs/data/first_basket_lineups.json for the page's "lineups confirmed" marker.

  python first_basket_lineups.py                     one pass (what health.py runs)
  python first_basket_lineups.py overlay             rebuild docs/data/first_basket_lineups.json from the checks file
  python first_basket_lineups.py seed 20261030 ...   resolve every player NBA.com lists on those dates into the crosswalk; report
  python first_basket_lineups.py audit               report-only: confirmed tens vs the graded starters file

Untouched: FP v2, ladders, model params (data/first_basket_params.json), grading code, first_basket_model.py.
"""
import csv
import datetime as dt
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_pos, read_json, write_json  # noqa: E402
import first_basket_model as fbm  # noqa: E402  (read-only use of the snapshot machinery: load_tables, walk, score_game, why_lines, build_today)

CHECKS = DATA_DIR / "first_basket_lineups.csv"
XWALK = DATA_DIR / "nbacom_player_ids.csv"
AUDIT = DATA_DIR / "first_basket_lineup_audit.csv"
STARTERS = DATA_DIR / "first_shot_starters.csv"          # ESPN starter truth, written by grading (read-only here)
OUT = SITE_DATA / "first_basket_lineups.json"
WINDOW = dt.timedelta(minutes=150)
SOURCE = "nbacom"
NBACOM = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{d}.json"
NBA_TO_ESPN = {"UTA": "UTAH", "NOP": "NO", "SAS": "SA", "GSW": "GS", "NYK": "NY", "WAS": "WSH"}
CHECK_FIELDS = ["date", "event_id", "tip", "checked_at", "status", "lineup_status", "team", "out", "in", "projected_logged_at", "refreshed_logged_at", "top_before", "top_after", "note"]
XWALK_FIELDS = ["nba_id", "player_id", "name", "team", "matched_by", "added_at"]
AUDIT_FIELDS = ["date", "event_id", "team", "confirmed_ids", "actual_ids", "out", "in", "audited_at"]
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (arc-lab research)", "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com", "Accept": "application/json"})


def ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def clean(x):
    return None if x is None or (isinstance(x, float) and x != x) else x


def norm_name(s):
    """'Bogdan Bogdanović' / 'Jabari Smith Jr.' / "Jae'Sean Tate" -> 'bogdan bogdanovic' / 'jabari smith' / 'jaesean tate'."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'’]", "", s.lower()); s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(t for t in s.split() if t not in SUFFIXES)


# ---------------------------------------------------------------- 1. watch: NBA.com's daily lineups
def nbacom_daily(date):
    """The daily lineups file for the slate date (YYYY-MM-DD, ET). None when NBA.com has not published it (404)."""
    r = S.get(NBACOM.format(d=date.replace("-", "")), timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _entry(p):
    return {"nba_id": str(p.get("personId")), "name": p.get("playerName") or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
            "pos": (p.get("position") or "").strip(), "status": p.get("lineupStatus"), "stamp": p.get("timestamp")}


def nbacom_games(doc):
    """{(away, home) in the pipeline's abbreviations: {game_id, status_text, teams: {abbr: {confirmed, starters, listed, statuses}}}}.
    starters = the players NBA.com lists with a position (its first five); confirmed = exactly five, all stamped Confirmed."""
    out = {}
    for g in (doc or {}).get("games") or []:
        teams, key = {}, []
        for side in ("awayTeam", "homeTeam"):
            t = g.get(side) or {}; abbr = NBA_TO_ESPN.get(t.get("teamAbbreviation"), t.get("teamAbbreviation")); key.append(abbr)
            allp = [_entry(p) for p in t.get("players") or []]; five = [p for p in allp if p["pos"]]
            teams[abbr] = {"starters": five, "bench": [p for p in allp if not p["pos"]], "listed": len(allp), "statuses": sorted({p["status"] for p in allp if p["status"]}),
                           "confirmed": len(five) == 5 and all(p["status"] == "Confirmed" for p in five)}
        out[tuple(key)] = {"game_id": g.get("gameId"), "status_text": g.get("gameStatusText"), "teams": teams}
    return out


# ---------------------------------------------------------------- 1b. player-id crosswalk
class Unresolved(Exception):
    pass


class Universe:
    """The pipeline's player identity per team: tonight's slate roster, tonight's projections and every player the starters
    file last saw on that team. {team: {player_id: name}} plus name / position lookups."""

    def __init__(self):
        self.by_team, self.names, self.pos = {}, {}, {}
        P = read_json(SITE_DATA / "projections.json", None) or {}
        for p in P.get("players") or []:
            self._add(p.get("team"), str(p["id"]), p.get("name"), p.get("pos"))
        slate = read_json(DATA_DIR / "slate.json", {}) or {}
        for team, lst in (slate.get("rosters") or {}).items():
            for x in lst:
                self._add(team, str(x["id"]), x.get("name"), x.get("pos"))
        if STARTERS.exists():
            st = pd.read_csv(STARTERS, dtype={"player_id": str}, usecols=["date", "team", "player_id", "player", "pos"])
            for r in st.sort_values("date").drop_duplicates("player_id", keep="last").itertuples():
                self._add(r.team, r.player_id, r.player, r.pos)

    def _add(self, team, pid, name, pos):
        if not team or not pid or not name:
            return
        self.by_team.setdefault(team, {})[pid] = name; self.names.setdefault(pid, name)
        if pos and pid not in self.pos:
            self.pos[pid] = pos


class Crosswalk:
    """data/nbacom_player_ids.csv: NBA.com personId -> pipeline player id. Append-only; a runtime name match is appended at once."""

    def __init__(self, path=XWALK):
        self.path = Path(path); self.map = {}
        if self.path.exists():
            for r in csv.DictReader(open(self.path, encoding="utf-8")):
                self.map[str(r["nba_id"])] = str(r["player_id"])

    def resolve(self, p, team, uni, now=None):
        """(player_id, how) for a listed player on `team`. Crosswalk hit, else a unique normalised-name match in the team's universe
        (appended). No match or more than one -> Unresolved (the caller skips the whole game)."""
        if p["nba_id"] in self.map:
            return self.map[p["nba_id"]], "crosswalk"
        want = norm_name(p["name"])
        cands = sorted({pid for pid, nm in uni.by_team.get(team, {}).items() if norm_name(nm) == want})
        if len(cands) == 1:
            self.append(p["nba_id"], cands[0], uni.names.get(cands[0], p["name"]), team, "name+team", now)
            return cands[0], "name+team"
        if not cands:
            raise Unresolved(f"{p['name']} (NBA.com id {p['nba_id']}, {team}) matches nobody in {team}'s roster universe ({len(uni.by_team.get(team, {}))} players)")
        raise Unresolved(f"{p['name']} (NBA.com id {p['nba_id']}, {team}) is ambiguous: {', '.join(f'{uni.names.get(c)} #{c}' for c in cands)}")

    def append(self, nba_id, pid, name, team, how, now=None):
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=XWALK_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"nba_id": nba_id, "player_id": pid, "name": name, "team": team, "matched_by": how, "added_at": ts(now or fbm.now())})
        self.map[str(nba_id)] = str(pid)


def resolve_game(ng, uni, xw, now=None):
    """{team: [five {id, name, pos (NBA.com's), nba_id, matched_by}]} for a confirmed NBA.com game, or raises Unresolved on the
    first starter that cannot be resolved (no partial result)."""
    out = {}
    for team, t in ng["teams"].items():
        out[team] = []
        for p in t["starters"]:
            pid, how = xw.resolve(p, team, uni, now)
            out[team].append({"id": pid, "name": uni.names.get(pid, p["name"]), "pos": p["pos"], "nba_id": p["nba_id"], "matched_by": how})
    return out


def checks_done(date):
    """event_id -> final status (identical / changed / unresolved) already logged for the date; those games are not read again."""
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
        """Name and position of a confirmed starter from the PIPELINE's identity: tonight's projections, then the slate roster, then
        his latest starters-file position; NBA.com's listed position only if the pipeline has none. His attempt history enters by
        id through Tallies.features (own history)."""
        pid = x["id"]; p = self.players.get(pid) or self.roster.get(pid) or {}
        return p.get("name") or x["name"] or pid, norm_pos(p.get("pos") or self.last_pos.get(pid) or x["pos"])


def reprice(ctx, game, confirmed, now, diff, source=None):
    """Single-game fast path. game: the first_basket_today entry (event_id, tip, home, away) plus date. Appends the new snapshot
    (ten rows, logged_at = now) and its why line (stamped confirmed / nbacom, with the diff, the confirmed ids and NBA.com's
    positions); returns (rows, S, G)."""
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
    line = {"date": date, "event_id": game["event_id"], "logged_at": stamp, "why": why, "p_tip_home": round(float(S["p_tip"]), 3),
            "lineup_status": "confirmed", "lineup_source": SOURCE, "confirmed_at": stamp, "lineup_diff": diff,
            "confirmed_ids": {t: [x["id"] for x in confirmed[t]] for t in confirmed},
            "nbacom_positions": {x["id"]: x["pos"] for t in confirmed for x in confirmed[t] if x.get("pos")}, **(source or {})}
    with open(fbm.WHY, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
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
        if st == "unresolved":
            games[e] = {"status": "unresolved", "lineup_status": "confirmed", "lineup_source": SOURCE, "confirmed_at": r0.checked_at, "polls": int(len(g)), "note": clean(r0.note)}
            continue
        games[e] = {"status": st, "lineup_status": "confirmed", "lineup_source": SOURCE, "confirmed_at": r0.checked_at, "polls": int(len(g)), "projected_logged_at": clean(r0.projected_logged_at),
                    "refreshed_logged_at": clean(r0.refreshed_logged_at) if st == "changed" else None, "top_before": clean(r0.top_before), "top_after": clean(r0.top_after),
                    "diff": {r.team: {"out": [x for x in str(clean(r.out) or "").split("; ") if x], "in": [x for x in str(clean(r.get("in")) or "").split("; ") if x]}
                             for _, r in fin.iterrows() if st == "changed" and isinstance(r.team, str) and r.team}}
    write_json(OUT, {"date": date, "generated": ts(now or fbm.now()), "window_minutes": int(WINDOW.total_seconds() // 60), "lineup_source": SOURCE, "games": games, "checks": int(len(c)),
                     "source": "data/first_basket_lineups.csv (health workflow, inside 150 minutes of tip): NBA.com's daily lineups (the five listed with a position, all stamped Confirmed) vs the game's last pre-tip snapshot",
                     "note": "changed = a new snapshot was appended to data/first_basket_log.csv from the confirmed ten (its logged_at is refreshed_logged_at); identical = prices stand; "
                             "pending = NBA.com had not stamped the five Confirmed at the last read; unresolved = confirmed but a starter could not be matched to the pipeline's "
                             "player ids, so the game was not repriced. The last pre-tip snapshot grades, as before."})
    return games


# ---------------------------------------------------------------- 5. audit (report-only): confirmed tens vs ESPN's graded starters
def audit(today, now=None):
    """For every confirmed snapshot of an earlier date whose game the starters file has labelled and that is not audited yet:
    log confirmed vs actual ids per team (data/first_basket_lineup_audit.csv). Report-only; grading and the late-scratch void
    are untouched. Returns the mismatches found this pass."""
    if not fbm.WHY.exists() or not STARTERS.exists():
        return []
    done = set()
    if AUDIT.exists():
        a = pd.read_csv(AUDIT, dtype={"event_id": str}); done = set(zip(a.event_id, a.team))
    todo = {}
    for line in open(fbm.WHY, encoding="utf-8"):
        try:
            w = json.loads(line)
        except ValueError:
            continue
        if w.get("lineup_source") == SOURCE and w.get("confirmed_ids") and w.get("date", "") < today:
            todo[w["event_id"]] = w   # the last confirmed snapshot of the game
    todo = {e: w for e, w in todo.items() if any((e, t) not in done for t in w["confirmed_ids"])}
    if not todo:
        return []
    st = pd.read_csv(STARTERS, dtype={"event_id": str, "player_id": str}, usecols=["event_id", "team", "player_id", "player"])
    st = st[st.event_id.isin(todo)]
    rows, mism = [], []
    for e, w in todo.items():
        g = st[st.event_id == e]
        if g.empty:
            continue   # not graded yet; next pass
        names = dict(zip(g.player_id, g.player))
        for team, ids in w["confirmed_ids"].items():
            if (e, team) in done:
                continue
            actual = sorted(g[g.team == team].player_id); conf = sorted(ids)
            out = sorted(names.get(p, p) for p in set(conf) - set(actual)); inn = sorted(names.get(p, p) for p in set(actual) - set(conf))
            rows.append({"date": w["date"], "event_id": e, "team": team, "confirmed_ids": " ".join(conf), "actual_ids": " ".join(actual), "out": "; ".join(out), "in": "; ".join(inn), "audited_at": ts(now or fbm.now())})
            if out or inn:
                mism.append(f"{w['date']} {e} {team}: confirmed had {', '.join(out) or '-'}; ESPN started {', '.join(inn) or '-'} (late-scratch void governs)")
    if rows:
        new = not AUDIT.exists()
        with open(AUDIT, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            if new:
                wr.writeheader()
            wr.writerows(rows)
        print(f"[fb-lineups] audit: {len(rows)} confirmed team-fives checked against the graded starters, {len(mism)} mismatch" + ("" if len(mism) == 1 else "es"))
        for m in mism:
            print(f"[fb-lineups] audit mismatch (report-only): {m}")
    return mism


# ---------------------------------------------------------------- one pass
def watch(now=None, fetch_fn=nbacom_daily, odds_fn=pull_odds, xwalk_path=XWALK):
    now = now or fbm.now()
    FB = read_json(fbm.TODAY, None)
    if not FB or not FB.get("games"):
        print("[fb-lineups] no logged slate (docs/data/first_basket_today.json)"); return {"status": "no_slate"}
    date = FB["date"]
    try:
        audit_mism = audit(date, now)
    except Exception as e:  # noqa: BLE001
        audit_mism = []; print(f"[fb-lineups] audit failed (report-only, ignored): {e!r}")
    done = checks_done(date)
    due = [g for g in FB["games"] if g["event_id"] not in done and now < parse_ts(g["tip"]) <= now + WINDOW]
    if not due:
        nxt = sorted(parse_ts(g["tip"]) for g in FB["games"] if parse_ts(g["tip"]) > now)
        print(f"[fb-lineups] {date}: nothing due at {now:%H:%M}Z ({len(done)} games checked" + (f", next tip {nxt[0]:%H:%M}Z" if nxt else ", no tip ahead") + ")")
        return {"status": "idle", "date": date, "checked": len(done), "audit": audit_mism}
    if not fbm.LOG.exists():
        print("[fb-lineups] no first-basket log"); return {"status": "no_log"}
    log = pd.read_csv(fbm.LOG, dtype={"event_id": str, "player_id": str})
    snap = fbm.last_pretip(log, date)
    checks, refreshed, results, details, alarms = [], [], {}, {}, []
    stamp = ts(now); err = ""
    try:
        doc = fetch_fn(date)
    except Exception as ex:  # noqa: BLE001
        doc = False; err = str(ex)[:100]
    if doc is False or doc is None:
        why = f"NBA.com daily lineups unreachable: {err}" if doc is False else f"NBA.com has not published the {date} daily lineups file yet (404)"
        for g in due:
            checks.append({"date": date, "event_id": g["event_id"], "tip": g["tip"], "checked_at": stamp, "status": "pending", "lineup_status": "unknown", "note": why}); results[g["event_id"]] = "pending"
        print(f"[fb-lineups] {why}; {len(due)} due game(s) again next pass")
        log_checks(checks); publish_overlay(date, now)
        return {"status": "checked", "date": date, "due": [g["event_id"] for g in due], "results": results, "refreshed": [], "details": {}, "alarms": [], "audit": audit_mism}
    games = nbacom_games(doc); uni = Universe(); xw = Crosswalk(xwalk_path); ctx = None
    for g in due:
        e = g["event_id"]; base = {"date": date, "event_id": e, "tip": g["tip"], "checked_at": stamp}; label = f"{g['away']}@{g['home']}"
        ng = games.get((g["away"], g["home"]))
        if ng is None:
            listed = ", ".join(f"{a}@{h}" for a, h in games)
            checks.append({**base, "status": "pending", "lineup_status": "unknown", "note": f"no NBA.com game matches {label} on {date} (listed: {listed or 'none'})"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: no NBA.com game matches by team pair (listed: {listed or 'none'}); skipped, again next pass"); continue
        if not all(t["confirmed"] for t in ng["teams"].values()):
            state = "; ".join(f"{t}: {len(x['starters'])} with a position, status {'/'.join(x['statuses']) or '-'}" for t, x in ng["teams"].items())
            checks.append({**base, "status": "pending", "lineup_status": "projected", "note": f"NBA.com lists the game but the five are not stamped Confirmed ({state})"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: listed, not confirmed ({state}; tip {parse_ts(g['tip']) - now} away); again next pass"); continue
        try:
            conf = resolve_game(ng, uni, xw, now)
        except Unresolved as ex:
            note = f"confirmed on NBA.com but not repriced: {ex}"
            checks.append({**base, "status": "unresolved", "lineup_status": "confirmed", "note": note}); results[e] = "unresolved"
            alarms.append(f"{label} ({date}): {note}. The projected snapshot stands; add the pair to data/nbacom_player_ids.csv for next time.")
            print(f"[fb-lineups] {label}: UNRESOLVED, refresh skipped entirely: {ex}"); continue
        rows = snap[snap.event_id == e]
        if rows.empty:
            checks.append({**base, "status": "pending", "lineup_status": "confirmed", "note": "no pre-tip snapshot in the log for this game: nothing to compare"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: confirmed but no snapshot in the log; skipped"); continue
        if set(conf) != set(rows.team):
            checks.append({**base, "status": "pending", "lineup_status": "confirmed", "note": f"NBA.com teams {sorted(conf)} do not match the log's {sorted(set(rows.team))}"}); results[e] = "pending"
            print(f"[fb-lineups] {label}: team mismatch {sorted(conf)} vs {sorted(set(rows.team))}; skipped"); continue
        stamps = sorted({p["stamp"] for t in ng["teams"].values() for p in t["starters"] if p.get("stamp")})
        src = {"nbacom_game_id": ng["game_id"], "nbacom_stamp": stamps[-1] if stamps else None, "matched_by": {x["id"]: x["matched_by"] for t in conf for x in conf[t]}}
        diff = lineup_diff(rows, conf); before = rows.sort_values("p_first_basket", ascending=False).player.iloc[0]; proj_at = rows.logged_at.iloc[0]
        if not diff:
            checks.append({**base, "status": "identical", "lineup_status": "confirmed", "projected_logged_at": proj_at, "top_before": before, "top_after": before,
                           "note": f"{SOURCE}: confirmed ten = projected ten (stamp {src['nbacom_stamp']}); prices stand, no snapshot"}); results[e] = "identical"
            print(f"[fb-lineups] {label}: lineups confirmed (NBA.com stamp {src['nbacom_stamp']}), identical to the {proj_at} snapshot; no snapshot, no change"); continue
        ctx = ctx or Ctx()
        new_rows, S_, G = reprice(ctx, {**g, "date": date}, conf, now, diff, src)
        after = max(new_rows, key=lambda r: r["p_first_basket"])["player"]
        for team, d in diff.items():
            checks.append({**base, "status": "changed", "lineup_status": "confirmed", "team": team, "out": "; ".join(d["out"]), "in": "; ".join(d["in"]), "projected_logged_at": proj_at,
                           "refreshed_logged_at": stamp, "top_before": before, "top_after": after,
                           "note": f"{SOURCE}: repriced from the confirmed ten (stamp {src['nbacom_stamp']}); new snapshot appended"})
        refreshed.append(e); results[e] = "changed"
        details[e] = {"matchup": label, "diff": diff, "diff_text": diff_text(diff), "top_before": before, "top_after": after, "logged_at": stamp,
                      "top_p_after": max(r["p_first_basket"] for r in new_rows)}
        print(f"[fb-lineups] {label}: lineups confirmed (NBA.com stamp {src['nbacom_stamp']}) with changes ({diff_text(diff)}); repriced, top pick {before} -> {after}, snapshot {stamp} appended")
    if checks:
        log_checks(checks)
    if refreshed:
        regenerate(ctx, date, refreshed, odds_fn)
    publish_overlay(date, now)
    return {"status": "checked", "date": date, "due": [g["event_id"] for g in due], "results": results, "refreshed": refreshed, "details": details, "alarms": alarms, "audit": audit_mism}


# ---------------------------------------------------------------- seed: every player NBA.com lists on the given dates -> crosswalk
def seed(dates, fetch_fn=nbacom_daily, xwalk_path=XWALK, now=None):
    """Resolve every player NBA.com lists (starters first, then the bench) for the dates (YYYYMMDD or YYYY-MM-DD) into the crosswalk.
    Prints and returns the tally; nothing is guessed: an unresolved or ambiguous player is reported, not written."""
    uni = Universe(); xw = Crosswalk(xwalk_path)
    tally = {"dates": {}, "starters_total": 0, "starters_resolved": 0, "bench_total": 0, "bench_resolved": 0, "unresolved": [], "added": 0}
    for d in dates:
        d = f"{d[:4]}-{d[4:6]}-{d[6:]}" if "-" not in d else d
        doc = fetch_fn(d)
        if not doc:
            print(f"[fb-lineups] seed {d}: no NBA.com file"); continue
        n0 = len(xw.map); games = nbacom_games(doc); ds = {"games": len(games), "starters": 0, "starters_resolved": 0, "bench": 0, "bench_resolved": 0}
        for key, ng in games.items():
            for team in key:
                t = ng["teams"][team]
                for kind, lst in (("starters", t["starters"]), ("bench", t["bench"])):
                    for p in lst:
                        ds[kind] += 1
                        try:
                            xw.resolve(p, team, uni, now); ds[f"{kind}_resolved"] += 1
                        except Unresolved as ex:
                            tally["unresolved"].append(f"{d} {kind}: {ex}")
        tally["dates"][d] = ds; tally["added"] += len(xw.map) - n0
        for k in ("starters", "bench"):
            tally[f"{k}_total"] += ds[k]; tally[f"{k}_resolved"] += ds[f"{k}_resolved"]
        print(f"[fb-lineups] seed {d}: {ds['games']} games, starters {ds['starters_resolved']}/{ds['starters']} resolved, bench {ds['bench_resolved']}/{ds['bench']}")
    print(f"[fb-lineups] seed: crosswalk now {len(xw.map)} ids (+{tally['added']}); starters {tally['starters_resolved']}/{tally['starters_total']}, "
          f"bench {tally['bench_resolved']}/{tally['bench_total']}; unresolved {len(tally['unresolved'])}")
    for u in tally["unresolved"]:
        print(f"[fb-lineups]   unresolved: {u}")
    return tally


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "watch"
    if cmd == "overlay":
        FB = read_json(fbm.TODAY, None) or {}
        publish_overlay(FB.get("date") or dt.date.today().isoformat())
    elif cmd == "seed":
        seed(sys.argv[2:])
    elif cmd == "audit":
        FB = read_json(fbm.TODAY, None) or {}
        audit(FB.get("date") or dt.date.today().isoformat())
    else:
        watch()
