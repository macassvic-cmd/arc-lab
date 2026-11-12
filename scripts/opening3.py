"""Opening 3 (2026-11-06): the first three minutes of every game plus the game's firsts, from the play-by-play store
(scripts/pbp_store.py, ESPN summary plays compacted to data/pbp_raw/<event_id>.json.gz). Display layer only: no model, grading,
FP v2, ladder or first-basket file is read or written.

Runs in the morning grade pass right after the play-by-play store step, which has just cached the last three days' games. Every
cached game not yet collected is extracted once and appended (append-only by event id):

  data/opening3_games.csv     one row per game: jump ball, first FGM (+ assister), first assist, first rebound, first 3 (per team
                              and for the game), first dunk (per team and for the game), each with player / team / clock, and the
                              two teams' points inside the window
  data/opening3_players.csv   one row per player-game (everyone who played): window PTS / AST / OREB / DREB / REB / FGM / FGA /
                              3PM / 3PA / dunks / FTM, starter flag from the box score, and a flag for each first he recorded
  docs/data/opening3.json     leaderboards (top 12 by count, with starts and count/starts), per-season player averages (>= 5 games),
                              and the last 60 games' firsts

Classification uses STRUCTURED play fields only, never the play text (the 2026-10-31 lesson):
  window      period 1 and clock >= 9:00 remaining (elapsed <= 180 s; exactly 9:00 counts)
  FG attempt  shooting play with pointsAttempted 2 or 3; made = scoringPlay; a made 3 = pointsAttempted 3 (cross-checked scoreValue 3)
  free throw  shooting play with pointsAttempted 1
  dunk        made shooting play whose type text contains "Dunk"
  assist      the second participant of a made FG
  rebound     type id 155 (defensive) / 156 (offensive); no participant = team rebound, nobody credited
  jump ball   type id 615: participants [jumper, jumper, possession gainer]; the winner is the jumper on the gainer's team
Players are the box-score roster (id, name, team, starter, did-not-play); a participant missing from it keeps his id as his name.
Seasons are ESPN season years: 2025 = 2024-25, 2026 = 2025-26, 2027 = 2026-27.

  python opening3.py                 collect every cached game not yet in the store, then publish the JSON
  python opening3.py publish         rebuild docs/data/opening3.json from the store only
  python opening3.py check EVENT_ID  print one game's extraction (hand-check; caches the game if needed)
"""
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, write_json
import pbp_store as ps

GAMES = DATA_DIR / "opening3_games.csv"
PLAYERS = DATA_DIR / "opening3_players.csv"
OUT = SITE_DATA / "opening3.json"
WINDOW_SECS = 540.0          # clock remaining at the window's end: 12:00 - 3:00
RECENT = 60
TOP = 12
MIN_GAMES = 5
SEASON_MIN = 2025            # the three seasons in scope: 2024-25 (ESPN 2025), 2025-26, 2026-27; earlier box-score history is ignored
DREB, OREB, JUMP = "155", "156", "615"
FIRSTS = ("first_fgm", "first_ast", "first_reb", "first_3_game", "first_dunk_game")

GAME_COLS = ["event_id", "date", "season", "phase", "home", "away",
             "jump_winner", "jump_winner_team", "jump_gainer", "jump_gainer_team",
             "first_fgm_player", "first_fgm_team", "first_fgm_clock", "first_fgm_assist",
             "first_ast_player", "first_ast_team", "first_ast_clock",
             "first_reb_player", "first_reb_team", "first_reb_clock", "first_reb_kind",
             "first_3_game_player", "first_3_game_team", "first_3_game_clock", "first_3_home_player", "first_3_home_clock", "first_3_away_player", "first_3_away_clock",
             "first_dunk_game_player", "first_dunk_game_team", "first_dunk_game_clock", "first_dunk_home_player", "first_dunk_home_clock", "first_dunk_away_player", "first_dunk_away_clock",
             "win_pts_home", "win_pts_away", "window_plays", "collected_at"]
# Whole-game columns (2026-11-11 store extension). Every name carries "wg" = WHOLE GAME; the columns above are WINDOW-limited
# (first three minutes only) and are never altered. Clocks are period-aware: <name>_period + <name>_elapsed_s = game seconds
# elapsed since the opening tip (12-minute quarters, 5-minute overtimes), never a bare quarter clock.
WG_GAME_COLS = ["wg_first_point_team", "wg_first_point_player", "wg_first_point_is_ft", "wg_first_point_period", "wg_first_point_elapsed_s",
                "wg_first_3_game_team", "wg_first_3_game_player", "wg_first_3_game_period", "wg_first_3_game_elapsed_s",
                "wg_first_dunk_game_team", "wg_first_dunk_game_player", "wg_first_dunk_game_period", "wg_first_dunk_game_elapsed_s"] + \
               [f"wg_first_{k}_{side}_{f}" for k in ("3", "dunk", "fg") for side in ("home", "away") for f in ("player", "period", "elapsed_s")] + ["wg_plays", "wg_collected_at"]
WG_PLAYER_COLS = ["wg_f_point", "wg_f_3_game", "wg_f_3_team", "wg_f_dunk_game", "wg_f_dunk_team", "wg_f_fg_team"]
BASE_GAME_COLS = list(GAME_COLS)
GAME_COLS = GAME_COLS + WG_GAME_COLS
PLAYER_COLS = ["event_id", "date", "season", "player_id", "player", "team", "opp", "home", "starter",
               "win_pts", "win_ast", "win_oreb", "win_dreb", "win_reb", "win_fgm", "win_fga", "win_3pm", "win_3pa", "win_dunks", "win_ftm",
               "f_fgm", "f_ast", "f_reb", "f_3_game", "f_3_team", "f_dunk_game", "f_dunk_team", "jump_win", "jump_gain"]
BASE_PLAYER_COLS = list(PLAYER_COLS)
PLAYER_COLS = PLAYER_COLS + WG_PLAYER_COLS


def elapsed_s(period, secs):
    """Game seconds elapsed since the opening tip for a play at `secs` remaining in `period` (quarters 720 s, overtimes 300 s)."""
    period = int(period)
    return (period - 1) * 720.0 + (720.0 - secs) if period <= 4 else 2880.0 + (period - 5) * 300.0 + (300.0 - secs)


def extract_wg(doc, who, side_of, tid_abbr):
    """Whole-game firsts over EVERY play in sequence order: (game columns, {athlete id: {flag: 1}}). Made shots only
    (scoringPlay); a dunk = a made shooting play whose type text contains "Dunk"; first point = the first made shot of any
    kind, free throws included."""
    def name(pid): return who[pid]["name"] if pid in who else str(pid)
    def team_of(pid, play_team): return who[pid]["team"] if pid in who else tid_abbr.get(str(play_team), str(play_team))
    plays = [p for p in sorted(doc["plays"], key=lambda p: p["seq"]) if p.get("period") is not None and p.get("secs") is not None]
    g = {c: None for c in WG_GAME_COLS}; g["wg_plays"] = len(plays); g["wg_collected_at"] = now_iso()
    fl = defaultdict(dict)
    def put(key, p, shooter, with_team):
        if g[key + "_player"] is not None:
            return False
        g[key + "_player"] = name(shooter); g[key + "_period"] = int(p["period"]); g[key + "_elapsed_s"] = round(elapsed_s(p["period"], p["secs"]), 1)
        if with_team:
            g[key + "_team"] = team_of(shooter, p.get("team"))
        return True
    for p in plays:
        ath = [str(a) for a in (p.get("ath") or []) if a]
        if not p.get("shooting") or not ath or not p.get("scoring"):
            continue
        att = int(p.get("att") or 0); val = int(p.get("value") or 0); shooter = ath[0]
        side = side_of.get(str(p.get("team")) if p.get("team") is not None else None)
        if att not in (1, 2, 3):
            continue
        if put("wg_first_point", p, shooter, True):
            g["wg_first_point_is_ft"] = int(att == 1); fl[shooter]["wg_f_point"] = 1
        if att == 1:
            continue
        if side and put(f"wg_first_fg_{side}", p, shooter, False):
            fl[shooter]["wg_f_fg_team"] = 1
        if att == 3 and val == 3:
            if put("wg_first_3_game", p, shooter, True): fl[shooter]["wg_f_3_game"] = 1
            if side and put(f"wg_first_3_{side}", p, shooter, False): fl[shooter]["wg_f_3_team"] = 1
        if "dunk" in (p.get("type") or "").lower():
            if put("wg_first_dunk_game", p, shooter, True): fl[shooter]["wg_f_dunk_game"] = 1
            if side and put(f"wg_first_dunk_{side}", p, shooter, False): fl[shooter]["wg_f_dunk_team"] = 1
    return g, fl


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- extraction

def extract(doc):
    """(game row, [player rows]) for one cached game, or (None, reason) when it cannot be extracted."""
    if not doc or doc.get("status") != "ok" or not doc.get("plays"):
        return None, "no_plays"
    teams = doc["meta"]["teams"]
    if not teams.get("home") or not teams.get("away"):
        return None, "no_teams"
    tid_abbr = {str(teams[s]["id"]): teams[s]["abbr"] for s in ("home", "away")}
    side_of = {str(teams["home"]["id"]): "home", str(teams["away"]["id"]): "away"}
    who = {}                                   # athlete id -> {name, team abbr, starter, played}
    for tid, roster in (doc.get("roster") or {}).items():
        for a in roster:
            if a.get("id"):
                who[str(a["id"])] = {"name": a.get("name") or str(a["id"]), "team": tid_abbr.get(str(tid), str(tid)), "starter": int(a.get("starter") or 0),
                                     "played": not a.get("dnp")}
    def name(pid): return who[pid]["name"] if pid in who else str(pid)
    def team_of(pid, play_team): return who[pid]["team"] if pid in who else tid_abbr.get(str(play_team), str(play_team))

    window = [p for p in sorted(doc["plays"], key=lambda p: p["seq"]) if p.get("period") == 1 and p.get("secs") is not None and p["secs"] >= WINDOW_SECS]
    g = {c: None for c in GAME_COLS}
    g.update({"event_id": str(doc["event_id"]), "date": doc["date"], "season": int(doc["season"]), "phase": doc.get("phase") or "regular",
              "home": teams["home"]["abbr"], "away": teams["away"]["abbr"], "win_pts_home": 0, "win_pts_away": 0, "window_plays": len(window), "collected_at": now_iso()})
    st = defaultdict(lambda: defaultdict(int))  # athlete id -> window tallies / first flags
    seen_3_team, seen_dunk_team = set(), set()
    for p in window:
        ath = [str(a) for a in (p.get("ath") or []) if a]
        tid = str(p.get("team")) if p.get("team") is not None else None
        side = side_of.get(tid)
        clock = p.get("clock")
        typ = p.get("type") or ""
        if p.get("type_id") == JUMP and g["jump_gainer"] is None and len(ath) >= 3:
            gainer = ath[2]; jumpers = ath[:2]
            g["jump_gainer"], g["jump_gainer_team"] = name(gainer), team_of(gainer, tid); st[gainer]["jump_gain"] = 1
            winner = next((j for j in jumpers if j in who and who[j]["team"] == g["jump_gainer_team"]), None)
            if winner:
                g["jump_winner"], g["jump_winner_team"] = name(winner), who[winner]["team"]; st[winner]["jump_win"] = 1
            continue
        if p.get("type_id") in (DREB, OREB):
            if ath:                            # a player rebound; an empty participant list is a team rebound, nobody credited
                pid = ath[0]; kind = "oreb" if p["type_id"] == OREB else "dreb"
                st[pid]["win_" + kind] += 1; st[pid]["win_reb"] += 1
                if g["first_reb_player"] is None:
                    g["first_reb_player"], g["first_reb_team"], g["first_reb_clock"], g["first_reb_kind"] = name(pid), team_of(pid, tid), clock, kind; st[pid]["f_reb"] = 1
            continue
        if not p.get("shooting") or not ath:
            continue
        att = int(p.get("att") or 0)
        made = bool(p.get("scoring"))
        val = int(p.get("value") or 0)
        shooter = ath[0]
        if att == 1:                           # free throw
            if made:
                st[shooter]["win_ftm"] += 1; st[shooter]["win_pts"] += val
                if side: g["win_pts_" + side] += val
            continue
        if att not in (2, 3):
            continue
        st[shooter]["win_fga"] += 1
        if att == 3:
            st[shooter]["win_3pa"] += 1
        if not made:
            continue
        st[shooter]["win_fgm"] += 1; st[shooter]["win_pts"] += val
        if side: g["win_pts_" + side] += val
        shooter_team = team_of(shooter, tid)
        assister = ath[1] if len(ath) > 1 else None
        if assister:
            st[assister]["win_ast"] += 1
            if g["first_ast_player"] is None:
                g["first_ast_player"], g["first_ast_team"], g["first_ast_clock"] = name(assister), team_of(assister, tid), clock; st[assister]["f_ast"] = 1
        if g["first_fgm_player"] is None:
            g["first_fgm_player"], g["first_fgm_team"], g["first_fgm_clock"] = name(shooter), shooter_team, clock
            g["first_fgm_assist"] = name(assister) if assister else None; st[shooter]["f_fgm"] = 1
        if att == 3 and val == 3:
            st[shooter]["win_3pm"] += 1
            if g["first_3_game_player"] is None:
                g["first_3_game_player"], g["first_3_game_team"], g["first_3_game_clock"] = name(shooter), shooter_team, clock; st[shooter]["f_3_game"] = 1
            if side and side not in seen_3_team:
                seen_3_team.add(side); g[f"first_3_{side}_player"], g[f"first_3_{side}_clock"] = name(shooter), clock; st[shooter]["f_3_team"] = 1
        if "dunk" in typ.lower():
            st[shooter]["win_dunks"] += 1
            if g["first_dunk_game_player"] is None:
                g["first_dunk_game_player"], g["first_dunk_game_team"], g["first_dunk_game_clock"] = name(shooter), shooter_team, clock; st[shooter]["f_dunk_game"] = 1
            if side and side not in seen_dunk_team:
                seen_dunk_team.add(side); g[f"first_dunk_{side}_player"], g[f"first_dunk_{side}_clock"] = name(shooter), clock; st[shooter]["f_dunk_team"] = 1
    wg, wg_flags = extract_wg(doc, who, side_of, tid_abbr)
    g.update(wg)
    for pid, f in wg_flags.items():
        st[pid].update(f)
    rows = []
    ids = [pid for pid, a in who.items() if a["played"]] + [pid for pid in st if pid not in who]
    for pid in ids:
        a = who.get(pid, {"name": str(pid), "team": None, "starter": 0, "played": True})
        home = 1 if a["team"] == g["home"] else 0
        opp = g["away"] if home else g["home"]
        r = {"event_id": g["event_id"], "date": g["date"], "season": g["season"], "player_id": pid, "player": a["name"], "team": a["team"], "opp": opp, "home": home, "starter": a["starter"]}
        for c in PLAYER_COLS[9:]:
            r[c] = int(st[pid][c]) if pid in st else 0
        rows.append(r)
    return g, rows


# ---------------------------------------------------------------- store

def read_csv(path, cols):
    if path.exists():
        return pd.read_csv(path, dtype={"event_id": str, "player_id": str})
    return pd.DataFrame(columns=cols)


def append_rows(path, cols, rows):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def candidate_events():
    """(event_id, date, season, phase) of every known finished game from SEASON_MIN on: box-score games plus the enumerated postseason games."""
    ev = ps.events(); ev["phase"] = "regular"
    if ps.EXTRA.exists():
        x = pd.read_csv(ps.EXTRA, dtype={"event_id": str})
        ev = pd.concat([ev, x[x.phase == "postseason"][["event_id", "date", "season", "phase"]]])
    ev = ev[ev.season.astype(int) >= SEASON_MIN]
    return ev.drop_duplicates("event_id").sort_values(["date", "event_id"]).reset_index(drop=True)


def collect(fetch_missing=False, sleep=0.0):
    """Extract every candidate game that is cached (or fetch it when fetch_missing) and not yet in the store. Append-only."""
    done = set(read_csv(GAMES, GAME_COLS).event_id.astype(str))
    ev = candidate_events()
    todo = ev[~ev.event_id.astype(str).isin(done)]
    added, skipped = [], defaultdict(int)
    for r in todo.itertuples():
        doc = ps.load(r.event_id)
        if doc is None and fetch_missing:
            ps.store(r.event_id, r.date, r.season, r.phase); doc = ps.load(r.event_id)
            if sleep: import time; time.sleep(sleep)
        if doc is None:
            skipped["not_cached"] += 1; continue
        g, rows = extract(doc)
        if g is None:
            skipped[rows] += 1; continue
        append_rows(GAMES, GAME_COLS, [g]); append_rows(PLAYERS, PLAYER_COLS, rows); added.append(g["event_id"])
    print(f"[opening3] {len(done)} games already collected; {len(todo)} candidates; added {len(added)}; skipped {dict(skipped)}")
    return added


def backfill_wg(fetch_missing=True):
    """Idempotent backfill of the whole-game columns for every game already in the store, from the play-by-play cache.
    Existing columns are carried over value for value (verified after writing); only the wg columns are computed."""
    import time
    old_g = pd.read_csv(GAMES, dtype=str, keep_default_na=False); old_p = pd.read_csv(PLAYERS, dtype=str, keep_default_na=False)
    ev = candidate_events().set_index("event_id")
    grow, prow, problems = {}, {}, defaultdict(list)
    for e in old_g.event_id:
        doc = ps.load(e)
        if doc is None and fetch_missing and e in ev.index:
            r = ev.loc[e]; ps.store(e, r.date, r.season, r.phase); doc = ps.load(e); time.sleep(0.12)
        g, rows = extract(doc) if doc is not None else (None, "not_cached")
        if g is None:
            problems[rows].append(e); continue
        grow[e] = {c: g[c] for c in WG_GAME_COLS}
        for r in rows:
            prow[(e, str(r["player_id"]))] = {c: r[c] for c in WG_PLAYER_COLS}
    fmt = lambda v: "" if v is None else str(v)
    new_g = old_g[BASE_GAME_COLS].copy()
    for c in WG_GAME_COLS:
        new_g[c] = [fmt(grow.get(e, {}).get(c)) for e in old_g.event_id]
    new_p = old_p[BASE_PLAYER_COLS].copy()
    keys = list(zip(old_p.event_id, old_p.player_id)); keyset = set(keys)
    no_row = [k for k in prow if any(prow[k].values()) and k not in keyset]
    for c in WG_PLAYER_COLS:
        new_p[c] = [fmt(prow[k][c]) if k in prow else ("" if k[0] not in grow else "0") for k in keys]
    new_g.to_csv(GAMES, index=False, lineterminator="\n"); new_p.to_csv(PLAYERS, index=False, lineterminator="\n")
    chk_g = pd.read_csv(GAMES, dtype=str, keep_default_na=False); chk_p = pd.read_csv(PLAYERS, dtype=str, keep_default_na=False)
    assert chk_g[BASE_GAME_COLS].equals(old_g[BASE_GAME_COLS]) and chk_p[BASE_PLAYER_COLS].equals(old_p[BASE_PLAYER_COLS]), "existing columns changed"
    print(f"[opening3] whole-game backfill: {len(grow)} of {len(old_g)} games; not extracted {dict(problems)}; "
          f"flagged players with no store row: {len(no_row)} {no_row[:10]}; existing columns verified unchanged")


# ---------------------------------------------------------------- publish

def publish(out=OUT):
    games = read_csv(GAMES, GAME_COLS)
    players = read_csv(PLAYERS, PLAYER_COLS)
    if games.empty:
        write_json(out, {"games": 0, "leaderboards": {}, "players": {}, "recent_firsts": []}); return
    games = games.sort_values(["date", "event_id"])
    for c in ("season",):
        games[c] = games[c].astype(int); players[c] = players[c].astype(int)
    num = [c for c in PLAYER_COLS[9:]]
    players[num] = players[num].fillna(0).astype(int)
    players["starter"] = players["starter"].fillna(0).astype(int)
    leaderboards, table = {}, {}
    for season, ps_ in players.groupby("season"):
        key = str(season)
        last_team = ps_.sort_values(["date", "event_id"]).groupby("player_id")["team"].last()
        names = ps_.sort_values(["date", "event_id"]).groupby("player_id")["player"].last()
        agg = ps_.groupby("player_id").agg(games=("event_id", "count"), starts=("starter", "sum"), win_pts=("win_pts", "mean"), win_ast=("win_ast", "mean"), win_reb=("win_reb", "mean"),
                                           f_fgm=("f_fgm", "sum"), f_ast=("f_ast", "sum"), f_reb=("f_reb", "sum"), f_3_game=("f_3_game", "sum"), f_dunk_game=("f_dunk_game", "sum"))
        lb = {}
        for k in FIRSTS:
            col = "f_" + k[len("first_"):]
            top = agg[agg[col] > 0].copy()
            top["rate"] = top.apply(lambda r: (r[col] / r["starts"]) if r["starts"] > 0 else None, axis=1)
            top = top.sort_values([col, "rate"], ascending=[False, False], na_position="last").head(TOP)
            lb[k] = [{"player": names[pid], "team": last_team[pid], "count": int(r[col]), "starts": int(r["starts"]), "rate": (None if r["rate"] is None or pd.isna(r["rate"]) else round(float(r["rate"]), 4))}
                     for pid, r in top.iterrows()]
        leaderboards[key] = lb
        ok = agg[agg.games >= MIN_GAMES].sort_values(["win_pts", "win_ast"], ascending=False)
        table[key] = [{"player": names[pid], "team": last_team[pid], "games": int(r.games), "starts": int(r.starts), "win_pts": round(float(r.win_pts), 2), "win_ast": round(float(r.win_ast), 2), "win_reb": round(float(r.win_reb), 2)}
                      for pid, r in ok.iterrows()]
    recent = []
    for r in games.tail(RECENT).iloc[::-1].itertuples():
        def v(x): return None if (x is None or (isinstance(x, float) and pd.isna(x))) else x
        recent.append({"date": r.date, "season": int(r.season), "home": r.home, "away": r.away,
                       "first_fgm_player": v(r.first_fgm_player), "first_fgm_clock": v(r.first_fgm_clock), "first_ast_player": v(r.first_ast_player), "first_reb_player": v(r.first_reb_player),
                       "first_3_game_player": v(r.first_3_game_player), "first_3_game_clock": v(r.first_3_game_clock), "first_dunk_game_player": v(r.first_dunk_game_player), "first_dunk_game_clock": v(r.first_dunk_game_clock),
                       "win_pts_home": int(r.win_pts_home), "win_pts_away": int(r.win_pts_away)})
    doc = {"games": int(len(games)), "leaderboards": leaderboards, "players": table, "recent_firsts": recent}
    write_json(out, doc)
    print(f"[opening3] published {out.name}: {len(games)} games, seasons {sorted(leaderboards)}, {len(recent)} recent")
    return doc


def check(event_id):
    ev = candidate_events(); row = ev[ev.event_id.astype(str) == str(event_id)]
    if ps.load(event_id) is None and not row.empty:
        r = row.iloc[0]; ps.store(r.event_id, r.date, r.season, r.phase)
    g, rows = extract(ps.load(event_id))
    if g is None:
        print("cannot extract:", rows); return
    for k in GAME_COLS:
        if g[k] not in (None, ""): print(f"{k:24s} {g[k]}")
    cols = [c for c in PLAYER_COLS[9:]]
    print(pd.DataFrame(rows).query("win_pts>0 or win_reb>0 or win_ast>0 or win_fga>0 or jump_win>0 or jump_gain>0")[["player", "team", "starter"] + cols].to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "check":
        check(a[1])
    elif a and a[0] == "publish":
        publish()
    elif a and a[0] == "backfill-wg":
        backfill_wg()
    else:
        collect(fetch_missing="--fetch" in a, sleep=0.12)
        publish()
