"""First-shot labels (backtests/first_basket_v2_rule.md, section 1). Labels first, model second.

  python first_shot.py backfill      label every stored game (data/pbp_raw) not yet labelled: regular season 2024-25 onward from the
                                     box-score files, postseason / preseason games from data/pbp_extra_events.csv (pbp_store.py --phase)

Outputs (append-only, one pass per game):
  data/first_shot_history.csv     one row per regular-season / postseason game: each team's first field-goal attempt, the game's
                                  first attempt, the first made field goal (the graded target), jump-ball jumpers and possession
  data/first_shot_preseason.csv   the same columns for preseason games (separate table, small sample, current starters)
  data/first_shot_sequence.csv    one row per field-goal attempt from the tip up to and including the first made field goal
  data/first_shot_starters.csv    the ten starters of every labelled game (team, position), so the model needs no raw play store
  docs/data/first_shot_coverage.json   coverage per season and phase, reported before the model runs
Shot type: `3` when ESPN pointsAttempted is 3; else `rim` when the play type names a layup, dunk, tip, putback, finger roll,
alley oop or hook; else `mid`. Free throws are not attempts.
"""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_pos, write_json
from pbp_store import EXTRA, RAW, events, load

HIST = DATA_DIR / "first_shot_history.csv"
PRE = DATA_DIR / "first_shot_preseason.csv"
SEQ = DATA_DIR / "first_shot_sequence.csv"
STARTERS = DATA_DIR / "first_shot_starters.csv"
COVERAGE = SITE_DATA / "first_shot_coverage.json"
FIRST_SEASON = 2025
RIM_WORDS = ("layup", "dunk", "tip", "putback", "finger roll", "alley oop", "hook")
GAME_FIELDS = ["event_id", "date", "season", "phase", "home", "away",
               "jump_home_id", "jump_away_id", "jump_gainer_id", "jump_gainer_team", "jump_text",
               "home_fga_id", "home_fga_player", "home_fga_type", "home_fga_made", "home_fga_secs",
               "away_fga_id", "away_fga_player", "away_fga_type", "away_fga_made", "away_fga_secs",
               "first_fga_team", "first_fga_id", "first_fga_type", "first_fga_made", "first_fga_secs",
               "first_fg_team", "first_fg_id", "first_fg_player", "first_fg_pos", "first_fg_started", "first_fg_type", "first_fg_secs",
               "attempts_to_first_fg", "label_clean", "reason"]
SEQ_FIELDS = ["event_id", "date", "season", "phase", "idx", "team", "opp", "home", "player_id", "player", "pos", "started", "shot_type", "made", "secs_elapsed", "same_team_as_prev"]
STARTER_FIELDS = ["event_id", "date", "season", "phase", "team", "opp", "home", "player_id", "player", "pos"]


def shot_type(p):
    if (p.get("att") or 0) == 3:
        return "3"
    t = (p.get("type") or "").lower()
    return "rim" if any(w in t for w in RIM_WORDS) else "mid"


def is_fga(p):
    return bool(p.get("shooting")) and "free throw" not in (p.get("type") or "").lower() and (p.get("att") or 0) >= 2


def elapsed(p):
    return None if p.get("secs") is None else round(720 - p["secs"], 1)


def game_box(doc, box):
    """Rows (player_id, player, pos, starter, team abbr) for this game: the box-score file when present, else the stored roster."""
    if box is not None and len(box):
        return box
    teams = doc["meta"]["teams"]; abbr = {t["id"]: t["abbr"] for t in teams.values()}
    rows = [{"player_id": str(a["id"]), "player": a["name"], "pos": norm_pos(a.get("pos")), "starter": int(a.get("starter") or 0), "team": abbr.get(tid)}
            for tid, lst in (doc.get("roster") or {}).items() for a in lst if not a.get("dnp")]
    return pd.DataFrame(rows, columns=["player_id", "player", "pos", "starter", "team"])


def label(doc, box):
    """Returns (game_row, sequence_rows, starter_rows)."""
    plays = sorted(doc.get("plays", []), key=lambda p: p["seq"])
    teams = doc["meta"]["teams"]; abbr = {t["id"]: t["abbr"] for t in teams.values()}
    home, away = teams.get("home", {}).get("abbr"), teams.get("away", {}).get("abbr")
    g = {"event_id": doc["event_id"], "date": doc["date"], "season": doc["season"], "phase": doc.get("phase", "regular"), "home": home, "away": away}
    bx = game_box(doc, box)
    by_id = {str(r.player_id): r for r in bx.itertuples()}
    starters = [{"event_id": g["event_id"], "date": g["date"], "season": g["season"], "phase": g["phase"], "team": r.team, "opp": away if r.team == home else home,
                 "home": int(r.team == home), "player_id": str(r.player_id), "player": r.player, "pos": r.pos} for r in bx[bx.starter == 1].itertuples()]
    if not plays:
        return {**g, "label_clean": 0, "reason": "no_plays"}, [], starters
    p1 = [p for p in plays if p["period"] == 1]
    jump = next((p for p in p1 if (p["type"] or "").lower().startswith("jump")), None)
    if jump:
        ids = [str(a) for a in jump.get("ath", []) if a]
        jumpers = ids[:2]
        g["jump_gainer_id"] = ids[2] if len(ids) > 2 else None
        g["jump_gainer_team"] = abbr.get(jump.get("team")); g["jump_text"] = jump.get("text")
        for j in jumpers:
            r = by_id.get(j)
            if r is not None:
                g["jump_home_id" if r.team == home else "jump_away_id"] = j
    seq, team_first, seq_done = [], {}, False
    for p in p1:   # the sequence stops at the first made field goal; each team's first attempt is still found if it comes later
        if not is_fga(p):
            continue
        pid = str(p["ath"][0]) if p.get("ath") and p["ath"][0] else None
        team = abbr.get(p.get("team")); r = by_id.get(pid)
        made = int(bool(p.get("scoring")) and (p.get("value") or 0) >= 2)
        row = {"event_id": g["event_id"], "date": g["date"], "season": g["season"], "phase": g["phase"], "idx": len(seq) + 1, "team": team, "opp": away if team == home else home,
               "home": int(team == home), "player_id": pid, "player": r.player if r is not None else None, "pos": r.pos if r is not None else None,
               "started": int(r.starter) if r is not None else None, "shot_type": shot_type(p), "made": made, "secs_elapsed": elapsed(p),
               "same_team_as_prev": int(seq[-1]["team"] == team) if seq else None}
        if not seq_done:
            seq.append(row)
            if made:
                seq_done = True
        if team in (home, away) and team not in team_first:
            team_first[team] = row
        if seq_done and len(team_first) == 2:
            break
    for side, team in (("home", home), ("away", away)):
        f = team_first.get(team)
        if f:
            g.update({f"{side}_fga_id": f["player_id"], f"{side}_fga_player": f["player"], f"{side}_fga_type": f["shot_type"], f"{side}_fga_made": f["made"], f"{side}_fga_secs": f["secs_elapsed"]})
    if seq:
        f = seq[0]
        g.update({"first_fga_team": f["team"], "first_fga_id": f["player_id"], "first_fga_type": f["shot_type"], "first_fga_made": f["made"], "first_fga_secs": f["secs_elapsed"]})
    last = seq[-1] if seq else None
    if last is None or not last["made"]:
        return {**g, "label_clean": 0, "reason": "no_made_field_goal_in_period_1"}, seq, starters
    g.update({"first_fg_team": last["team"], "first_fg_id": last["player_id"], "first_fg_type": last["shot_type"], "first_fg_secs": last["secs_elapsed"], "attempts_to_first_fg": last["idx"]})
    if last["player_id"] is None:
        return {**g, "label_clean": 0, "reason": "scoring_play_without_participant"}, seq, starters
    if last["player"] is None:
        return {**g, "label_clean": 0, "reason": "scorer_not_in_box_score"}, seq, starters
    g.update({"first_fg_player": last["player"], "first_fg_pos": last["pos"], "first_fg_started": last["started"], "label_clean": 1, "reason": "ok"})
    return g, seq, starters


def all_events():
    ev = events()
    ev = ev[ev.season >= FIRST_SEASON].copy(); ev["phase"] = "regular"
    if EXTRA.exists():
        x = pd.read_csv(EXTRA, dtype={"event_id": str})
        ev = pd.concat([ev[["event_id", "date", "season", "phase"]], x[["event_id", "date", "season", "phase"]]]).drop_duplicates("event_id")
    return ev.sort_values(["date", "event_id"]).reset_index(drop=True)


def backfill():
    ev = all_events()
    done = set()
    for p in (HIST, PRE):
        if p.exists():
            done |= set(pd.read_csv(p, dtype={"event_id": str}, usecols=["event_id"]).event_id)
    frames = [pd.read_csv(DATA_DIR / fn, dtype={"event_id": str, "player_id": str}, usecols=["event_id", "player_id", "player", "team", "pos", "starter"])
              for fn in ("player_games.csv", "player_games_history.csv") if (DATA_DIR / fn).exists()]
    box = pd.concat(frames).drop_duplicates(["event_id", "player_id"]) if frames else pd.DataFrame(columns=["event_id", "player_id", "player", "team", "pos", "starter"])
    by_ev = {e: g for e, g in box.groupby("event_id")}
    files = {}
    for path, fields in ((HIST, GAME_FIELDS), (PRE, GAME_FIELDS), (SEQ, SEQ_FIELDS), (STARTERS, STARTER_FIELDS)):
        new = not path.exists()
        f = open(path, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        files[path] = (f, w)
    added = 0
    try:
        for r in ev.itertuples():
            if r.event_id in done:
                continue
            doc = load(r.event_id)
            if doc is None:
                continue   # not stored yet: labelled on a later run
            doc.setdefault("phase", r.phase)
            g, seq, starters = label(doc, by_ev.get(r.event_id))
            files[PRE if r.phase == "preseason" else HIST][1].writerow({k: g.get(k) for k in GAME_FIELDS})
            files[SEQ][1].writerows(seq); files[STARTERS][1].writerows(starters); added += 1
    finally:
        for f, _ in files.values():
            f.close()
    print(f"[first-shot] labelled {added} new games")
    coverage(ev)


def coverage(ev):
    h = pd.concat([pd.read_csv(p, dtype={"event_id": str}) for p in (HIST, PRE) if p.exists()])
    s = pd.read_csv(SEQ, dtype={"event_id": str, "player_id": str}) if SEQ.exists() else pd.DataFrame()
    rep = {"title": "First shot: label coverage", "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rule_file": "backtests/first_basket_v2_rule.md",
           "target": "first made field goal of the game (free throws excluded)", "blocks": []}
    for (season, phase), g in ev.groupby(["season", "phase"]):
        hs = h[(h.season == season) & (h.phase == phase)]; clean = hs[hs.label_clean == 1]
        ss = s[s.event_id.isin(hs.event_id)] if len(s) else s
        first = ss[ss.idx == 1] if len(ss) else ss
        tip_known = hs[hs.jump_gainer_team.notna()]
        blk = {"season": f"{season - 1}-{str(season)[2:]}", "phase": phase, "games": int(len(g)), "games_with_play_file": int(sum((RAW / f"{e}.json.gz").exists() for e in g.event_id)),
               "games_labelled": int(len(hs)), "gradable": int(len(clean)), "gradable_share": round(len(clean) / len(g), 4) if len(g) else None,
               "reasons": {k: int(v) for k, v in hs.reason.value_counts().items()},
               "jump_ball_captured_share": round(float(hs.jump_gainer_team.notna().mean()), 4) if len(hs) else None,
               "first_attempt_by_possession_team_share": round(float((tip_known.first_fga_team == tip_known.jump_gainer_team).mean()), 4) if len(tip_known) else None,
               "first_fg_by_possession_team_share": round(float((tip_known.first_fg_team == tip_known.jump_gainer_team)[tip_known.label_clean == 1].mean()), 4) if len(tip_known) else None,
               "first_fg_scorer_started_share": round(float(clean.first_fg_started.mean()), 4) if len(clean) else None,
               "median_attempts_to_first_fg": float(clean.attempts_to_first_fg.median()) if len(clean) else None,
               "median_secs_to_first_fg": round(float(clean.first_fg_secs.median()), 1) if len(clean) else None,
               "first_attempt_type_mix": {k: round(float(v), 4) for k, v in first.shot_type.value_counts(normalize=True).items()} if len(first) else {},
               "make_rate_by_type_opening_sequence": {k: round(float(v), 4) for k, v in ss.groupby("shot_type").made.mean().items()} if len(ss) else {},
               "same_team_after_miss_share": round(float(ss[ss.idx > 1].same_team_as_prev.mean()), 4) if len(ss) and (ss.idx > 1).any() else None}
        rep["blocks"].append(blk)
        print(f"[first-shot] {blk['season']} {phase}: {blk['gradable']}/{blk['games']} gradable ({blk['gradable_share']}), reasons {blk['reasons']}")
    write_json(COVERAGE, rep)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="backfill", choices=["backfill"])
    ap.parse_args()
    backfill()
