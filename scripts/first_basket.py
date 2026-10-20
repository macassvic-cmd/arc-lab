"""First basket, label pipeline and coverage (backtests/ladder_v2_rule.md, section 4). Pipeline first, model second.

  python first_basket.py backfill [--seasons 2025 2026]   label every stored game (data/pbp_raw) not yet in
                                                           data/first_basket_history.csv; write docs/data/first_basket_coverage.json
Label: the first play with scoringPlay true in period 1; the scorer is its first participant athlete id. A label is clean when
that id is a player in the game's box score. Jump-ball participants and the tip winner are stored for the model.
"""
import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, write_json
from pbp_store import RAW, events, load

HIST = DATA_DIR / "first_basket_history.csv"
FIELDS = ["event_id", "date", "season", "home", "away", "scorer_id", "scorer", "scorer_team", "scorer_started", "play_type", "is_ft", "secs_elapsed", "points",
          "tip_team", "jump_ids", "jump_text", "label_clean", "reason"]
JUMP_RE = re.compile(r"^(.*?) vs\. (.*?) \((.*?) gains possession\)")


def label(doc, box):
    """box: DataFrame of this game's box rows (player_id, player, team, starter)."""
    plays = sorted(doc.get("plays", []), key=lambda p: p["seq"])
    teams = doc["meta"]["teams"]; abbr = {t["id"]: t["abbr"] for t in teams.values()}
    out = {"event_id": doc["event_id"], "date": doc["date"], "season": doc["season"], "home": teams.get("home", {}).get("abbr"), "away": teams.get("away", {}).get("abbr")}
    if not plays:
        return {**out, "label_clean": 0, "reason": "no_plays"}
    jump = next((p for p in plays if p["period"] == 1 and (p["type"] or "").lower().startswith("jump")), None)
    if jump:
        out["tip_team"] = abbr.get(jump.get("team")); out["jump_ids"] = "|".join(str(a) for a in jump.get("ath", []) if a); out["jump_text"] = jump.get("text")
    first = next((p for p in plays if p["period"] == 1 and p["scoring"] and (p.get("value") or 0) > 0), None)
    if first is None:
        return {**out, "label_clean": 0, "reason": "no_scoring_play_in_period_1"}
    sid = str(first["ath"][0]) if first.get("ath") and first["ath"][0] else None
    out.update({"play_type": first["type"], "is_ft": int("free throw" in (first["type"] or "").lower()), "secs_elapsed": None if first["secs"] is None else round(720 - first["secs"], 1), "points": first.get("value")})
    if sid is None:
        return {**out, "label_clean": 0, "reason": "scoring_play_without_participant"}
    row = box[box.player_id == sid]
    if row.empty:
        return {**out, "scorer_id": sid, "scorer_team": abbr.get(first.get("team")), "label_clean": 0, "reason": "scorer_not_in_box_score"}
    r = row.iloc[0]
    return {**out, "scorer_id": sid, "scorer": r.player, "scorer_team": r.team, "scorer_started": int(r.starter), "label_clean": 1, "reason": "ok"}


FIRST_SEASON = 2025   # labels cover 2024-25 onward (the rule's backfill seasons plus every live season)


def backfill(seasons=None):
    ev = events(seasons)
    if not seasons:
        ev = ev[ev.season >= FIRST_SEASON].reset_index(drop=True)
    done = set(pd.read_csv(HIST, dtype={"event_id": str}).event_id) if HIST.exists() else set()
    frames = [pd.read_csv(DATA_DIR / fn, dtype={"event_id": str, "player_id": str}, usecols=["event_id", "player_id", "player", "team", "starter"]) for fn in ("player_games.csv", "player_games_history.csv") if (DATA_DIR / fn).exists()]
    box = pd.concat(frames).drop_duplicates(["event_id", "player_id"])
    by_ev = {e: g for e, g in box.groupby("event_id")}
    new_file = not HIST.exists()
    added = 0
    with open(HIST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in ev.itertuples():
            if r.event_id in done:
                continue
            doc = load(r.event_id)
            if doc is None:
                continue   # not stored yet: labelled on a later run
            w.writerow({k: v for k, v in label(doc, by_ev.get(r.event_id, box.iloc[0:0])).items() if k in FIELDS}); added += 1
    print(f"[first-basket] labelled {added} new games -> {HIST.name}")
    coverage(ev)


def coverage(ev):
    h = pd.read_csv(HIST, dtype={"event_id": str, "scorer_id": str})
    rep = {"title": "First basket: label coverage", "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rule_file": "backtests/ladder_v2_rule.md", "seasons": {}}
    for season, g in ev.groupby("season"):
        hs = h[h.season == season]; clean = hs[hs.label_clean == 1]
        rep["seasons"][f"{season - 1}-{str(season)[2:]}"] = {
            "games_in_box_scores": int(len(g)), "games_with_play_file": int(sum((RAW / f"{e}.json.gz").exists() for e in g.event_id)), "games_labelled": int(len(hs)),
            "clean_labels": int(len(clean)), "clean_share_of_games": round(len(clean) / len(g), 4) if len(g) else None,
            "reasons": {k: int(v) for k, v in hs.reason.value_counts().items()},
            "first_score_free_throw_share": round(float(clean.is_ft.mean()), 4) if len(clean) else None,
            "scorer_started_share": round(float(clean.scorer_started.mean()), 4) if len(clean) else None,
            "tip_winner_team_scored_first_share": round(float((clean.tip_team == clean.scorer_team)[clean.tip_team.notna()].mean()), 4) if clean.tip_team.notna().any() else None,
            "median_secs_to_first_score": round(float(clean.secs_elapsed.median()), 1) if len(clean) else None}
    write_json(SITE_DATA / "first_basket_coverage.json", rep)
    for k, v in rep["seasons"].items():
        print(f"[first-basket] {k}: {v['clean_labels']}/{v['games_in_box_scores']} clean ({v['clean_share_of_games']}), reasons {v['reasons']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="backfill", choices=["backfill"])
    ap.add_argument("--seasons", nargs="*")
    a = ap.parse_args()
    backfill(a.seasons)
