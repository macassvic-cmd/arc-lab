"""v3 data foundation, item 1: derived play-by-play tables from data/pbp_raw (pbp_store.py). No credits, no modelling.

  python pbp_tables.py                 all seasons in the cache
  python pbp_tables.py --seasons 2025  one season

Outputs:
  data/pbp_plays_<season>.csv.gz   compact plays, one file per season
  data/stints.csv.gz               one row per (game, team, lineup segment, player): a segment is a stretch with a constant five
  data/rotation_games.csv.gz       one row per player-game: first entry, stint count, minutes by score-margin bucket
  data/rotation_profile.csv        per player-season descriptive profile
  data/lineups.csv.gz              per season / team / five-man unit: minutes, possessions, points, net rating, thin flag
  docs/data/v3_coverage.json       coverage report (numbers only), read by the Backtest page

Lineup resolution: period 1 = box-score starters; later periods = players who act (any non-substitution play, incl. the OT
jump ball) or are substituted OUT before being substituted IN during that period, filled from the five on the floor at the end of
the previous period. A team-period is "clean" when that yields exactly five and every substitution is consistent (entering player
not already on, leaving player on). Score margin is the team's score minus the opponent's at the start of each interval.
"""
import argparse
import datetime as dt
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, write_json
from pbp_store import RAW

PERIOD_SECS = {1: 720, 2: 720, 3: 720, 4: 720}   # overtime periods: 300
CLOSE, MID = 10, 15                                # |margin| <= 10 close, 10-15 mid, > 15 in Q4+ garbage
THIN_MINUTES = 25
MIN_TOL, TEAM_TOL, SHARE_OK = 1.5, 3.0, 0.90      # "complete PBP" thresholds (see coverage)
PLAY_COLS = ["event_id", "season", "date", "seq", "period", "secs", "type_id", "type", "text", "team", "scoring", "value", "att", "shooting", "home", "away", "ath", "x", "y", "wall"]


def period_len(p):
    return PERIOD_SECS.get(p, 300)


def elapsed(period, secs):
    """Game minutes elapsed at a clock reading."""
    before = sum(period_len(q) for q in range(1, period))
    return (before + period_len(period) - secs) / 60.0


def load_box():
    frames = []
    for fn in ("player_games.csv", "player_games_history.csv"):
        p = DATA_DIR / fn
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"event_id": str, "player_id": str}))
    b = pd.concat(frames).drop_duplicates(["event_id", "player_id"])
    b["min"] = pd.to_numeric(b["min"], errors="coerce").fillna(0)
    return b


def is_turnover(t):
    t = t or ""
    return ("Turnover" in t and t != "No Turnover") or t in ("Traveling", "Lane")


def is_fga(p):
    return bool(p["shooting"]) and not (p["type"] or "").startswith("Free Throw")


def is_fta(p):
    return (p["type"] or "").startswith("Free Throw")


class Game:
    """Walk one game's plays per team, tracking the five on the floor."""

    def __init__(self, doc, box_team_of, box_min):
        self.doc, self.eid, self.season, self.date = doc, doc["event_id"], doc["season"], doc["date"]
        t = doc["meta"]["teams"]
        self.home_id, self.away_id = t["home"]["id"], t["away"]["id"]
        self.abbr = {self.home_id: t["home"]["abbr"], self.away_id: t["away"]["abbr"]}
        self.team_of = box_team_of          # athlete id -> team abbr (box score)
        self.box_min = box_min              # (abbr, athlete id) -> minutes
        self.plays = sorted(doc["plays"], key=lambda p: p["seq"])
        self.periods = sorted({p["period"] for p in self.plays if p["period"]})
        self.segments = []                  # team lineup segments
        self.player_stints = defaultdict(list)   # (team, pid) -> [(period, start, end)]
        self.buckets = defaultdict(lambda: [0.0, 0.0, 0.0])   # (team, pid) -> close / mid / garbage minutes
        self.clean = {}                     # (team, period) -> bool
        self.reasons = defaultdict(int)
        self.actions_off_court = 0

    def margin(self, team_id, home, away):
        return (home - away) if team_id == self.home_id else (away - home)

    def resolve_start(self, team_id, period, prev_five):
        abbr = self.abbr[team_id]
        if period == 1:
            five = list(self.doc["starters"].get(team_id, []))
            return five, len(five) == 5, "starters"
        status = {}
        for p in self.plays:
            if p["period"] != period:
                continue
            if p["type"] == "Substitution":
                if p["team"] != team_id or len(p["ath"]) < 2:
                    continue
                enter, leave = p["ath"][0], p["ath"][1]
                status.setdefault(leave, "on"); status.setdefault(enter, "in")
            else:
                for a in p["ath"]:
                    if self.team_of.get(a) == abbr:
                        status.setdefault(a, "on")
        on = [a for a, s in status.items() if s == "on"]   # insertion order = first evidence in the period
        reason = "evidence"
        if len(on) > 5:                                      # a substitution the feed never logged: keep the five seen earliest, flag
            self.reasons[f"period start over-resolved ({len(on)} players by evidence; earliest five kept) [team-periods]"] += 1
            return on[:5], False, "evidence-trimmed"
        if len(on) < 5:
            for a in prev_five:
                if a not in status and len(on) < 5:
                    on.append(a); reason = "evidence+carry"
        return on, len(on) == 5, reason

    def run(self):
        prev_five = {}
        for period in self.periods:
            plays = [p for p in self.plays if p["period"] == period]
            on, start_t, seg_start, seg_margin, ok = {}, {}, {}, {}, {}
            for tid in (self.home_id, self.away_id):
                five, clean, how = self.resolve_start(tid, period, prev_five.get(tid, []))
                ok[tid] = clean
                if not clean and len(five) != 5:
                    self.reasons[f"period start unresolved ({len(five)} players) [team-periods]"] += 1
                on[tid] = list(five)
                start_t[tid] = {a: period_len(period) for a in five}
                seg_start[tid] = period_len(period); seg_margin[tid] = 0
            t_prev, home, away = period_len(period), None, None
            for p in plays:
                if home is None:
                    home, away = p["home"], p["away"]
                    for tid in on:
                        seg_margin[tid] = self.margin(tid, home, away)
                t = p["secs"] if p["secs"] is not None else t_prev
                t = min(t, t_prev)   # clocks never run backwards inside a period
                dt_min = (t_prev - t) / 60.0
                if dt_min > 0:
                    for tid, five in on.items():
                        m = abs(self.margin(tid, home, away))
                        slot = 2 if (period >= 4 and m > MID) else (0 if m <= CLOSE else 1)
                        for a in five:
                            self.buckets[(self.abbr[tid], a)][slot] += dt_min
                if p["type"] == "Substitution" and p["team"] in on and len(p["ath"]) >= 2:
                    tid = p["team"]; enter, leave = p["ath"][0], p["ath"][1]
                    five = on[tid]
                    if leave not in five:
                        ok[tid] = False; self.reasons["sub out of a player not on the floor [substitutions]"] += 1
                    if enter in five:
                        ok[tid] = False; self.reasons["sub in of a player already on the floor [substitutions]"] += 1
                    self.close_segment(tid, period, seg_start[tid], t, five, seg_margin[tid], self.margin(tid, home, away))
                    if leave in five:
                        five.remove(leave)
                        self.player_stints[(self.abbr[tid], leave)].append((period, start_t[tid].pop(leave, period_len(period)), t))
                    if enter not in five:
                        five.append(enter); start_t[tid][enter] = t
                    seg_start[tid] = t; seg_margin[tid] = self.margin(tid, home, away)
                elif p["type"] != "Substitution":
                    for a in p["ath"]:
                        tm = self.team_of.get(a)
                        tid = self.home_id if tm == self.abbr[self.home_id] else (self.away_id if tm == self.abbr[self.away_id] else None)
                        if tid is not None and a not in on[tid]:
                            self.actions_off_court += 1
                home, away, t_prev = p["home"], p["away"], t
            for tid, five in on.items():
                self.close_segment(tid, period, seg_start[tid], 0.0, five, seg_margin[tid], self.margin(tid, home or 0, away or 0))
                for a in five:
                    self.player_stints[(self.abbr[tid], a)].append((period, start_t[tid].get(a, period_len(period)), 0.0))
                self.clean[(self.abbr[tid], period)] = ok[tid] and len(five) == 5
                if len(five) != 5 and ok[tid]:
                    self.reasons[f"period ends with {len(five)} on the floor [team-periods]"] += 1
                prev_five[tid] = list(five)

    def close_segment(self, tid, period, start, end, five, m0, m1):
        if end < start:
            self.segments.append(dict(event_id=self.eid, season=self.season, date=self.date, team=self.abbr[tid], period=period, start_secs=start, end_secs=end,
                                      minutes=(start - end) / 60.0, lineup="-".join(sorted(five)), n_on=len(five), margin_start=m0, margin_end=m1))

    def stats_for_segments(self):
        """Possession components and points per segment, for the lineup table (team's own plays while the unit is on)."""
        out = []
        for s in self.segments:
            tid = self.home_id if s["team"] == self.abbr[self.home_id] else self.away_id
            opp = self.away_id if tid == self.home_id else self.home_id
            agg = dict(fga=0, fta=0, oreb=0, tov=0, pts=0, ofga=0, ofta=0, ooreb=0, otov=0, opts=0)
            for p in self.plays:
                if p["period"] != s["period"] or p["secs"] is None or not (s["end_secs"] <= p["secs"] <= s["start_secs"]):
                    continue
                if p["secs"] == s["start_secs"] and p["type"] == "Substitution":
                    continue
                pre = "" if p["team"] == tid else ("o" if p["team"] == opp else None)
                if pre is None:
                    continue
                agg[pre + "fga"] += is_fga(p); agg[pre + "fta"] += is_fta(p); agg[pre + "oreb"] += p["type"] == "Offensive Rebound"
                agg[pre + "tov"] += is_turnover(p["type"]); agg[pre + "pts"] += (p["value"] or 0) if p["scoring"] else 0
            out.append({**s, **agg})
        return out


def build(seasons=None):
    box = load_box()
    if seasons:
        box = box[box.season.isin(seasons)]
    team_of_all = {e: dict(zip(g.player_id, g.team)) for e, g in box.groupby("event_id")}
    min_all = {e: dict(zip(zip(g.team, g.player_id), g["min"])) for e, g in box.groupby("event_id")}
    games = box.drop_duplicates("event_id")[["event_id", "season", "date"]].sort_values("date")
    seg_rows, rot_rows, cov = [], [], defaultdict(lambda: defaultdict(int))
    reasons = defaultdict(int)
    writers = {}   # season -> (gzip handle, csv writer): plays stream straight to disk, never held in memory

    def plays_writer(season):
        if season not in writers:
            import csv
            fh = gzip.open(DATA_DIR / f"pbp_plays_{season}.csv.gz", "wt", encoding="utf-8", newline="")
            w = csv.writer(fh); w.writerow(PLAY_COLS); writers[season] = (fh, w)
        return writers[season][1]

    for g in games.itertuples():
        c = cov[int(g.season)]
        c["games_box"] += 1
        fn = RAW / f"{g.event_id}.json.gz"
        if not fn.exists():
            reasons["no play file stored [games]"] += 1; continue
        c["games_with_file"] += 1
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            doc = json.load(f)
        if not doc["plays"]:
            reasons["file without plays [games]"] += 1; continue
        c["games_with_plays"] += 1
        pw = plays_writer(int(g.season))
        for p in doc["plays"]:
            pw.writerow([g.event_id, int(g.season), g.date, p["seq"], p["period"], p["secs"], p["type_id"], p["type"], p["text"], p["team"], int(p["scoring"]), p["value"], p["att"], int(p["shooting"]), p["home"], p["away"], "|".join(a for a in p["ath"] if a), None if (p["x"] or 0) < -1e6 else p["x"], None if (p["y"] or 0) < -1e6 else p["y"], p["wall"]])
        gm = Game(doc, team_of_all.get(g.event_id, {}), min_all.get(g.event_id, {}))
        try:
            gm.run()
        except Exception as e:   # never let one malformed feed stop the build
            reasons[f"walk error: {type(e).__name__}"] += 1; continue
        for k, v in gm.reasons.items():
            reasons[k] += v
        seg_rows.extend(gm.stats_for_segments())
        # player-game rotation rows + completeness checks
        pbp_min = defaultdict(float)
        for (team, pid), st in gm.player_stints.items():
            pbp_min[(team, pid)] = sum((a - b) / 60.0 for _, a, b in st)
        box_g = box[box.event_id == g.event_id]
        played = box_g[box_g["min"] > 0]
        within, team_sum_ok = 0, True
        game_len = sum(period_len(p) for p in gm.periods) / 60.0
        for tm in box_g.team.unique():
            tot = sum(v for (t, _), v in pbp_min.items() if t == tm)
            if abs(tot - 5 * game_len) > TEAM_TOL:
                team_sum_ok = False
        starters = {(r.team, r.player_id) for r in box_g.itertuples() if int(r.starter) == 1}
        for r in played.itertuples():
            key = (r.team, r.player_id); st = sorted(gm.player_stints.get(key, []))
            # merge consecutive stints (same period, touching) into continuous on-court stretches
            merged = []
            for per, a, b in st:
                if merged and merged[-1][0] == per and abs(merged[-1][2] - a) < 1e-6:
                    merged[-1][2] = b
                else:
                    merged.append([per, a, b])
            # a stretch spanning a period break counts once
            stretches = 0; prev_end = None
            for per, a, b in merged:
                if not (prev_end is not None and prev_end[1] == 0.0 and a == period_len(per) and per == prev_end[0] + 1):
                    stretches += 1
                prev_end = (per, b)
            first = 0.0 if key in starters else (elapsed(merged[0][0], merged[0][1]) if merged else None)
            bk = gm.buckets.get(key, [0.0, 0.0, 0.0])
            gap = abs(pbp_min.get(key, 0.0) - float(r.min))
            within += gap <= MIN_TOL
            rot_rows.append(dict(event_id=g.event_id, season=int(g.season), date=g.date, team=r.team, player_id=r.player_id, starter=int(r.starter),
                                 first_entry_minute=None if first is None else round(first, 2), stint_count=stretches,
                                 minutes_close=round(bk[0], 2), minutes_mid=round(bk[1], 2), minutes_garbage=round(bk[2], 2),
                                 minutes_total_pbp=round(pbp_min.get(key, 0.0), 2), box_minutes=float(r.min), abs_minutes_gap=round(gap, 2),
                                 pbp_found=int(key in gm.player_stints)))
        all_clean = all(gm.clean.get((tm, p), False) for tm in box_g.team.unique() for p in gm.periods)
        share_ok = within / max(len(played), 1)
        complete = all_clean and share_ok >= SHARE_OK and team_sum_ok
        c["games_complete"] += int(complete)
        c["games_all_periods_clean"] += int(all_clean)
        c["games_minutes_match"] += int(share_ok >= SHARE_OK)
        c["games_team_sum_ok"] += int(team_sum_ok)
        c["actions_off_court"] += gm.actions_off_court
        if not complete:
            if not all_clean: reasons["incomplete: a team-period lineup did not resolve cleanly [games]"] += 1
            if share_ok < SHARE_OK: reasons["incomplete: < 90% of players within 1.5 min of box minutes [games]"] += 1
            if not team_sum_ok: reasons["incomplete: team stint minutes off by > 3 from 5 x game length [games]"] += 1
    # ---- write tables
    for fh, _ in writers.values():
        fh.close()
    seg = pd.DataFrame(seg_rows)
    stints = seg.loc[seg.index.repeat(seg.n_on)].copy() if len(seg) else seg
    if len(seg):
        stints["player_id"] = [a for lu, n in zip(seg.lineup, seg.n_on) for a in (lu.split("-") + [""] * n)[:n]]
        stints = stints[["event_id", "season", "date", "team", "player_id", "period", "start_secs", "end_secs", "minutes", "lineup", "margin_start", "margin_end"]].rename(columns={"minutes": "stint_minutes", "start_secs": "start_secs_remaining", "end_secs": "end_secs_remaining"})
        stints["stint_minutes"] = stints.stint_minutes.round(3)
        with gzip.open(DATA_DIR / "stints.csv.gz", "wt", encoding="utf-8", newline="") as f:
            stints.to_csv(f, index=False)
    rot = pd.DataFrame(rot_rows)
    with gzip.open(DATA_DIR / "rotation_games.csv.gz", "wt", encoding="utf-8", newline="") as f:
        rot.to_csv(f, index=False)
    prof = profile(rot)
    prof.to_csv(DATA_DIR / "rotation_profile.csv", index=False)
    lu = lineups(seg)
    with gzip.open(DATA_DIR / "lineups.csv.gz", "wt", encoding="utf-8", newline="") as f:
        lu.to_csv(f, index=False)
    return cov, reasons, rot, lu, seg


def profile(rot):
    if rot.empty:
        return pd.DataFrame()
    g = rot.groupby(["season", "player_id"])
    tot = g[["minutes_close", "minutes_mid", "minutes_garbage", "minutes_total_pbp"]].sum()
    out = pd.DataFrame({"games": g.size(), "starts": g.starter.sum(),
                        "median_first_sub_minute": rot[rot.starter == 0].groupby(["season", "player_id"]).first_entry_minute.median(),
                        "mean_stint_count": g.stint_count.mean().round(2),
                        "share_close": (tot.minutes_close / tot.minutes_total_pbp.replace(0, np.nan)).round(3),
                        "share_mid": (tot.minutes_mid / tot.minutes_total_pbp.replace(0, np.nan)).round(3),
                        "share_garbage": (tot.minutes_garbage / tot.minutes_total_pbp.replace(0, np.nan)).round(3),
                        "minutes_pbp": tot.minutes_total_pbp.round(1)}).reset_index()
    out["median_first_sub_minute"] = out.median_first_sub_minute.round(2)
    return out


def lineups(seg):
    """Per season / team / five-man unit. Possessions = FGA - OREB + TOV + 0.44 FTA from the plays while the unit is on (own and
       opponent's). Net rating = points per 100 own possessions - points allowed per 100 opponent possessions."""
    if seg.empty:
        return pd.DataFrame()
    s = seg[seg.n_on == 5]
    g = s.groupby(["season", "team", "lineup"]).agg(minutes=("minutes", "sum"), segments=("minutes", "size"), games=("event_id", "nunique"),
                                                      fga=("fga", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"), pts=("pts", "sum"),
                                                      ofga=("ofga", "sum"), ofta=("ofta", "sum"), ooreb=("ooreb", "sum"), otov=("otov", "sum"), opts=("opts", "sum")).reset_index()
    g["poss"] = (g.fga - g.oreb + g.tov + 0.44 * g.fta).round(1)
    g["opp_poss"] = (g.ofga - g.ooreb + g.otov + 0.44 * g.ofta).round(1)
    g["ortg"] = (100 * g.pts / g.poss.replace(0, np.nan)).round(1)
    g["drtg"] = (100 * g.opts / g.opp_poss.replace(0, np.nan)).round(1)
    g["net_rating"] = (g.ortg - g.drtg).round(1)
    g["thin"] = (g.minutes < THIN_MINUTES).astype(int)
    g["minutes"] = g.minutes.round(2)
    return g.sort_values(["season", "team", "minutes"], ascending=[True, True, False])[["season", "team", "lineup", "games", "segments", "minutes", "poss", "opp_poss", "pts", "opts", "ortg", "drtg", "net_rating", "thin"]]


def coverage(cov, reasons, rot, lu):
    by_season = []
    for season in sorted(cov):
        c = cov[season]
        n = c["games_box"]
        r = rot[rot.season == season]
        by_season.append(dict(season=f"{season - 1}-{str(season)[2:]}", games_box=n, games_with_file=c["games_with_file"], games_with_plays=c["games_with_plays"],
                              games_complete=c["games_complete"], share_complete=round(c["games_complete"] / n, 4) if n else None,
                              games_all_periods_clean=c["games_all_periods_clean"], games_minutes_match=c["games_minutes_match"], games_team_sum_ok=c["games_team_sum_ok"],
                              player_games_box=int(len(r)), player_games_joined=int(r.pbp_found.sum()), join_rate=round(float(r.pbp_found.mean()), 4) if len(r) else None,
                              player_games_within_1_5_min=int((r.abs_minutes_gap <= MIN_TOL).sum()), median_abs_minutes_gap=round(float(r.abs_minutes_gap.median()), 2) if len(r) else None,
                              actions_off_court=c["actions_off_court"],
                              lineups=int((lu.season == season).sum()) if len(lu) else 0, lineups_thin=int(((lu.season == season) & (lu.thin == 1)).sum()) if len(lu) else 0,
                              lineups_thin_share=round(float(lu[lu.season == season].thin.mean()), 4) if len(lu) and (lu.season == season).any() else None))
    return dict(title="v3 data foundation: play-by-play coverage", generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                source="ESPN summary API `plays` (site.api.espn.com), stored by scripts/pbp_store.py; zero Odds API credits",
                complete_definition=f"every team-period lineup resolved cleanly (exactly five; every substitution consistent) AND >= {int(SHARE_OK * 100)}% of the players who played have stint minutes within {MIN_TOL} min of their box minutes AND each team's stint minutes within {TEAM_TOL} min of 5 x game length",
                by_season=by_season, failure_reasons=sorted(({"reason": k, "count": int(v)} for k, v in reasons.items()), key=lambda x: -x["count"]),
                notes=["Period-1 fives are the box-score starters; later periods resolve from who acts or is subbed out before being subbed in, filled from the previous period's closing five.",
                       "Score-margin buckets: close = |margin| <= 10, mid = 10-15, garbage = > 15 in the 4th quarter or overtime; minutes accrue per play interval at the margin before the play.",
                       "Lineup possessions = FGA - OREB + TOV + 0.44 FTA from plays while the unit is on; units under 25 minutes are flagged thin.",
                       "actions_off_court counts plays whose participant was not on the tracked floor (free-throw / substitution ordering in the feed, or a lineup error); reported, not used."])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int)
    a = ap.parse_args()
    import time; t0 = time.time()
    cov, reasons, rot, lu, seg = build(a.seasons)
    rep = coverage(cov, reasons, rot, lu)
    prev = {}
    try:
        with open(SITE_DATA / "v3_coverage.json") as f:
            prev = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if prev.get("game_context"):
        rep["game_context"] = prev["game_context"]
    write_json(SITE_DATA / "v3_coverage.json", rep)
    for s in rep["by_season"]:
        print(f"[pbp_tables] {s}")
    print(f"[pbp_tables] reasons {rep['failure_reasons']} | segments {len(seg)} | rotation rows {len(rot)} | lineups {len(lu)} | {time.time() - t0:.0f}s")
