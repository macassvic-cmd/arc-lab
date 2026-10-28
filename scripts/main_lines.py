"""Underdog main-line sidecar for the Performance page's "playable rungs" view (display and reporting only, 2026-10-28).

  python main_lines.py          # record tonight's main lines from the board capture and backfill 3PM history; rewrite the sidecar

data/main_lines.csv: date, event_id, player_id, stat, line, captured_at. One row per player-game-stat; the LAST capture before
that game's tip wins. Two sources, merged idempotently on every run:
  * docs/data/lines.json: the board capture lines.json is built from (the Underdog "balanced" main line per player and stat),
    read at ladder snapshot time. Only PTS / REB / AST / 3PM lines, only for games that have not tipped at the capture time.
  * data/ud_prices.csv: the 3PM pricing history (one row per side per capture, since 2026-10-23), which backfills the main 3PM
    line for slates before this sidecar existed. player_id comes from the ladder log's row for the same date, event and name.
Nothing here touches the model, the ladder log (append-only, never edited), or the locked recalibration pipeline, whose
statistics keep computing on the full log. ladder_log.py grade reads this file to build the playable-rungs slice:
a rung is playable when rung >= floor(main line) + 1 (the line's over and the true alts above it; everything at or below
the line is a negative alt and is excluded). Player-games with no recorded main line are excluded under that view and counted.
"""
import datetime as dt
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_name, read_json  # noqa: E402
from odds import team_abbr  # noqa: E402

OUT = DATA_DIR / "main_lines.csv"
LINES = SITE_DATA / "lines.json"
PROJ = SITE_DATA / "projections.json"
UD_PRICES = DATA_DIR / "ud_prices.csv"
LADDER_LOG = DATA_DIR / "ladder_log.csv"
FIELDS = ["date", "event_id", "player_id", "stat", "line", "captured_at"]
STATS = {"3PM": "tpm", "PTS": "pts", "REB": "reb", "AST": "ast"}   # lines.json stat -> ladder stat code (FP is not a ladder stat)
KEY = ["date", "event_id", "player_id", "stat"]


def parse_ts(s):
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def iso(t):
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def playable_floor(line):
    """Lowest playable rung for a main line: floor(line) + 1 (2.5 -> 3+, 3.0 -> 4+)."""
    return int(math.floor(float(line))) + 1


def from_lines(doc, P):
    """Main lines on the board capture for tonight's slate, pre-tip only."""
    if not doc or not P or not P.get("games"):
        return []
    cap_s = doc.get("captured_at") or doc.get("updated")
    if not cap_s:
        return []
    cap = parse_ts(cap_s)
    games = {}
    for g in P["games"]:
        games[frozenset({team_abbr(g["home"]), team_abbr(g["away"])})] = g
    players = {}
    for p in P["players"]:
        if p.get("inactive"):
            continue
        players.setdefault((norm_name(p["name"]), team_abbr(p["team"])), p)
    rows, skipped = [], 0
    for ln in doc.get("lines") or []:
        stat = STATS.get(ln.get("stat"))
        if not stat:
            continue
        tm, op = team_abbr(ln.get("team")), team_abbr(ln.get("opp"))
        g = games.get(frozenset({tm, op}))
        p = players.get((norm_name(ln.get("player")), tm))
        if not g or not p:
            skipped += 1; continue
        try:
            line = float(ln["line"]); tip = parse_ts(g["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if cap >= tip:
            continue   # a post-tip capture never records a main line
        rows.append({"date": P["date"], "event_id": str(g["event_id"]), "player_id": str(p["id"]), "stat": stat, "line": line, "captured_at": iso(cap)})
    print(f"[main-lines] board capture {iso(cap)}: {len(rows)} pre-tip main lines for {P.get('date')} ({skipped} lines not matched to the slate)")
    return rows


def from_ud_prices():
    """Backfill: the 3PM main line per player-game from data/ud_prices.csv, pre-tip captures only, player_id via the ladder log."""
    if not UD_PRICES.exists() or not LADDER_LOG.exists():
        return []
    u = pd.read_csv(UD_PRICES, dtype={"event_id": str}, usecols=["captured_at", "date", "event_id", "player", "line"])
    lg = pd.read_csv(LADDER_LOG, dtype={"event_id": str, "player_id": str}, usecols=["date", "event_id", "player_id", "player", "tip"]).drop_duplicates(["date", "event_id", "player_id"])
    lg["key"] = lg.player.map(norm_name)
    u["key"] = u.player.map(norm_name)
    m = u.merge(lg[["date", "event_id", "player_id", "key", "tip"]], on=["date", "event_id", "key"], how="inner")
    m = m[pd.to_datetime(m.captured_at, utc=True) < pd.to_datetime(m.tip, utc=True)]
    m["captured_at"] = pd.to_datetime(m.captured_at, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    m["stat"] = "tpm"
    rows = m[FIELDS].to_dict("records")
    print(f"[main-lines] ud_prices backfill: {len(u)} price rows -> {len(rows)} pre-tip 3PM observations ({u.key.nunique() - m.key.nunique()} names not in the ladder log)")
    return rows


def record(out=OUT):
    old = pd.read_csv(out, dtype={"event_id": str, "player_id": str}) if out.exists() else pd.DataFrame(columns=FIELDS)
    new = from_lines(read_json(LINES, None), read_json(PROJ, None)) + from_ud_prices()
    d = pd.concat([old, pd.DataFrame(new, columns=FIELDS)], ignore_index=True)
    d["line"] = d.line.astype(float)
    d = d.sort_values("captured_at", kind="stable").drop_duplicates(KEY, keep="last").sort_values(KEY + ["captured_at"])
    d[FIELDS].to_csv(out, index=False)
    by = d.groupby("stat").size().to_dict()
    print(f"[main-lines] {out.name}: {len(d)} player-game-stats ({by}), {d.date.nunique()} dates ({d.date.min()} .. {d.date.max()}); last pre-tip capture wins")
    return d


if __name__ == "__main__":
    record()
