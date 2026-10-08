"""Turn a pirate-bets Underdog board snapshot into docs/data/lines.json.

Input:  board_snapshots/underdog.json (pirate-bets canonical contract: {"platform",
        "captured_at", "entries": [{"player", "league", "market", "line", "team",
        "opponent", "game", "scheduled_at", "game_status", "platform_ids": {...}}]})
Output: docs/data/lines.json in the shape of lines.example.json, plus sharp odds
        from OddsBlaze on each line when ODDSBLAZE_API_KEY is set:
  {"updated", "book": "Underdog", "sharp_book": "pinnacle"|null,
   "lines": [{"player", "stat", "line", "team", "opp", "start", "ud_id",
              "sharp": {"book", "line", "over", "under", "fair_over", "exact"} | null}]}

CLI:
  python underdog_lines.py --snapshot <path/to/underdog.json> [--out ../docs/data/lines.json] [--push] [--no-odds]
Library (what the pirate-bets scraper hook calls):
  write_lines(snapshot_dict, out_path, push=False, odds=True) -> summary dict
"""
import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SITE_DATA, norm_name, write_json  # noqa: E402
from odds import OddsBlaze, devig, team_abbr  # noqa: E402

log = logging.getLogger("arc.lines")
DEFAULT_OUT = SITE_DATA / "lines.json"

# Underdog display_stat -> Arc Lab stat code. Season-long markets ("Regular Season
# Points Per Game", "Triple Doubles") and anything else fall through and are ignored.
MARKETS = {
    "points": "PTS", "pts": "PTS",
    "rebounds": "REB", "rebs": "REB",
    "assists": "AST", "asts": "AST",
    "pts + rebs + asts": "PRA", "points + rebounds + assists": "PRA", "pts+rebs+asts": "PRA", "pra": "PRA",
    "3-pointers made": "3PM", "3 pointers made": "3PM", "3-pt made": "3PM", "3 pt made": "3PM",
    "three pointers made": "3PM", "three-pointers made": "3PM", "threes made": "3PM", "3pm": "3PM", "3pt made": "3PM",
}


def stat_for(market):
    return MARKETS.get((market or "").strip().lower())


def nba_lines(snapshot):
    """Entries worth showing: NBA, a stat we model, main ("balanced") line, game not started."""
    out = []
    for e in snapshot.get("entries") or []:
        if (e.get("league") or "").upper() != "NBA":
            continue
        stat = stat_for(e.get("market"))
        if not stat:
            continue
        pid = e.get("platform_ids") or {}
        if pid.get("line_type") not in (None, "balanced"):
            continue
        if (e.get("game_status") or "scheduled") not in ("scheduled", "pregame", "not_started"):
            continue
        try:
            line = float(e["line"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "player": e["player"], "stat": stat, "line": line,
            "team": team_abbr(e.get("team")) or e.get("team"),
            "opp": team_abbr(e.get("opponent")) or e.get("opponent"),
            "start": e.get("scheduled_at"), "ud_id": e.get("source_id"),
            "sharp": None,
        })
    # one row per player/stat: keep the lowest-id-stable first occurrence
    seen, uniq = set(), []
    for row in out:
        k = (norm_name(row["player"]), row["stat"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(row)
    return uniq


def attach_sharp(lines, props):
    """props: OddsBlaze.player_props() index. Exact-line match first, else the nearest
    posted line (flagged exact=false) so the reader still sees where the sharp number sits."""
    hits = 0
    for row in lines:
        rec = props.get((norm_name(row["player"]), row["stat"]))
        if not rec or not rec["lines"]:
            continue
        exact = row["line"] in rec["lines"]
        line = row["line"] if exact else min(rec["lines"], key=lambda l: (abs(l - row["line"]), l))
        sides = rec["lines"][line]
        row["sharp"] = {
            "book": rec["book"], "line": line, "over": sides.get("over"), "under": sides.get("under"),
            "fair_over": (lambda p: None if p is None else round(p, 4))(devig(sides.get("over"), sides.get("under"))),
            "exact": exact,
        }
        hits += 1
    return hits


def build(snapshot, odds=True, client=None):
    lines = nba_lines(snapshot)
    sharp_book, hits = None, 0
    if odds and lines:
        client = client or OddsBlaze()
        if client.enabled:
            props = client.player_props()
            if props:
                hits = attach_sharp(lines, props)
                sharp_book = next(iter(props.values()))["book"]
        else:
            log.info("no ODDSBLAZE_API_KEY - lines written without sharp odds")
    return {
        "updated": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "book": "Underdog", "captured_at": snapshot.get("captured_at"), "sharp_book": sharp_book,
        "lines": lines,
    }, hits


def git_push(path):
    """Commit docs/data/lines.json to the arc-lab repo and push, so Pages picks it up.
    Quiet no-op when nothing changed. Returns True on push."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    rel = os.path.relpath(path, repo)

    def git(*a, check=True):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, timeout=120, check=check)

    git("add", rel)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    git("commit", "-q", "-m", f"lines: {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ}")
    for _ in range(3):
        if git("pull", "--rebase", "-q", check=False).returncode == 0 and git("push", "-q", check=False).returncode == 0:
            return True
    raise RuntimeError("git push failed after 3 attempts")


def write_lines(snapshot, out_path=DEFAULT_OUT, push=False, odds=True, client=None):
    doc, hits = build(snapshot, odds=odds, client=client)
    write_json(Path(out_path), doc)
    pushed = git_push(str(out_path)) if push else False
    summary = {"lines": len(doc["lines"]), "sharp": hits, "sharp_book": doc["sharp_book"], "pushed": pushed, "path": str(out_path)}
    log.info("lines.json: %(lines)d NBA lines, %(sharp)d with sharp odds (%(sharp_book)s), pushed=%(pushed)s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="pirate-bets board_snapshots/underdog.json")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--push", action="store_true", help="commit and push lines.json to the arc-lab repo")
    ap.add_argument("--no-odds", action="store_true", help="skip OddsBlaze even if a key is set")
    args = ap.parse_args()
    with open(args.snapshot, encoding="utf-8") as f:
        snapshot = json.load(f)
    s = write_lines(snapshot, out_path=Path(args.out), push=args.push, odds=not args.no_odds)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
