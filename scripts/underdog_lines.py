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
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter  # noqa: E402
from common import DATA_DIR, SITE_DATA, norm_name, write_json  # noqa: E402
from odds import OddsBlaze, devig, team_abbr  # noqa: E402

log = logging.getLogger("arc.lines")
DEFAULT_OUT = SITE_DATA / "lines.json"
BOARD_LATEST = DATA_DIR / "board_latest.json"   # repo copy of the capture's NBA entries (fantasy_points rule addendum 2026-10-23)
BOARD_REFRESH_SECONDS = 3600                     # an unchanged board is re-stamped at most hourly, so pushes stay quiet
PP_LATEST = DATA_DIR / "pp_fantasy_lines.json"   # repo copy of pirate-bets' PrizePicks Fantasy Score extract (fantasy_points rule amendment 2026-10-25)
PP_FILE_ENV = "ARC_LAB_PP_FILE"                  # optional explicit path to pp_fantasy_lines.jsonl

# Underdog display_stat -> Arc Lab stat code. Scope reset 2026-10-22 trimmed this to 3PM + Fantasy Points; CEO ruling
# 2026-10-27 restores the main Points / Rebounds / Assists lines for LINE PUBLISHING only (Props tab, Ladder tonight strip).
# Exact market names on purpose: "1Q Points", "Points + Rebounds", "Pts + Rebs + Asts" (PRA), "Rebounds + Assists",
# double-double / triple-double and every other combo or period market fall through and are ignored. No PRA, ever.
MARKETS = {
    "fantasy points": "FP", "fantasy score": "FP", "fantasy pts": "FP",
    "3-pointers made": "3PM", "3 pointers made": "3PM", "3-pt made": "3PM", "3 pt made": "3PM",
    "three pointers made": "3PM", "three-pointers made": "3PM", "threes made": "3PM", "3pm": "3PM", "3pt made": "3PM",
    "points": "PTS", "pts": "PTS",
    "rebounds": "REB", "rebs": "REB", "reb": "REB",
    "assists": "AST", "asts": "AST", "ast": "AST",
}
PUBLISHED_STATS = ("3PM", "PTS", "REB", "AST", "FP")


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


def write_board(snapshot, out_path=BOARD_LATEST):
    """Copy the capture's NBA entries (every market, prices and multipliers included) into the repo as
    data/board_latest.json with the capture time, so the Actions pricing step can check freshness and price against it.
    Rewritten when the NBA board content changed or the stored copy's capture is over BOARD_REFRESH_SECONDS old."""
    nba = [e for e in snapshot.get("entries") or [] if (e.get("league") or "").upper() == "NBA"]
    digest = hashlib.sha1(json.dumps(nba, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    now = dt.datetime.now(dt.timezone.utc)
    try:
        with open(out_path, encoding="utf-8") as f:
            old = json.load(f)
        old_cap = dt.datetime.fromisoformat(str(old.get("captured_at")).replace("Z", "+00:00"))
        if old.get("board_sha1") == digest and (now - old_cap).total_seconds() < BOARD_REFRESH_SECONDS:
            return False
    except Exception:  # noqa: BLE001  (no copy yet, or unreadable: write it)
        pass
    markets = Counter((e.get("market") or "").strip() for e in nba)
    write_json(Path(out_path), {
        "platform": snapshot.get("platform") or "underdog", "league": "NBA",
        "captured_at": snapshot.get("captured_at"), "source_updated_at": snapshot.get("source_updated_at"),
        "stored_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "nba_entries": len(nba), "markets": dict(markets.most_common()), "board_sha1": digest, "entries": nba,
    })
    return True


def pp_candidates(snapshot_path):
    """Where pirate-bets' PrizePicks pull writes pp_fantasy_lines.jsonl: an explicit ARC_LAB_PP_FILE, the board_snapshots
    folder of the Underdog snapshot we were handed, and the Desktop pirate-bets checkout."""
    out = []
    if os.environ.get(PP_FILE_ENV):
        out.append(Path(os.environ[PP_FILE_ENV]))
    if snapshot_path:
        out.append(Path(snapshot_path).resolve().parent / "pp_fantasy_lines.jsonl")
    out.append(Path.home() / "Desktop" / "pirate-bets" / "reader" / "board_snapshots" / "pp_fantasy_lines.jsonl")
    return out


def write_pp(snapshot_path=None, out_path=PP_LATEST):
    """Copy the PrizePicks Fantasy Score extract (one JSON object per line) into the repo as data/pp_fantasy_lines.json with
    its capture time (the file's modification time, UTC). The newest existing candidate wins; rewritten only when the
    content changed. Returns True when written."""
    found = [p for p in pp_candidates(snapshot_path) if p.is_file()]
    if not found:
        return False
    src = max(found, key=lambda p: p.stat().st_mtime)
    raw = src.read_bytes()
    digest = hashlib.sha1(raw).hexdigest()
    try:
        with open(out_path, encoding="utf-8") as f:
            if json.load(f).get("sha1") == digest:
                return False
    except Exception:  # noqa: BLE001  (no copy yet, or unreadable: write it)
        pass
    lines, bad = [], 0
    for ln in raw.decode("utf-8", "replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            lines.append(json.loads(ln))
        except ValueError:
            bad += 1
    iso = lambda d: d.replace(microsecond=0).isoformat().replace("+00:00", "Z")  # noqa: E731
    write_json(Path(out_path), {
        "platform": "prizepicks", "source_path": str(src),
        "captured_at": iso(dt.datetime.fromtimestamp(src.stat().st_mtime, dt.timezone.utc)),
        "stored_at": iso(dt.datetime.now(dt.timezone.utc)), "n_lines": len(lines), "bad_lines": bad, "sha1": digest, "lines": lines,
    })
    log.info("pp_fantasy_lines.json: %d lines copied from %s (%d unparseable)", len(lines), src, bad)
    return True


def git_push(paths):
    """Commit the given files to the arc-lab repo and push, so Pages and Actions pick them up.
    Quiet no-op when nothing changed. Returns True on push."""
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    repo = str(Path(__file__).resolve().parent.parent)

    def git(*a, check=True):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, timeout=120, check=check)

    for path in paths:
        if os.path.exists(path):
            git("add", os.path.relpath(os.path.abspath(path), repo))
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    staged = git("diff", "--cached", "--name-only", check=False).stdout.split()
    git("commit", "-q", "-m", f"board: {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} ({', '.join(staged) or 'lines'})")
    for _ in range(3):
        if git("pull", "--rebase", "-q", check=False).returncode == 0 and git("push", "-q", check=False).returncode == 0:
            return True
    raise RuntimeError("git push failed after 3 attempts")


def write_lines(snapshot, out_path=DEFAULT_OUT, push=False, odds=True, client=None, snapshot_path=None):
    doc, hits = build(snapshot, odds=odds, client=client)
    write_json(Path(out_path), doc)
    board = write_board(snapshot)
    try:
        pp = write_pp(snapshot_path)
    except Exception as e:  # noqa: BLE001  (never let the PrizePicks copy break the Underdog hook)
        log.warning("pp_fantasy_lines.json not written: %s", e); pp = False
    pushed = git_push([str(out_path), str(BOARD_LATEST), str(PP_LATEST)]) if push else False
    by_stat = {s: sum(1 for l in doc["lines"] if l["stat"] == s) for s in PUBLISHED_STATS}
    summary = {"lines": len(doc["lines"]), "by_stat": by_stat, "sharp": hits, "sharp_book": doc["sharp_book"], "board_written": board, "pp_written": pp, "pushed": pushed, "path": str(out_path)}
    log.info("lines.json: %(lines)d NBA lines %(by_stat)s, %(sharp)d with sharp odds (%(sharp_book)s), board copy written=%(board_written)s, pp copy written=%(pp_written)s, pushed=%(pushed)s", summary)
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
    s = write_lines(snapshot, out_path=Path(args.out), push=args.push, odds=not args.no_odds, snapshot_path=args.snapshot)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
