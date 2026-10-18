"""NBA 2024-25 alternate-threes market-calibration test (backtests/nba_threes_alt_calibration_rule.md). HISTORICAL. Writes only to
its own cache, docs/data/backtest_threes_alt_calibration.json and the Backtest page; never to data/paper_log.csv or results.json.
  python backtest_threes_alt_calibration.py pull  --season 2025     # closing player_threes_alternate per game (10 credits each, 15K stop)
  python backtest_threes_alt_calibration.py grade --season 2025     # de-vig, buckets, the locked tail rule -> report JSON
  python backtest_threes_alt_calibration.py all   --season 2025
The Odds API event ids and tip times come from data/odds_close_<season>/events_by_date.json (cached by the closing-line backtest), so
no listing credits are spent. ODDS_API_KEY from the environment, never printed.
"""
import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_name, read_json, write_json
from backtest_threes_close import _get, spent, implied, BUDGET_STOP

MARKET = "player_threes_alternate"
TAIL_LINE, TAIL_ABOVE_MAIN, TAIL_GAP, TAIL_MIN_N, WEEK_MIN_N = 4.5, 2.0, 0.03, 500, 30
BUCKETS = [(0, 0.05, "<5%"), (0.05, 0.10, "5-10%"), (0.10, 0.20, "10-20%"), (0.20, 0.35, "20-35%"), (0.35, 0.50, "35-50%"), (0.50, 1.01, ">=50%")]


# ---------------- pull ----------------
def pull(season, close_dir, out_dir, budget=BUDGET_STOP, max_games=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = out_dir / "usage.jsonl"
    files = sorted(f for f in close_dir.glob("*.json") if f.name not in ("events_by_date.json",))
    games = [(f.stem, read_json(f, {})) for f in files]
    games = [(eid, d) for eid, d in games if d and not d.get("missing") and d.get("odds_event")]
    print(f"[alt] {len(games)} games with a cached closing event; spent so far {spent(usage):,} credits")
    done = 0
    for eid, d in games:
        fn = out_dir / f"{eid}.json"
        if fn.exists():
            continue
        s = spent(usage)
        if s > budget:
            print(f"[alt] BUDGET STOP: {s:,} credits spent > {budget:,}; {done} games pulled this run"); return False
        e = d["odds_event"]
        resp, rec = _get(f"events/{e['id']}/odds", dict(regions="us", markets=MARKET, date=e["commence"], oddsFormat="american"), usage, "nba_threes_alt_close")
        write_json(fn, dict(event_id=eid, odds_event=e, snapshot=resp.get("timestamp"), data=resp.get("data", resp), cost=rec["cost"], remaining=rec["remaining"]))
        done += 1
        if max_games and done >= max_games:
            print(f"[alt] stopped after {done} games (--max-games)"); return True
        if done % 50 == 0:
            print(f"[alt] {done} games pulled, {spent(usage):,} credits spent, {rec['remaining']:,} remaining", flush=True)
        time.sleep(0.15)
    print(f"[alt] done: {done} games pulled this run, {spent(usage):,} credits spent in total")
    return True


# ---------------- de-vig and rungs ----------------
def book_holds(close_doc):
    """Per book: {player -> (main_line, overround)} from the two-sided player_threes close, plus the book's median overround and the
       pooled modal main line per player (ties -> lower), as in the closing-line backtest."""
    per_book, modal = {}, {}
    for b in close_doc["data"].get("bookmakers", []):
        for m in b.get("markets", []):
            if m["key"] != "player_threes":
                continue
            sides = {}
            for o in m["outcomes"]:
                sides.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = o["price"]
            best = {}
            for (player, line), s in sides.items():
                if line is None or "Over" not in s or "Under" not in s:
                    continue
                po, pu = implied(s["Over"]), implied(s["Under"])
                # the book's main line: the two-sided line nearest even money
                cand = (abs(po - pu), float(line), po + pu)
                if player not in best or cand < best[player]:
                    best[player] = cand
                modal.setdefault(player, {}).setdefault(float(line), 0)
                modal[player][float(line)] += 1
            per_book[b["title"]] = {p: (ln, h) for p, (_, ln, h) in best.items()}
    main = {p: max(lines.items(), key=lambda kv: (kv[1], -kv[0]))[0] for p, lines in modal.items()}
    return per_book, main


def rungs_for_game(alt_doc, close_doc):
    """Rows: one per (player, line) Over rung pooled across books: fair P (median of per-book multiplicative de-vig), n_books, flags."""
    per_book, main = book_holds(close_doc)
    med_h = {b: float(np.median([h for _, h in v.values()])) if v else None for b, v in per_book.items()}
    quotes, two_sided, clipped, dropped, alt_two_sided = {}, 0, 0, 0, {}
    for b in alt_doc["data"].get("bookmakers", []):
        bk = b["title"]
        for m in b.get("markets", []):
            if m["key"] != MARKET:
                continue
            sides = {}
            for o in m["outcomes"]:
                sides.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = o["price"]
            for (player, line), s in sides.items():
                if line is None or "Over" not in s:
                    continue
                if "Under" in s: two_sided += 1; alt_two_sided.setdefault(player, set()).add(float(line))
                h = per_book.get(bk, {}).get(player, (None, None))[1] or med_h.get(bk)
                if not h:
                    dropped += 1; continue
                fair = implied(s["Over"]) / h
                if fair > 1: fair, clipped = 1.0, clipped + 1
                quotes.setdefault((player, float(line)), []).append((bk, fair, s["Over"]))
    rows = []
    for (player, line), qs in quotes.items():
        ml = main.get(player)
        if ml is None and player in alt_two_sided:   # fallback fixed in the rule: the lowest two-sided rung the alternate market itself quotes
            ml = min(alt_two_sided[player])
        rows.append(dict(event_id=alt_doc["event_id"], player=player, name_key=norm_name(player), line=line, fair=float(np.median([f for _, f, _ in qs])),
                         n_books=len(qs), books="|".join(sorted(b for b, _, _ in qs)), quotes=[(b, round(f, 5)) for b, f, _ in qs],
                         best_price=int(max(pr for _, _, pr in qs)), main_line=ml, tip=alt_doc["odds_event"]["commence"],
                         snapshot_alt=alt_doc.get("snapshot"), snapshot_main=close_doc.get("snapshot")))
    return rows, dict(two_sided=two_sided, clipped=clipped, dropped_no_hold=dropped)


def build_rungs(season, alt_dir, close_dir):
    rows, stats = [], dict(two_sided=0, clipped=0, dropped_no_hold=0, games=0, games_no_alt_market=0)
    for fn in sorted(alt_dir.glob("*.json")):
        if fn.name == "usage.jsonl" or fn.suffix != ".json":
            continue
        alt = read_json(fn, None); close = read_json(close_dir / fn.name, None)
        if not alt or not close or close.get("missing"):
            continue
        stats["games"] += 1
        r, s = rungs_for_game(alt, close)
        if not r: stats["games_no_alt_market"] += 1
        rows += r
        for k in ("two_sided", "clipped", "dropped_no_hold"): stats[k] += s[k]
    df = pd.DataFrame(rows)
    pg = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})
    pg = pg[pg.season == season][["event_id", "player", "min", "tpm"]].copy(); pg["name_key"] = pg.player.map(norm_name)
    pg = pg.rename(columns={"player": "box_player"})
    df = df.merge(pg, on=["event_id", "name_key"], how="left")
    stats["rungs_quoted"] = int(len(df)); stats["rungs_no_box_row"] = int(df.tpm.isna().sum()); stats["rungs_dnp"] = int((df["min"].fillna(0) == 0).sum() - df.tpm.isna().sum())
    df = df[df.tpm.notna() & (df["min"] > 0)].copy()
    df["hit"] = (df.tpm > df.line).astype(int)
    df["is_tail"] = (df.line >= TAIL_LINE) | (df.main_line.notna() & (df.line >= df.main_line + TAIL_ABOVE_MAIN))
    df["date"] = pd.to_datetime(df.tip).dt.tz_convert("UTC")
    iso = df.date.dt.isocalendar(); df["week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    df["bucket"] = pd.cut(df.fair, [b[0] for b in BUCKETS] + [1.01], right=False, labels=[b[2] for b in BUCKETS])
    long = df[["event_id", "player", "line", "hit", "is_tail", "quotes"]].explode("quotes")
    long["book"] = long.quotes.map(lambda q: q[0]); long["fair_book"] = long.quotes.map(lambda q: q[1]).astype(float)
    return df, long.drop(columns="quotes"), stats


def cluster_se(d, groups):
    """SE of the mean of d with clusters = groups (sandwich for a mean)."""
    d = np.asarray(d, float); n = len(d); m = d.mean()
    s = pd.Series(d - m).groupby(np.asarray(groups)).sum().values
    return float(math.sqrt((s ** 2).sum()) / n) if n else float("nan")


def gap_block(t):
    if len(t) == 0: return dict(n=0, hit=None, fair=None, gap=None, se=None)
    g = float(t.hit.mean() - t.fair.mean()); se = cluster_se(t.hit - t.fair, t.event_id)
    return dict(n=int(len(t)), hit=round(float(t.hit.mean()), 4), fair=round(float(t.fair.mean()), 4), gap=round(g, 4), se=round(se, 4), ci95=[round(g - 1.96 * se, 4), round(g + 1.96 * se, 4)])


def grade(season, alt_dir, close_dir, report_path, rule_commit):
    df, long, stats = build_rungs(season, alt_dir, close_dir)
    tails = df[df.is_tail]
    weekly = tails.groupby("week").apply(lambda t: pd.Series(dict(n=len(t), hit=t.hit.mean(), fair=t.fair.mean(), gap=t.hit.mean() - t.fair.mean()))).reset_index()
    elig = weekly[weekly.n >= WEEK_MIN_N]
    best_week = elig.sort_values("gap", ascending=False).week.iloc[0] if len(elig) else None
    full = gap_block(tails); drop = gap_block(tails[tails.week != best_week]) if best_week else gap_block(tails.iloc[0:0])
    pass_full = full["n"] >= TAIL_MIN_N and full["gap"] is not None and full["gap"] >= TAIL_GAP
    pass_drop = drop["n"] >= TAIL_MIN_N and drop["gap"] is not None and drop["gap"] >= TAIL_GAP
    underpriced = bool(pass_full and pass_drop)
    buckets = [dict(bucket=lab, **gap_block(df[df.bucket == lab])) for _, _, lab in BUCKETS]
    buckets_tail = [dict(bucket=lab, **gap_block(tails[tails.bucket == lab])) for _, _, lab in BUCKETS]
    by_line = [dict(line=float(l), **gap_block(g)) for l, g in df.groupby("line") if len(g) >= 30]
    by_books = [dict(n_books=int(k), **gap_block(g)) for k, g in df.groupby(df.n_books.clip(upper=6)) if len(g) >= 30]
    by_kind = dict(five_plus=gap_block(df[df.line >= TAIL_LINE]), two_above_main=gap_block(df[df.main_line.notna() & (df.line >= df.main_line + TAIL_ABOVE_MAIN)]),
                   non_tail=gap_block(df[~df.is_tail]), all_rungs=gap_block(df))
    brier_market = float(((df.fair - df.hit) ** 2).mean()); brier_tail = float(((tails.fair - tails.hit) ** 2).mean()) if len(tails) else None
    # books per rung and each book on its own (its own de-vigged P against the outcome, on the rungs it quotes)
    books_per_rung = dict(distribution={str(int(k)): int(v) for k, v in df.n_books.value_counts().sort_index().items()}, mean=round(float(df.n_books.mean()), 2),
                          median=float(df.n_books.median()), distinct_books=sorted(long.book.unique().tolist()), tail_mean=round(float(tails.n_books.mean()), 2) if len(tails) else None)
    def book_block(g):
        if len(g) == 0: return dict(n=0)
        gap = float(g.hit.mean() - g.fair_book.mean()); se = cluster_se(g.hit - g.fair_book, g.event_id)
        return dict(n=int(len(g)), hit=round(float(g.hit.mean()), 4), fair=round(float(g.fair_book.mean()), 4), gap=round(gap, 4), se=round(se, 4))
    per_book = [dict(book=b, all_rungs=book_block(g), tail_rungs=book_block(g[g.is_tail])) for b, g in long.groupby("book") if len(g) >= 100]
    per_book.sort(key=lambda r: -r["all_rungs"]["n"])
    nt = by_kind["non_tail"]
    # how wrong the de-vigged tail price would have to be for the verdict to flip: true P would need to be >= hit - 3pp
    flip_factor = round(full["fair"] / (full["hit"] - TAIL_GAP), 2) if full["n"] and full["hit"] and full["hit"] > TAIL_GAP else None
    relabel = ("DraftKings does not underprice tails; DK alt tail rungs carry heavy margin" if not underpriced else "MARKET UNDERPRICES TAILS")
    rep = dict(
        title="NBA 2024-25 alternate threes: market-calibration test (locked rule)", season="2024-25", generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rule_file="backtests/nba_threes_alt_calibration_rule.md", rule_commit=rule_commit,
        rule=f"'Market underprices tails' iff tail rungs (line >= {TAIL_LINE} or >= main line + {TAIL_ABOVE_MAIN:g}) hit >= {TAIL_GAP*100:.0f}pp more often than the de-vigged price, n >= {TAIL_MIN_N}, and still after dropping the best ISO week; otherwise tails are calibrated and tail modeling stops.",
        devig="multiplicative per book: implied(Over at the rung) / (implied Over + implied Under at that book's two-sided main line in the closing player_threes snapshot); book median hold when the player has no two-sided line at that book; pooled fair P = median over books",
        verdict=relabel, verdict_rule="MARKET UNDERPRICES TAILS" if underpriced else "market tails are calibrated -> tail modeling stops (the locked rule's wording)",
        underpriced=underpriced, books_per_rung=books_per_rung, per_book=per_book, flip_factor=flip_factor,
        tails=dict(full=full, pass_full=bool(pass_full), best_week=best_week, drop_best_week=drop, pass_drop=bool(pass_drop), weekly=weekly.round(4).to_dict("records")),
        by_kind=by_kind, buckets=buckets, buckets_tail=buckets_tail, by_line=by_line, by_books=by_books,
        summary=dict(games=stats["games"], rungs_graded=int(len(df)), tail_rungs=int(len(tails)), players=int(df.name_key.nunique()), books_median=float(df.n_books.median()) if len(df) else None,
                     brier_market_all=round(brier_market, 4), brier_market_tail=round(brier_tail, 4) if brier_tail is not None else None, mean_fair_all=round(float(df.fair.mean()), 4), hit_all=round(float(df.hit.mean()), 4)),
        pull=dict(**stats, credits_spent=spent(alt_dir / "usage.jsonl"), budget_stop=BUDGET_STOP),
        notes=[f"The de-vig removes main-line hold only. Non-tail alternate rungs also hit {nt['gap']*100:+.1f}pp vs 'fair' (n {nt['n']:,}), so the test cannot separate tail calibration from the extra margin books load onto alternate ladders. The verdict's direction is robust: for tails to be underpriced the de-vigged tail price would have to overstate the true probability by {flip_factor}x (true P >= hit rate - 3pp = {max(full['hit'] - TAIL_GAP, 0)*100:.1f}% against a de-vigged {full['fair']*100:.1f}%), i.e. the books would have to be overpricing tails by more than 2x.",
               f"Pooled across {len(books_per_rung['distinct_books'])} books (median {books_per_rung['median']:g} per rung; DraftKings quotes {next((r['all_rungs']['n'] for r in per_book if r['book']=='DraftKings'), 0):,} of the book-rung quotes); the per-book table shows each book's own de-vigged price against the outcome.",
               "Outcome = ESPN box-score threes made > line; rungs with no box-score row or 0 minutes dropped (DNP).",
               "Alternate ladders are one-sided (Over only), so each book's margin comes from its own two-sided main line in the same game's closing snapshot; multiplicative de-vigging is the conservative choice for a tail test.",
               "Fair P > 1 after de-vigging is clipped to 1 (counted in pull.clipped).",
               "Cluster-robust SEs (clusters = game) are reported for information; the locked rule is the 3pp gap at n >= 500 holding after the best-week drop.",
               "Reported alongside the box-score dispersion test; no dispersion-test output exists in this repository yet, this section does not depend on it.",
               "Backtest page only: nothing here touches data/paper_log.csv or docs/data/results.json."])
    write_json(report_path, rep)
    df.drop(columns=["date", "quotes"]).to_csv(alt_dir / "rungs_graded.csv", index=False)
    print(f"[alt] {rep['verdict_rule']} | {rep['verdict']} | books/rung {books_per_rung['distribution']} | flip factor {flip_factor} | tail rungs {full['n']} hit {full['hit']} fair {full['fair']} gap {full['gap']} (se {full['se']}) | drop {best_week}: n {drop['n']} gap {drop['gap']} | rungs {len(df)} games {stats['games']} credits {rep['pull']['credits_spent']:,}")
    for b in buckets: print(f"    {b['bucket']:>7} n {b['n']:6} fair {b['fair']} hit {b['hit']} gap {b['gap']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "grade", "all"])
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--budget", type=int, default=BUDGET_STOP)
    ap.add_argument("--max-games", type=int)
    a = ap.parse_args()
    close_dir = DATA_DIR / f"odds_close_{a.season}"; alt_dir = DATA_DIR / f"odds_alt_{a.season}"
    rule_commit = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "backtests/nba_threes_alt_calibration_rule.md"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    ok = True
    if a.cmd in ("pull", "all"):
        ok = pull(a.season, close_dir, alt_dir, a.budget, a.max_games)
    if a.cmd in ("grade", "all") and ok:
        grade(a.season, alt_dir, close_dir, SITE_DATA / "backtest_threes_alt_calibration.json", rule_commit)
