"""Pick6 slip builder, pipeline side (2026-11-14). DISPLAY AND LOGGING ONLY: the model's first-3-minutes leaders matched to
tonight's DraftKings Pick6 board. No prices, no edges, no win probabilities: every rate here is history, and the method is
not priced or backtested.

  python pick6_builder.py build    candidate legs for tonight -> docs/data/pick6_tonight.json (run after every board capture)
  python pick6_builder.py record   refresh only the record block of the published file (after the slips are logged)
  python pick6_builder.py grade    morning: grade the logged standard slips from the Opening 3 store, fill the outcomes into
                                   data/pick6_slips.jsonl, publish the record block

The SLIP ASSEMBLY (exclusivity, caps, fill order) is NOT here: it lives once in docs/pick6_builder.js, used by the page's Build
button and by scripts/pick6_slips.cjs (Node) for the standard logged slips.

Inputs: data/pick6_board.json (pick6_capture.py), the Opening 3 store (data/opening3_players.csv, data/opening3_games.csv:
starts only) and tonight's player pool from docs/data/projections.json (the active players of tonight's teams).

Rules as ordered:
  firsts (1st Point, Team 1st FG, 1st Three, Team 1st Three, 1st Dunk): YES only. Leaders are ranked on TONIGHT'S TEAM ONLY:
      by the count of that first with tonight's team; if the best current-team count in the unit is under 3, by per-start rate
      with tonight's team. Career numbers are context, never the rank. Game-level markets have one leader per game, team-level
      markets one per team. Only the leader is ever offered; the leg's rank is his margin over the unit's #2.
  window (Points / Rebounds / Assists 1st 3 Mins): MORE only, default posted line only. Each listed player is a candidate; rank
      and rate = the share of his starts in which he cleared tonight's line. Basis = current team when that sample has
      n >= 15, otherwise career (labelled). Minimum sample n = 20 starts: below it he is not a candidate at all.
  join: a leg exists only if that player + market is on tonight's board, the wanted side is offered, and the card is neither
      paused nor locked. Leaders not listed are counted in "not on Pick6 tonight".

Readings this build chose (not fixed by the order; stamped in the JSON):
  - the per-start-rate fallback needs at least 5 starts with tonight's team (a 1-for-1 "rate" is not a ranking)
  - legs are ordered count-ranked units first (by count margin), then rate-ranked units (by rate margin)
  - a current-team window sample of 15-19 starts is the basis by the first rule but fails the minimum of 20: not a candidate
"""
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, norm_name, read_json, write_json  # noqa: E402
from teams import team_abbr  # noqa: E402

BOARD = DATA_DIR / "pick6_board.json"
SLIPS = DATA_DIR / "pick6_slips.jsonl"
OUT = SITE_DATA / "pick6_tonight.json"
ET = ZoneInfo("America/New_York")
FIRSTS = {"1st Point": ("wg_f_point", "game", "scored the game's first point", "wg_first_point_player"),
          "Team 1st FG": ("wg_f_fg_team", "team", "made his team's first field goal", None),
          "1st Three": ("wg_f_3_game", "game", "made the game's first three", "wg_first_3_game_player"),
          "Team 1st Three": ("wg_f_3_team", "team", "made his team's first three", None),
          "1st Dunk": ("wg_f_dunk_game", "game", "had the game's first dunk", "wg_first_dunk_game_player")}
WINDOW = {"Points 1st 3 Mins": ("win_pts", "points"), "Rebounds 1st 3 Mins": ("win_reb", "rebounds"), "Assists 1st 3 Mins": ("win_ast", "assists")}
MARKETS = list(FIRSTS) + list(WINDOW)
COUNT_MIN, RATE_MIN_STARTS, TEAM_BASIS_N, WINDOW_MIN_N = 3, 5, 15, 20
SIZES = [3, 4, 6, 8]
RANKINGS = ["standard", "value"]          # standard = leader's margin / hit share; value = historical rate x multiplier
PAYOUT_FORMULA = {
    "source": "DraftKings support article KB0010250 'Picks with Multipliers and Standings Points' (v13, 2026-08-25) and the Pick6 page's own entry details for the pick group; see data/pick6_rules.md",
    "base_multiplier": "Base Multiplier = Standard Base Multiplier of the sport and pick tier x the Pick Multiplier of all the picks in the lineup, capped at a Max Base Multiplier",
    "standings_points": "Standings Points = entry fee x the Pick Multipliers of the correct picks; a winner's share of the prize pool is its Standings Points over all winners' Standings Points",
    "label": "Base Payout multiple (DraftKings' guaranteed minimum per $1 if every leg hits; Extra Winnings from the prize pool come on top and cannot be known in advance)",
    "partial_tier_reading": "for a tier with a missed pick (5 of 6, 7 of 8, ...) the article's wording 'all the picks in your lineup' is applied literally; its worked example covers only an all-correct lineup"}
PAYOUT_NOTE = ("Pick6 is a prize pool: winners share it in proportion to Standings Points. The multiplier product shown here is a Standings Points basis, "
               "not a payout and not a price. See data/pick6_rules.md for what is and is not published about payouts.")


def now():
    return dt.datetime.now(dt.timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    s = str(s)
    if "." in s:                                   # DraftKings writes seven fractional digits
        head, rest = s.split(".", 1)
        tz = rest[rest.find("+"):] if "+" in rest else ("+00:00" if rest.endswith("Z") else "")
        s = head + tz
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def starts():
    p = pd.read_csv(DATA_DIR / "opening3_players.csv", dtype={"event_id": str, "player_id": str})
    p = p[p.starter == 1].copy()
    p["key"] = p.player.map(norm_name)
    return p


def build():
    board = read_json(BOARD, None)
    proj = read_json(SITE_DATA / "projections.json", {}) or {}
    record = record_block()
    base = {"title": "Pick6", "generated": iso(now()), "markets": MARKETS, "sizes": SIZES, "rankings": RANKINGS, "record": record, "logged": [],
            "payout_note": PAYOUT_NOTE, "payout": {**PAYOUT_FORMULA, "table": (board or {}).get("payout_table") if isinstance(board, dict) else None},
            "label": "The model's first-3-minutes leaders matched to tonight's Pick6 board. Historical rates only: no prices, no edges, no win probabilities. This method is not priced or backtested.",
            "rules": {"firsts": "leaders ranked on tonight's team only (count; per-start rate when the best current-team count is under 3); career is context", "window": "More only, default line; share of starts clearing tonight's line; current team when n >= 15 else career; minimum n = 20",
                      "caps": "max 1 leg per player, max 2 legs per game; game-level markets one leg per game, team-level markets one leg per team",
                      "readings": [f"per-start-rate ranking needs at least {RATE_MIN_STARTS} starts with tonight's team", "legs ordered count-ranked units first, then rate-ranked units",
                                   "a current-team window sample of 15-19 starts is not a candidate (basis by the first rule, under the minimum of 20)"]}}
    if not board or not board.get("captured_at") or not board.get("complete"):
        write_json(OUT, {**base, "fresh": False, "why_not": "no complete Pick6 board capture on file", "candidates": [], "not_on_pick6": []}); print("[pick6] no complete board: published an empty file"); return
    cards = [c for m in board["markets"].values() for c in m["cards"]]
    tips = [parse(c["start"]) for c in cards if c.get("start")]
    date = min(tips).astimezone(ET).date().isoformat()
    first_tip, last_tip = min(tips), max(tips)
    S = starts()
    pool = {}                                           # ESPN team code -> tonight's active players (our own pool, independent of the board)
    for pl in proj.get("players") or []:
        if not pl.get("inactive") and proj.get("date") == date:
            pool.setdefault(pl["team"], {})[norm_name(pl["name"])] = pl["name"]
    leaders_ok = bool(pool)
    listed = {}                                         # (market, player key, team) -> card
    for name in MARKETS:
        for c in board["markets"][name]["cards"]:
            listed[(name, norm_name(c["player"]), team_abbr(c["team"]))] = c
    games = {}                                          # "DEN @ NOP" -> (away code, home code, start)
    for c in cards:
        if c.get("game") and c["game"] not in games:
            a, h = [team_abbr(x.strip()) for x in c["game"].split("@")]
            games[c["game"]] = (a, h, c["start"])
    cands, missing, notes = [], [], []

    def offered(card, side):
        if not card or card.get("paused") or card.get("locked"):
            return None
        return next((s for s in card["sides"] if s["side"] == side), None)

    # ---- firsts
    for market, (flag, scope, phrase, _) in FIRSTS.items():
        for game, (away, home, start) in games.items():
            units = [[away, home]] if scope == "game" else [[away], [home]]
            for teams in units:
                rows = []
                for t in teams:
                    for key, disp in pool.get(t, {}).items():
                        s = S[(S.key == key) & (S.team == t)]
                        if not len(s):
                            continue
                        car = S[S.key == key]
                        rows.append({"key": key, "player": disp, "team": t, "n": int(len(s)), "c": int(s[flag].sum()), "career_n": int(len(car)), "career_c": int(car[flag].sum())})
                if not rows:
                    continue
                if max(r["c"] for r in rows) >= COUNT_MIN:
                    mode = "count"; rows.sort(key=lambda r: (-r["c"], -(r["c"] / r["n"]), r["player"]))
                    margin = rows[0]["c"] - (rows[1]["c"] if len(rows) > 1 else 0)
                else:
                    mode = "rate"; rows = [r for r in rows if r["n"] >= RATE_MIN_STARTS]
                    if not rows or rows[0]["c"] == 0 and all(r["c"] == 0 for r in rows):
                        continue
                    rows.sort(key=lambda r: (-(r["c"] / r["n"]), -r["n"], r["player"]))
                    margin = round(rows[0]["c"] / rows[0]["n"] - (rows[1]["c"] / rows[1]["n"] if len(rows) > 1 else 0), 4)
                L = rows[0]; second = rows[1] if len(rows) > 1 else None
                card = listed.get((market, L["key"], L["team"]))
                side = offered(card, "Yes")
                unit = game if scope == "game" else f"{L['team']} ({game})"
                if not side:
                    missing.append({"market": market, "player": L["player"], "team": L["team"], "game": game,
                                    "why": "not listed" if not card else "paused" if card.get("paused") else "locked" if card.get("locked") else "no Yes side offered"})
                    continue
                cands.append({"market": market, "type": "first", "scope": scope, "unit": unit, "player": card["player"], "player_key": L["key"], "team": L["team"], "game": game, "start": start,
                              "side": "Yes", "line": None, "multiplier": side["multiplier"], "rate": round(L["c"] / L["n"], 4), "hits": L["c"], "n": L["n"], "basis": f"with {L['team']}",
                              "rank_mode": mode, "margin": margin, "second": (f"{second['player']} {second['c']} of {second['n']}" if second else None),
                              "text": f"{phrase} in {L['c']} of {L['n']} starts with {L['team']}", "career": f"career in the dataset: {L['career_c']} of {L['career_n']} starts"})
    # ---- window
    for market, (col, word) in WINDOW.items():
        for c in board["markets"][market]["cards"]:
            key, t = norm_name(c["player"]), team_abbr(c["team"])
            side = offered(c, "More")
            if not side or c.get("line") is None:
                continue                                 # a side that is not offered is never built
            team_s = S[(S.key == key) & (S.team == t)]; car = S[S.key == key]
            s, basis = (team_s, f"with {t}") if len(team_s) >= TEAM_BASIS_N else (car, "career in the dataset")
            if len(s) < WINDOW_MIN_N:
                continue
            hits = int((s[col] > float(c["line"])).sum()); n = int(len(s))
            cands.append({"market": market, "type": "window", "scope": "player", "unit": c["player"], "player": c["player"], "player_key": key, "team": t, "game": c["game"], "start": c["start"],
                          "side": "More", "line": c["line"], "multiplier": side["multiplier"], "rate": round(hits / n, 4), "hits": hits, "n": n, "basis": basis,
                          "rank_mode": "share", "margin": round(hits / n, 4), "second": None,
                          "text": f"over {c['line']} {word} in the first 3 minutes in {hits} of {n} starts ({basis})", "career": f"career in the dataset: {int((car[col] > float(c['line'])).sum())} of {int(len(car))} starts"})
    # ---- order inside each market: firsts count-ranked units first by count margin, then rate-ranked by rate margin; window by share
    def order(c):
        return (0 if c["rank_mode"] == "count" else 1 if c["rank_mode"] == "rate" else 0, -float(c["margin"]), -c["rate"], c["player"])
    out = []
    for m in MARKETS:
        ms = [x for x in cands if x["market"] == m]
        for c in ms:
            c["value"] = round(c["rate"] * float(c["multiplier"]), 4)       # historical rate x multiplier (not a payout, not an edge)
        by_value = {id(c): i for i, c in enumerate(sorted(ms, key=lambda c: (-c["value"], -c["rate"], c["player"])))}
        for i, c in enumerate(sorted(ms, key=order)):
            c["order"] = i; c["order_value"] = by_value[id(c)]; c["id"] = f"{m}|{c['player_key']}|{c['team']}"; out.append(c)
    doc = {**base, "date": date, "board_captured_at": board["captured_at"], "first_tip": iso(first_tip), "last_tip": iso(last_tip),
           "leaders_date": proj.get("date"), "fresh": bool(leaders_ok), "why_not": "" if leaders_ok else f"tonight's leaders are missing (the projection pool is dated {proj.get('date')}, the board {date})",
           "games": [{"game": g, "start": v[2]} for g, v in games.items()], "board_summary": board.get("summary"),
           "logged": logged_for(date), "candidates": out, "not_on_pick6": missing, "counts": {m: sum(1 for c in out if c["market"] == m) for m in MARKETS}}
    write_json(OUT, doc)
    print(f"[pick6] {date}: board {board['captured_at']}, {len(out)} candidate legs {doc['counts']}, {len(missing)} leaders not on Pick6 tonight, fresh={doc['fresh']} {doc['why_not']}")
    return doc


# ---------------------------------------------------------------- grading and the record

def read_slips():
    if not SLIPS.exists():
        return []
    return [json.loads(l) for l in open(SLIPS, encoding="utf-8") if l.strip()]


def standard(slips):
    """The record's slips: per (date, size, ranking), the LAST record built from a board captured before the slate's first tip.
    Records written before 2026-11-14's two-ranking change carry no ranking: they are the standard ranking."""
    best = {}
    for i, s in enumerate(slips):
        if s.get("board_captured_at") and s.get("first_tip") and s["board_captured_at"] < s["first_tip"]:
            best[(s["date"], s["size"], s.get("ranking") or "standard")] = i
    return sorted(best.values())


def grade():
    """Grade per data/pick6_rules.md. A void leg (did not start / minimum play unmet) does NOT void the slip: the slip drops a
    level and is graded on its non-void legs, flagged `reduced`. A slip left with 0 or 1 non-void legs is a REFUND: excluded
    from the hit and return tallies and counted separately. no_event legs keep the labelled convention (counted as a miss)."""
    slips = read_slips()
    if not slips:
        print("[pick6] no logged slips yet"); build_record_only(); return
    P = pd.read_csv(DATA_DIR / "opening3_players.csv", dtype={"event_id": str, "player_id": str}); P["key"] = P.player.map(norm_name)
    G = pd.read_csv(DATA_DIR / "opening3_games.csv", dtype={"event_id": str})
    n = 0
    for i in standard(slips):
        s = slips[i]
        if s.get("graded"):
            continue
        day = P[P.date == s["date"]]; games = G[G.date == s["date"]]
        need = {tuple(sorted(team_abbr(x.strip()) for x in l["game"].split("@"))) for l in s["legs"]}
        have = {tuple(sorted((r.home, r.away))) for r in games.itertuples()}
        if not need <= have:
            continue                                     # the store does not hold every game of the slip yet
        for l in s["legs"]:
            row = day[(day.key == l["player_key"]) & (day.team == l["team"])]
            if not len(row) or int(row.iloc[0].starter) != 1:
                l["outcome"] = "void"; l["note"] = "did not start"; continue
            r = row.iloc[0]
            if l["market"] in FIRSTS:
                flag, _, _, gcol = FIRSTS[l["market"]]
                g = games[games.event_id == r.event_id].iloc[0]
                if gcol and (pd.isna(g[gcol]) or str(g[gcol]) == ""):
                    l["outcome"] = "miss"; l["note"] = "no_event: the game had none (counted as a miss, rule not published)"
                else:
                    l["outcome"] = "hit" if int(r[flag]) == 1 else "miss"
            else:
                val = float(r[WINDOW[l["market"]][0]]); l["actual"] = val
                l["outcome"] = "hit" if val > float(l["line"]) else "miss"
        live = [l for l in s["legs"] if l["outcome"] != "void"]
        voids = len(s["legs"]) - len(live)
        refund = len(live) <= 1
        fully = (not refund) and all(l["outcome"] == "hit" for l in live)
        mult = 1.0
        for l in live:
            mult *= float(l["multiplier"])
        base, basis = base_payout(s.get("payout_table"), live)
        s["graded"] = {"at": iso(now()), "base_payout_multiple": base, "base_payout_basis": basis, "hits": sum(l["outcome"] == "hit" for l in live), "live_legs": len(live), "void_legs": voids,
                       "status": "refund" if refund else "graded", "reduced": bool(voids and not refund), "graded_size": None if refund else len(live),
                       "fully_hit": fully, "multiplier_product_live": None if refund else round(mult, 4),
                       "multiplier_product_returned": round(mult, 4) if fully else 0.0, "units_staked": 0 if refund else 1}
        n += 1
    if n:
        with open(SLIPS, "w", encoding="utf-8", newline="\n") as f:
            for s in slips:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[pick6] graded {n} standard slips")
    build_record_only()


def base_payout(table, live):
    """DraftKings' Base Payout multiple for a graded slip, per the published formula, at its REDUCED size (non-void legs):
    Standard Base Multiplier of the tier (pick level = non-void legs, correct = hits) x the Pick Multipliers of the lineup's
    picks, capped at the tier's Max Base Multiplier. (0.0, 'no tier') when the finish is not a paying tier; (None, ...) when the
    slip was logged without a payout table or is a refund."""
    if not table or len(live) <= 1:
        return None, "refund" if len(live) <= 1 else "no payout table on the logged slip"
    hits = sum(l["outcome"] == "hit" for l in live)
    tier = next((t for t in table.get(str(len(live))) or [] if int(t["correct"]) == hits), None)
    if not tier:
        return 0.0, "no paying tier"
    m = float(tier["standard_multiplier"])
    for l in live:
        m *= float(l["multiplier"])
    return round(min(m, float(tier["max_base_multiplier"])), 4), ("all correct" if hits == len(live) else "partial tier (literal reading of the published formula)")


def tally(std):
    g = [s for s in std if s.get("graded")]
    live = [s for s in g if (s["graded"].get("status") or "graded") != "refund" and s["graded"].get("live_legs", 0) > 1]
    refunds = len(g) - len(live)
    by = {}
    for s in live:
        for l in s["legs"]:
            b = by.setdefault(l["market"], {"hit": 0, "miss": 0, "void": 0}); b[l["outcome"]] += 1

    def returned(s):
        gr = s["graded"]
        if "multiplier_product_returned" in gr:
            return float(gr["multiplier_product_returned"])
        if not gr.get("fully_hit"):
            return 0.0
        m = 1.0
        for l in s["legs"]:
            if l.get("outcome") != "void":
                m *= float(l["multiplier"])
        return m
    return {"slips_built": len(std), "slips_graded": len(live), "slips_pending": len(std) - len(g), "refunds": refunds,
            "reduced": sum(1 for s in live if s["graded"].get("reduced") or s["graded"].get("void_legs")), "slips_fully_hit": sum(bool(s["graded"]["fully_hit"]) for s in live),
            "units_staked": len(live), "multiplier_product_returned": round(sum(returned(s) for s in live), 2),
            "base_payout_returned": round(sum(float(s["graded"].get("base_payout_multiple") or 0) for s in live), 2),
            "base_payout_from_partial_tiers": round(sum(float(s["graded"].get("base_payout_multiple") or 0) for s in live if str(s["graded"].get("base_payout_basis", "")).startswith("partial")), 2),
            "slips_without_payout_table": sum(1 for s in live if s["graded"].get("base_payout_multiple") is None),
            "by_size": {str(z): {"graded": sum(1 for s in live if s["size"] == z), "fully_hit": sum(1 for s in live if s["size"] == z and s["graded"]["fully_hit"])} for z in SIZES},
            "leg_hit_rate_by_market": {m: {**v, "rate": round(v["hit"] / (v["hit"] + v["miss"]), 4) if v["hit"] + v["miss"] else None} for m, v in by.items()}}


def record_block():
    slips = read_slips(); std = [slips[i] for i in standard(slips)]
    out = {"label": "Logged standard slips only (sizes 3/4/6/8 under both rankings, all markets on, last pre-tip board). Custom builds in the browser are not logged or graded.",
           "return_label": "multiplier product returned = the sum, over fully-hit slips, of their multiplier product at a flat 1 unit per slip. A Standings Points basis, not a payout.",
           "base_payout_label": "Base Payout returned = the sum of DraftKings' guaranteed Base Payout multiples of the graded slips at 1 unit each, by the published formula (data/pick6_rules.md). Extra Winnings from the prize pool are not included: they depend on the other winners.",
           "void_rule": "A void leg drops the slip a level: it is graded on its non-void legs (reduced). A slip left with 0 or 1 non-void legs is a refund, counted separately.",
           "overall": tally(std), "by_ranking": {r: tally([s for s in std if (s.get("ranking") or "standard") == r]) for r in RANKINGS}}
    out.update({k: out["overall"][k] for k in ("slips_built", "slips_graded", "slips_pending", "slips_fully_hit", "leg_hit_rate_by_market")})   # the fields the first tab version read
    g = [s for s in std if s.get("graded")]
    out["recent"] = [{"date": s["date"], "size": s["size"], "ranking": s.get("ranking") or "standard", "status": s["graded"].get("status") or "graded", "hits": s["graded"]["hits"],
                      "live_legs": s["graded"]["live_legs"], "void_legs": s["graded"]["void_legs"], "fully_hit": s["graded"]["fully_hit"]} for s in g[-16:]]
    return out


def logged_for(date):
    """Tonight's logged standard slips (the last record per size and ranking), for display next to the built slip."""
    slips = read_slips(); best = {}
    for s in slips:
        if s["date"] == date:
            best[(s.get("ranking") or "standard", s["size"])] = s
    out = []
    for (ranking, size), s in sorted(best.items()):
        rate = 1.0; mult = 1.0
        for l in s["legs"]:
            rate *= float(l["rate"]); mult *= float(l["multiplier"])
        tier = next((t for t in ((s.get("payout_table") or {}).get(str(len(s["legs"]))) or []) if int(t["correct"]) == len(s["legs"])), None)
        out.append({"base_multiple": round(min(float(tier["standard_multiplier"]) * mult, float(tier["max_base_multiplier"])), 4) if tier else None,
                    "ranking": ranking, "size": size, "built_at": s.get("built_at"), "board_captured_at": s.get("board_captured_at"), "complete": s.get("complete"),
                    "multiplier_product": round(mult, 4), "breakeven_multiple": round(1 / rate, 2) if rate > 0 else None, "graded": s.get("graded"),
                    "legs": [{**l, "value": l.get("value", round(float(l["rate"]) * float(l["multiplier"]), 4))} for l in s["legs"]]})
    return out


def build_record_only():
    doc = read_json(OUT, None)
    if doc:
        doc["record"] = record_block(); doc["logged"] = logged_for(doc.get("date")) if doc.get("date") else []; doc["generated"] = iso(now()); write_json(OUT, doc)
    else:
        build()


if __name__ == "__main__":
    {"build": build, "grade": grade, "record": build_record_only}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
