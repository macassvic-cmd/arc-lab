"""Sam Method, Part 2: the shadow optimizer (CEO order 2026-11-12, optimizer spec v5 + Amendment A1). PRE-GATE SHADOW, PAPER ONLY.
Reads one capture (sam_capture.py), runs exactly ONE search per stat, logs the selections, re-polls their asks at +5 minutes,
grades them nightly from the box scores and publishes docs/data/sam_method.json for the page tab. It reads the ladder model's
published numbers; it writes nothing the ladder, FP, first-basket or any existing tab reads.

Rules as locked in the order (kept here so the file is self-contained):
  universe     Kalshi markets of the capture whose game had not tipped, per side, passing the DEPTH RULE at capture: the side's
               ask exists, ask size >= 100 contracts, yes spread (yes_ask - yes_bid) <= $0.10
  model P      Kalshi "X or more" = the ladder model's LIVE PUBLISHED P(stat >= X) for every stat (backtests/sam_method_guard_rule.md
               section 1, CEO ruling 2026-11-12). A recalibration map in shadow prices nothing: the Part A points map stays out
               until it ships, and then arrives through the published P. A strike that is not a ladder rung cannot be priced:
               skipped, counted.
  guard_v1     registered before the first counted run (same rule file, section 3): a candidate is excluded when its claimed edge
               exceeds 10.0 points, or when the player's projected minutes moved more than 3.0 from the slate's morning build
               (or cannot be checked). Counted per run; excluded legs are never logged.
  fee          taker fee per contract = 0.07 x ask x (1 - ask); fee-adjusted break-even = ask + fee, per side
  candidate    the side's P beats its own fee-adjusted break-even by >= 3 points. Both sides are searched.
  eligibility  (i) >= 20 current-season games, same team, role stable; or (ii) >= 41 prior-season games with the same team and
               >= 10 current games confirming the role (minutes within 5, field-goal attempts within 20%). Everyone else
               (rookies, team-changers, role-changers) is unselectable. The base rate of a rung needs >= 30 games and >= 3
               observed hits; rate used = max(observed, half the league rate of that rung-stat).
  tail flags   a Yes leg is excluded when P(yes) >= 1.75 x the usable base rate; a No leg when P(yes) <= base rate / 1.75
  fixed runs   one run per stat per slate; at most the top 3 player-game-stat clusters by per-leg z = edge / sqrt(BE (1 - BE)).
               A cluster counts once: its largest-edge leg is kept, its other candidate legs are logged as duplicates.
               No re-runs: a slate that already has runs in the log is never searched again.

Readings this build had to choose (the spec text does not fix them; each is stamped in the published JSON):
  - model_p_raw / model_p_shrunk are the probability OF THE LOGGED SIDE (for a No leg, 1 - P(yes)), so P, break-even and hit are
    all in the same direction. model_p_raw = the ladder's published P; model_p_shrunk = the P the search used.
  - "role stable" in clause (i) = the last 10 games against the earlier ones, by the same minutes / attempts bands as clause (ii).
  - base-rate games = the player's games (minutes > 0) with tonight's team, prior season + current season. League rate = the
    share of all player-games (minutes > 0) in those two seasons at or above the rung.
  - a rung with no usable base rate (under 30 games or under 3 hits) cannot be tail-checked and is excluded, counted.
  - ask_survived = the side's ask at +5 minutes is still at or below the decision ask.
  - the 35 locked columns have no off_schedule field: an off-schedule run carries it in run_id (suffix -offsched) and in
    data/sam_method/runs_log.csv; such rows are excluded from every gate count.
  - grading fills only hit / scratch_status / status on the logged row; no other field of a logged row ever changes.

  python sam_optimizer.py run <capture.json.gz> [--no-wait] [--out-root DIR]   search, +5 re-poll, log, publish
  python sam_optimizer.py grade                                               nightly: grade pending rows, publish
  python sam_optimizer.py publish                                             rebuild docs/data/sam_method.json only
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SITE_DATA, read_json, write_json  # noqa: E402
import sam_capture as sc  # noqa: E402

STATS = ["pts", "reb", "ast", "tpm"]
LABEL = {"pts": "PTS", "reb": "REB", "ast": "AST", "tpm": "3PM"}
MIN_SIZE, MAX_SPREAD, EDGE_FLOOR, FEE, TAIL, TOP = 100.0, 0.10, 0.03, 0.07, 1.75, 3
METHOD_VERSION, P_SOURCE = "sam_v1+guard_v1", "live_published_p"   # backtests/sam_method_guard_rule.md; a change = a version bump and a new record
GUARD_EDGE, GUARD_MINUTES = 0.10, 3.0
A_FIT, A_CHECK = 14, 7
COLS = ["date", "venue", "run_id", "cluster_id", "stat", "player", "team", "game_id", "rung", "side", "model_p_raw", "model_p_shrunk",
        "calibrated_book_value", "decision_ts", "reference_price", "reference_source", "kalshi_mid", "ask", "ask_size_fillable", "spread",
        "fee_per_contract", "fee_adj_breakeven", "claimed_edge_pp", "ranking_z", "candidates_searched", "base_rate_eligible",
        "base_rate_clause", "tail_claim_flag", "tail_claim_reason", "duplicate_of", "ask_plus5", "ask_survived", "scratch_status", "hit", "status"]
assert len(COLS) == 35
RUN_COLS = ["date", "run_id", "stat", "off_schedule", "decision_ts", "capture_at", "markets", "tipped_excluded", "side_legs", "depth_pass", "unpriced_rung",
            "no_model_player", "candidates_searched", "candidates", "ineligible_player", "no_base_rate", "tail_excluded", "clusters_selected", "legs_logged", "note",
            "method_version", "p_source", "guard_edge", "guard_minutes"]
ROOT_DATA, ROOT_SITE = DATA_DIR, SITE_DATA


def paths():
    d = ROOT_DATA / "sam_method"
    return {"dir": d, "log": d / "shadow_log.csv", "runs": d / "runs_log.csv", "capture_log": d / "capture_log.csv", "his": d / "his_record.json",
            "book_gap": d / "book_gap_log.csv", "out": ROOT_SITE / "sam_method.json"}


def read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame()


def append(p, cols, rows):
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows([{c: ("" if r.get(c) is None else r.get(c)) for c in cols} for r in rows])


def fee(ask):
    return FEE * ask * (1 - ask)


# ---------------------------------------------------------------- base rates and eligibility

class Box:
    def __init__(self):
        g = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})
        g = g[g["min"] > 0].sort_values(["date", "event_id"])
        self.cur = int(g.season.max()); self.g = g[g.season >= self.cur - 1]
        self.by = {pid: x for pid, x in self.g.groupby("player_id")}
        self._league = {}

    def league(self, stat, x):
        if (stat, x) not in self._league:
            self._league[(stat, x)] = float((self.g[stat] >= x).mean())
        return self._league[(stat, x)]

    def eligibility(self, pid, team):
        """(clause or None, reason). Clause (i): >= 20 current games, same team, role stable. Clause (ii): >= 41 prior-season
        games with this team and >= 10 current games confirming the role (minutes within 5, attempts within 20%)."""
        x = self.by.get(str(pid))
        if x is None:
            return None, "no box-score history (rookie or new to the league)"
        cur, prior = x[x.season == self.cur], x[x.season == self.cur - 1]
        if len(cur) and set(cur.team) != {team}:
            return None, "changed team this season"
        def same_role(a, b):
            if not len(a) or not len(b):
                return False
            fa, fb = float(a.fga.mean()), float(b.fga.mean())
            return abs(float(a["min"].mean()) - float(b["min"].mean())) <= 5 and (fb > 0 and abs(fa - fb) / fb <= 0.20)
        if len(cur) >= 20:
            return ("i", "") if same_role(cur.tail(10), cur.iloc[:-10]) else (None, "role changed inside this season")
        pt = prior[prior.team == team]
        if len(pt) < 41:
            return None, ("rookie or no prior season" if not len(prior) else f"{len(pt)} prior-season games with {team} (needs 41)")
        if len(cur) < 10:
            return None, f"{len(cur)} current-season games (needs 10)"
        return ("ii", "") if same_role(cur, pt) else (None, "role not confirmed (minutes beyond 5 or attempts beyond 20% of last season)")

    def base_rate(self, pid, team, stat, x):
        """(usable rate or None, games, hits): this team's games, prior + current season; needs >= 30 games and >= 3 hits."""
        g = self.by.get(str(pid))
        if g is None:
            return None, 0, 0
        g = g[g.team == team]; n = len(g); hits = int((g[stat] >= x).sum())
        if n < 30 or hits < 3:
            return None, n, hits
        return max(hits / n, 0.5 * self.league(stat, x)), n, hits


# ---------------------------------------------------------------- the search

def search(cap):
    """All four runs for one capture: (legs to log, run summaries). Pure: no file is read after the inputs, nothing is written."""
    box = Box()
    slate = read_json(DATA_DIR / "slate.json", {}) or {}
    game_id = {f"{sc.team_abbr(g['away'])}@{sc.team_abbr(g['home'])}": str(g["event_id"]) for g in slate.get("games") or []}
    model = {(p["player_key"], p["team"]): p for p in cap["model"]["players"]}
    date, off = cap["slate_date"], bool(cap.get("off_schedule"))
    decision_ts = sc.iso(sc.now())
    legs, runs = [], []
    for stat in STATS:
        run_id = f"{date}-{LABEL[stat]}" + ("-offsched" if off else "")
        c = dict.fromkeys(["markets", "tipped_excluded", "side_legs", "depth_pass", "unpriced_rung", "no_model_player", "candidates_searched", "candidates",
                           "ineligible_player", "no_base_rate", "tail_excluded", "guard_edge", "guard_minutes"], 0)
        cands = []
        for m in (r for r in cap["kalshi"]["markets"] if r["stat"] == stat):
            c["markets"] += 1
            if m.get("tipped_at_capture"):
                c["tipped_excluded"] += 1; continue
            p = model.get((m["player_key"], m["team"]))
            rung = ((p or {}).get("ladder") or {}).get(stat, {}).get(str(m["x"]))
            for side in ("yes", "no"):
                c["side_legs"] += 1
                ask, size = m[f"{side}_ask"], m[f"{side}_ask_size"]
                if not (ask is not None and 0 < ask < 1 and size is not None and size >= MIN_SIZE and m["yes_ask"] is not None and m["yes_bid"] is not None
                        and m["yes_ask"] - m["yes_bid"] <= MAX_SPREAD + 1e-9):
                    continue
                c["depth_pass"] += 1
                if p is None:
                    c["no_model_player"] += 1; continue
                if not rung:
                    c["unpriced_rung"] += 1; continue
                c["candidates_searched"] += 1
                p_pub = float(rung["p"]); p_use = p_pub            # the live published P, every stat (rule section 1)
                ps_raw, ps = (p_pub, p_use) if side == "yes" else (1 - p_pub, 1 - p_use)
                f = fee(ask); be = ask + f; edge = ps - be
                if edge < EDGE_FLOOR - 1e-12:
                    continue
                c["candidates"] += 1
                if edge > GUARD_EDGE + 1e-12:                    # guard_v1, edge cap
                    c["guard_edge"] += 1; continue
                mm = p.get("min_morning")
                if mm is None or p.get("min") is None or abs(float(p["min"]) - float(mm)) > GUARD_MINUTES + 1e-9:   # guard_v1, minutes move (fails closed)
                    c["guard_minutes"] += 1; continue
                clause, why = box.eligibility(p["player_id"], m["team"])
                if not clause:
                    c["ineligible_player"] += 1; continue
                br, n, hits = box.base_rate(p["player_id"], m["team"], stat, m["x"])
                if br is None:
                    c["no_base_rate"] += 1; continue
                if (side == "yes" and p_use >= TAIL * br) or (side == "no" and p_use <= br / TAIL):
                    c["tail_excluded"] += 1; continue
                ref = (m.get("reference") or {}).get(side) or {}
                mid = (m["yes_bid"] + m["yes_ask"]) / 2
                cands.append({"date": date, "venue": "kalshi", "run_id": run_id, "cluster_id": f"{date}|{p['player_id']}|{m['game']}|{LABEL[stat]}", "stat": LABEL[stat],
                              "player": p["player"], "team": m["team"], "game_id": game_id.get(m["game"], m["game"]), "rung": m["x"], "side": side,
                              "model_p_raw": round(ps_raw, 4), "model_p_shrunk": round(ps, 4), "calibrated_book_value": "", "decision_ts": decision_ts,
                              "reference_price": ref.get("price", ""), "reference_source": ref.get("book", ""), "kalshi_mid": round(mid if side == "yes" else 1 - mid, 4),
                              "ask": ask, "ask_size_fillable": size, "spread": round(m["yes_ask"] - m["yes_bid"], 4), "fee_per_contract": round(f, 5),
                              "fee_adj_breakeven": round(be, 5), "claimed_edge_pp": round(edge * 100, 2), "ranking_z": round(edge / math.sqrt(be * (1 - be)), 4),
                              "base_rate_eligible": 1, "base_rate_clause": clause, "tail_claim_flag": 0, "tail_claim_reason": "", "duplicate_of": "",
                              "ask_plus5": "", "ask_survived": "", "scratch_status": "", "hit": "", "status": "pending",
                              "_ticker": m["ticker"], "_base_rate": round(br, 4), "_base_n": n, "_base_hits": hits, "_raw_p": True})
        by_cluster = {}
        for x in cands:
            by_cluster.setdefault(x["cluster_id"], []).append(x)
        ranked = sorted(by_cluster.values(), key=lambda v: -max(x["ranking_z"] for x in v))[:TOP]
        out = []
        for v in ranked:
            keep = max(v, key=lambda x: (x["claimed_edge_pp"], x["ranking_z"]))
            for x in sorted(v, key=lambda x: x is not keep):
                x["candidates_searched"] = c["candidates_searched"]
                if x is not keep:
                    x["duplicate_of"] = f"{keep['cluster_id']}|{keep['rung']}|{keep['side']}"
                out.append(x)
        legs += out
        runs.append({"date": date, "run_id": run_id, "stat": LABEL[stat], "off_schedule": int(off), "decision_ts": decision_ts, "capture_at": cap["captured_at"], **c,
                     "clusters_selected": len(ranked), "legs_logged": len(out), "method_version": METHOD_VERSION, "p_source": P_SOURCE,
                     "note": "" if ranked else ("no candidate cleared the +3pp floor" if not c["candidates"] else "every candidate failed the outlier guard, eligibility, base rate or the tail flag")})
    return legs, runs


def repoll(legs, cap, wait=True):
    """Selected markets' asks at capture + 5 minutes. Returns the raw responses (stored privately by the caller)."""
    if not legs:
        return []
    due = sc.parse(cap["captured_at"]) + dt.timedelta(minutes=5)
    if wait:
        time.sleep(max(0.0, (due - sc.now()).total_seconds()))
    raw = []
    for x in legs:
        try:
            m = sc.kalshi_market(x["_ticker"])
        except RuntimeError:
            m = None
        raw.append({"ticker": x["_ticker"], "polled_at": sc.iso(sc.now()), "market": m})
        a = sc.fnum((m or {}).get("yes_ask_dollars" if x["side"] == "yes" else "no_ask_dollars"))
        x["ask_plus5"] = "" if a is None else a
        x["ask_survived"] = int(a is not None and 0 < a < 1 and a <= float(x["ask"]) + 1e-9)
    return raw


def run(capture_path, wait=True):
    P = paths()
    with gzip.open(capture_path, "rt", encoding="utf-8") as f:
        cap = json.load(f)
    done = read_csv(P["runs"])
    if len(done) and (done.date == cap["slate_date"]).any():
        print(f"[sam-optimizer] {cap['slate_date']} already has its four runs in the log: no re-run"); publish(); return
    legs, runs = search(cap)
    raw = repoll(legs, cap, wait)
    if raw:
        side = Path(capture_path).with_name(Path(capture_path).name.replace(".json.gz", ".repoll.json.gz"))
        with gzip.open(side, "wt", encoding="utf-8") as f:
            json.dump({"slate_date": cap["slate_date"], "repoll": raw}, f, separators=(",", ":"))
    append(P["log"], COLS, legs); append(P["runs"], RUN_COLS, runs)
    tonight = {"slate_date": cap["slate_date"], "captured_at": cap["captured_at"], "captured_at_pt": cap.get("captured_at_pt"), "off_schedule": bool(cap.get("off_schedule")),
               "model_snapshot": cap["model"].get("updated"), "sportsbook": cap["sportsbook"].get("status"),
               "extra": {f"{x['cluster_id']}|{x['rung']}|{x['side']}": {"raw_p": x["_raw_p"], "base_rate": x["_base_rate"], "base_n": x["_base_n"], "base_hits": x["_base_hits"], "ticker": x["_ticker"]} for x in legs}}
    write_json(P["dir"] / "tonight_meta.json", tonight)
    for r in runs:
        print(f"[sam-optimizer] {r['run_id']}: markets {r['markets']}, side-legs {r['side_legs']}, depth pass {r['depth_pass']}, unpriced rung {r['unpriced_rung']}, no model player {r['no_model_player']}, "
              f"searched {r['candidates_searched']}, candidates {r['candidates']} (guard edge {r['guard_edge']}, guard minutes {r['guard_minutes']}, ineligible {r['ineligible_player']}, no base rate {r['no_base_rate']}, tail {r['tail_excluded']}), "
              f"clusters {r['clusters_selected']}, legs logged {r['legs_logged']} {r['note']}")
    for x in legs:
        print(f"   {x['stat']:3s} {x['player']:24s} {x['rung']}+ {x['side']:3s} P {x['model_p_shrunk']:.3f} ask {float(x['ask']):.2f} BE {x['fee_adj_breakeven']:.4f} edge {x['claimed_edge_pp']:+.1f}pp z {x['ranking_z']:.3f} "
              f"size {float(x['ask_size_fillable']):.0f} +5 {x['ask_plus5']} survived {x['ask_survived']} {'DUPLICATE' if x['duplicate_of'] else ''}")
    publish()


# ---------------------------------------------------------------- grading and the tab file

def grade():
    P = paths(); log = read_csv(P["log"])
    if not len(log):
        print("[sam-optimizer] nothing logged yet"); publish(); return
    box = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})
    ingested = set(box.event_id); n = 0
    col = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "tpm"}
    for i, r in log[log.status == "pending"].iterrows():
        if r.game_id not in ingested:
            continue
        pid = r.cluster_id.split("|")[1]
        b = box[(box.event_id == r.game_id) & (box.player_id == pid)]
        if not len(b) or float(b.iloc[0]["min"]) <= 0:
            log.loc[i, ["scratch_status", "status"]] = ["inactive" if not len(b) else "did_not_enter", "void"]
        else:
            yes = int(float(b.iloc[0][col[r.stat]]) >= float(r.rung))
            log.loc[i, ["hit", "status"]] = [str(yes if r.side == "yes" else 1 - yes), "graded"]
        n += 1
    if n:
        log[COLS].to_csv(P["log"], index=False, lineterminator="\n")
    print(f"[sam-optimizer] graded {n} logged legs; {int((log.status == 'pending').sum())} pending")
    publish()


def record(rows):
    """Cluster record over primary rows (duplicates never count). Descriptive only until the gate window opens."""
    p = rows[rows.duplicate_of == ""]
    g = p[p.status == "graded"]
    out = {"clusters_logged": int(len(p)), "graded": int(len(g)), "pending": int((p.status == "pending").sum()), "void": int((p.status == "void").sum()),
           "hits": None, "hit_rate": None, "mean_breakeven": None, "z_edge": None, "z_over": None, "ask_survival_rate": None}
    sv = p[p.ask_survived != ""]
    if len(sv):
        out["ask_survival_rate"] = round(float(sv.ask_survived.astype(int).mean()), 4); out["ask_survival_n"] = int(len(sv))
    if len(g):
        h, be, pm = g.hit.astype(float), g.fee_adj_breakeven.astype(float), g.model_p_shrunk.astype(float)
        out.update({"hits": int(h.sum()), "hit_rate": round(float(h.mean()), 4), "mean_breakeven": round(float(be.mean()), 4),
                    "z_edge": round(float((h.sum() - be.sum()) / math.sqrt((be * (1 - be)).sum())), 3),
                    "z_over": round(float((pm.sum() - h.sum()) / math.sqrt((pm * (1 - pm)).sum())), 3)})
    return out


def publish():
    P = paths(); log, runs, caps = read_csv(P["log"]), read_csv(P["runs"]), read_csv(P["capture_log"])
    meta = read_json(P["dir"] / "tonight_meta.json", {}) or {}
    for col in RUN_COLS:
        if len(runs) and col not in runs.columns:
            runs[col] = ""
    on = runs[runs.off_schedule == "0"] if len(runs) else runs
    slates_on = int(on.date.nunique()) if len(on) else 0
    latest = runs.date.max() if len(runs) else None
    tonight = []
    for r in (runs[runs.date == latest].to_dict("records") if latest else []):
        legs = log[log.run_id == r["run_id"]].to_dict("records") if len(log) else []
        for x in legs:
            x.update((meta.get("extra") or {}).get(f"{x['cluster_id']}|{x['rung']}|{x['side']}", {}))
        tonight.append({**r, "legs": legs})
    skipped = caps[caps.status.str.startswith("skipped")].to_dict("records") if len(caps) else []
    doc = {"title": "Sam Method", "generated": sc.iso(sc.now()),
           "banner": "PRE-GATE SHADOW — PAPER ONLY. Base model calibration is under repair; these are the tool's logged shadow selections, not plays.",
           "gate": {"a_fit_target": A_FIT, "a_check_target": A_CHECK, "on_schedule_slates": slates_on, "off_schedule_slates": int(runs[runs.off_schedule == "1"].date.nunique()) if len(runs) else 0,
                    "a_fit_progress": min(slates_on, A_FIT), "a_check_progress": max(0, min(slates_on - A_FIT, A_CHECK)),
                    "step0": "Step-0 pre-read (existing graded ladder data): playable bands priced 55–70% hit 40–47% — base-model halt zone, ≥ +10pp.",
                    "note": "Only slates captured on schedule (13:00–13:25 PT) count toward A-fit and A-check; a manual run is stamped off_schedule and counts toward nothing."},
           "tonight": {"slate_date": latest, **{k: meta.get(k) for k in ("captured_at", "captured_at_pt", "off_schedule", "model_snapshot", "sportsbook")}, "runs": tonight} if latest and meta.get("slate_date") == latest else
                      ({"slate_date": latest, "runs": tonight} if latest else None),
           "record": {"label": "Descriptive only, pre-gate. Z_edge and Z_over are running tallies, not a test.",
                      "on_schedule": record(log[~log.run_id.str.endswith("-offsched")]) if len(log) else None,
                      "off_schedule": record(log[log.run_id.str.endswith("-offsched")]) if len(log) else None},
           "skipped_slates": skipped[-10:],
           "book_gap": {"status": "not_run", "label": "Book vs Kalshi (model-free) — PAPER ONLY",
                        "reason": "The archive curve needs the Oct 18–Nov 11 alt-price archive, which is not in this repository; nothing is fitted or logged until it is supplied.",
                        "rows": read_csv(P["book_gap"]).to_dict("records")[-60:] if P["book_gap"].exists() else []},
           "his_record": read_json(P["his"], None) if P["his"].exists() else None,
           "his_record_placeholder": "Grading in progress — 20 posted ladders transcribed.",
           "method": ["The tool searches every Kalshi single-leg rung (points, rebounds, assists, threes) for the biggest gap between our model's probability and the price, after Kalshi's fee.",
                      "It is Sam's method — a tool-driven search of alt rungs at fixed targets — rebuilt on Kalshi singles; sportsbook prices are reference only, never a venue.",
                      "Model probabilities are the ladder's live published numbers for every stat; the points recalibration map is in shadow and prices nothing until it ships.",
                      "Legs are ranked by per-leg z (edge over the break-even's standard deviation) in four fixed runs a day, one per stat, top three player-game clusters each, no re-runs.",
                      "Everything here is paper. Staking is a separate CEO decision after the locked gate."],
           "guard": {"version": METHOD_VERSION, "rule": "backtests/sam_method_guard_rule.md", "edge_cap_points": GUARD_EDGE * 100, "minutes_move": GUARD_MINUTES,
                     "text": f"Outlier guard (registered before the first counted run): a leg is dropped when its claimed edge is above {GUARD_EDGE * 100:.0f} points or the player's projected minutes moved more than {GUARD_MINUTES:.0f} from the morning build. The largest gaps are usually the model's own projection errors."},
           "readings": ["P shown is the probability of the listed side.", "A strike that is not a ladder rung is not priced.", "Off-schedule (manual) runs are shown but count toward no gate."]}
    write_json(P["out"], doc)
    print(f"[sam-optimizer] published {P['out'].name}: {len(log)} logged legs, {slates_on} on-schedule slates")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "grade", "publish"]); ap.add_argument("capture", nargs="?")
    ap.add_argument("--no-wait", action="store_true", help="re-poll immediately (testing only)")
    ap.add_argument("--out-root", help="write logs and the tab JSON under this directory instead of the repository (dry runs)")
    a = ap.parse_args()
    if a.out_root:
        ROOT_DATA, ROOT_SITE = Path(a.out_root) / "data", Path(a.out_root) / "docs_data"
        ROOT_SITE.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        run(a.capture, wait=not a.no_wait)
    elif a.cmd == "grade":
        grade()
    else:
        publish()
