"""Fantasy Points v1 (backtests/fantasy_points_v1_rule.md). Underdog scoring FP = PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV.

  python fantasy_points.py priors                 estimate the frozen constants once (2024-25): STL/BLK/TOV league rates by position,
                                                  NB dispersion per stat, within-player residual rate correlations -> data/fp_priors.json
  python fantasy_points.py snapshot               append tonight's pre-tip FP distributions to data/fp_log.csv (append-only)
  python fantasy_points.py grade                  grade logged player-games and priced lines vs box scores -> data/fp_grade.csv,
                                                  docs/data/fp_performance.json
  python fantasy_points.py price <underdog.json>  pricing study from the pirate-bets Underdog board capture: FP lines (if the capture
                                                  carries the market) -> data/fp_prices.csv; 3PM lines -> data/ud_prices.csv;
                                                  docs/data/fp_pricing.json for the tab. Never scrapes the app.

build_model.main calls profile_fp / simulate_fp per player and ships the result in projections.json (`fp`, `stl`, `blk`, `tov`).
The ladder engine's own draws are untouched: this module uses its own generator and its own copula draw.
"""
import csv
import datetime as dt
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, POSITIONS, SITE_DATA, norm_name, read_json, write_json

RULE = "backtests/fantasy_points_v1_rule.md"
PRIORS = DATA_DIR / "fp_priors.json"
LOG = DATA_DIR / "fp_log.csv"
GRADE = DATA_DIR / "fp_grade.csv"
FP_PRICES = DATA_DIR / "fp_prices.csv"
UD_PRICES = DATA_DIR / "ud_prices.csv"
PERF = SITE_DATA / "fp_performance.json"
PRICING = SITE_DATA / "fp_pricing.json"
PP_FILE = DATA_DIR / "pp_fantasy_lines.json"      # repo copy of pirate-bets/reader/board_snapshots/pp_fantasy_lines.jsonl (scraper hook)
PP_PRICES = DATA_DIR / "pp_prices.csv"            # PrizePicks standard Fantasy Score lines priced from the FP log (amendment 2026-10-25)
PP_PRICING = SITE_DATA / "pp_pricing.json"
PP_MAX_AGE_H = 26                                 # a copy older than this (or missing) = "no PP lines today"; never fails the build

WEIGHTS = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "tov": -1.0}
COMP = ["tpa", "two", "reb", "ast", "stl", "blk", "tov"]       # copula components (section 2)
NEW = ["stl", "blk", "tov"]
NB_STATS = ["reb", "ast", "stl", "blk", "tov"]
N_SIM, K_RATE, K_RHO, BE_STD, SEED = 10000, 120.0, 40, 0.55, 20261022
RATE_HALF_LIFE, PRIOR_SEASON_W, PRIOR_SEASON = 12, 0.5, 2025   # engine weighting; priors season = 2024-25
QS = np.arange(1, 100)                                          # percentiles logged
FP_MARKETS = {"fantasy points", "fantasy score", "fantasy pts"}
THREES_MARKETS = {"3-pointers made", "3 pointers made", "three pointers made", "three-pointers made", "threes made", "3pm", "3-pt made", "3 pt made", "3pt made"}
PMF_TOP = {"stl": 10, "blk": 10, "tov": 12}
THIN, MIN_DECILE, GATE_N = 30, 20, 200
MAX_CAPTURE_AGE_H = 12                                          # addendum 2026-10-23: never price a capture older than this (or than the slate)
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001  (no tz database: fall back to fixed EST)
    ET = dt.timezone(dt.timedelta(hours=-5))
LABEL = "Hit rate measures accuracy, not profit. Pricing study, paper only."
LOG_FIELDS = ["date", "event_id", "tip", "player_id", "player", "team", "opp", "pos", "min", "fp_mean", "fp_sd", "fp_q", "stl_mean", "blk_mean", "tov_mean",
              "stl_pmf", "blk_pmf", "tov_pmf", "logged_at"]
FP_PRICE_FIELDS = ["captured_at", "date", "event_id", "player", "team", "opp", "market", "line", "side", "mult", "decimal", "convention", "p_model", "breakeven", "edge", "fp_logged_at", "source_id"]
UD_PRICE_FIELDS = ["captured_at", "date", "event_id", "player", "team", "opp", "line", "side", "mult", "decimal", "convention", "p_model", "p_source", "breakeven", "edge", "source_id"]
GRADE_FIELDS = ["date", "event_id", "tip", "player_id", "player", "team", "opp", "pos", "min_proj", "min_actual", "fp_mean", "fp_actual", "pit", "stl_actual", "blk_actual",
                "tov_actual", "void", "line_source", "market", "line", "side", "mult", "p_model", "breakeven", "hit", "logged_at", "captured_at", "source"]
PP_PRICE_FIELDS = ["captured_at", "date", "event_id", "player", "team", "opp", "market", "line", "side", "p_model", "scheduled_at", "game_status", "source_id", "fp_logged_at"]


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def ts(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def fp_of(d):
    return sum(w * d[s] for s, w in WEIGHTS.items())


# ---------------- numerics (no scipy on the runner) ----------------
def norm_cdf(z):
    """Abramowitz-Stegun 7.1.26 erf, vectorized; |error| < 1.5e-7."""
    x = np.abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * y)


def count_ppf(u, mu, alpha):
    """Inverse CDF of a negative-binomial count (mean mu, variance mu + alpha mu^2; Poisson when alpha = 0), elementwise over
    arrays u (uniforms) and mu. Walks the pmf recurrence until the cdf passes u."""
    mu = np.maximum(mu, 1e-9)
    if alpha <= 0:
        pk = np.exp(-mu); ratio = lambda k: mu / (k + 1)
    else:
        r = 1 / alpha; p = r / (r + mu)
        pk = p ** r; ratio = lambda k: (k + r) / (k + 1) * (1 - p)
    cdf = pk.copy(); out = np.zeros_like(u, dtype=np.int64); done = u < cdf
    kmax = int(np.ceil(mu.max() + 12 * math.sqrt(mu.max() * (1 + alpha * mu.max())) + 10))
    for k in range(kmax):
        pk = pk * ratio(k); cdf = cdf + pk
        newly = (~done) & (u < cdf); out[newly] = k + 1; done |= newly
        if done.all():
            break
    out[~done] = kmax
    return out


def nearest_pd(R):
    w, v = np.linalg.eigh(R)
    R2 = (v * np.maximum(w, 1e-6)) @ v.T
    d = np.sqrt(np.diag(R2)); R2 = R2 / np.outer(d, d)
    return (R2 + R2.T) / 2


# ---------------- priors (estimated once, frozen) ----------------
def residual_corr(d):
    """Within-player residual correlation of per-minute component rates. d has the raw box-score columns + 'two'."""
    rates = pd.DataFrame({c: d[c] / d["min"] for c in COMP}, index=d.index)
    res = rates - rates.groupby(d.player_id).transform("mean")
    C = np.array(res.corr().reindex(index=COMP, columns=COMP).fillna(0.0).to_numpy(), dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


def priors():
    if PRIORS.exists():
        print(f"[fp] {PRIORS} exists: frozen, not re-estimated"); return
    import build_model as bm
    df = bm.load_games(include_history=True)
    d = df[(df.season == PRIOR_SEASON) & (df["min"] >= 8)].copy()
    d["two"] = (d.pts - 3 * d.tpm) / 2
    n = d.groupby("player_id").event_id.transform("size"); d = d[n >= 20].copy()
    out = {"rule_file": RULE, "estimated": ts(now()), "season": "2024-25", "n_rows": int(len(d)), "n_players": int(d.player_id.nunique()),
           "components": COMP, "K_rate_minutes": K_RATE, "K_rho_games": K_RHO, "n_sim": N_SIM, "weights": WEIGHTS,
           "rate_pos": {}, "alpha": {}, "rho_pos": {}, "n_pos": {}}
    for pos in POSITIONS + ["ALL"]:
        dp = d if pos == "ALL" else d[d.pos == pos]
        out["n_pos"][pos] = int(len(dp))
        out["rate_pos"][pos] = {s: round(float(dp[s].sum() / dp["min"].sum()), 6) for s in NEW} if len(dp) else None
        out["rho_pos"][pos] = np.round(residual_corr(dp), 4).tolist() if len(dp) >= 200 else None
    g = d.groupby("player_id")
    for s in NB_STATS:
        tot, mins = g[s].transform("sum"), g["min"].transform("sum")
        mu = (tot - d[s]) / (mins - d["min"]).clip(lower=1) * d["min"]
        out["alpha"][s] = round(max(0.0, float(((d[s] - mu) ** 2 - mu).sum() / (mu ** 2).sum())), 5)
    write_json(PRIORS, out)
    print(f"[fp] priors frozen: {out['n_rows']} rows, {out['n_players']} players; alpha {out['alpha']}; rates ALL {out['rate_pos']['ALL']}")


def load_priors():
    return read_json(PRIORS, None)


# ---------------- per-player profile and simulation ----------------
def profile_fp(g, cur, pos, pri):
    """STL/BLK/TOV per-minute rates (engine weighting, EB-shrunk to the position prior with K_RATE minutes) and the player's
    blended copula correlation matrix. g = his games of the current + prior season."""
    g = g.sort_values("date", ascending=False)
    k = np.arange(len(g)); w = 0.5 ** (k / RATE_HALF_LIFE) * np.where(g.season.values == cur, 1.0, PRIOR_SEASON_W)
    M = float((g["min"].values * w).sum())
    prior = (pri["rate_pos"].get(pos) or pri["rate_pos"]["ALL"])
    rates = {}
    for s in NEW:
        raw = float((g[s].values * w).sum() / M) if M > 0 else prior[s]
        rates[s] = (M * raw + K_RATE * prior[s]) / (M + K_RATE)
    gg = g[g["min"] >= 8].assign(two=lambda d: (d.pts - 3 * d.tpm) / 2)
    n = int(len(gg))
    R_pos = np.array(pri["rho_pos"].get(pos) or pri["rho_pos"]["ALL"])
    R_pl = residual_corr(gg) if n >= 5 else R_pos
    R = nearest_pd((n * R_pl + K_RHO * R_pos) / (n + K_RHO))
    return {"rates": rates, "R": R, "n_corr": n}


def simulate_fp(p, prof, opp, dvp, absent_starters, pri, rng):
    """Section 2: one minutes draw per sim, 7 copula components, FP per sim. p is the engine's player dict (min, sd_min, rates, pct, pos)."""
    import build_model as bm
    N = N_SIM
    mins = np.clip(rng.normal(p["min"], p["sd_min"], N), 0, 48)
    f = {s: bm.dvp_factor(dvp, opp, p["pos"], s) for s in ("tpa", "tpm", "reb", "ast", "pts")}
    n_abs = min(absent_starters, bm.USAGE_MAX_ABSENT)
    u = {s: 1 + bm.USAGE_BUMP[s] * n_abs for s in bm.USAGE_BUMP}
    r = p["rates"]
    mu = {"tpa": mins * r["tpa"] * f["tpa"] * u["tpa"], "two": np.maximum(mins * r["non3"] * f["pts"] * u["pts"], 0) / 2,
          "reb": mins * r["reb"] * f["reb"] * u["reb"], "ast": mins * r["ast"] * f["ast"] * u["ast"],
          "stl": mins * prof["rates"]["stl"], "blk": mins * prof["rates"]["blk"], "tov": mins * prof["rates"]["tov"]}
    L = np.linalg.cholesky(prof["R"])
    U = np.clip(norm_cdf(rng.standard_normal((N, len(COMP))) @ L.T), 1e-9, 1 - 1e-9)
    alpha = pri["alpha"]
    x = {c: count_ppf(U[:, i], mu[c], 0.0 if c in ("tpa", "two") else alpha[c]) for i, c in enumerate(COMP)}
    make_p = float(np.clip(p["pct"] * f["tpm"] / max(f["tpa"], 0.5), 0.12, 0.6))
    tpm = rng.binomial(x["tpa"], make_p)
    pts = 3 * tpm + 2 * x["two"]
    fp = pts + 1.2 * x["reb"] + 1.5 * x["ast"] + 3 * x["stl"] + 3 * x["blk"] - x["tov"]
    q = np.percentile(fp, QS)
    out = {"fp": {"mean": round(float(fp.mean()), 2), "sd": round(float(fp.std()), 2), "p10": round(float(q[9]), 1), "p50": round(float(q[49]), 1), "p90": round(float(q[89]), 1),
                  "q": [round(float(v), 1) for v in q]}}
    for s in NEW:
        out[s] = {"mean": round(float(x[s].mean()), 2), "pmf": bm.pmf(x[s], PMF_TOP[s])}
    return out


# ---------------- log ----------------
def append(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def snapshot():
    P = read_json(SITE_DATA / "projections.json", None)
    if not P or not P.get("games"):
        print("[fp] no slate"); return 0
    ev = {}
    for g in P["games"]:
        ev[(g["home"], g["away"])] = g; ev[(g["away"], g["home"])] = g
    t = now(); rows, skipped = [], 0
    for p in P["players"]:
        if p.get("inactive") or "fp" not in p:
            continue
        g = ev.get((p["team"], p["opp"]))
        if not g or not g.get("start"):
            continue
        tip = parse_ts(g["start"])
        if t >= tip:
            skipped += 1; continue
        rows.append({"date": P["date"], "event_id": g["event_id"], "tip": ts(tip), "player_id": p["id"], "player": p["name"], "team": p["team"], "opp": p["opp"], "pos": p["pos"],
                     "min": p["min"], "fp_mean": p["fp"]["mean"], "fp_sd": p["fp"]["sd"], "fp_q": json.dumps(p["fp"]["q"]),
                     "stl_mean": p["stl"]["mean"], "blk_mean": p["blk"]["mean"], "tov_mean": p["tov"]["mean"],
                     "stl_pmf": json.dumps(p["stl"]["pmf"]), "blk_pmf": json.dumps(p["blk"]["pmf"]), "tov_pmf": json.dumps(p["tov"]["pmf"]), "logged_at": ts(t)})
    if rows:
        append(LOG, LOG_FIELDS, rows)
    print(f"[fp] appended {len(rows)} rows for {P['date']} ({skipped} players skipped: game already tipped)")
    return len(rows)


def last_pretip(log):
    d = log[pd.to_datetime(log.logged_at, utc=True) < pd.to_datetime(log.tip, utc=True)]
    return d.sort_values("logged_at").drop_duplicates(["date", "event_id", "player_id"], keep="last")


def cdf_from_q(q, x):
    """P(FP <= x) from the 99 logged percentiles (linear between them, 0/1 beyond the 1st/99th)."""
    q = np.asarray(q, dtype=float)
    if x < q[0]:
        return 0.0
    if x > q[-1]:
        return 1.0
    return float(np.interp(x, q, QS / 100.0))


# ---------------- pricing study ----------------
def team_abbr(x):
    from odds import team_abbr as ta
    return ta(x) or x


def leg_mult(e, side):
    try:
        return float(((e.get("platform_ids") or {}).get("payout_multipliers") or {}).get(side) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def leg_price(e, side):
    """(mult, decimal, convention, break-even) per the 2026-10-23 addendum: a side is book-style when the capture carries a
    decimal price for it and its payout multiplier is 1.0 (break-even 1/decimal); every other side is a pick'em leg (0.55/mult)."""
    m = leg_mult(e, side)
    pr = ((e.get("platform_ids") or {}).get("prices") or {}).get(side) or {}
    try:
        dec = float(pr.get("decimal")) if pr.get("decimal") not in (None, "") else None
    except (TypeError, ValueError):
        dec = None
    if dec is not None and dec > 1.0 and abs(m - 1.0) < 1e-9:
        return m, dec, "book", 1.0 / dec
    return m, dec, "pickem", (BE_STD / m if m > 0 else None)


def capture_check(cap, slate_date, t):
    """Addendum 2026-10-23 item 2: the capture's US/Eastern date must be on the slate date or later and the capture at most
    MAX_CAPTURE_AGE_H hours old at pricing time. Returns (ok, reason)."""
    try:
        c = dt.datetime.fromisoformat(str(cap).replace("Z", "+00:00"))
        if c.tzinfo is None:
            c = c.replace(tzinfo=dt.timezone.utc)
    except Exception:  # noqa: BLE001
        return False, "capture has no parseable captured_at"
    cap_et = c.astimezone(ET).date().isoformat()
    if slate_date and cap_et < str(slate_date):
        return False, f"capture dated {cap_et} ET is older than the slate {slate_date}"
    age_h = (t - c).total_seconds() / 3600
    if age_h > MAX_CAPTURE_AGE_H:
        return False, f"capture is {age_h:.1f} h old (limit {MAX_CAPTURE_AGE_H} h)"
    return True, f"capture {age_h * 60:.0f} min old, dated {cap_et} ET, slate {slate_date}"


def price(snapshot_path):
    snap = json.load(open(snapshot_path, encoding="utf-8"))
    cap = snap.get("captured_at") or ts(now())
    nba = [e for e in snap.get("entries") or [] if (e.get("league") or "").upper() == "NBA"]
    markets = Counter((e.get("market") or "").strip() for e in nba)
    P = read_json(SITE_DATA / "projections.json", None) or {"date": None, "games": [], "players": []}
    games = []
    for g in P["games"]:
        games.append(g)
    def match_game(e):
        # canonicalize BOTH sides through odds.team_abbr: Underdog's and ESPN's abbreviations differ (NYK/NY, GSW/GS, ...)
        a, b = team_abbr(e.get("team")), team_abbr(e.get("opponent"))
        for g in games:
            if {team_abbr(g["home"]), team_abbr(g["away"])} == {a, b}:
                try:
                    if abs((parse_ts(e["scheduled_at"]) - parse_ts(g["start"])).total_seconds()) <= 12 * 3600:
                        return g
                except Exception:
                    return g
        return None
    players = {norm_name(p["name"]): p for p in P["players"] if not p.get("inactive")}
    log = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str}) if LOG.exists() else None
    fp_rows, ud_rows, unmatched, tipped = [], [], 0, 0
    t = now()
    ok, why = capture_check(cap, P.get("date"), t)
    fp_entries = [e for e in nba if (e.get("market") or "").strip().lower() in FP_MARKETS]
    th_entries = [e for e in nba if (e.get("market") or "").strip().lower() in THREES_MARKETS]
    for e in (fp_entries + th_entries) if ok else []:
        g = match_game(e)
        if g is None:
            unmatched += 1; continue
        if t >= parse_ts(g["start"]):
            tipped += 1; continue
        try:
            line = float(e["line"])
        except (KeyError, TypeError, ValueError):
            continue
        key = norm_name(e.get("player"))
        is_fp = e in fp_entries
        tm = g["home"] if team_abbr(e.get("team")) == team_abbr(g["home"]) else g["away"]   # site (ESPN) abbreviations in the output
        op = g["away"] if tm == g["home"] else g["home"]
        for side in e.get("available_directions") or ["higher", "lower"]:
            m, dec, conv, be = leg_price(e, side)
            if is_fp:
                if log is None:
                    continue
                d = log[(log.date == P["date"]) & (log.event_id == g["event_id"]) & (log.player.map(norm_name) == key)]
                d = d[pd.to_datetime(d.logged_at, utc=True) < pd.to_datetime(d.tip, utc=True)]
                if d.empty:
                    continue
                row = d.sort_values("logged_at").iloc[-1]
                c = cdf_from_q(json.loads(row.fp_q), line); p_m = (1 - c) if side == "higher" else c
                fp_rows.append({"captured_at": cap, "date": P["date"], "event_id": g["event_id"], "player": e["player"], "team": tm, "opp": op,
                                "market": e["market"], "line": line, "side": side, "mult": m, "decimal": dec, "convention": conv, "p_model": round(p_m, 4), "breakeven": round(be, 4) if be else None,
                                "edge": round(p_m - be, 4) if be else None, "fp_logged_at": row.logged_at, "source_id": e.get("source_id")})
            else:
                p = players.get(key)
                if not p or "tpm" not in p:
                    continue
                k = int(math.floor(line)) + 1   # higher = at least k threes
                lad = (p.get("ladder") or {}).get("tpm", {}).get(str(k))
                p_hi = float(lad["p"]) if lad else float(sum(p["tpm"]["pmf"][k:]))
                src = "ladder" if lad else "pmf"
                p_m = p_hi if side == "higher" else 1 - p_hi
                ud_rows.append({"captured_at": cap, "date": P["date"], "event_id": g["event_id"], "player": e["player"], "team": tm, "opp": op,
                                "line": line, "side": side, "mult": m, "decimal": dec, "convention": conv, "p_model": round(p_m, 4), "p_source": src, "breakeven": round(be, 4) if be else None,
                                "edge": round(p_m - be, 4) if be else None, "source_id": e.get("source_id")})
    def fresh(path, rows):
        if not rows:
            return 0
        if path.exists():
            old = pd.read_csv(path, usecols=["captured_at"])
            if (old.captured_at == cap).any():
                return 0   # this capture is already logged
        append(path, FP_PRICE_FIELDS if path is FP_PRICES else UD_PRICE_FIELDS, rows); return len(rows)
    n_fp, n_ud = fresh(FP_PRICES, fp_rows), fresh(UD_PRICES, ud_rows)
    conv = Counter(r["convention"] for r in fp_rows + ud_rows)
    doc = {"captured_at": cap, "source_updated_at": snap.get("source_updated_at"), "stored_at": snap.get("stored_at"), "priced_at": ts(t), "slate_date": P.get("date"),
           "capture_ok": ok, "capture_check": why, "max_capture_age_h": MAX_CAPTURE_AGE_H, "nba_entries": len(nba), "markets": dict(markets.most_common()),
           "fp_market_strings_checked": sorted(FP_MARKETS), "fp_market_present": bool(fp_entries), "fp_entries": len(fp_entries),
           "fp_lines_priced": len(fp_rows), "threes_entries": len(th_entries), "threes_priced": len(ud_rows), "unmatched_entries": unmatched, "tipped_entries": tipped,
           "conventions": dict(conv),
           "note": (f"Not priced: {why}. Pricing never runs against a capture older than the slate it prices (addendum 2026-10-23)." if not ok else
                    f"No Fantasy Points lines on the latest board capture (captured {cap}). Underdog posts this market intermittently; nothing is priced and no other source is used (rule section 4)."
                    if not fp_entries else "Fantasy Points lines priced against the last pre-tip FP log row."),
           "breakeven_rule": f"book-style side (decimal price posted, multiplier 1.0): 1 / decimal; pick'em leg: {BE_STD} / leg multiplier", "label": LABEL, "rule_file": RULE,
           "fp_lines": sorted(fp_rows, key=lambda r: -(r["edge"] or -9))[:400], "threes": sorted(ud_rows, key=lambda r: -(r["edge"] or -9))[:400]}
    write_json(PRICING, doc)
    print(f"[fp] board {cap} ({'ok' if ok else 'REFUSED'}: {why}): {len(nba)} NBA entries, markets {dict(markets.most_common(6))}...; FP market present: {bool(fp_entries)} "
          f"({len(fp_entries)} entries, {n_fp} new price rows); 3PM {len(th_entries)} entries -> {n_ud} new rows; conventions {dict(conv)}; {unmatched} entries not on tonight's slate, {tipped} tipped")
    return doc


# ---------------- grading ----------------
def price_pp(path):
    """Amendment 2026-10-25: price PrizePicks standard Fantasy Score lines from the repo copy written by the scraper hook.
    Standard lines only; a line counts only when its capture precedes scheduled_at; per player and game the last entry in the
    file wins; P(over) = 1 - CDF(line) from the player's last pre-tip FP log row (the Underdog interpolation). Both sides are
    logged like the Underdog rows. A missing, unreadable or stale (> PP_MAX_AGE_H) copy means "no PP lines today"."""
    t = now()
    doc = {"generated": ts(t), "platform": "prizepicks", "source_file": None, "captured_at": None, "stored_at": None, "ok": False, "reason": None,
           "max_age_h": PP_MAX_AGE_H, "file_lines": 0, "standard": 0, "non_standard_excluded": 0, "not_fantasy_score": 0, "captured_after_tip": 0, "pre_tip": 0,
           "matched": 0, "priced_rows": 0, "new_rows": 0, "unmatched": [], "lines": [], "label": LABEL, "rule_file": RULE,
           "note": "PrizePicks standard Fantasy Score lines (rule amendment 2026-10-25). P(over) from the last pre-tip FP log row, same interpolation as the Underdog pricing. "
                   "Graded with source = PP; the 200-line gate counts UD + PP combined."}

    def finish(reason=None):
        if reason:
            doc["reason"] = reason
        write_json(PP_PRICING, doc)
        print(f"[fp] prizepicks: {'ok' if doc['ok'] else 'no lines'}: {doc['reason'] or ''} file_lines={doc['file_lines']} standard={doc['standard']} "
              f"non_standard_excluded={doc['non_standard_excluded']} pre_tip={doc['pre_tip']} matched={doc['matched']} priced_rows={doc['priced_rows']} new_rows={doc['new_rows']} "
              f"unmatched={len(doc['unmatched'])}")
        return doc

    p = Path(path)
    if not p.exists():
        return finish("no data/pp_fantasy_lines.json yet: the scraper hook has not delivered a PrizePicks copy")
    try:
        cp = json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return finish(f"unreadable copy: {e}")
    cap = cp.get("captured_at")
    doc.update({"source_file": cp.get("source_path"), "captured_at": cap, "stored_at": cp.get("stored_at")})
    try:
        age_h = (t - parse_ts(cap)).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        return finish("copy carries no capture time")
    if age_h > PP_MAX_AGE_H:
        return finish(f"copy captured {age_h:.1f} h ago, over the {PP_MAX_AGE_H} h limit: no PP lines today")
    raw = cp.get("lines") or []
    doc["file_lines"] = len(raw)
    std = [r for r in raw if str(r.get("odds_type") or "").strip().lower() == "standard"]
    doc["standard"], doc["non_standard_excluded"] = len(std), len(raw) - len(std)
    if doc["non_standard_excluded"]:
        print(f"[fp] prizepicks WARNING: {doc['non_standard_excluded']} non-standard (demon/goblin) rows in the copy; excluded here, upstream filter expected them gone")
    fs = [r for r in std if str(r.get("stat_type") or "").strip().lower() in FP_MARKETS]
    doc["not_fantasy_score"] = len(std) - len(fs)
    if not LOG.exists():
        return finish("no FP log yet")
    snap = last_pretip(pd.read_csv(LOG, dtype={"event_id": str, "player_id": str}))
    idx = {(r.date, norm_name(r.player)): r for r in snap.itertuples()}
    last = {}   # per player + game date, the last entry in the file (file order = pull order)
    for r in fs:
        try:
            sched = parse_ts(r["scheduled_at"])
        except Exception:  # noqa: BLE001
            doc["unmatched"].append({"player": r.get("player"), "team": r.get("team"), "why": "no scheduled_at"}); continue
        last[(norm_name(r.get("player")), sched.astimezone(ET).date().isoformat())] = (r, sched)
    rows, out = [], []
    for (key, date), (r, sched) in last.items():
        lcap = r.get("captured_at") or r.get("pulled_at") or r.get("snapshot_at") or r.get("extracted_at") or cap
        try:
            lcap_dt = parse_ts(lcap)
        except Exception:  # noqa: BLE001
            lcap_dt = parse_ts(cap)
        if lcap_dt >= sched:
            doc["captured_after_tip"] += 1; continue
        doc["pre_tip"] += 1
        row = idx.get((date, key))
        if row is None:
            doc["unmatched"].append({"player": r.get("player"), "team": r.get("team"), "date": date, "why": "no pre-tip FP log row for that slate"}); continue
        ta = team_abbr(r.get("team"))
        if ta and ta != team_abbr(row.team):
            doc["unmatched"].append({"player": r.get("player"), "team": r.get("team"), "date": date, "why": f"team mismatch (log says {row.team})"}); continue
        try:
            line = float(r["line"])
        except (KeyError, TypeError, ValueError):
            doc["unmatched"].append({"player": r.get("player"), "team": r.get("team"), "date": date, "why": "no numeric line"}); continue
        doc["matched"] += 1
        c = cdf_from_q(json.loads(row.fp_q), line); p_over = 1 - c
        for side, p_m in (("higher", p_over), ("lower", c)):
            rows.append({"captured_at": ts(lcap_dt), "date": date, "event_id": row.event_id, "player": r.get("player"), "team": row.team, "opp": row.opp,
                         "market": r.get("stat_type"), "line": line, "side": side, "p_model": round(p_m, 4), "scheduled_at": ts(sched),
                         "game_status": r.get("game_status"), "source_id": r.get("source_id"), "fp_logged_at": row.logged_at})
        out.append({"date": date, "event_id": row.event_id, "player": r.get("player"), "team": row.team, "opp": row.opp, "line": line, "p_over": round(p_over, 4),
                    "scheduled_at": ts(sched), "captured_at": ts(lcap_dt), "source_id": r.get("source_id")})
    doc["priced_rows"] = len(rows)
    if rows:
        seen = set()
        if PP_PRICES.exists():
            old = pd.read_csv(PP_PRICES, dtype=str, usecols=["captured_at", "source_id", "side", "player", "date"])
            seen = set(zip(old.captured_at, old.source_id.fillna(""), old.side, old.player, old.date))
        new = [x for x in rows if (x["captured_at"], str(x["source_id"] or ""), x["side"], x["player"], x["date"]) not in seen]
        if new:
            append(PP_PRICES, PP_PRICE_FIELDS, new)
        doc["new_rows"] = len(new)
    doc["ok"] = True
    doc["lines"] = sorted(out, key=lambda x: (x["date"], -x["line"]), reverse=True)[:400]
    return finish()


# ---------------- grading ----------------
def buckets(p, h, thin=THIN):
    out = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        m = (p >= lo) & (p < hi) if i < 9 else (p >= lo)
        n = int(m.sum())
        out.append({"lo": lo, "hi": hi, "n": n, "pred": round(float(p[m].mean()), 4) if n else None, "actual": round(float(h[m].mean()), 4) if n else None,
                    "gap": round(float(p[m].mean() - h[m].mean()), 4) if n else None, "thin": n < thin})
    return out


def decile_verdict(bk, n_total):
    """Section 6: at n >= GATE_N, PASS if the n-weighted mean |gap| over deciles with n >= MIN_DECILE is <= 0.05 and none exceeds 0.10."""
    if n_total < GATE_N:
        return {"status": "collecting", "graded": int(n_total), "needed": GATE_N}
    use = [b for b in bk if b["n"] >= MIN_DECILE]
    if not use:
        return {"status": "collecting", "graded": int(n_total), "needed": GATE_N, "note": "no decile with 20+ rows"}
    wmae = sum(b["n"] * abs(b["gap"]) for b in use) / sum(b["n"] for b in use); worst = max(abs(b["gap"]) for b in use)
    return {"status": "PASS" if wmae <= 0.05 and worst <= 0.10 else "FAIL", "graded": int(n_total), "weighted_mean_abs_gap": round(wmae, 4), "worst_gap": round(worst, 4)}


def grade():
    if not LOG.exists():
        if PERF.exists():
            PERF.unlink()
        print("[fp] no log yet"); return
    log = pd.read_csv(LOG, dtype={"event_id": str, "player_id": str})
    snap = last_pretip(log)
    box = pd.read_csv(DATA_DIR / "player_games.csv", dtype={"event_id": str, "player_id": str})
    box = box.set_index(["event_id", "player_id"])
    prices = pd.read_csv(FP_PRICES, dtype={"event_id": str}) if FP_PRICES.exists() else pd.DataFrame(columns=FP_PRICE_FIELDS)
    pp_prices = pd.read_csv(PP_PRICES, dtype={"event_id": str}) if PP_PRICES.exists() else pd.DataFrame(columns=PP_PRICE_FIELDS)
    rows, pit, marg = [], [], {s: {k: ([], []) for k in ((1, 2) if s != "tov" else (2, 3))} for s in NEW}
    for r in snap.itertuples():
        if (r.event_id, r.player_id) not in box.index:
            continue   # box score not in yet: pending
        b = box.loc[(r.event_id, r.player_id)]
        void = int(float(b["min"]) <= 0)
        actual = None if void else float(fp_of(b)); q = json.loads(r.fp_q)
        base = {"date": r.date, "event_id": r.event_id, "tip": r.tip, "player_id": r.player_id, "player": r.player, "team": r.team, "opp": r.opp, "pos": r.pos, "min_proj": r.min,
                "min_actual": float(b["min"]), "fp_mean": r.fp_mean, "fp_actual": None if void else round(actual, 1), "pit": None if void else round(cdf_from_q(q, actual), 4),
                "stl_actual": int(b.stl), "blk_actual": int(b.blk), "tov_actual": int(b.tov), "void": void, "logged_at": r.logged_at}
        if not void:
            pit.append(base["pit"])
            for s in NEW:
                pmf = json.loads(getattr(r, f"{s}_pmf"))
                for k, (pp, hh) in marg[s].items():
                    pp.append(float(sum(pmf[k:]))); hh.append(int(int(b[s]) >= k))
        pr = prices[(prices.date == r.date) & (prices.event_id == r.event_id) & (prices.player.map(norm_name) == norm_name(r.player)) & (prices.captured_at < r.tip)]
        pr = pr.sort_values("captured_at").drop_duplicates(["line", "side"], keep="last")
        ppr = pp_prices[(pp_prices.date == r.date) & (pp_prices.event_id == r.event_id) & (pp_prices.player.map(norm_name) == norm_name(r.player)) & (pp_prices.captured_at < r.tip)]
        if len(ppr):
            ppr = ppr[ppr.captured_at == ppr.captured_at.max()].drop_duplicates(["line", "side"], keep="last")   # the LAST pre-tip capture's line (amendment 2026-10-25)
        graded = False
        if len(pr) and not void:
            for x in pr.itertuples():
                hit = None if actual == x.line else int(actual > x.line if x.side == "higher" else actual < x.line)
                rows.append({**base, "line_source": "underdog", "market": x.market, "line": x.line, "side": x.side, "mult": x.mult, "p_model": x.p_model, "breakeven": x.breakeven, "hit": hit, "captured_at": x.captured_at, "source": "UD"})
            graded = True
        if len(ppr) and not void:   # a player with both a UD and a PP line grades both: two lines, one actual
            for x in ppr.itertuples():
                hit = None if actual == x.line else int(actual > x.line if x.side == "higher" else actual < x.line)
                rows.append({**base, "line_source": "prizepicks", "market": x.market, "line": x.line, "side": x.side, "mult": "", "p_model": x.p_model, "breakeven": "", "hit": hit, "captured_at": x.captured_at, "source": "PP"})
            graded = True
        if not graded:
            med = float(q[49]); p_hi = 1 - cdf_from_q(q, med)
            rows.append({**base, "line_source": "model_median", "market": "", "line": round(med, 1), "side": "higher", "mult": "", "p_model": round(p_hi, 4), "breakeven": "",
                         "hit": None if void or actual == med else int(actual > med), "captured_at": "", "source": ""})
    R = pd.DataFrame(rows, columns=GRADE_FIELDS); R.to_csv(GRADE, index=False)
    gl = R[R.line_source.isin(["underdog", "prizepicks"]) & R.hit.notna()]   # the gate record: UD + PP combined (amendment 2026-10-25)

    def line_stats(d, gate=True):
        pp_, hh_ = d.p_model.astype(float).values, d.hit.astype(int).values
        bk_ = buckets(pp_, hh_) if len(d) else buckets(np.array([]), np.array([]))
        o = {"n": int(len(d)), "hits": int(hh_.sum()) if len(d) else 0, "hit_rate": round(float(hh_.mean()), 4) if len(d) else None,
             "mean_p": round(float(pp_.mean()), 4) if len(d) else None, "by_decile": bk_}
        o["gate" if gate else "decile_read"] = decile_verdict(bk_, len(d))
        return o
    lines_doc = line_stats(gl)
    lines_doc["scope"] = "combined UD + PP: the gate counter and verdict (rule amendment 2026-10-25)"
    lines_doc["sources"] = {s: int((gl.source == s).sum()) for s in ("UD", "PP")}
    by_source = {s: {**line_stats(gl[gl.source == s], gate=False), "label": {"UD": "Underdog", "PP": "PrizePicks"}[s],
                     "note": "same grading code path and format as the combined table; shown alongside so neither book's record hides in the merge; not the gate"} for s in ("UD", "PP")}
    pit_arr = np.array(pit)
    pit_tab = [{"lo": i / 10, "hi": (i + 1) / 10, "n": int(((pit_arr >= i / 10) & ((pit_arr < (i + 1) / 10) if i < 9 else True)).sum()) if len(pit_arr) else 0, "expected_share": 0.1} for i in range(10)]
    pg = R.drop_duplicates(["date", "event_id", "player_id"])
    marg_doc = {}
    for s in NEW:
        marg_doc[s] = {}
        for k, (pp, hh) in marg[s].items():
            pp, hh = np.array(pp), np.array(hh)
            marg_doc[s][f"at_least_{k}"] = {"n": int(len(pp)), "buckets": buckets(pp, hh) if len(pp) else [], "verdict": decile_verdict(buckets(pp, hh) if len(pp) else [], len(pp))}
    doc = {"generated": ts(now()), "rule_file": RULE, "label": LABEL,
           "player_games": int(len(pg)), "void": int(pg.void.sum()), "pending_in_log": int(len(snap) - len(pg)),
           "lines": lines_doc, "by_source": by_source,
           "model_median_lines": {"n": int((R.line_source == "model_median").sum()), "note": "graded against the model's own median for calibration only; never count toward the 200"},
           "pit": {"n": int(len(pit_arr)), "deciles": pit_tab, "mean": round(float(pit_arr.mean()), 4) if len(pit_arr) else None,
                   "note": "PIT = CDF(actual FP) from the logged percentiles; a calibrated distribution puts 10% of player-games in every decile"},
           "marginals": marg_doc,
           "recent": pg.sort_values(["date", "fp_mean"], ascending=[False, False]).head(60)[["date", "player", "team", "opp", "min_proj", "min_actual", "fp_mean", "fp_actual", "pit", "void"]].to_dict("records"),
           "method": ["FP = PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - TOV (Underdog). Graded snapshot = the last FP distribution logged before tip (data/fp_log.csv, append-only).",
                      "Underdog FP lines are graded from data/fp_prices.csv (last pre-tip capture per line and side); a push is void. The verdict at 200 graded lines is calibration by decile, not hit rate.",
                      "Amendment 2026-10-25: PrizePicks standard Fantasy Score lines (data/pp_prices.csv, last pre-tip capture per player and game, both sides) are graded through the same code path against the same box-score FP, tagged source = PP. The 200-line counter and the decile verdict apply to UD + PP combined; UD-only and PP-only tables are shown alongside. PrizePicks scoring is identical to Underdog's.",
                      "A player with no box-score minutes is void (shown, excluded). No tuning on these results, ever (backtests/fantasy_points_v1_rule.md)."]}
    write_json(PERF, doc)
    print(f"[fp] graded {len(pg)} player-games ({int(pg.void.sum())} void), {len(gl)} FP lines (UD {lines_doc['sources']['UD']}, PP {lines_doc['sources']['PP']}; gate {doc['lines']['gate']['status']}), PIT mean {doc['pit']['mean']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    if cmd == "price":
        price(sys.argv[2])
    elif cmd == "price_pp":
        price_pp(sys.argv[2] if len(sys.argv) > 2 else PP_FILE)
    else:
        {"priors": priors, "snapshot": snapshot, "grade": grade}[cmd]()
