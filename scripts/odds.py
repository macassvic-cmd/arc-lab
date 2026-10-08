"""OddsBlaze client for Arc Lab: sharp player-prop prices and game spreads.

Key comes from ODDSBLAZE_API_KEY (environment, or a .env at the repo root).
Everything degrades to "no odds" when the key is missing or a call fails, so the
site and the model keep working without it.

Endpoint (https://docs.oddsblaze.com/endpoints/odds):
  GET https://odds.oddsblaze.com/?key&sportsbook&league[&market&market_contains&main&live&price&event]
  -> {"updated", "league", "sportsbook", "events": [{"id", "teams": {"away": {...}, "home": {...}},
      "date", "live", "odds": [{"id", "market", "name", "price", "main", "sgp",
      "selection": {"name", "side", "line"}, "player": {"id", "name", "team", ...}}]}]}
"""
import logging
import os
import re
import time
import unicodedata

import requests

log = logging.getLogger("arc.odds")

ODDS_URL = "https://odds.oddsblaze.com/"
TIMEOUT = 30
MIN_SPACING = 1.3            # OddsBlaze 429s bursts even under the 30/min cap
SHARP_BOOKS = [b.strip() for b in os.environ.get("ODDSBLAZE_SHARP_BOOKS", "pinnacle,circa,draftkings,fanduel").split(",") if b.strip()]
STATS = ["3PM", "PTS", "REB", "AST", "PRA"]

# ESPN's abbreviations are the canonical ones everywhere in Arc Lab.
ESPN_TEAMS = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans", "NY": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SA": "San Antonio Spurs", "SAC": "Sacramento Kings", "TOR": "Toronto Raptors",
    "UTAH": "Utah Jazz", "WSH": "Washington Wizards",
}
TEAM_ALIASES = {"GSW": "GS", "NOP": "NO", "NOR": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH",
                "PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
_NAME_TO_ABBR = {v.lower(): k for k, v in ESPN_TEAMS.items()}
_NAME_TO_ABBR.update({v.split()[-1].lower(): k for k, v in ESPN_TEAMS.items()})  # "Warriors"
_NAME_TO_ABBR["trail blazers"] = "POR"
_NAME_TO_ABBR["los angeles clippers"] = "LAC"


def team_abbr(value):
    """Any spelling of a team (ESPN/Underdog/OddsBlaze abbreviation, full or nick name) -> ESPN abbreviation."""
    if not value:
        return None
    s = str(value).strip()
    u = s.upper()
    if u in ESPN_TEAMS:
        return u
    if u in TEAM_ALIASES:
        return TEAM_ALIASES[u]
    return _NAME_TO_ABBR.get(s.lower())


def norm_name(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def american_to_prob(price):
    try:
        a = float(str(price).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def devig(over_price, under_price):
    """Fair P(over) from a two-way price (multiplicative devig). None if either side is missing."""
    po, pu = american_to_prob(over_price), american_to_prob(under_price)
    if po is None or pu is None or po + pu <= 0:
        return None
    return po / (po + pu)


def stat_for_market(market):
    """OddsBlaze market name -> Arc Lab stat code, or None for anything we don't model
    (quarters, halves, doubles, combos other than PRA)."""
    m = (market or "").lower()
    if not m or any(w in m for w in ("1st", "first", "2nd", "3rd", "4th", "quarter", "half", "double", "team ")):
        return None
    pts, reb, ast = "point" in m, "rebound" in m, "assist" in m
    three = "three" in m or "3-point" in m or "3 point" in m or "3pt" in m or "3-pt" in m
    if three:
        return "3PM" if not (reb or ast) else None
    if pts and reb and ast:
        return "PRA"
    if pts and not reb and not ast:
        return "PTS"
    if reb and not pts and not ast:
        return "REB"
    if ast and not pts and not reb:
        return "AST"
    return None


class OddsBlaze:
    def __init__(self, key=None):
        self.key = key or os.environ.get("ODDSBLAZE_API_KEY", "").strip() or _key_from_dotenv()
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "arc-lab/0.1 (prop research)"
        self._last = 0.0

    @property
    def enabled(self):
        return bool(self.key)

    def get(self, sportsbook, league="nba", **params):
        """One odds call. Returns the parsed body, or None on any failure (logged, never raised)."""
        if not self.enabled:
            return None
        q = {"key": self.key, "sportsbook": sportsbook, "league": league}
        q.update({k: v for k, v in params.items() if v is not None})
        for attempt in range(3):
            gap = MIN_SPACING - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            try:
                r = self.s.get(ODDS_URL, params=q, timeout=TIMEOUT)
            except requests.RequestException as e:
                log.warning("oddsblaze %s/%s: %s", sportsbook, league, e)
                time.sleep(2 * (attempt + 1))
                continue
            finally:
                self._last = time.monotonic()
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    log.warning("oddsblaze %s/%s: non-JSON body", sportsbook, league)
                    return None
            if r.status_code in (401, 403):
                log.warning("oddsblaze: key rejected (HTTP %s) - set a valid ODDSBLAZE_API_KEY", r.status_code)
                return None
            if r.status_code in (408, 425, 429, 500, 502, 503, 504):
                try:
                    wait = float(r.headers.get("Retry-After") or 0)
                except ValueError:
                    wait = 0
                time.sleep(min(wait or 3 * (attempt + 1), 60))
                continue
            log.warning("oddsblaze %s/%s: HTTP %s", sportsbook, league, r.status_code)
            return None
        return None

    # ---------- player props ----------
    def player_props(self, league="nba", books=None):
        """{(norm player, stat): {"book", "player", "main", "lines": {line: {"over": price, "under": price}}}}
        from the first book in `books` that answers with player markets."""
        for book in books or SHARP_BOOKS:
            body = self.get(book, league, market_contains="Player")
            idx = _index_props(body, book)
            if idx:
                log.info("oddsblaze: %d player/stat props from %s", len(idx), book)
                return idx
        return {}

    # ---------- spreads / totals ----------
    def spreads(self, league="nba", books=None):
        """{(away_abbr, home_abbr): {"home_spread": -4.5, "total": 228.5, "book": "pinnacle", "start": iso}}.
        home_spread < 0 means the home team is favored."""
        for book in books or SHARP_BOOKS:
            body = self.get(book, league, market_contains="Spread")
            out = _index_spreads(body, book)
            if out:
                _add_totals(out, self.get(book, league, market_contains="Total"))
                log.info("oddsblaze: spreads for %d games from %s", len(out), book)
                return out
        return {}


def _key_from_dotenv():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(root, ".env"), os.path.join(os.getcwd(), ".env")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"^\s*ODDSBLAZE_API_KEY\s*=\s*(.+?)\s*$", line)
                    if m:
                        return m.group(1).strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _index_props(body, book):
    idx = {}
    for ev in (body or {}).get("events") or []:
        for o in ev.get("odds") or []:
            stat = stat_for_market(o.get("market"))
            if not stat:
                continue
            sel = o.get("selection") or {}
            name = (o.get("player") or {}).get("name") or sel.get("name")
            side = (sel.get("side") or "").lower()
            if not name or side not in ("over", "under") or sel.get("line") is None:
                continue
            try:
                line = float(sel["line"])
            except (TypeError, ValueError):
                continue
            rec = idx.setdefault((norm_name(name), stat), {"book": book, "player": name, "main": None, "lines": {}})
            rec["lines"].setdefault(line, {})[side] = o.get("price")
            if o.get("main"):
                rec["main"] = line
    return idx


def _game_key(ev):
    teams = ev.get("teams") or {}
    away_t, home_t = teams.get("away") or {}, teams.get("home") or {}
    away = team_abbr(away_t.get("abbreviation")) or team_abbr(away_t.get("name"))
    home = team_abbr(home_t.get("abbreviation")) or team_abbr(home_t.get("name"))
    return (away, home) if away and home else None


_PERIOD_WORDS = ("1st", "first", "2nd", "3rd", "4th", "quarter", "half")


def _index_spreads(body, book):
    out = {}
    for ev in (body or {}).get("events") or []:
        key = _game_key(ev)
        if not key:
            continue
        away, home = key
        cands = []
        for o in ev.get("odds") or []:
            m = (o.get("market") or "").lower()
            if "spread" not in m or any(w in m for w in _PERIOD_WORDS):
                continue
            sel = o.get("selection") or {}
            side = team_abbr(sel.get("name"))
            try:
                line = float(sel.get("line"))
            except (TypeError, ValueError):
                continue
            if side == home:
                cands.append((bool(o.get("main")), line, o.get("price")))
            elif side == away:
                cands.append((bool(o.get("main")), -line, o.get("price")))
        if not cands:
            continue
        mains = [c for c in cands if c[0]]
        if mains:
            home_spread = mains[0][1]
        else:  # no main flag: the most evenly priced line is the main one
            home_spread = min(cands, key=lambda c: abs((american_to_prob(c[2]) or 0.5) - 0.5))[1]
        out[(away, home)] = {"home_spread": home_spread, "total": None, "book": book, "start": ev.get("date")}
    return out


def _add_totals(out, body):
    for ev in (body or {}).get("events") or []:
        key = _game_key(ev)
        if key not in out:
            continue
        best = None
        for o in ev.get("odds") or []:
            m = (o.get("market") or "").lower()
            if "total" not in m or "team" in m or "player" in m or any(w in m for w in _PERIOD_WORDS):
                continue
            sel = o.get("selection") or {}
            try:
                line = float(sel.get("line"))
            except (TypeError, ValueError):
                continue
            score = (0 if o.get("main") else 1, abs((american_to_prob(o.get("price")) or 0.5) - 0.5))
            if best is None or score < best[0]:
                best = (score, line)
        if best:
            out[key]["total"] = best[1]
