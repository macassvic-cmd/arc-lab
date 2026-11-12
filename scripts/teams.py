"""Team abbreviations: ESPN's are the canonical ones everywhere in Arc Lab. Standard library only (the pirate-bets scraper hook
runs underdog_lines.py with a venv that has requests but no pandas)."""

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
    """Any spelling of a team (ESPN/Underdog/sportsbook abbreviation, full or nick name) -> ESPN abbreviation."""
    if not value:
        return None
    s = str(value).strip()
    u = s.upper()
    if u in ESPN_TEAMS:
        return u
    if u in TEAM_ALIASES:
        return TEAM_ALIASES[u]
    return _NAME_TO_ABBR.get(s.lower())
