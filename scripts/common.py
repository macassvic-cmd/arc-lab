"""Shared helpers for Arc Lab."""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # raw game logs (not served)
SITE_DATA = ROOT / "docs" / "data"  # JSON the site reads
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
# ESPN abbreviations for the 30 teams. Anything else (All-Star STARS/STRIPES/WORLD, Rising Stars) is not a real game.
NBA_TEAMS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL", "MEM",
             "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH"}

# ESPN sometimes lists generic G / F / G-F etc. Map them to a single bucket.
POS_MAP = {"PG": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "C": "C",
           "G": "SG", "F": "SF", "G-F": "SG", "F-G": "SF", "F-C": "PF", "C-F": "C"}


def norm_pos(abbr):
    return POS_MAP.get((abbr or "").upper().strip(), "SF")


def norm_name(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


# ---- injury / lineup news classifier (first match = primary tag) ----
RULES = [
    ("OUT", r"\b(ruled out|will not (play|return)|won'?t (play|return)|not available|unavailable|"
            r"will miss|sidelined|DNP|out(?! of)\b)"),
    ("DOUBTFUL", r"\bdoubtful\b"),
    ("QUESTIONABLE", r"\b(questionable|game[- ]time decision|GTD)\b"),
    ("MIN RESTRICTION", r"\b(minutes? (restriction|limit)|limited minutes)\b"),
    ("STARTING", r"\b(will start|to start|starting|starting lineup|enter the lineup)\b"),
    ("AVAILABLE", r"\b(will play|is available|upgraded|cleared|probable|will return|returns?|active)\b"),
]
RULES = [(t, re.compile(p, re.IGNORECASE)) for t, p in RULES]
NAME_RE = re.compile(r"^([A-Z][\w'.\-]+(?:\s(?:[A-Z][\w'.\-]+|Jr\.|Sr\.|III|II|IV)){1,3})")


def classify(text):
    tags = [t for t, rx in RULES if rx.search(text)]
    return (tags[0] if tags else "NEWS"), tags


def player_from_text(text):
    m = NAME_RE.match(text.strip())
    return m.group(1) if m else None
