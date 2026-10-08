"""Assign PG/SG/SF/PF/C from how players play, since ESPN only gives G/F/C.

Checked 2026-10-08: ESPN box scores, the roster endpoint and the athlete endpoints all
carry generic positions (rosters: G 279, F 240, C 92 of 611). So every player is sorted
within ESPN's own group by per-36 rates over the backfill:

  G  -> PG when assists/36 >= 5.5, else SG
  F  -> C  when rebounds/36 >= 9.5 and blocks/36 >= 2.0 (rim-protecting "forwards", e.g. Wembanyama)
        PF when rebounds/36 >= 7.5, or rebounds/36 >= 6.5 with 3PA/36 < 3.5 (interior forwards)
        SF otherwise
  C  -> C

Players under MIN_MINUTES keep the group default (SG / SF / C) and are marked low_minutes.

Output: data/positions.csv, one row per player with the inputs, pos_auto, and an empty
`override` column. Put PG/SG/SF/PF/C in `override` to pin a player; regenerating keeps it.
build_model.py reads `pos` (= override or pos_auto) keyed by player_id.

  python positions.py            # rebuild data/positions.csv from data/player_games.csv
"""
import pandas as pd

from common import DATA_DIR, POSITIONS

POSITIONS_CSV = DATA_DIR / "positions.csv"
MIN_MINUTES = 150
PG_AST36 = 5.5
C_REB36, C_BLK36 = 9.5, 2.0
PF_REB36 = 7.5
PF_REB36_LOW, PF_TPA36_MAX = 6.5, 3.5
GROUP = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}
GROUP_DEFAULT = {"G": "SG", "F": "SF", "C": "C"}


def classify(group, r):
    if r["min"] < MIN_MINUTES:
        return GROUP_DEFAULT[group]
    if group == "G":
        return "PG" if r["ast36"] >= PG_AST36 else "SG"
    if group == "F":
        if r["reb36"] >= C_REB36 and r["blk36"] >= C_BLK36:
            return "C"
        if r["reb36"] >= PF_REB36 or (r["reb36"] >= PF_REB36_LOW and r["tpa36"] < PF_TPA36_MAX):
            return "PF"
        return "SF"
    return "C"


def build(df):
    """df: player_games rows (pos already normalized to PG/SG/SF/PF/C). Returns the positions table."""
    d = df.copy()
    d["group"] = d["pos"].map(GROUP).fillna("F")
    agg = d.groupby("player_id").agg(
        player=("player", "last"), gp=("event_id", "nunique"), min=("min", "sum"),
        ast=("ast", "sum"), reb=("reb", "sum"), tpa=("tpa", "sum"), blk=("blk", "sum"),
        group=("group", lambda s: s.value_counts().index[0]),
    ).reset_index()
    for s in ("ast", "reb", "tpa", "blk"):
        agg[f"{s}36"] = (agg[s] / agg["min"].clip(lower=1) * 36).round(2)
    agg["espn_pos"] = agg["group"]
    agg["low_minutes"] = agg["min"] < MIN_MINUTES
    agg["pos_auto"] = [classify(g, r) for g, r in zip(agg["group"], agg.to_dict("records"))]
    out = agg[["player_id", "player", "espn_pos", "gp", "min", "ast36", "reb36", "tpa36", "blk36", "low_minutes", "pos_auto"]].copy()
    out["min"] = out["min"].round(0).astype(int)
    out["override"] = ""
    return out.sort_values(["pos_auto", "min"], ascending=[True, False]).reset_index(drop=True)


def load_overrides(path=POSITIONS_CSV):
    try:
        prev = pd.read_csv(path, dtype={"player_id": str, "override": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}
    prev["override"] = prev["override"].fillna("").str.strip().str.upper()
    return {pid: o for pid, o in zip(prev["player_id"], prev["override"]) if o in POSITIONS}


def write(df, path=POSITIONS_CSV):
    table = build(df)
    overrides = load_overrides(path)
    table["override"] = table["player_id"].map(overrides).fillna("")
    table["pos"] = table["override"].where(table["override"] != "", table["pos_auto"])
    table.to_csv(path, index=False, lineterminator="\n")
    return table


def position_map(path=POSITIONS_CSV):
    """{player_id: pos} from data/positions.csv (override wins), {} if the file is absent."""
    try:
        t = pd.read_csv(path, dtype={"player_id": str, "override": str, "pos": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}
    return dict(zip(t["player_id"], t["pos"]))


if __name__ == "__main__":
    from build_model import load_games
    games = load_games(apply_positions=False)
    table = write(games)
    counts = table.groupby("pos").size().reindex(POSITIONS).fillna(0).astype(int)
    print("[positions] players per position:", counts.to_dict(), "| overrides:", int((table["override"] != "").sum()))
