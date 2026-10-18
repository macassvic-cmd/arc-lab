"""Play-style positions for a historical replay, built from the replayed season and the one before it only (no later-season
data), written beside the live table.   python build_positions_season.py --season 2025 --out ../data/positions_2025.csv"""
import argparse
from pathlib import Path

import build_model as bm
import positions as P

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
df = bm.load_games(apply_positions=False, include_history=True)
df = df[(df.season <= a.season) & (df.season >= a.season - 1)].copy()
tab = P.write(df, Path(a.out))
print(f"positions from seasons {a.season - 1}-{a.season}: {len(tab)} players -> {a.out}")
