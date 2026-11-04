"""Slips tonight replay fixtures (canonical v2 acceptance a-c): python tests/slips_replay/check.py   (from the repo root; writes nothing)."""
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import slips as S  # noqa: E402

bad = 0
for d in sorted(p for p in HERE.iterdir() if p.is_dir()):
    e = json.load(open(d / "expected.json"))
    c = S.compute(dt.datetime.fromisoformat(e["now"].replace("Z", "+00:00")), d / "strip_log.csv", d / "fp_v2_log.csv", d / "lines.json", None, e["date"])
    got = [f"{x['player']} {x['stat']} {x['side']} {x['line']}" for x in c["slips"].get(8, [])]
    nested = got[:3] == [f"{x['player']} {x['stat']} {x['side']} {x['line']}" for x in c["slips"].get(3, [])] and got[:6] == [f"{x['player']} {x['stat']} {x['side']} {x['line']}" for x in c["slips"].get(6, [])]
    excl = sorted([x for x in c["legs"] if x["out"] in ("minutes floor", "no minutes projection") and x["p"] >= S.P_FLOOR], key=S.rank_key)
    ok = got == e["expected_8_leg"] and nested and bool(excl) and excl[0]["player"] == e["excluded_by_minutes_top"]
    bad += not ok
    print(f"{e['date']}: {'PASS' if ok else 'FAIL'} ({len(c['legs'])} legs, {len(c['ranked'])} qualifying; top minutes-floor exclusion {excl[0]['player'] if excl else None} at {excl[0]['min'] if excl else None} min)")
    if not ok:
        print("  expected:", e["expected_8_leg"]); print("  got:     ", got)
sys.exit(1 if bad else 0)
