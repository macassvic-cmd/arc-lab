"""Publish-path helper for the Sam Method workflow (2026-11-16, after the 2026-11-15 commit-step failure).

The job can sit for hours waiting for 13:00 PT, so main moves under it; a rebase of the job's commit then conflicts on the
derived JSON. This helper makes the commit independent of how old the job's tree is:

  python sam_commit.py save    remember the rows this job APPENDED to the append-only logs (everything past what HEAD has)
  python sam_commit.py apply   after `git reset --hard origin/main`: append those rows to the fresh files (skipping any that
                               are already there), restore tonight_meta.json; the caller then regenerates the derived JSON
                               with `sam_optimizer.py publish` instead of merging it

Only whole appended lines move. Rows that upstream changed in place (the morning grade fills hit / status) are kept as upstream
has them."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["data/sam_method/shadow_log.csv", "data/sam_method/runs_log.csv", "data/sam_method/capture_log.csv", "data/oddsapi_props_credits.csv"]
META = "data/sam_method/tonight_meta.json"
STASH = Path("/tmp/sam_commit.json") if sys.platform != "win32" else ROOT / ".sam_commit.json"


def lines(text):
    return [l for l in text.replace("\r\n", "\n").split("\n") if l.strip()]


def save():
    out = {"new": {}, "meta": None}
    for f in FILES:
        p = ROOT / f
        if not p.exists():
            continue
        head = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
        base = set(lines(head.stdout)) if head.returncode == 0 else set()
        new = [l for l in lines(p.read_text(encoding="utf-8")) if l not in base]
        if new:
            out["new"][f] = {"header": lines(p.read_text(encoding="utf-8"))[0], "rows": new}
    if (ROOT / META).exists():
        out["meta"] = (ROOT / META).read_text(encoding="utf-8")
    STASH.write_text(json.dumps(out), encoding="utf-8")
    print("[sam-commit] saved", {f: len(v["rows"]) for f, v in out["new"].items()}, "new rows")


def apply():
    out = json.loads(STASH.read_text(encoding="utf-8"))
    for f, v in out["new"].items():
        p = ROOT / f
        have = lines(p.read_text(encoding="utf-8")) if p.exists() else []
        add = [l for l in v["rows"] if l not in set(have) and l != v["header"]]
        if not have:
            add = [v["header"]] + add
        if add:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8", newline="\n") as fh:
                fh.write("".join(l + "\n" for l in add))
        print(f"[sam-commit] {f}: appended {len(add)} rows")
    if out.get("meta"):
        (ROOT / META).write_text(out["meta"], encoding="utf-8")


if __name__ == "__main__":
    {"save": save, "apply": apply}[sys.argv[1]]()
