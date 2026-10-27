"""
Build-time guard (2026-10-27): every docs/data/*.json must parse under the rules a browser's JSON.parse applies -
no NaN, no Infinity. Python's json.load accepts those tokens, so a file can look fine to the builder and still make a
page render nothing. Exit 1 names each bad file; the workflow runs this before committing.

    python validate_site_json.py            # all of docs/data
    python validate_site_json.py a.json b   # specific files
"""

import sys
from pathlib import Path

from common import SITE_DATA, check_json_file


def main(argv):
    paths = [Path(a) for a in argv] or sorted(SITE_DATA.glob("*.json"))
    bad = []
    for p in paths:
        try:
            check_json_file(p)
        except (ValueError, OSError) as e:
            bad.append(str(e))
    print(f"[validate] {len(paths) - len(bad)} of {len(paths)} published JSON files parse strictly")
    for b in bad:
        print(f"[validate] INVALID {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
