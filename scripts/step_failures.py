"""Post the update-data run's failed product steps to Discord.

The workflow runs each product's grade / snapshot / pricing step with continue-on-error, so one product's failure cannot
stop another's pre-tip logging (2026-10-28). This step runs last with `if: always()` and STEPS = ${{ toJSON(steps) }}: every
step with an id and outcome "failure" is named in one Discord message (DISCORD_WEBHOOK_URL), with the run link. Nothing is
posted when every step succeeded. Exit 0 always: reporting must not fail the run.

  STEPS='{"fp2_grade": {"outcome": "failure", ...}}' python step_failures.py
"""
import json
import os
import sys

from health import notify


def failed_steps(steps):
    return sorted(k for k, v in (steps or {}).items() if isinstance(v, dict) and v.get("outcome") == "failure")


def main():
    try:
        steps = json.loads(os.environ.get("STEPS") or "{}")
    except ValueError as e:
        print(f"[step-failures] STEPS is not JSON ({e}); nothing reported")
        return 0
    bad = failed_steps(steps)
    run = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    if not bad:
        print(f"[step-failures] all {len(steps)} identified steps succeeded")
        return 0
    text = f"❌ Arc Lab update-data: {len(bad)} product step(s) failed, the rest of the run continued: {', '.join(bad)}. {run}"
    print(f"[step-failures] {text}")
    notify(text)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
