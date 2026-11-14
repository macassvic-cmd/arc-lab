"""DraftKings Pick6 NBA board capture, PC side. LOGGED OUT, SLOW, a stock visible Chrome with its own never-signed-in profile.

Why it looks like this (2026-11-14): the eight opening markets load in the browser after the page opens, and DraftKings' edge
refuses a headless browser. A normal Chrome window with a real profile is not refused. This script opens that window, loads
the NBA board, clicks the Quick Hits category and then each of the eight market pills with human-scale pauses, and reads the
data the PAGE ITSELF fetched for each click (React Router `.data` responses). No account is ever used, nothing about the
browser is altered or disguised, and if the board is refused the script stops and says so. It never runs through anyone's
DraftKings login: the profile directory is its own (%USERPROFILE%\\pick6-chrome-profile).

Needs: Google Chrome and the `websockets` package (the pirate-bets scraper venv has it). Standard library otherwise.

  python pick6_capture.py [--out ../data/pick6_board.json] [--keep-raw DIR]

Output (derived fields only; raw responses are never written into the repo): one card per player per market with the posted
line, the sides offered and each side's payout multiplier, paused / locked flags, and the capture timestamp.
"""
import argparse
import asyncio
import base64
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe") if os.path.exists(p)), None)
PROFILE = os.path.expandvars(r"%USERPROFILE%\pick6-chrome-profile")
PORT = 9333
HOME = "https://pick6.draftkings.com/?sport=NBA"
CATEGORY = "Quick Hits"
# pill label on the page -> the market's canonical name in our files
MARKETS = [("1st Point", "1st Point"), ("Team 1st FG", "Team 1st FG"), ("1st Three", "1st Three"), ("Team 1st Three", "Team 1st Three"), ("1st Dunk", "1st Dunk"),
           ("Points 1st 3 Mins", "Points 1st 3 Mins"), ("Rebounds 1st 3 mins", "Rebounds 1st 3 Mins"), ("Assists 1st 3 Mins", "Assists 1st 3 Mins")]
WINDOW = {"Points 1st 3 Mins", "Rebounds 1st 3 Mins", "Assists 1st 3 Mins"}
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "pick6_board.json"


def decode(arr):
    """React Router turbo-stream array -> plain objects."""
    sys.setrecursionlimit(100000)
    cache = {}
    def val(i):
        if isinstance(i, bool) or not isinstance(i, int):
            return i
        if i < 0:
            return None
        if i in cache:
            return cache[i]
        v = arr[i]
        if isinstance(v, dict):
            o = {}; cache[i] = o
            for k, x in v.items():
                o[arr[int(k[1:])] if k.startswith("_") else k] = val(x)
            return o
        if isinstance(v, list):
            if v and isinstance(v[0], str) and len(v[0]) == 1 and v[0].isupper():
                cache[i] = v; return v
            o = []; cache[i] = o
            o.extend(val(x) for x in v)
            return o
        cache[i] = v
        return v
    return val(0)


def find(o, key, depth=0):
    if depth > 7 or not isinstance(o, dict):
        return None
    if key in o:
        return o
    for v in o.values():
        r = find(v, key, depth + 1)
        if r is not None:
            return r
    return None


def pick_cards(text):
    try:                                               # the per-market list is plain JSON
        d = json.loads(text)
        pc = find(d, "pickCardByPickableId") if isinstance(d, dict) else None
        if pc:
            return pc
    except ValueError:
        pass
    # the category's own route data is a turbo-stream array, one per line
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("["):
            try:
                pc = find(decode(json.loads(line)), "pickCardByPickableId")
            except (ValueError, IndexError, KeyError, RecursionError):
                continue
            if pc:
                return pc
    return None


def cards_of(pc, market):
    """Derived rows for one market pill: the card's DEFAULT line only (the line adjuster is out of scope)."""
    rows = []
    for card in pc["pickCardByPickableId"].values():
        ent = (card.get("entities") or [{}])[0]; dk = str(ent.get("dkId"))
        info = (pc.get("entityInfoByDkId") or {}).get(dk) or {}
        comp = (pc.get("competitionById") or {}).get(str((ent.get("compIds") or [None])[0])) or {}
        ec = (comp.get("entityCompByDkId") or {}).get(dk) or {}
        team = ((pc.get("displayTeamById") or {}).get(str(ec.get("teamId"))) or {}).get("name")
        mk = next((m for m in card.get("activePickableMarkets") or [] if m.get("pickableMarketId") == card.get("defaultPickableMarketId")), None) or (card.get("activePickableMarkets") or [None])[0]
        if not mk:
            rows.append({"player": info.get("fullName"), "team": team, "game": comp.get("name"), "market": market, "locked": True, "paused": False, "line": None, "sides": [], "dk_market": None})
            continue
        defn = (pc.get("pickSixMarketById") or {}).get(str(mk.get("pickSixMarketId"))) or {}
        names = defn.get("propositionNameById") or {}
        sides = [{"side": names.get(str(s.get("statLinePropositionId")), str(s.get("statLinePropositionId"))), "multiplier": s.get("standingsMultiplier")} for s in mk.get("activeSelections") or []
                 if s.get("standingsMultiplier") is not None]
        rows.append({"player": info.get("fullName"), "player_short": info.get("name"), "team": team, "position": ec.get("position"), "game": comp.get("name"), "start": comp.get("startTime"),
                     "status_tags": [x.get("text") for x in ec.get("statuses") or []], "market": market, "dk_market": defn.get("name"),
                     "line": None if defn.get("hideTargetLine") else mk.get("targetValue"), "sides": sides,
                     "paused": bool(mk.get("isPaused")), "locked": not sides,                 # locked = no side with a multiplier: not listed for the builder
                     "alt_lines": max(0, len(card.get("activePickableMarkets") or []) - 1)})
    return rows


async def capture(keep_raw=None):
    import websockets
    if not CHROME:
        sys.exit("[pick6] Google Chrome not found")
    proc = subprocess.Popen([CHROME, f"--user-data-dir={PROFILE}", f"--remote-debugging-port={PORT}", "--remote-allow-origins=*", "--no-first-run", "--no-default-browser-check",
                             "--window-size=1280,900", "about:blank"])
    board = {"captured_at": None, "source": "pick6.draftkings.com, logged out, visible Chrome with its own profile", "category": CATEGORY, "markets": {}, "notes": []}
    try:
        tabs = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2)); break
            except Exception:
                time.sleep(0.5)
        if not tabs:
            sys.exit("[pick6] Chrome did not open its DevTools port")
        page = next(t for t in tabs if t.get("type") == "page")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            n = 0; results = {}; data_reqs = []
            async def pump(seconds):
                end = time.time() + seconds
                while time.time() < end:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(0.05, end - time.time())))
                    except asyncio.TimeoutError:
                        break
                    if msg.get("method") == "Network.responseReceived":
                        r = msg["params"]; u = r["response"]["url"]
                        path = u.split("?")[0]
                        # what the page itself fetches: the category's route data, and one pick-card list per market pill
                        if ("pick6.draftkings.com" in u and path.endswith(".data")) or ("api.draftkings.com/pick6/" in u and path.endswith("/pickcards")):
                            data_reqs.append((r["requestId"], r["response"]["status"]))
                    elif "id" in msg:
                        results[msg["id"]] = msg
            async def call(method, wait=20, **params):
                nonlocal n
                n += 1; i = n
                await ws.send(json.dumps({"id": i, "method": method, "params": params}))
                end = time.time() + wait
                while i not in results and time.time() < end:
                    await pump(0.3)
                return results.pop(i, {})
            async def js(expr):
                return ((await call("Runtime.evaluate", expression=expr, returnByValue=True)).get("result") or {}).get("result", {}).get("value")
            async def body(rid):
                b = (await call("Network.getResponseBody", requestId=rid)).get("result")
                if not b:
                    return None
                return (base64.b64decode(b["body"]) if b.get("base64Encoded") else b["body"].encode("utf-8")).decode("utf-8", "replace")
            PRESS = ("(()=>{const want=%s;const all=[...document.querySelectorAll('button,[role=tab],[role=button],a,li,div,span')].filter(e=>(e.innerText||'').trim()===want);"
                     "if(!all.length)return null;const e=all.find(x=>x.tagName==='BUTTON'||x.getAttribute('role')==='tab'||x.getAttribute('role')==='button')||all[all.length-1];"
                     "e.scrollIntoView({block:'center',inline:'center',behavior:'instant'});if(%s){e.click();}return e.tagName+'|'+(e.getAttribute('aria-selected')||'');})()")
            async def click(label, wait_for=10):
                """Press the pill whose visible text is exactly `label` (waits up to `wait_for` s for it to appear): scroll it into
                view, pause, then activate the button the way a click does. Returns False when no such pill is on the page."""
                seen = None
                for _ in range(int(wait_for / 0.5)):
                    seen = await js(PRESS % (json.dumps(label), "false"))
                    if seen:
                        break
                    await pump(0.5)
                if not seen:
                    return False
                await pump(0.9)
                return bool(await js(PRESS % (json.dumps(label), "true")))
            await call("Network.enable"); await call("Page.enable")
            await call("Page.navigate", url=HOME)
            for _ in range(9):                                        # up to 45 s for the page to render
                await pump(5)
                st = await js("document.readyState+'|'+(document.body?document.body.innerText.length:0)+'|'+document.title")
                if st and st.startswith("complete") and int(st.split("|")[1]) > 2000:
                    break
            title = await js("document.title") or ""
            text = await js("document.body?document.body.innerText:''") or ""
            if "Access Denied" in title or "Access Denied" in text[:400] or len(text) < 500:
                board["notes"].append(f"refused or empty page (title {title!r}, {len(text)} chars): stopped, nothing captured")
                return board
            if "Log In" not in text and "Sign Up" not in text:
                board["notes"].append("the page does not show a logged-out header: stopped (this capture must never run inside an account)")
                return board
            if not await click(CATEGORY):
                board["notes"].append(f"category pill {CATEGORY!r} not found on the page")
                return board
            await pump(6)
            for label, name in MARKETS:
                before = len(data_reqs)
                if not await click(label):
                    board["markets"][name] = {"listed": False, "cards": [], "note": f"pill {label!r} not on the page"}
                    continue
                await pump(6.5)                                       # a person reading the list
                pc = None
                # the first pill is already selected when the category opens: its cards came with the category's own data
                pool = data_reqs[before:] or (data_reqs[-1:] if label == MARKETS[0][0] else [])
                for rid, status in reversed(pool):
                    t = await body(rid)
                    pc = pick_cards(t) if t else None
                    if pc:
                        if keep_raw:
                            Path(keep_raw).mkdir(parents=True, exist_ok=True); (Path(keep_raw) / f"{name.replace(' ', '_')}.data").write_text(t, encoding="utf-8")
                        break
                if not pc:
                    board["markets"][name] = {"listed": False, "cards": [], "note": "no data response followed the click"}
                    continue
                rows = cards_of(pc, name)
                board["markets"][name] = {"listed": True, "cards": rows, "dk_markets": sorted({c["dk_market"] for c in rows if c.get("dk_market")})}
            board["captured_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                await call("Browser.close", wait=4)
            except Exception:
                pass
    finally:
        time.sleep(1)
        if proc.poll() is None:
            proc.terminate()
    return board


def summarize(board):
    out = {}
    for name, m in board["markets"].items():
        cards = m["cards"]; live = [c for c in cards if not c["locked"] and not c["paused"]]
        sides = sorted({s["side"] for c in cards for s in c["sides"]})
        mult = [s["multiplier"] for c in live for s in c["sides"]]
        out[name] = {"cards": len(cards), "open": len(live), "paused": sum(c["paused"] for c in cards), "locked": sum(c["locked"] for c in cards), "sides_offered": sides,
                     "cards_with_less": sum(any(s["side"] == "Less" for s in c["sides"]) for c in cards), "multiplier_min": min(mult) if mult else None, "multiplier_max": max(mult) if mult else None,
                     "games": len({c["game"] for c in cards}), "lines": sorted({c["line"] for c in cards if c["line"] is not None})[:8]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(DEFAULT_OUT)); ap.add_argument("--keep-raw")
    a = ap.parse_args()
    board = asyncio.run(capture(a.keep_raw))
    board["summary"] = summarize(board)
    complete = board["captured_at"] and all(board["markets"].get(n, {}).get("listed") for _, n in MARKETS)
    board["complete"] = bool(complete)
    if board["captured_at"]:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(board, f, indent=1, ensure_ascii=False)
        os.replace(tmp, a.out)
    print(json.dumps({"captured_at": board["captured_at"], "complete": board["complete"], "notes": board["notes"], "summary": board["summary"]}, indent=1))
    sys.exit(0 if complete else 2)
