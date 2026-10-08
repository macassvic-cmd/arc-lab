"""Append one injury/lineup news item to docs/data/news.json.

Called by the news workflow with the repository_dispatch payload in $NEWS_PAYLOAD:
  {"text": "<notification text>", "ts": "<optional ISO time>"}
"""
import datetime as dt
import json
import os
import re

from common import SITE_DATA, classify, player_from_text, read_json, write_json

NEWS = SITE_DATA / "news.json"
KEEP = 400


def clean(text):
    text = re.sub(r"^\s*Underdog\s*NBA\s*[:\-]?\s*", "", text or "", flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def main():
    payload = json.loads(os.environ.get("NEWS_PAYLOAD") or "{}")
    text = clean(payload.get("text", ""))
    if not text:
        print("empty payload")
        return
    now = dt.datetime.utcnow().replace(microsecond=0)
    ts = payload.get("ts") or now.isoformat() + "Z"
    news = read_json(NEWS, [])
    recent = {n["text"] for n in news[-50:]}
    if text in recent:
        print("duplicate, skipped")
        return
    tag, tags = classify(text)
    news.append({"ts": ts, "received": now.isoformat() + "Z", "text": text,
                 "tag": tag, "tags": tags, "player": player_from_text(text)})
    write_json(NEWS, news[-KEEP:])
    print(f"[{tag}] {text[:100]}")


if __name__ == "__main__":
    main()
