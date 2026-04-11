#!/usr/bin/env python3
"""Zotero Web API helper for the manuscript bibliography.

Handles dedup + batched item creation via the Zotero Web API
(https://api.zotero.org). Deliberately avoids the local connector
(127.0.0.1:23119) because it has no idempotency and caused duplicate
imports on 2026-04-09.

Env (auto-loaded from ./.env by walking up from the script dir):
    ZOTERO_API_KEY   Web API key
    ZOTERO_USER_ID   numeric user id (or group id)

CLI:
    python zotero_api.py dedup <doi-or-query>
    python zotero_api.py add <items.json> [--yes] [--dry-run]

The <items.json> file is a JSON list of Zotero item objects, each at
minimum containing `itemType`, `title`, `creators`, and `DOI`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.zotero.org"
ZOTERO_API_VERSION = "3"


# ---------- env + http ----------

def _load_env() -> tuple[str, str]:
    """Auto-load .env by walking up from this file; return (api_key, user_id)."""
    here = pathlib.Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        f = p / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
    key = os.environ.get("ZOTERO_API_KEY")
    uid = os.environ.get("ZOTERO_USER_ID")
    if not key or not uid:
        sys.exit("Missing ZOTERO_API_KEY or ZOTERO_USER_ID (set in .env)")
    return key, uid


def _headers(api_key: str, write: bool = False) -> dict:
    h = {
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Authorization": f"Bearer {api_key}",
    }
    if write:
        h["Content-Type"] = "application/json"
    return h


def _request(method: str, url: str, headers: dict, body=None) -> tuple[int, str]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ---------- public API ----------

def search(query: str, qmode: str = "everything", limit: int = 10) -> list:
    """Free-text search against the user's library."""
    key, uid = _load_env()
    qs = urllib.parse.urlencode(
        {"q": query, "qmode": qmode, "format": "json", "limit": limit}
    )
    url = f"{API_BASE}/users/{uid}/items?{qs}"
    status, body = _request("GET", url, _headers(key))
    if status != 200:
        sys.exit(f"Search failed ({status}): {body[:300]}")
    return json.loads(body)


def search_by_doi(doi: str) -> list:
    """Search by DOI — DO NOT use for dedup during add.

    Zotero's `q=` with qmode=everything indexes title/creators/year/full-text/
    notes/tags but NOT the DOI field, so DOI-string search is unreliable.
    Kept for CLI exploration only. Use `find_duplicate(item)` for reliable
    dedup during import.
    """
    if not doi:
        return []
    hits = search(doi)
    doi_l = doi.lower()
    return [h for h in hits if (h["data"].get("DOI") or "").lower() == doi_l]


def _title_probe(title: str) -> str:
    """Extract a distinctive query chunk from a title (first ~6 words, no punctuation)."""
    import re
    words = re.findall(r"\w+", title)
    return " ".join(words[:6])


def find_duplicate(item: dict) -> list:
    """Reliable dedup for an item we are about to import.

    Strategy: search by a distinctive title fragment (title IS indexed),
    then filter hits by exact DOI (preferred) or exact title match.
    """
    title = (item.get("title") or "").strip()
    doi = (item.get("DOI") or "").strip().lower()
    if not title:
        return []
    hits = search(_title_probe(title), limit=25)
    if doi:
        exact = [h for h in hits if (h["data"].get("DOI") or "").lower() == doi]
        if exact:
            return exact
    # Fallback: exact title match (case-insensitive)
    t_l = title.lower()
    return [h for h in hits if (h["data"].get("title") or "").strip().lower() == t_l]


def create_items(items: list) -> dict:
    """POST a batch of Zotero items. Automatically chunks into batches of 50."""
    if not items:
        return {"successful": {}, "failed": {}, "unchanged": {}}
    key, uid = _load_env()
    url = f"{API_BASE}/users/{uid}/items"
    merged: dict = {"successful": {}, "failed": {}, "unchanged": {}}
    for start in range(0, len(items), 50):
        batch = items[start : start + 50]
        if start > 0:
            print(f"  (sending batch {start // 50 + 1}...)", flush=True)
        status, body = _request("POST", url, _headers(key, write=True), batch)
        if status not in (200, 201):
            sys.exit(f"Create failed ({status}): {body[:500]}")
        result = json.loads(body)
        for bucket in ("successful", "failed", "unchanged"):
            for k, v in result.get(bucket, {}).items():
                merged[bucket][str(start + int(k))] = v
    return merged


# ---------- internal-field stripping ----------

def strip_internal_fields(items: list[dict]) -> list[dict]:
    """Remove _-prefixed keys (e.g. _verification) before sending to Zotero."""
    return [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]


# ---------- item builders ----------

def creator(first: str, last: str, kind: str = "author") -> dict:
    return {"creatorType": kind, "firstName": first, "lastName": last}


def journal_article(
    title: str,
    authors: list[tuple[str, str]],
    journal: str,
    year: str,
    *,
    volume: str = "",
    issue: str = "",
    pages: str = "",
    doi: str = "",
    url: str = "",
    extra: str = "",
) -> dict:
    return {
        "itemType": "journalArticle",
        "title": title,
        "creators": [creator(f, l) for f, l in authors],
        "publicationTitle": journal,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "date": year,
        "DOI": doi,
        "url": url or (f"https://doi.org/{doi}" if doi else ""),
        "extra": extra,
    }


def conference_paper(
    title: str,
    authors: list[tuple[str, str]],
    proceedings: str,
    year: str,
    *,
    pages: str = "",
    doi: str = "",
    url: str = "",
    extra: str = "",
) -> dict:
    return {
        "itemType": "conferencePaper",
        "title": title,
        "creators": [creator(f, l) for f, l in authors],
        "proceedingsTitle": proceedings,
        "pages": pages,
        "date": year,
        "DOI": doi,
        "url": url or (f"https://doi.org/{doi}" if doi else ""),
        "extra": extra,
    }


def preprint(
    title: str,
    authors: list[tuple[str, str]],
    year: str,
    *,
    arxiv_id: str = "",
    doi: str = "",
    url: str = "",
    extra: str = "",
) -> dict:
    return {
        "itemType": "preprint",
        "title": title,
        "creators": [creator(f, l) for f, l in authors],
        "date": year,
        "DOI": doi,
        "url": url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
        "extra": f"arXiv:{arxiv_id}" + (f"\n{extra}" if extra else "") if arxiv_id else extra,
    }


# ---------- CLI ----------

def cmd_dedup(args):
    q = args.query
    hits = search_by_doi(q) if q.startswith("10.") else search(q)
    print(f"{len(hits)} hit(s)")
    for h in hits:
        d = h["data"]
        print(f"  {h['key']}  {d.get('DOI') or '-'}  {(d.get('title') or '')[:80]}")


def cmd_add(args):
    items = json.loads(pathlib.Path(args.file).read_text())
    if not isinstance(items, list):
        sys.exit("Input JSON must be a list of Zotero item objects")

    # Strip _verification and other internal fields before dedup/POST
    clean_items = strip_internal_fields(items)

    to_post, skipped = [], []
    for it in clean_items:
        hits = find_duplicate(it)
        (skipped if hits else to_post).append((it, hits))

    to_post_items = [it for it, _ in to_post]
    print(f"To import: {len(to_post_items)}  |  Already in library: {len(skipped)}")
    for it, hits in skipped:
        print(f"  SKIP  {(it.get('title') or '')[:70]}  -> {hits[0]['key']}")
    for it, _ in to_post:
        print(f"  NEW   {(it.get('title') or '')[:70]}")

    if args.dry_run or not to_post_items:
        return
    if not args.yes:
        reply = input("Proceed with import? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return

    result = create_items(to_post_items)
    succ = result.get("successful", {})
    fail = result.get("failed", {})
    unch = result.get("unchanged", {})
    print(f"\nImported: {len(succ)}  Failed: {len(fail)}  Unchanged: {len(unch)}")
    for k, v in succ.items():
        print(f"  OK    {v['key']}  {(v['data'].get('title') or '')[:70]}")
    for k, v in fail.items():
        print(f"  FAIL  idx={k}  {v}")


def main():
    p = argparse.ArgumentParser(description="Zotero Web API helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_d = sub.add_parser("dedup", help="Search library by DOI or free text")
    p_d.add_argument("query")
    p_d.set_defaults(func=cmd_dedup)

    p_a = sub.add_parser("add", help="Add items from a JSON file (with dedup)")
    p_a.add_argument("file")
    p_a.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p_a.add_argument("--dry-run", action="store_true", help="Show plan, do not POST")
    p_a.set_defaults(func=cmd_add)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
