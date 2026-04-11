# Zotero

This folder is used to manage the **bibliography for the manuscript**. It holds the list of references relevant to the paper and is the interface between our source files and the Zotero reference manager itself (Zotero desktop app + Web API). Anything we cite in `paper.md`, `methods.md`, `supplementary_information.md`, or figure captions should be tracked here, and mirrored in the Zotero library so BibTeX export stays in sync.

The user will provide the Zotero Web API key separately — do **not** hardcode it. Read it from the environment when needed and never print it in logs or commits.

## Environment

- **Zotero desktop app**: standard macOS/Linux install; local data dir `~/Zotero/` with `zotero.sqlite` (read-only if touched directly).
- **Local connector** (used by the browser extension and scripts): `http://127.0.0.1:23119`. Ping: `GET /connector/ping` — must return 200 before any import attempt.
- **Zotero Web API**: `https://api.zotero.org` — preferred for deduplication, deletion, and targeting a specific collection. Requires the API key and user/group ID.

## Hard rules

1. **Never import unverified metadata — zero tolerance for hallucinated references.** Every single field — title, authors (full names, order, spelling), venue, year, volume, issue, pages, DOI — must be **independently verified online** against the publisher page, DOI resolver, or arXiv listing before import. Do not rely on memory, training data, or `references_needed.md` as a source of truth; those are candidates only. If you cannot pull up an authoritative online source that confirms every field, the reference is **unverified** and must not be imported (see rule 5).
2. **Never import without explicit user confirmation.** Present the fully verified list first; wait for the user to say go.
3. **Never create duplicates.** Always dedup against the existing library before importing.
4. **Always flag preprint vs. published.** If a paper exists as both, prefer the published version and note the arXiv ID in the entry.
5. **Never import unverifiable references.** If a reference cannot be fully verified online (e.g., DOI does not resolve, publisher page is unavailable, author list cannot be confirmed), it must **not** be added to Zotero. Instead, set its `_verification.status` to `"flagged"` with notes in `unverified_references.json`. It stays there until the user manually resolves it.

## Three-stage import pipeline

The pipeline uses three files to move references from candidates to Zotero, with an explicit verification checkpoint between structuring and importing:

```
references_needed.md            (1) Candidate list — rough info, grouped by section
        ↓  Stage 1: Populate
unverified_references.json      (2) Structured JSON — all fields filled, NOT yet verified
        ↓  Stage 2: Verify
verified_references.json        (3) Verified JSON — every field confirmed online, ready to import
        ↓  Stage 3: Import
Zotero library                  (4) Final destination
```

All Zotero API interaction goes through `manuscript/zotero/zotero_api.py` (Web API, not the local connector — see "Duplicate prevention" section for why).

### Stage 1: Populate (`references_needed.md` → `unverified_references.json`)

Take candidates from `references_needed.md` and build structured Zotero-shaped JSON objects with all fields filled in (best-effort). Each entry gets a `_verification` metadata block:

```json
{
  "itemType": "journalArticle",
  "title": "...",
  "creators": [...],
  "DOI": "...",
  "_verification": {
    "status": "unverified",
    "source": "",
    "notes": ""
  }
}
```

This stage is explicitly allowed to contain errors — it is a structured draft, not a source of truth. Use the `journal_article()`, `conference_paper()`, or `preprint()` builders from `zotero_api.py` and add `_verification` manually. Append new entries to `unverified_references.json`.

### Stage 2: Verify (`unverified_references.json` → `verified_references.json`)

**This is the most critical step — zero tolerance for hallucinated metadata.**

For each entry in `unverified_references.json`:

1. **Fetch the authoritative source online** — publisher page (via DOI resolver), arXiv abstract page, or Google Scholar. You MUST actually visit/fetch the page.
2. **Compare every field** against the source: exact title, every author (full first name, last name, correct spelling, correct order), venue name, year, volume, issue, pages, DOI. **No field may come from memory or training data alone.**
3. **Correct any errors** in the JSON.
4. **Record provenance**: set `_verification.status` to `"verified"` and `_verification.source` to the URL you checked.
5. **Move the corrected entry** to `verified_references.json`.

If a reference **cannot be fully verified** (DOI doesn't resolve, publisher page unavailable, author list unconfirmable):
- Set `_verification.status` to `"flagged"` with `_verification.notes` explaining what failed.
- **Leave it in `unverified_references.json`** — do NOT move it to verified. It stays flagged until the user manually resolves it.
- Do **not** import flagged references into Zotero under any circumstances.

### Stage 3: Import (`verified_references.json` → Zotero)

1. **Dry-run**: `python manuscript/zotero/zotero_api.py add manuscript/zotero/verified_references.json --dry-run`
   Shows `NEW` vs `SKIP -> <key>` for each entry. The `_verification` fields are automatically stripped before dedup/POST.
2. **Present** the dry-run output to the user and **wait for explicit confirmation**.
3. **Import**: `python manuscript/zotero/zotero_api.py add manuscript/zotero/verified_references.json --yes`
4. **Re-run the dry-run** to confirm dedup now catches the just-imported items (idempotency smoke test).

Already-imported items stay in `verified_references.json` as a record — the dedup check skips them on future runs.

### File roles summary

| File | Role | Mutability |
|---|---|---|
| `references_needed.md` | Candidate list (rough, grouped by section) | Append new candidates; remove once populated into JSON |
| `unverified_references.json` | Structured but unverified; also holds flagged items | Entries move out as they get verified |
| `verified_references.json` | Verified and ready for import; also serves as archive | Append-only (verified entries accumulate here) |

The old connector path (`POST http://127.0.0.1:23119/connector/import`) is **disfavored**: it caused the 2026-04-09 duplication incident, has no delete endpoint, and has no idempotency. Only fall back to it if the Web API is unavailable.

## Helper library: `zotero_api.py`

Located at `manuscript/zotero/zotero_api.py`. Standard library only (no extra deps). Key surfaces:

- `search(query, qmode="everything", limit=10)` — free-text search over the user's library.
- `find_duplicate(item)` — **reliable dedup.** Searches by a distinctive fragment of the item's title (title IS indexed), then filters hits by exact DOI match (preferred) or exact title match. Use this, not `search_by_doi`, during any import flow.
- `search_by_doi(doi)` — kept for CLI exploration only. See the gotcha below — this is NOT reliable on its own.
- `create_items(items)` — POST a batch (max 50) to `/users/<id>/items`.
- `journal_article(...)`, `conference_paper(...)`, `preprint(...)` — builders that return Zotero-shaped dicts. Use these to populate `unverified_references.json`; add `_verification` metadata manually.
- `strip_internal_fields(items)` — removes `_verification` and any other `_`-prefixed keys before POST. Called automatically by `cmd_add`.
- CLI: `dedup <query>` and `add <file.json> [--yes] [--dry-run]`.

Refine this file over time — add `--collection <key>` for targeted imports, add a `delete <key>` command (the connector can't delete, but the Web API can via `DELETE /users/<id>/items/<key>`). Treat it as the canonical entry point, not a throwaway.

## Duplicate prevention and Zotero-search gotchas

**2026-04-09 incident:** 8 unverified references were imported via the connector and then re-imported, leaving duplicates in the library. Cleanup had to go through the Web API or manual UI deletion. Rule (3) is non-negotiable because of this.

**2026-04-10 gotcha (discovered during the 3-paper test import):** Zotero's `q=` parameter with `qmode=everything` indexes **title, creators, year, full-text, notes, and tags** — but **NOT the `DOI` field**. A naive "search by DOI string" returns 0 hits even for items that clearly have that DOI (confirmed by direct `GET /items/<key>`). This silent failure mode is exactly what lets duplicates slip through. **Always dedup via `find_duplicate()`** in `zotero_api.py`, which searches by a title fragment (indexed) and filters by DOI.

**Env-in-subshell gotcha:** The Claude Code Bash tool spawns fresh subshells that do NOT inherit parent-shell env, so `$ZOTERO_API_KEY` may be empty even if `printenv` showed it a moment ago. Either `source .env` at the top of the command, or rely on `zotero_api.py`'s built-in `.env` auto-loader. The helper walks up from the script's own directory looking for `.env`, so it works regardless of cwd.

## Test imports already in the library

Imported 2026-04-10 as the helper's first live test run (3 items, one POST, zero duplicates):

| Key | Paper |
|---|---|
| `PCSW6VET` | Yamins et al. 2014 — *Performance-optimized hierarchical models* (PNAS) |
| `T22MPRXV` | Khaligh-Razavi & Kriegeskorte 2014 — *Deep supervised, but not unsupervised…* (PLoS Comp Bio) |
| `2DH23EKH` | Güçlü & van Gerven 2015 — *Deep neural networks reveal a gradient…* (J Neurosci) |

These keys are stable — reference them in BibTeX export or future `--collection` targeting rather than re-creating the items.

## BibTeX conventions

- Wrap acronyms and proper nouns in double braces: `{{AlexNet}}`, `{{ImageNet}}`, `{{NSD}}` — otherwise BibTeX will lowercase them under most styles.
- Always include `doi`, `url`, and (for arXiv entries) `eprint` + `archivePrefix = {arXiv}`.
- Prefer published venues over preprints when both exist.
- Citation key format: `firstauthor<year><shortname>` — e.g., `allen2022nsd`, `hebart2023things`.

## Zotero Web API reference

Credentials live in the project `.env` as `ZOTERO_API_KEY` and `ZOTERO_USER_ID`. `zotero_api.py` auto-loads them. **Never print the API key** in terminal output, commit messages, or committed files — use the env var, not the literal value.

Useful endpoints (all under `https://api.zotero.org`):
- `GET /users/<id>/items?q=<query>&qmode=everything&format=json` — free-text search. Remember: does **not** index DOI.
- `GET /users/<id>/items/<key>?format=json` — direct item lookup; authoritative.
- `POST /users/<id>/items` — body is a JSON list of item objects; returns `successful` / `failed` / `unchanged` maps keyed by the original list index. One item per list element, full stop.
- `DELETE /users/<id>/items/<key>` — delete an item (the connector can't do this). Requires an `If-Unmodified-Since-Version` header — fetch it first via a `GET` on the item or from the `Last-Modified-Version` response header.
- Headers: `Zotero-API-Version: 3`, `Authorization: Bearer <key>`, `Content-Type: application/json` on writes.

## Note on filename

The user originally asked for `Claude.md5` (voice-dictation typo). The correct filename is `CLAUDE.md` — that is what tooling and other project-level instructions look for.
