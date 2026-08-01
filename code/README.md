# Message Notification Router — code

Pipeline layers:

| Layer | Location | Status |
|---|---|---|
| 1. Multimodal preprocessing | `code/preprocess/` | built |
| 2. Context assembly | `code/context/` | not yet |
| 3. LLM router | `code/router/` | not yet |
| 4. Deterministic guard | `code/guard/` | not yet |

## Setup

```bash
pip install -r code/requirements.txt
cp .env.example .env    # then add your key
```

Layer 1 uses Gemini's native multimodal input (audio and image bytes are sent
inline — there is no separate ASR or OCR dependency). It reads
`GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `GOOGLE_GENAI_API_KEY` from the
environment, and `MEDIA_MODEL` to override the model (default
`gemini-2.5-flash`).

## Layer 1 — media preprocessing

Resolves every `media_id` in `messages.csv` and `message_history.csv` to a file
via `images.csv` / `voice_notes.csv`, interprets it, and caches the result.

```bash
python code/preprocess/run.py                  # messages + history (30 distinct files)
python code/preprocess/run.py --scope messages # messages only (19 files)
python code/preprocess/run.py --dry-run        # resolve and report, no model calls
python code/preprocess/run.py --force          # ignore cache, reprocess
python code/preprocess/run.py --only img_003,vn_008
```

Output is written to `cache/media_interpretations.json`, keyed on `media_id`.

**Voice notes** produce a verbatim `transcript` (original language preserved,
including Hindi/English mixing) plus a one-line `intent_summary` describing what
the speaker wants from the listener.

**Images** produce `extracted_text` (verbatim OCR — prices, URLs, UPI IDs, offer
codes, small print), a short `visual_description`, and a `layout` classification
of `stock_promo` / `personal_screenshot` / `other`. Layout matters because a
designed promo poster and a personal screenshot carry different priority and
risk priors even when their text is similar.

Both prompts are extraction-only. Dataset media includes scam posters and fake
payment screenshots whose content is adversarial by construction, so the model is
instructed to report what the media says rather than act on it, and to make no
routing judgement. That decision belongs to Layer 3.

### Key-optional

The stage runs without credentials. With no key, every file yields a typed
`no_api_key` placeholder that is **not** cached, so adding a key and re-running
processes exactly the files that still need it. Cached interpretations make
Layers 2–4 fully reproducible offline.

### Failure handling

Nothing here can take down a run. Every outcome is a well-formed record carrying
a `status`, and downstream layers branch on that status rather than on the
presence of a field:

| status | meaning | cached? |
|---|---|---|
| `ok` | interpreted successfully | yes |
| `missing_index` | `media_id` absent from `images.csv` / `voice_notes.csv` | yes |
| `missing_file` | indexed, but not on disk | yes |
| `unreadable` | present but empty or unopenable | yes |
| `no_api_key` | no credentials configured | no |
| `api_error` | model call failed after retries | no |
| `bad_response` | model replied in an unexpected shape | no |

Environment-dependent failures are deliberately not cached — otherwise a later
run with a working key would keep serving empty placeholders. A row whose media
could not be interpreted routes on message metadata alone.

`sha256` is stored per entry. If the bytes behind a cached `media_id` ever
change, the run reports it as stale and reprocesses rather than silently serving
an interpretation of a different file.

## Tests

```bash
python code/preprocess/test_preprocess.py
```

56 assertions, no network and no API key required. The real dataset media is
complete — all 30 referenced files resolve and none are missing — so the failure
paths are exercised against a synthetic dataset built in a temp directory:
unindexed ids, indexed-but-absent files, and empty files. The end-to-end case
asserts the run exits 0 with three broken media files present.
