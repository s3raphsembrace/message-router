# Message Notification Router — code

Pipeline layers:

| Layer | Location | Status |
|---|---|---|
| 1. Multimodal preprocessing | `code/preprocess/` | built |
| 2. Context assembly | `code/context/` | built |
| 3. LLM router | `code/router/` | not yet |
| 4. Deterministic guard | `code/guard/` | not yet |

Design evidence for every threshold below lives in [`code/analysis/`](analysis/).

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

## Layer 2 — context assembly

Turns one message row into a compact JSON decision context. No API key, no
network, fully deterministic.

```bash
python code/context/run.py --stats            # token report over all 110
python code/context/run.py --show msg_066     # inspect one context
python code/context/run.py --out contexts.json
python code/context/run.py --samples --stats  # same, over sample_messages.csv
```

Context layout:

| block | contents |
|---|---|
| `message` | what arrived; interpreted media kept in a separate labelled sub-block |
| `user` | quiet hours, 30d opens/replies/dismissals/reports, baseline notification load |
| `sender` | group / business / personal identity and structural facts |
| `rapport_with_this_sender` | how this user has reacted to **this** counterpart before |
| `evidence_candidates` | numbered shortlist, each with the user's recorded reaction |
| `signals` | deterministic facts the model cannot verify for itself |

### Budget

Default 1,400 tokens (~4 chars/token heuristic). Measured over all 110 messages:
**min 461, p50 756, p95 936, max 963** — nothing truncated, ~82k tokens for the
whole run. When a context does exceed budget it degrades in a fixed order:
shorten evidence text → drop lowest-ranked evidence → trim the incoming message
(never below a 500-char floor) → trim interpreted media text. The incoming
message is touched last because truncating what actually arrived would change the
decision.

Raw tables are never dumped: 756 daily-load rows become three numbers, and a
user's history becomes a ranked shortlist of at most 6 with the reaction attached.

### Retrieval

No embeddings, no vector store. A user has at most 32 historical messages (median
15), so the candidate pool is ranked exhaustively:

```
2.00 * same counterpart  +  1.50 * token overlap  +  0.60 * recency  +  0.25 * same conversation kind
```

Ties break on `message_id` so ordering is stable across runs. Each candidate
carries a plain-language reaction line — *"not opened, dismissed, MUTED sender
after this"* — because raw history is weak but history joined with
`message_events` is close to decisive.

Candidates are **numbered**, and the router selects indices rather than typing
message ids. `evidence_ids_for()` maps indices back and drops anything invalid,
which makes a hallucinated evidence id structurally impossible.

### Signals, and the compute-once boundary

`signals.py` computes every fact the model cannot check for itself — domain ages,
report counts, opt-out timestamps, mute state, direct address, repetition counts.
The router decides *with* these in hand, and the Layer 4 guard later re-reads the
same values, so a decision and its enforcement can never rest on different
numbers.

The scam signature is a **four-way conjunction** (`domain_mismatch` AND
`not verified` AND `young_domain` AND `high_reports`). Partial matches stay
features: a verified 4,400-day-old sender using a link shortener is labelled
`digest/promotion` in the ground truth, not scam.

`in_quiet_hours` and the softer engagement ratios are reported as features and
never enforced — see [`analysis/`](analysis/) for why.

### A dataset quirk worth knowing

`daily_notification_summary.csv` covers **2026-07-04 → 07-17**, while
`messages.csv` spans **07-18 → 07-31**. They do not overlap at all, and every
user has exactly 14 rows over that same fixed period. A rolling lookback would
therefore give a message dated 07-18 thirteen days of history and one dated 07-31
none — an artifact of table coverage, not user behaviour. It is treated as a
fixed baseline profile so every message gets the same comparable signal, and the
period is stated in the context so the router does not read it as "right now".

## Tests

```bash
python code/preprocess/test_preprocess.py   # 56 assertions
python code/context/test_context.py         # 78 assertions
```

134 assertions total, no network and no API key.

The real dataset is healthy in both stages — all 30 media files resolve, and no
context exceeds the token budget — so the failure paths would otherwise never
execute. Layer 1 fault-injects a synthetic dataset in a temp directory (unindexed
ids, indexed-but-absent files, empty files) and asserts the run exits 0 with
three broken media present; Layer 2 forces the truncation ladder with an
artificially tight budget.

Layer 2 assertions are anchored
on the specific real rows whose correct handling was established in
[`analysis/`](analysis/): the same-poster opt-out collision (`msg_065` vs
`msg_066`), the verified-but-mismatched sender that must not be flagged as scam
(`msg_041`), the impersonation that must (`msg_019`), the muted-group urgent
mention (`msg_056`), and the muted-group chain-spam mention that is *also* both
muted and directly addressed (`msg_040`) — proving signals alone cannot decide
that row, and the ladder in Layers 3–4 has to.
