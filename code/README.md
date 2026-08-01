# Message Notification Router — code

Pipeline layers:

| Layer | Location | Status |
|---|---|---|
| 1. Multimodal preprocessing | `code/preprocess/` | built |
| 2. Context assembly | `code/context/` | built |
| 3. LLM router | `code/router/` | prompt, validator, re-ask loop, writer; **model client next** |
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
python code/context/run.py --stats                 # token + evidence report over all 110
python code/context/run.py --show msg_066          # inspect one context
python code/context/run.py --top-k 10              # more evidence candidates
python code/context/run.py --same-sender-only      # strict: no fallback tier
python code/context/run.py --out contexts.json
python code/context/run.py --samples --stats       # same, over sample_messages.csv
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
**min 234, p50 632, p95 854, max 936** — nothing truncated, ~67k tokens for the
whole run. When a context does exceed budget it degrades in a fixed order:
shorten evidence text → drop lowest-ranked evidence → trim the incoming message
(never below a 500-char floor) → trim interpreted media text. The incoming
message is touched last because truncating what actually arrived would change the
decision.

Raw tables are never dumped: 756 daily-load rows become three numbers, and a
user's history becomes a ranked shortlist of at most 6 with the reaction attached.

### Retrieval

Fully deterministic and model-free — candidate selection happens before the LLM
sees anything. No embeddings, no vector store: a user has at most 32 historical
messages (median 15), so the pool is filtered and ranked exhaustively.

**Two tiers.**

1. **Primary** — same user **and** same sender/group/business. This is the pool
   the evidence question is actually about.
2. **Fallback** — same user, different counterpart, but a near-verbatim resend of
   the same template (similarity ≥ 0.60). Only tops up when the primary tier is
   thinner than K, and never outranks it. Disable with `--same-sender-only`.

Within the pool:

```
3.0 * primary tier  +  1.6 * reaction salience  +  1.2 * token overlap  +  0.8 * recency
```

Ties break on `message_id`, so ordering is stable across runs.

**Reaction salience** is the ranking's centre of gravity, since what the user
*did* about a past message is stronger evidence than the message itself:

| reaction | weight |
|---|---|
| reported | 1.00 |
| muted sender after | 0.90 |
| dismissed, never opened | 0.65 |
| opened and replied | 0.50 |
| opened only | 0.25 |
| no recorded reaction | 0.00 |

The scale is deliberately **symmetric**. Ranking only on negative reactions would
starve the `notify` cases of supporting evidence — *"opened in 2 minutes and
replied"* justifies an interrupt exactly as much as *"muted the sender after
this"* justifies suppressing one. What is scored is how **decisive** a reaction
is, not how bad it was.

**K is configurable** — `--top-k` on the CLI, `shortlist_limit=` on
`build_context()`, default 6.

Candidates are **numbered**, and the router selects indices rather than typing
message ids. `evidence_ids_for()` maps indices back and drops anything invalid,
which makes a hallucinated evidence id structurally impossible.

#### What the filter costs, measured

Restricting to the same counterpart leaves **7 of 110 messages with no evidence
at all** (`msg_089`–`msg_096`); those correctly emit `none`. A further 7 messages
would have dropped a near-identical template from a different sender — **4 of
which carry a MUTE or REPORT reaction**, the most decisive evidence available for
that decision — which is why the fallback tier exists. It fires on exactly 7
contexts and contributes 8 rows, so it is a top-up, not a back door.

`msg_066` is the clearest case. Two same-sender rows, then two fallback rows at
similarity 1.00 that both read *"not opened, dismissed, MUTED sender after
this"*. Strict mode drops it from 4 candidates to 2.

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
python code/context/test_context.py         # 98 assertions
python code/router/test_router.py           # 123 assertions
```

277 assertions total, no network and no API key.

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

## Layer 3 — router prompt (prompt + schema; model client next)

```bash
python code/router/run.py --system            # the system prompt
python code/router/run.py --show msg_066      # the exact prompt for one message
python code/router/run.py --stats             # prompt sizes across all 110
```

System prompt ~1,362 tokens (identical on every call, so it caches); user prompt
min 294 / p50 792 / max 1,150. About 2,150 tokens per call.

The prompt states the **precedence ladder explicitly** rather than leaving it to
be inferred — risk floor, then direct-address override, then baseline — because
the three rows it exists for all invert under a naive "weigh everything" reading:
a scam inside a trusted group, an `@`-mention that is chain spam, and a verified
sender on a mismatched domain.

### Evidence policy

`evidence_message_ids` is graded on relevance and the scorer rewards `none`, so
the policy is enforced in **two independent places** — instructed in the prompt
and enforced in the parser, because an instruction is a preference and the scorer
is not.

The rules come from the labelled data rather than intuition. Of 30 samples:

| evidence cited | samples |
|---|---|
| none | 2 |
| exactly 1 id | 25 |
| 2 ids | 3 |
| 3+ ids | **0** |

So the prompt says: cite **exactly one** in the normal case, a second only when it
adds distinct support, **never three or more**, and prefer an empty list over a
weak citation — stating outright that an empty list is a correct answer scored as
`none`, and that an irrelevant citation is worse than citing nothing. The
candidate list is described as *"a menu, not a quota"*, since a model offered K
candidates will otherwise fill K.

`decision.py` enforces the same thing independently: `MAX_EVIDENCE_IDS = 2`,
indices resolved against the candidates actually offered, invalid or duplicated
indices dropped, and an empty result serialised as the literal string `none`.

### Expected `none` rate

**7 of 110 messages are offered no candidates at all** (`msg_089`–`msg_096`) and
are structurally forced to `none` — 6.4%, which lines up almost exactly with the
labelled rate of 2/30 = 6.7%. A further 9 rows have only weak candidates, so a
`none` rate somewhere in 6–15% is the expected shape. Materially more than that
means the model is under-citing.

### Failure handling

`parse_response()` never raises. Plain text, an empty string, a JSON array, a
null, an invalid action, a junk confidence — each degrades to a well-formed row
with a `parse_notes` entry recording what happened, so one bad response cannot
drop a message from `output.csv`.

The prompt also carries an injection guard: message text, voice transcripts and
image OCR are data written by a possibly adversarial sender, and the model is
told not to follow instructions found inside them.

> **Note on module naming:** the router's schema module is `decision.py`, not
> `schema.py`. `code/preprocess/schema.py` is already on `sys.path` by the time
> the router imports, and two modules named `schema` resolve to whichever
> directory landed first.

## Output contract

```bash
python code/main.py                  # route all, write dataset/output.csv, verify
python code/main.py --out /tmp/o.csv # write elsewhere
python code/main.py --verify-only    # re-check an existing file
```

`writer.py` reproduces the shipped blank template byte-for-byte: UTF-8, **no
BOM**, **CRLF** line endings, the six columns in fixed order, and one row per
`message_id` in `dataset/messages.csv` **in that file's order**. A missing
prediction raises rather than shipping a short file.

`verify_output()` re-reads what was written and independently checks BOM, line
endings, header, row count, row order, duplicate ids, closed vocabulary for
`action` and `message_type`, `confidence` parseable and within [0,1], and a
non-blank evidence cell. Every run self-verifies after writing.

### Validation, re-ask, fallback

`validate.py` is strict and repairs nothing — that is the point. Silent repair
would hide exactly the failures the re-ask exists to correct.

| check | error code |
|---|---|
| decodes as a JSON object | `not_json`, `not_object` |
| exactly the five required keys, no extras | `missing_keys`, `unexpected_keys` |
| `action` in the 3-value vocabulary | `action_out_of_vocab` |
| `message_type` in the 11-value vocabulary | `type_out_of_vocab` |
| `reason` a non-empty string | `reason_empty` |
| `confidence` a real number in [0,1] (rejects `true`, NaN) | `confidence_not_number`, `confidence_out_of_range` |
| every evidence index drawn from the offered shortlist | `evidence_not_in_shortlist` |
| evidence is an array; at most 2 distinct | `evidence_not_array`, `evidence_too_many` |

On any violation the loop **re-asks exactly once**, appending the specific
validator messages to the user turn — each names the offending value and states
the allowed set, because a generic "try again" tends to reproduce the same
defect. If the retry is still invalid, the row becomes the safe default:
**`digest` / `unknown` / `0.5` / `none`**. `digest` is chosen because it neither
interrupts the user nor suppresses something they may have needed.

Re-asks are counted, along with model calls, first-try successes, recoveries,
failures, and a histogram of which violations actually occurred:

```
messages          : 110
model calls       : 220
valid first try   : 0
re-asks           : 110
  succeeded       : 103
  still invalid   : 7
fallback rows     : 7
```

The model is injected as `call_model(system, user)`, so the entire loop is
tested with scripted responses and no key. `call_model=None` means no model is
configured — those rows take the default and are counted separately from a
failed re-ask, since no call was made.

> ⚠️ **`dataset/output.csv` is currently a placeholder.** The Layer 3 model
> client is not wired, so all 110 rows are the safe default. The file is
> contract-valid — which proves the plumbing — but contains no real routing. The
> run prints a loud banner saying so. Re-run `python code/main.py` once the
> client lands.
