"""History retrieval.

No embeddings, no vector store. A user has at most 32 historical messages (median
15, largest payload ~900 tokens), so the entire candidate pool is already small
enough to rank exhaustively -- semantic search would add a dependency and
non-determinism to solve a problem that does not exist at this scale.

Ranking is deterministic and explainable, which matters because the selected ids
become `evidence_message_ids` in the output and are graded on relevance.
"""

from loaders import counterpart_of, parse_ts
from textutil import condense, jaccard, tokens

# --------------------------------------------------------------------------
# ranking weights
# --------------------------------------------------------------------------
# Candidates are filtered to the same user and same counterpart first, so
# "same sender" is a gate rather than a weight. Within that pool, ordering is by
# how much a past message actually tells us.
W_REACTION = 1.6        # what the user DID about it -- the strongest evidence
W_TEXT_SIMILARITY = 1.2  # is this the same template being resent?
W_RECENCY = 0.8         # older behaviour is weaker behaviour
W_PRIMARY_TIER = 3.0    # same-sender evidence always outranks a fallback match

DEFAULT_SHORTLIST = 6
RECENCY_HALFLIFE_DAYS = 45.0

# A cross-sender message must be at least this similar to qualify as fallback
# evidence. Set high on purpose: the only cross-sender rows worth citing are
# near-verbatim resends of the same template.
FALLBACK_SIMILARITY = 0.6

# --------------------------------------------------------------------------
# reaction salience
# --------------------------------------------------------------------------
# Symmetric by construction, and verifiably so: every negative tier has a positive
# tier of equal weight. An earlier version claimed symmetry while scoring
# `reported` at 1.0 against `opened+replied` at 0.5, which surfaced negative
# history first on 59 of 110 messages and biased the router toward mute on exactly
# the axis that gets graded for evidence relevance.
#
# The tiers pair by how deliberate the act was, not by whether it was favourable:
#
#   DELIBERATE   reported            <->  replied            1.00
#   STRONG       muted sender after  <->  opened quickly     0.90
#   PASSIVE      dismissed, unopened <->  opened             0.65
#   NONE         no recorded reaction                        0.00
#
# What is scored is how DECISIVE a past reaction is, not how bad it was. "Replied
# within 2 minutes" justifies an interrupt exactly as much as "muted the sender
# after this" justifies suppressing one.
REACTION_DELIBERATE = 1.0
REACTION_STRONG = 0.9
REACTION_PASSIVE = 0.65
REACTION_NONE = 0.0

# Backwards-compatible aliases; the pairing above is the source of truth.
REACTION_REPORTED = REACTION_DELIBERATE
REACTION_REPLIED = REACTION_DELIBERATE
REACTION_MUTED_AFTER = REACTION_STRONG
REACTION_OPENED_FAST = REACTION_STRONG
REACTION_DISMISSED_UNOPENED = REACTION_PASSIVE
REACTION_OPENED = REACTION_PASSIVE

# An open this fast is an engagement signal in its own right, not just an open.
# reaction_time_minutes was previously carried through the whole pipeline and
# never used for anything.
FAST_REACTION_MINUTES = 10.0


def reaction_salience(event):
    """How much a past reaction tells us, in [0, 1]. Highest signal wins.

    Checked most-decisive first. A message that was both opened and reported
    scores as reported: the report is the deliberate act.
    """
    if not event:
        return REACTION_NONE

    # deliberate acts, either direction
    if event.get("message_reported") == "1":
        return REACTION_DELIBERATE
    if event.get("message_replied") == "1":
        return REACTION_DELIBERATE

    opened = event.get("message_opened") == "1"

    # strong signals, either direction
    if event.get("muted_after_message") == "1":
        return REACTION_STRONG
    if opened:
        raw = event.get("reaction_time_minutes")
        try:
            if raw not in (None, "") and float(raw) <= FAST_REACTION_MINUTES:
                return REACTION_STRONG
        except (TypeError, ValueError):
            pass

    # passive signals, either direction
    if event.get("notification_dismissed") == "1" and not opened:
        return REACTION_PASSIVE
    if opened:
        return REACTION_PASSIVE
    return REACTION_NONE


def _comparable_text(row, dataset):
    """Tokens for a history row, falling back to its media interpretation.

    A media-only history row has empty message_text, so scoring it directly gave
    a similarity of 0 against everything -- meaning a resent scam poster could
    never match the same poster arriving today, which is precisely the repetition
    the retrieval exists to find.
    """
    text = row.get("message_text") or ""
    if text.strip():
        return tokens(text)
    media = dataset.media.get(row.get("media_id") or "")
    if media is not None and media.ok:
        return tokens(media.router_text())
    return set()


def _recency_score(then, now):
    """1.0 for same-day, decaying by half every RECENCY_HALFLIFE_DAYS."""
    if then is None or now is None:
        return 0.0
    days = abs((now - then).days)
    return 0.5 ** (days / RECENCY_HALFLIFE_DAYS)


def describe_reaction(event):
    """One human-readable line of what the user did with a past message.

    This is the whole point of the join: raw history says what arrived, the event
    says whether it landed. `muted_after` on a near-identical past promo is close
    to decisive on its own.
    """
    if not event:
        return "no recorded reaction"
    parts = []
    if event.get("message_opened") == "1":
        mins = event.get("reaction_time_minutes")
        parts.append("opened in %s min" % mins if mins else "opened")
    else:
        parts.append("not opened")
    if event.get("message_replied") == "1":
        parts.append("replied")
    if event.get("notification_dismissed") == "1":
        parts.append("dismissed")
    if event.get("muted_after_message") == "1":
        parts.append("MUTED sender after this")
    if event.get("message_reported") == "1":
        parts.append("REPORTED this")
    return ", ".join(parts)


def shortlist(message, dataset, limit=DEFAULT_SHORTLIST, text_chars=180,
              allow_fallback=True):
    """Deterministic, model-free evidence selection.

    Two tiers, in order:

      1. PRIMARY  -- same user AND same sender/group/business. This is the pool
         the evidence question is actually about: "have I heard from this
         counterpart before, and what did I do about it?"
      2. FALLBACK -- same user, different counterpart, but a near-verbatim resend
         of the same template (>= FALLBACK_SIMILARITY). Only used to top up when
         the primary tier is thinner than `limit`, and never outranks it.

    The fallback exists because it is load-bearing on real rows: 7 of 110 messages
    have no same-counterpart history at all, and 7 more would otherwise drop a
    near-identical template -- 4 of which carry a MUTE or REPORT reaction, i.e.
    the single most decisive evidence available for that decision. Pass
    allow_fallback=False for a strict same-sender-only shortlist.

    Returns a list of dicts carrying a 1-based `idx`. The router selects indices;
    it never types a message id, which makes a hallucinated evidence id
    structurally impossible.
    """
    user_id = message["user_id"]
    target_cp = counterpart_of(message)
    now = parse_ts(message.get("created_at"))

    media = dataset.media.get(message.get("media_id") or "")
    media_text = media.router_text() if (media is not None and media.ok) else ""
    # Two framings, scored independently and combined with max(). Comparing
    # text+OCR against a text-only history row is an unfair denominator: the extra
    # OCR tokens inflate the union and depress similarity, which silently dropped
    # two sim=1.00 muted duplicates from msg_066's evidence once Layer 1 ran.
    target_text = tokens(message.get("message_text") or "")
    target_full = tokens(((message.get("message_text") or "") + " " + media_text).strip())

    scored = []
    for row in dataset.user_history(user_id):
        primary = bool(target_cp) and counterpart_of(row) == target_cp
        row_tokens = _comparable_text(row, dataset)
        sim = max(jaccard(target_text, row_tokens) if target_text else 0.0,
                  jaccard(target_full, row_tokens) if target_full else 0.0)

        if not primary:
            if not allow_fallback or sim < FALLBACK_SIMILARITY:
                continue

        event = dataset.event(user_id, row["message_id"])
        score = (W_PRIMARY_TIER * (1.0 if primary else 0.0)
                 + W_REACTION * reaction_salience(event)
                 + W_TEXT_SIMILARITY * sim
                 + W_RECENCY * _recency_score(parse_ts(row.get("created_at")), now))
        scored.append((score, sim, primary, row, event))

    # Deterministic ordering: score desc, then message_id asc to break ties stably.
    scored.sort(key=lambda t: (-t[0], t[3]["message_id"]))

    out = []
    for idx, (score, sim, primary, row, event) in enumerate(scored[:limit], start=1):
        text = row.get("message_text") or ""
        if not text.strip() and row.get("media_id"):
            m = dataset.media.get(row["media_id"])
            text = (m.router_text() if (m is not None and m.ok)
                    else "(%s message, not interpreted)" % (row.get("media_type") or "media"))
        out.append({
            "idx": idx,
            "message_id": row["message_id"],
            "when": row.get("created_at"),
            "same_sender": bool(primary),
            "text_similarity": round(sim, 2),
            "text": condense(text, text_chars),
            "user_reaction": describe_reaction(event),
        })
    return out


def evidence_ids_for(candidates, indices):
    """Map router-selected indices back to message ids, dropping anything invalid."""
    by_idx = {c["idx"]: c["message_id"] for c in candidates}
    seen, out = set(), []
    for i in indices:
        try:
            key = int(i)
        except (TypeError, ValueError):
            continue
        mid = by_idx.get(key)
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out
