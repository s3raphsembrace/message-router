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
# Deliberately symmetric. Ranking only by negative reactions would starve the
# notify cases of supporting evidence -- "the user opened this in 2 minutes and
# replied" justifies an interrupt exactly as much as "the user muted the sender
# after this" justifies suppressing one. What is being scored is how DECISIVE a
# past reaction is, not how bad it was.
REACTION_REPORTED = 1.0
REACTION_MUTED_AFTER = 0.9
REACTION_DISMISSED_UNOPENED = 0.65
REACTION_OPENED_REPLIED = 0.5
REACTION_OPENED = 0.25
REACTION_NONE = 0.0


def reaction_salience(event):
    """How much a past reaction tells us, in [0, 1]. Highest signal wins."""
    if not event:
        return REACTION_NONE
    if event.get("message_reported") == "1":
        return REACTION_REPORTED
    if event.get("muted_after_message") == "1":
        return REACTION_MUTED_AFTER
    opened = event.get("message_opened") == "1"
    if event.get("notification_dismissed") == "1" and not opened:
        return REACTION_DISMISSED_UNOPENED
    if opened and event.get("message_replied") == "1":
        return REACTION_OPENED_REPLIED
    if opened:
        return REACTION_OPENED
    return REACTION_NONE


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
    target_tokens = tokens(((message.get("message_text") or "") + " " + media_text).strip())

    scored = []
    for row in dataset.user_history(user_id):
        primary = bool(target_cp) and counterpart_of(row) == target_cp
        sim = jaccard(target_tokens, tokens(row.get("message_text"))) if target_tokens else 0.0

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
