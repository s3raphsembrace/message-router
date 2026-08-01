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

# Same counterpart dominates: "have I heard from this sender before, and what did I
# do about it" is the question evidence is supposed to answer.
W_SAME_COUNTERPART = 2.0
W_TEXT_SIMILARITY = 1.5
W_RECENCY = 0.6
W_SAME_KIND = 0.25

DEFAULT_SHORTLIST = 6
RECENCY_HALFLIFE_DAYS = 45.0


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


def shortlist(message, dataset, limit=DEFAULT_SHORTLIST, text_chars=180):
    """Rank this user's history and return the top `limit` as evidence candidates.

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
        same_cp = counterpart_of(row) == target_cp and bool(target_cp)
        sim = jaccard(target_tokens, tokens(row.get("message_text"))) if target_tokens else 0.0
        rec = _recency_score(parse_ts(row.get("created_at")), now)
        same_kind = row.get("conversation_type") == message.get("conversation_type")

        score = (W_SAME_COUNTERPART * (1.0 if same_cp else 0.0)
                 + W_TEXT_SIMILARITY * sim
                 + W_RECENCY * rec
                 + W_SAME_KIND * (1.0 if same_kind else 0.0))
        if score <= 0:
            continue
        scored.append((score, sim, same_cp, row))

    # Deterministic ordering: score desc, then message_id asc to break ties stably.
    scored.sort(key=lambda t: (-t[0], t[3]["message_id"]))

    out = []
    for idx, (score, sim, same_cp, row) in enumerate(scored[:limit], start=1):
        event = dataset.event(user_id, row["message_id"])
        text = row.get("message_text") or ""
        if not text.strip() and row.get("media_id"):
            m = dataset.media.get(row["media_id"])
            text = (m.router_text() if (m is not None and m.ok)
                    else "(%s message, not interpreted)" % (row.get("media_type") or "media"))
        out.append({
            "idx": idx,
            "message_id": row["message_id"],
            "when": row.get("created_at"),
            "same_sender": bool(same_cp),
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
