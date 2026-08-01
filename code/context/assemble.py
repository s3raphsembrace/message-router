"""Decision context assembly.

Turns one message row into a compact JSON object carrying everything the router
needs and nothing it does not. Raw tables are never dumped -- 750 daily-load rows
become two numbers, and a user's history becomes a ranked shortlist with the
reaction attached to each item.

Layout:

    message   what arrived, with media interpretation kept separately labelled
    user      notification posture: quiet hours, 30d behaviour, current load
    sender    who it came from: group / business / personal identity
    rapport   how this user has reacted to THIS counterpart before
    evidence_candidates   numbered shortlist; the router selects indices
    signals   deterministic facts the model cannot verify for itself

The token budget is enforced by degrading in a fixed order, so a context is never
silently truncated mid-structure.
"""

import json
import math

from aggregates import notification_load, relative_label
from loaders import counterpart_of, parse_ts
from retrieve import DEFAULT_SHORTLIST, shortlist
from textutil import condense
import signals as signals_mod

DEFAULT_TOKEN_BUDGET = 1400

# ~4 characters per token. Deliberately a heuristic: the budget exists to keep
# contexts compact and comparable, and an exact tokenizer would add a dependency
# for precision that changes no decision.
CHARS_PER_TOKEN = 4.0

MESSAGE_TEXT_LIMIT = 1200
MESSAGE_TEXT_FLOOR = 500
MEDIA_FIELD_LIMIT = 700

# Degradation ladder, applied in order until the context fits. Evidence count
# steps are derived from K at runtime rather than fixed here.
_CANDIDATE_TEXT_STEPS = (180, 130, 90, 60)


def estimate_tokens(obj):
    """Approximate token count of the serialised context."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def _drop_empty(d):
    """Remove None / "" / empty-collection values. Absent means 'no signal here',
    which is cheaper and less confusing to the model than a wall of nulls."""
    out = {}
    for k, v in d.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        out[k] = v
    return out


def _media_block(media, field_limit=MEDIA_FIELD_LIMIT):
    """Layer 1 output, kept structurally separate from authored text."""
    if media is None:
        return {"status": "not_interpreted"}
    if not media.ok:
        return {"status": media.status, "note": "route on metadata alone"}
    block = {"status": "ok", "kind": media.kind,
             "interpretation_confidence": round(media.interp_confidence, 2)}
    if media.kind == "voice":
        block["transcript"] = condense(media.transcript, field_limit)
        block["intent_summary"] = condense(media.intent_summary, 200)
        if media.language:
            block["language"] = media.language
    else:
        block["extracted_text"] = condense(media.extracted_text, field_limit)
        block["visual_description"] = condense(media.visual_description, 250)
        block["layout"] = media.layout
    return _drop_empty(block)


def _user_block(dataset, message):
    """The user's own posture, with their base rate stated.

    `baseline_open_share` is the denominator everything else is read against. It
    is the difference between "opened 8 of 10" meaning unusually engaged and
    meaning below average -- across users the share spans 0.17 to 0.91.
    """
    user_id = message["user_id"]
    user = dataset.user(user_id)
    baselines = dataset.baselines

    block = {
        "user_id": user_id,
        "quiet_hours": user.get("do_not_disturb_window") or None,
        "opened_30d": user.get("messages_opened_30d"),
        "replied_30d": user.get("messages_replied_30d"),
        "dismissed_30d": user.get("notifications_dismissed_30d"),
        "reported_30d": user.get("messages_reported_30d"),
    }

    own_share = baselines["user_open_share"].get(user_id)
    if own_share is not None:
        block["baseline_open_share"] = round(own_share, 2)
        vs_others = relative_label(own_share, baselines["median_open_share"])
        if vs_others:
            block["baseline_vs_other_users"] = vs_others

    load = notification_load(dataset, user_id, message.get("created_at"))
    if load:
        own_load = baselines["user_daily_load"].get(user_id)
        vs_typical = relative_label(own_load, baselines["median_daily_load"])
        if vs_typical:
            load["vs_typical_user"] = vs_typical
        block["notification_load"] = load
    return _drop_empty(block)


def _sender_block(dataset, message):
    """Identity and structural facts. Trust metrics live in `signals` so the guard
    and the router read them from exactly one place."""
    conv = message.get("conversation_type")

    if conv == "group" and message.get("group_id"):
        group = dataset.group(message["group_id"])
        membership = dataset.member(message["group_id"], message["user_id"])
        try:
            volume = int(group.get("messages_30d") or 0)
        except (TypeError, ValueError):
            volume = None
        return _drop_empty({
            "type": "group",
            "group_name": group.get("group_name"),
            "group_type": group.get("group_type"),
            "member_count": group.get("member_count"),
            "messages_30d": group.get("messages_30d"),
            # 92 vs 742 messages/30d is the difference between a quiet family chat
            # and a firehose; the raw number alone does not say which this is.
            "volume_vs_typical_group": relative_label(
                volume, dataset.baselines["median_group_volume"]),
            "sender_user_id": message.get("sender_user_id"),
            "user_membership": _drop_empty({
                "role": membership.get("role"),
                "muted_by_user": membership.get("group_muted_by_user") == "1",
                "read_30d": membership.get("messages_read_30d"),
                "replies_30d": membership.get("replies_sent_30d"),
                "dismissed_30d": membership.get("notifications_dismissed_30d"),
            }),
        })

    if conv == "business" and message.get("business_id"):
        business = dataset.business(message["business_id"])
        return _drop_empty({
            "type": "business",
            "business_id": message["business_id"],
            "brand_name": business.get("brand_name") or business.get("display_name"),
            "category": business.get("category"),
            "messages_sent_30d": business.get("messages_sent_30d"),
        })

    sender_id = message.get("sender_user_id") or ""
    shared = dataset.groups_of_user.get(message["user_id"], set()) & dataset.groups_of_user.get(sender_id, set())
    return _drop_empty({
        "type": "personal",
        "sender_user_id": sender_id,
        # No contact list exists in the dataset; co-membership is the only available
        # proxy for whether these two people actually know each other.
        "shared_groups": len(shared),
    })


def build_context(message, dataset, reaction_stats, token_budget=DEFAULT_TOKEN_BUDGET,
                  shortlist_limit=DEFAULT_SHORTLIST, allow_fallback=True):
    """Assemble one decision context. Pure: no I/O, no model calls.

    `shortlist_limit` is K -- how many evidence candidates the router may choose
    from. `allow_fallback=False` restricts evidence strictly to the same
    sender/group/business.
    """
    media = dataset.media.get(message.get("media_id") or "") if message.get("media_id") else None
    history_rows = dataset.user_history(message["user_id"])
    stats = reaction_stats.get((message["user_id"], counterpart_of(message)))

    message_block = _drop_empty({
        "message_id": message["message_id"],
        "created_at": message.get("created_at"),
        "conversation_type": message.get("conversation_type"),
        "authored_text": condense(message.get("message_text"), MESSAGE_TEXT_LIMIT),
        "forwarded_count": message.get("forwarded_count"),
    })
    if message.get("media_type"):
        message_block["media"] = _media_block(media)

    context = {
        "message": message_block,
        "user": _user_block(dataset, message),
        "sender": _sender_block(dataset, message),
        "signals": _drop_empty(
            signals_mod.extract(message, dataset, reaction_stats, history_rows, media)),
    }

    if stats and stats.n:
        rapport = stats.summary()
        # State engagement with this sender against the user's own norm, not in
        # isolation. "0.80 vs 0.70 norm (above, 1.1x)" is actionable; "opened 8 of
        # 10" on its own is not, because the norm varies from 0.17 to 0.91.
        vs_own = relative_label(stats.open_rate,
                                dataset.baselines["user_open_share"].get(message["user_id"]))
        if vs_own:
            rapport["open_rate_vs_user_norm"] = vs_own
        context["rapport_with_this_sender"] = rapport

    candidates = shortlist(message, dataset, limit=shortlist_limit,
                           text_chars=_CANDIDATE_TEXT_STEPS[0], allow_fallback=allow_fallback)
    context["evidence_candidates"] = candidates

    return _fit_to_budget(context, message, dataset, token_budget, shortlist_limit,
                          allow_fallback)


def _fit_to_budget(context, message, dataset, budget, shortlist_limit, allow_fallback=True):
    """Shrink in a fixed order until the context fits, then record the outcome.

    Order matters: evidence detail is the most compressible thing here, and the
    incoming message text is the least -- truncating what actually arrived would
    change the decision, so it is touched last and never below a floor.
    """
    if estimate_tokens(context) <= budget:
        context["_meta"] = {"estimated_tokens": estimate_tokens(context), "truncated": False}
        return context

    # 1. shorten each evidence item's text
    for text_chars in _CANDIDATE_TEXT_STEPS[1:]:
        context["evidence_candidates"] = shortlist(
            message, dataset, limit=min(shortlist_limit, len(context["evidence_candidates"])),
            text_chars=text_chars, allow_fallback=allow_fallback)
        if estimate_tokens(context) <= budget:
            return _finish(context, True)

    # 2. drop the lowest-ranked evidence items, one at a time down to a floor of 2.
    # Derived from K rather than a fixed list so a larger K degrades gradually
    # instead of collapsing in one step.
    for count in range(len(context["evidence_candidates"]) - 1, 1, -1):
        if count >= len(context["evidence_candidates"]):
            continue
        context["evidence_candidates"] = shortlist(
            message, dataset, limit=count, text_chars=_CANDIDATE_TEXT_STEPS[-1],
            allow_fallback=allow_fallback)
        if estimate_tokens(context) <= budget:
            return _finish(context, True)

    # 3. finally, trim the incoming message itself -- never below the floor
    text = context["message"].get("authored_text") or ""
    if len(text) > MESSAGE_TEXT_FLOOR:
        context["message"]["authored_text"] = condense(text, MESSAGE_TEXT_FLOOR)
        if estimate_tokens(context) <= budget:
            return _finish(context, True)

    # 4. trim interpreted media text, which is derived rather than authored
    media_block = context["message"].get("media")
    if isinstance(media_block, dict):
        for key in ("transcript", "extracted_text"):
            if media_block.get(key):
                media_block[key] = condense(media_block[key], 300)
        if estimate_tokens(context) <= budget:
            return _finish(context, True)

    return _finish(context, True)


def _finish(context, truncated):
    context["_meta"] = {"estimated_tokens": estimate_tokens(context), "truncated": truncated}
    return context


def build_all(dataset, reaction_stats, rows=None, **kwargs):
    """Contexts for every message, keyed on message_id."""
    rows = dataset.messages if rows is None else rows
    return {r["message_id"]: build_context(r, dataset, reaction_stats, **kwargs) for r in rows}
