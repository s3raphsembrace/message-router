"""Derived deterministic signals.

Every fact the model cannot verify for itself is computed here and stated in the
context: domain ages, report counts, opt-out timestamps, mute state, direct
address. The router then decides *with* the evidence in hand rather than guessing,
and the Layer 4 guard re-reads the same values.

This is the "compute once, use twice" boundary from the design. Anything the guard
enforces must appear here first, so a decision and its enforcement can never be
based on different numbers.
"""

import datetime as dt

from loaders import parse_ts
from textutil import jaccard, mentions, tokens

# Thresholds for the scam conjunction. Individually none of these is sufficient --
# code/analysis/rule_validation.py shows a verified 4400-day-old sender using a link
# shortener that is labelled digest/promotion, not scam. Only the full conjunction
# is enforced.
YOUNG_DOMAIN_DAYS = 90
YOUNG_ACCOUNT_DAYS = 180
HIGH_REPORTS_30D = 20

# Token overlap above which two messages count as the same template resent.
REPEAT_SIMILARITY = 0.6

HIGH_FORWARD_COUNT = 5


def in_quiet_hours(window, when):
    """True if `when` falls inside the user's DND window. Handles midnight wrap.

    Reported as a feature only. Tested against the labelled samples it changes no
    decision -- zero samples fall inside a DND window, and all 8 affected live rows
    already route to digest or mute on content -- so enforcing it could only ever
    demote a correct notify. See code/analysis/rule_validation.py.
    """
    if not window or "-" not in window or when is None:
        return None
    try:
        start_s, end_s = window.split("-")
        start = dt.time(*map(int, start_s.split(":")))
        end = dt.time(*map(int, end_s.split(":")))
    except (ValueError, TypeError):
        return None
    t = when.time()
    return start <= t < end if start <= end else (t >= start or t < end)


def business_trust(business):
    """Verification and infrastructure-age facts about a business sender."""
    if not business:
        return {}

    def as_int(key):
        try:
            return int(business.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    official = (business.get("official_domain") or "").strip().lower()
    used = (business.get("domain_used_by_sender") or "").strip().lower()
    domain_age = as_int("domain_used_by_sender_age_days")
    account_age = as_int("account_age_days")
    reports = as_int("user_reports_30d")

    out = {
        "verified": business.get("verified") == "1",
        "category": business.get("category") or "",
        "official_domain": official,
        "domain_used_by_sender": used,
        "domain_mismatch": bool(official and used and official != used),
        "domain_age_days": domain_age,
        "account_age_days": account_age,
        "user_reports_30d": reports,
    }
    out["young_domain"] = domain_age < YOUNG_DOMAIN_DAYS
    out["young_account"] = account_age < YOUNG_ACCOUNT_DAYS
    out["high_reports"] = reports >= HIGH_REPORTS_30D
    # The full conjunction -- the only form the guard enforces.
    out["scam_signature"] = bool(
        out["domain_mismatch"] and not out["verified"]
        and out["young_domain"] and out["high_reports"]
    )
    return out


def promotion_consent(relationship):
    """Whether this user still wants promotional mail from this business."""
    if not relationship:
        return {"known_relationship": False}
    opted_out_at = (relationship.get("promotions_opted_out_at") or "").strip()
    return {
        "known_relationship": True,
        "why_user_knows_account": relationship.get("why_user_knows_account") or "",
        "allows_promotions": relationship.get("allows_promotions") == "1",
        "opted_out_of_promotions": bool(opted_out_at),
        "opted_out_at": opted_out_at or None,
        "last_activity_at": relationship.get("last_activity_at") or None,
        "activity_count_180d": relationship.get("activity_count_180d") or "0",
        "opened_30d": relationship.get("messages_opened_30d") or "0",
        "dismissed_30d": relationship.get("messages_dismissed_30d") or "0",
        "replied_30d": relationship.get("messages_replied_30d") or "0",
    }


def repetition(message_text, history_rows, threshold=REPEAT_SIMILARITY):
    """How many past messages to this user are near-duplicates of this one.

    A resent template the user has already ignored is a strong mute signal; the
    same template they engaged with is not. The count is reported here and the
    reactions travel with the retrieved shortlist.
    """
    if not message_text:
        return {"near_duplicates_in_history": 0, "duplicate_message_ids": []}
    target = tokens(message_text)
    if not target:
        return {"near_duplicates_in_history": 0, "duplicate_message_ids": []}
    hits = []
    for row in history_rows:
        if jaccard(target, tokens(row.get("message_text"))) >= threshold:
            hits.append(row["message_id"])
    return {"near_duplicates_in_history": len(hits), "duplicate_message_ids": hits[:5]}


def extract(message, dataset, reaction_stats, history_rows, media=None):
    """All deterministic signals for one message, as a flat dict."""
    user = dataset.user(message["user_id"])
    created = parse_ts(message.get("created_at"))
    text = message.get("message_text") or ""

    # Media text participates in mention and repetition detection, but stays
    # labelled separately in the context so the router can weigh authored text
    # differently from model-derived text.
    media_text = media.router_text() if (media is not None and media.ok) else ""
    combined = (text + "\n" + media_text).strip()

    try:
        forwarded = int(message.get("forwarded_count") or 0)
    except (TypeError, ValueError):
        forwarded = 0

    sig = {
        "forwarded_count": forwarded,
        "heavily_forwarded": forwarded >= HIGH_FORWARD_COUNT,
        "has_media": bool(message.get("media_type")),
        "media_interpreted": bool(media is not None and media.ok),
        "text_is_empty": not text.strip(),
    }

    quiet = in_quiet_hours(user.get("do_not_disturb_window"), created)
    if quiet is not None:
        sig["in_quiet_hours"] = quiet

    # Direct address. Never sufficient on its own to override a mute: one of the
    # two self-mentions inside muted groups is chain-forward spam.
    addressed = mentions(combined)
    sig["directly_addressed"] = message["user_id"] in addressed
    if addressed:
        sig["mentions"] = sorted(addressed)

    conv = message.get("conversation_type")
    if conv == "group" and message.get("group_id"):
        membership = dataset.member(message["group_id"], message["user_id"])
        sender_membership = dataset.member(message["group_id"], message.get("sender_user_id") or "")
        sig["group_muted_by_user"] = membership.get("group_muted_by_user") == "1"
        sig["user_role_in_group"] = membership.get("role") or "unknown"
        sig["sender_role_in_group"] = sender_membership.get("role") or "unknown"
        sig["sender_is_group_admin"] = sender_membership.get("role") == "admin"
    elif conv == "business" and message.get("business_id"):
        sig.update(business_trust(dataset.business(message["business_id"])))
        sig.update(promotion_consent(dataset.relationship(message["user_id"], message["business_id"])))

    stats = reaction_stats.get((message["user_id"], _counterpart(message)))
    sig["counterpart_unanimously_reported"] = bool(stats and stats.unanimously_reported)

    sig.update(repetition(combined or text, history_rows))
    return sig


def _counterpart(row):
    return row.get("business_id") or row.get("group_id") or row.get("sender_user_id") or ""
