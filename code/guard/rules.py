"""Deterministic override rules (Layer 4).

Every rule here decides on a fact the model cannot verify for itself -- an opt-out
timestamp, a domain age, a report count, a mute flag, a clock reading. Judgements
on continuous variables stay in Layer 2 as features for the router to weigh.

The layer is monotonic by construction: a rule may only hold an action where it is
or move it toward less interruption (notify -> digest -> mute). Nothing here can
promote a row, so the guard can make the system safer or more respectful of an
explicit preference, and can never make it noisier.
"""

from dataclasses import dataclass
from typing import Optional

# Ordered from most to least interrupting. The guard may only move down this scale.
ACTION_SEVERITY = {"notify": 2, "digest": 1, "mute": 0}

# Types that carry no standing value once the user has signalled they do not want
# the sender. A muted group's event notice is still worth a digest; its promo is not.
#
# `unknown` is deliberately NOT here. It used to be, which quietly turned the
# documented "cap at digest" in rule_opted_out and rule_muted_sender into a force
# to mute -- because the safe-default fallback row is typed `unknown`. A router
# failure then produced suppression rather than deferral, which is the opposite of
# what a safe default is for. It also muted msg_056, the muted-group urgent
# mention the problem statement calls out by name.
LOW_VALUE_TYPES = frozenset({"promotion", "greeting", "forward", "spam", "scam"})

# Types that can justify breaking through quiet hours.
QUIET_HOURS_EXEMPT_TYPES = frozenset({"urgent", "payment"})

REPORTED_POLICY_UNANIMOUS = "unanimous"
REPORTED_POLICY_ANY = "any"


@dataclass
class Override:
    """One rule's verdict. `message_type` None means leave the model's type alone."""
    rule: str
    action: str
    reason: str
    confidence: float
    message_type: Optional[str] = None
    terminal: bool = False           # stop evaluating further rules


def _sig(context):
    return context.get("signals") or {}


def _rapport(context):
    return context.get("rapport_with_this_sender") or {}


def _is_low_value(decision, context):
    """Low-value = the type carries no standing worth, or the user ignores this sender."""
    if decision.message_type in LOW_VALUE_TYPES:
        return True
    rapport = _rapport(context)
    seen = rapport.get("messages_seen") or 0
    return bool(seen >= 3 and (rapport.get("open_rate") or 0.0) == 0.0)


# ---------------------------------------------------------------------------
# force rules -- these overrule the model outright
# ---------------------------------------------------------------------------

def rule_scam_signature(decision, context, policy):
    """Full four-way scam conjunction -> mute/scam, whatever the model said.

    Deliberately the strict conjunction. A domain mismatch alone is labelled
    digest/promotion in the ground truth, so a looser trigger would mute
    legitimate senders on link shorteners.
    """
    if not _sig(context).get("scam_signature"):
        return None
    sig = _sig(context)
    return Override(
        rule="scam_signature",
        action="mute",
        message_type="scam",
        reason=("Sender uses %s instead of the brand's %s, is unverified, and has %s reports."
                % (sig.get("domain_used_by_sender") or "an unrecognised domain",
                   sig.get("official_domain") or "official domain",
                   sig.get("user_reports_30d"))),
        confidence=0.93,
        terminal=True,
    )


def rule_reported_sender(decision, context, policy):
    """A sender this user has reported -> mute/spam.

    Default policy is `unanimous`: every past message from this counterpart was
    reported and none was ever opened. The broad reading -- "has ever reported" --
    fires on 54% of messages and contradicts two labelled samples, one of which is
    notify/urgent from a sender the user opens 19 times out of 21. See
    code/analysis/rule_validation.py.
    """
    sig = _sig(context)
    rapport = _rapport(context)

    if policy.reported_policy == REPORTED_POLICY_ANY:
        fired = (rapport.get("reported") or 0) > 0
        detail = "reported %s of %s past messages" % (
            rapport.get("reported"), rapport.get("messages_seen"))
    else:
        fired = bool(sig.get("counterpart_unanimously_reported"))
        detail = "reported every past message from this sender and opened none"

    if not fired:
        return None
    return Override(
        rule="reported_sender",
        action="mute",
        message_type="scam" if sig.get("scam_signature") else "spam",
        reason="The user has %s, so this sender is suppressed." % detail,
        confidence=0.9,
        terminal=True,
    )


# ---------------------------------------------------------------------------
# preference rules -- cap, do not force
# ---------------------------------------------------------------------------

def rule_opted_out(decision, context, policy):
    """Explicit promotional opt-out -> cap at digest, or mute if low-value.

    Gated on the message actually being promotional. An opt-out covers marketing,
    not a live delivery or appointment update from the same business.
    """
    sig = _sig(context)
    if not sig.get("opted_out_of_promotions"):
        return None
    if decision.message_type not in ("promotion", "greeting", "forward", "unknown"):
        return None
    low = _is_low_value(decision, context)
    target = "mute" if low else "digest"
    if ACTION_SEVERITY[decision.action] <= ACTION_SEVERITY[target]:
        return None
    return Override(
        rule="opted_out_promotions",
        action=target,
        reason=("The user opted out of promotions from this brand on %s."
                % (sig.get("opted_out_at") or "record")),
        confidence=0.88,
    )


def rule_muted_sender(decision, context, policy):
    """User has muted this group -> cap at digest, or mute if low-value.

    A cap rather than a force: muting a group is not the same as refusing to be
    reached, and the router is allowed to have found a genuine direct request.
    That case is handled by the quiet-hours and direct-address exemptions below,
    and by the router's own stage 4.
    """
    sig = _sig(context)
    if not sig.get("group_muted_by_user"):
        return None
    # A direct, actionable request survives a mute -- this is the muted-family-group
    # case the problem statement calls out explicitly.
    if sig.get("directly_addressed") and decision.message_type in ("urgent", "payment", "event", "personal"):
        return None
    low = _is_low_value(decision, context)
    target = "mute" if low else "digest"
    if ACTION_SEVERITY[decision.action] <= ACTION_SEVERITY[target]:
        return None
    return Override(
        rule="muted_by_user",
        action=target,
        reason="The user has muted this conversation and the message carries no direct request.",
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# timing rule
# ---------------------------------------------------------------------------

def rule_quiet_hours(decision, context, policy):
    """Inside the user's DND window -> downgrade notify to digest.

    Exempt: an urgent message that directly addresses this user, and a payment
    from a trusted sender (a verified business, or a group admin). Without those
    carve-outs this rule can only ever demote a correct notify -- on this dataset
    it touches 8 rows, none of which is an emergency.
    """
    sig = _sig(context)
    if not sig.get("in_quiet_hours"):
        return None
    if decision.action != "notify":
        return None

    # Nothing flagged as a scam earns an exemption. In the full ladder the scam
    # rule is terminal and this is unreachable, but the exemption must not depend
    # on another rule having run first.
    if not sig.get("scam_signature"):
        if sig.get("directly_addressed") and decision.message_type in QUIET_HOURS_EXEMPT_TYPES:
            return None
        if decision.message_type == "payment" and (
                sig.get("sender_is_group_admin") or sig.get("verified")):
            return None

    return Override(
        rule="quiet_hours",
        action="digest",
        reason=("Delivered during the user's quiet hours (%s), so it waits for the digest."
                % (context.get("user", {}).get("quiet_hours") or "configured window")),
        confidence=min(0.82, max(0.6, decision.confidence)),
    )


# Evaluation order: force rules first and terminal, then preference caps, then
# timing. A later rule can only tighten what an earlier one left.
RULES = (
    rule_scam_signature,
    rule_reported_sender,
    rule_opted_out,
    rule_muted_sender,
    rule_quiet_hours,
)


class GuardPolicy(object):
    def __init__(self, reported_policy=REPORTED_POLICY_UNANIMOUS, enabled=None):
        if reported_policy not in (REPORTED_POLICY_UNANIMOUS, REPORTED_POLICY_ANY):
            raise ValueError("unknown reported_policy: %r" % reported_policy)
        self.reported_policy = reported_policy
        self.enabled = set(enabled) if enabled is not None else None

    def is_enabled(self, rule_name):
        return self.enabled is None or rule_name in self.enabled
