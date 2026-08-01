"""Applying the override rules, with a full audit trail.

Two invariants are enforced here rather than trusted:

1. MONOTONICITY. The guard may only hold an action or move it toward less
   interruption. If a rule ever tried to promote a row, that is a bug, and it is
   raised rather than silently shipped.

2. INTERNAL CONSISTENCY. When an override changes the action it also rewrites
   `reason` and `confidence`. A row reading action=mute with the model's original
   "trusted delivery update from a business the user orders from" would be
   self-contradicting, and `reason` and confidence calibration are both graded.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules import ACTION_SEVERITY, RULES, GuardPolicy       # noqa: E402


@dataclass
class OverrideRecord:
    """One rule firing on one message. This is the audit row."""
    message_id: str
    rule: str
    from_action: str
    to_action: str
    from_type: str
    to_type: str
    from_confidence: float
    to_confidence: float
    model_reason: str
    rule_reason: str
    model_fell_back: bool = False

    @property
    def changed_action(self) -> bool:
        return self.from_action != self.to_action

    @property
    def changed_type(self) -> bool:
        return self.from_type != self.to_type

    @property
    def disagreement(self) -> bool:
        """True when the rules contradicted the model, rather than merely
        restating what it already decided."""
        return self.changed_action or self.changed_type

    def to_row(self) -> dict:
        return {
            "message_id": self.message_id,
            "rule": self.rule,
            "from_action": self.from_action,
            "to_action": self.to_action,
            "from_type": self.from_type,
            "to_type": self.to_type,
            "from_confidence": "%.2f" % self.from_confidence,
            "to_confidence": "%.2f" % self.to_confidence,
            "disagreement": "1" if self.disagreement else "0",
            "model_fell_back": "1" if self.model_fell_back else "0",
            "model_reason": self.model_reason,
            "rule_reason": self.rule_reason,
        }


@dataclass
class GuardResult:
    decision: object
    records: List[OverrideRecord] = field(default_factory=list)

    @property
    def overridden(self) -> bool:
        return any(r.disagreement for r in self.records)


def apply_guard(decision, context, policy=None):
    """Run the rule ladder over one decision. Returns a GuardResult.

    The returned decision is a copy; the input is left untouched so the model's
    original verdict stays available for the audit.
    """
    policy = policy or GuardPolicy()
    import copy
    current = copy.deepcopy(decision)
    records = []

    for rule_fn in RULES:
        name = rule_fn.__name__.replace("rule_", "")
        if not policy.is_enabled(name):
            continue

        override = rule_fn(current, context, policy)
        if override is None:
            continue

        if ACTION_SEVERITY[override.action] > ACTION_SEVERITY[current.action]:
            raise AssertionError(
                "rule %r tried to promote %s -> %s on %s; the guard may only make "
                "things safer" % (override.rule, current.action, override.action,
                                  decision.message_id))

        record = OverrideRecord(
            message_id=decision.message_id,
            rule=override.rule,
            from_action=current.action,
            to_action=override.action,
            from_type=current.message_type,
            to_type=override.message_type or current.message_type,
            from_confidence=current.confidence,
            to_confidence=override.confidence,
            model_reason=current.reason,
            rule_reason=override.reason,
            model_fell_back=getattr(decision, "fell_back", False),
        )
        records.append(record)

        current.action = override.action
        if override.message_type:
            current.message_type = override.message_type
        # Rewrite reason and confidence together with the action, so the row never
        # states one thing and justifies another.
        current.reason = override.reason
        current.confidence = override.confidence
        current.notes = list(current.notes) + ["override: %s" % override.rule]

        if override.terminal:
            break

    if ACTION_SEVERITY[current.action] > ACTION_SEVERITY[decision.action]:
        raise AssertionError("guard promoted %s from %s to %s"
                             % (decision.message_id, decision.action, current.action))

    return GuardResult(decision=current, records=records)


def apply_guard_all(decisions, contexts, policy=None):
    """Run the guard over every decision. Returns (decisions, all_records)."""
    policy = policy or GuardPolicy()
    out, records = {}, []
    for message_id, decision in decisions.items():
        result = apply_guard(decision, contexts[message_id], policy)
        out[message_id] = result.decision
        records.extend(result.records)
    return out, records
