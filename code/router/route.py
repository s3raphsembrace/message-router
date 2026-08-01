"""Routing loop: validate, re-ask once, then fall back.

The model is injected as a callable `call_model(system, user) -> str | dict`, so
the loop is fully testable without a key and the Gemini client drops in unchanged.
`call_model=None` means no model is configured: every row takes the safe default,
and that is counted as a fallback with zero attempts rather than a failed re-ask.

Exactly one re-ask is allowed. The retry quotes the specific validator errors back
at the model -- a generic "try again" tends to reproduce the same defect.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision import build_decision, safe_default              # noqa: E402
from prompt import render                                      # noqa: E402
from validate import retry_instruction, validate               # noqa: E402

MAX_REASKS = 1


class RouteStats(object):
    """Counters for one run. `reasks` is the headline number."""

    def __init__(self):
        self.messages = 0
        self.model_calls = 0
        self.valid_first_try = 0
        self.reasks = 0
        self.reask_succeeded = 0
        self.reask_failed = 0
        self.fallbacks = 0
        self.no_model = 0
        self.violation_codes = Counter()
        self.first_failure_codes = Counter()

    def summary(self):
        return {
            "messages": self.messages,
            "model_calls": self.model_calls,
            "valid_first_try": self.valid_first_try,
            "reasks": self.reasks,
            "reask_succeeded": self.reask_succeeded,
            "reask_failed": self.reask_failed,
            "fallbacks": self.fallbacks,
            "no_model": self.no_model,
        }

    def report(self):
        lines = [
            "messages          : %d" % self.messages,
            "model calls       : %d" % self.model_calls,
            "valid first try   : %d" % self.valid_first_try,
            "re-asks           : %d" % self.reasks,
            "  succeeded       : %d" % self.reask_succeeded,
            "  still invalid   : %d" % self.reask_failed,
            "fallback rows     : %d" % self.fallbacks,
        ]
        if self.no_model:
            lines.append("no model available: %d" % self.no_model)
        if self.first_failure_codes:
            lines.append("first-attempt violations:")
            for code, n in self.first_failure_codes.most_common():
                lines.append("  %-28s %d" % (code, n))
        return "\n".join(lines)


def route_one(message_id, context, call_model, stats=None, max_reasks=MAX_REASKS):
    """Route one message. Always returns a RouterDecision; never raises."""
    stats = stats or RouteStats()
    stats.messages += 1
    candidates = context.get("evidence_candidates") or []

    if call_model is None:
        stats.no_model += 1
        stats.fallbacks += 1
        return safe_default(message_id, attempts=0, notes=["no model configured"])

    system, user = render(context)

    try:
        raw = call_model(system, user)
    except Exception as exc:                                   # transport/SDK failure
        stats.model_calls += 1
        stats.fallbacks += 1
        return safe_default(message_id, attempts=1,
                            notes=["model call failed: %s" % exc])
    stats.model_calls += 1

    payload, violations = validate(raw, candidates)
    if not violations:
        stats.valid_first_try += 1
        return build_decision(payload, message_id, candidates, attempts=1)

    for v in violations:
        stats.violation_codes[v.code] += 1
    stats.first_failure_codes[violations[0].code] += 1

    if max_reasks < 1:
        stats.fallbacks += 1
        return safe_default(message_id, attempts=1,
                            notes=[v.code for v in violations])

    # -- the single re-ask, with the specific errors injected ------------
    stats.reasks += 1
    retry_user = user + retry_instruction(violations)
    try:
        raw2 = call_model(system, retry_user)
    except Exception as exc:
        stats.reask_failed += 1
        stats.fallbacks += 1
        return safe_default(message_id, attempts=2,
                            notes=["re-ask call failed: %s" % exc])
    stats.model_calls += 1

    payload2, violations2 = validate(raw2, candidates)
    if not violations2:
        stats.reask_succeeded += 1
        decision = build_decision(payload2, message_id, candidates, attempts=2)
        decision.notes.append("recovered on re-ask after: %s"
                              % ", ".join(v.code for v in violations))
        return decision

    stats.reask_failed += 1
    stats.fallbacks += 1
    for v in violations2:
        stats.violation_codes[v.code] += 1
    return safe_default(message_id, attempts=2,
                        notes=["still invalid after re-ask: %s"
                               % ", ".join(v.code for v in violations2)])


def route_all(messages, contexts, call_model, stats=None, max_reasks=MAX_REASKS):
    """Route every message. Returns (decisions_by_id, stats)."""
    stats = stats or RouteStats()
    out = {}
    for row in messages:
        mid = row["message_id"]
        out[mid] = route_one(mid, contexts[mid], call_model, stats, max_reasks)
    return out, stats
