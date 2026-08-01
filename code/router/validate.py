"""Strict validation of the model's raw response.

Distinct from parsing on purpose. `decision.parse_response` is lenient -- it
repairs whatever it can so a row is never lost. This module is the opposite: it
reports precisely what is wrong and repairs nothing, so the caller can re-ask the
model with the specific defect quoted back at it.

Every message is written to be pasted straight into a retry prompt, which is why
they name the offending value and state the allowed set.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import ACTIONS, MESSAGE_TYPES                      # noqa: E402

REQUIRED_KEYS = ("action", "message_type", "reason", "confidence", "evidence_indices")

# Ceiling from the labelled data: 25 of 30 samples cite one id, 3 cite two,
# 2 cite none, none cites three or more.
MAX_EVIDENCE_IDS = 2

# Error codes, for counting which defects actually occur in a run.
E_NOT_JSON = "not_json"
E_NOT_OBJECT = "not_object"
E_MISSING_KEYS = "missing_keys"
E_UNEXPECTED_KEYS = "unexpected_keys"
E_ACTION = "action_out_of_vocab"
E_TYPE = "type_out_of_vocab"
E_REASON = "reason_empty"
E_CONFIDENCE_TYPE = "confidence_not_number"
E_CONFIDENCE_RANGE = "confidence_out_of_range"
E_EVIDENCE_TYPE = "evidence_not_array"
E_EVIDENCE_UNKNOWN = "evidence_not_in_shortlist"
E_EVIDENCE_COUNT = "evidence_too_many"


class Violation(object):
    __slots__ = ("code", "message")

    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __repr__(self):
        return "Violation(%s)" % self.code


def coerce_json(raw):
    """Return (payload, violation). payload is None when it could not be read."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return None, Violation(E_NOT_JSON, "The response was empty. Return a single JSON object.")
    if not isinstance(raw, str):
        return None, Violation(
            E_NOT_OBJECT,
            "The response was a %s. Return a single JSON object." % type(raw).__name__)
    text = raw.strip()
    if not text:
        return None, Violation(E_NOT_JSON, "The response was empty. Return a single JSON object.")
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        return None, Violation(
            E_NOT_JSON,
            "The response was not valid JSON (%s). Return a single JSON object and nothing else."
            % exc)
    if not isinstance(parsed, dict):
        return None, Violation(
            E_NOT_OBJECT,
            "The response was a JSON %s, not an object. Return a single JSON object with the "
            "five required keys." % type(parsed).__name__)
    return parsed, None


def validate(raw, candidates, max_evidence=MAX_EVIDENCE_IDS):
    """Validate a raw model response against the output contract.

    Returns (payload, violations). `payload` is the decoded object when it could
    be decoded at all, otherwise None. An empty violations list means the response
    is contract-clean and needs no repair.
    """
    payload, fatal = coerce_json(raw)
    if fatal is not None:
        return None, [fatal]

    violations = []
    keys = set(payload)

    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        violations.append(Violation(
            E_MISSING_KEYS,
            "Missing required key(s): %s. The object must contain exactly these five keys: %s."
            % (", ".join(missing), ", ".join(REQUIRED_KEYS))))

    unexpected = sorted(keys - set(REQUIRED_KEYS))
    if unexpected:
        violations.append(Violation(
            E_UNEXPECTED_KEYS,
            "Unexpected key(s): %s. Return exactly these five keys and no others: %s."
            % (", ".join(unexpected), ", ".join(REQUIRED_KEYS))))

    # -- action --------------------------------------------------------
    if "action" in keys:
        action = payload.get("action")
        if not isinstance(action, str) or action.strip().lower() not in ACTIONS:
            violations.append(Violation(
                E_ACTION,
                "action was %s. It must be exactly one of: %s."
                % (json.dumps(action), ", ".join(ACTIONS))))

    # -- message_type --------------------------------------------------
    if "message_type" in keys:
        mtype = payload.get("message_type")
        if not isinstance(mtype, str) or mtype.strip().lower() not in MESSAGE_TYPES:
            violations.append(Violation(
                E_TYPE,
                "message_type was %s. It must be exactly one of: %s."
                % (json.dumps(mtype), ", ".join(MESSAGE_TYPES))))

    # -- reason --------------------------------------------------------
    if "reason" in keys:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            violations.append(Violation(
                E_REASON,
                "reason was %s. It must be a non-empty sentence explaining why this user "
                "gets this action." % json.dumps(reason)))

    # -- confidence ----------------------------------------------------
    if "confidence" in keys:
        conf = payload.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            violations.append(Violation(
                E_CONFIDENCE_TYPE,
                "confidence was %s. It must be a number between 0 and 1, for example 0.82."
                % json.dumps(conf)))
        elif conf != conf or conf < 0.0 or conf > 1.0:                 # NaN or out of range
            violations.append(Violation(
                E_CONFIDENCE_RANGE,
                "confidence was %s, which is outside the allowed range. It must be between "
                "0 and 1 inclusive." % json.dumps(conf)))

    # -- evidence_indices ----------------------------------------------
    valid_indices = [c["idx"] for c in (candidates or [])]
    if "evidence_indices" in keys:
        indices = payload.get("evidence_indices")
        if not isinstance(indices, (list, tuple)):
            violations.append(Violation(
                E_EVIDENCE_TYPE,
                "evidence_indices was %s. It must be an array of integers, or [] when no "
                "candidate is relevant." % json.dumps(indices)))
        else:
            bad = []
            for item in indices:
                if isinstance(item, bool) or not isinstance(item, int):
                    bad.append(item)
                elif item not in valid_indices:
                    bad.append(item)
            if bad:
                allowed = (", ".join(str(i) for i in valid_indices) if valid_indices
                           else "(none were offered)")
                violations.append(Violation(
                    E_EVIDENCE_UNKNOWN,
                    "evidence_indices contained %s, which %s not offered in "
                    "evidence_candidates. Valid indices for this message: %s. Use [] if no "
                    "candidate genuinely supports the decision."
                    % (json.dumps(bad), "are" if len(bad) > 1 else "is", allowed)))
            elif len(set(indices)) > max_evidence:
                violations.append(Violation(
                    E_EVIDENCE_COUNT,
                    "evidence_indices had %d entries. Cite at most %d, and only those that "
                    "genuinely support this decision. Prefer [] over a weak citation."
                    % (len(set(indices)), max_evidence)))

    return payload, violations


def retry_instruction(violations):
    """The correction appended to the user turn on the single re-ask."""
    bullets = "\n".join("- %s" % v.message for v in violations)
    return (
        "\n\nYour previous response was rejected by the output validator for these "
        "reasons:\n\n%s\n\nReturn a corrected JSON object. Return only the JSON object, "
        "with exactly these five keys: %s." % (bullets, ", ".join(REQUIRED_KEYS)))
