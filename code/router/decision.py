"""Router decision: response schema and parsing (Layer 3).

Named `decision` rather than `schema` on purpose -- code/preprocess/schema.py is
already on sys.path by the time this imports, and two modules called `schema`
resolve to whichever directory landed first.

The model returns evidence as *indices* into the candidate list, never as message
ids. Parsing resolves those indices against the candidates that were actually
offered, so an out-of-range or invented index is dropped rather than becoming a
fabricated `evidence_message_ids` value.

Evidence policy is enforced here as well as instructed in the prompt, because an
instruction is a preference and the scorer is not. `MAX_EVIDENCE_IDS` reflects the
labelled data: of 30 samples, 25 cite exactly one id, 3 cite two, 2 cite none.
Nothing cites three.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "context"))

from prompt import ACTIONS, MESSAGE_TYPES                      # noqa: E402
from retrieve import evidence_ids_for                          # noqa: E402

NONE_EVIDENCE = "none"

# Hard ceiling. The ground truth never cites more than two.
MAX_EVIDENCE_IDS = 2

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": list(ACTIONS)},
        "message_type": {"type": "STRING", "enum": list(MESSAGE_TYPES)},
        "reason": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "evidence_indices": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["action", "message_type", "reason", "confidence", "evidence_indices"],
}


@dataclass
class RouterDecision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: List[str] = field(default_factory=list)
    parse_notes: List[str] = field(default_factory=list)

    @property
    def evidence_field(self) -> str:
        """The output.csv value: semicolon-joined ids, or the literal 'none'."""
        return ";".join(self.evidence_message_ids) if self.evidence_message_ids else NONE_EVIDENCE

    def to_row(self) -> dict:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": ("%.2f" % self.confidence).rstrip("0").rstrip("."),
            "evidence_message_ids": self.evidence_field,
        }


def _clamp01(value, default=0.5):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default, True
    if f != f:                                                  # NaN
        return default, True
    return max(0.0, min(1.0, f)), not (0.0 <= f <= 1.0)


def parse_response(raw, message_id, candidates):
    """Parse a model response into a RouterDecision.

    Never raises. Anything unusable degrades to a conservative, well-formed
    decision with a note explaining what happened, so one bad response cannot
    remove a row from output.csv.
    """
    notes = []

    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            return RouterDecision(
                message_id=message_id, action="digest", message_type="unknown",
                reason="Router response could not be parsed; defaulted to a safe low-priority action.",
                confidence=0.3, parse_notes=["unparseable response: %s" % exc])
    if not isinstance(payload, dict):
        return RouterDecision(
            message_id=message_id, action="digest", message_type="unknown",
            reason="Router response had an unexpected shape; defaulted to a safe low-priority action.",
            confidence=0.3, parse_notes=["expected object, got %s" % type(payload).__name__])

    action = str(payload.get("action") or "").strip().lower()
    if action not in ACTIONS:
        notes.append("invalid action %r -> digest" % action)
        action = "digest"

    message_type = str(payload.get("message_type") or "").strip().lower()
    if message_type not in MESSAGE_TYPES:
        notes.append("invalid message_type %r -> unknown" % message_type)
        message_type = "unknown"

    reason = " ".join(str(payload.get("reason") or "").split())
    if not reason:
        notes.append("empty reason")
        reason = "No explanation was produced for this routing decision."

    confidence, adjusted = _clamp01(payload.get("confidence"))
    if adjusted:
        notes.append("confidence %r clamped to %.2f" % (payload.get("confidence"), confidence))

    # -- evidence -------------------------------------------------------
    indices = payload.get("evidence_indices")
    if indices is None:
        indices = []
    if not isinstance(indices, (list, tuple)):
        notes.append("evidence_indices was %s -> none" % type(indices).__name__)
        indices = []

    ids = evidence_ids_for(candidates, indices)
    dropped = len(indices) - len(ids)
    if dropped > 0:
        notes.append("%d evidence index/indices invalid or duplicated" % dropped)

    if len(ids) > MAX_EVIDENCE_IDS:
        notes.append("evidence truncated from %d to %d (no padding)" % (len(ids), MAX_EVIDENCE_IDS))
        ids = ids[:MAX_EVIDENCE_IDS]

    return RouterDecision(
        message_id=message_id, action=action, message_type=message_type,
        reason=reason, confidence=confidence, evidence_message_ids=ids,
        parse_notes=notes)
