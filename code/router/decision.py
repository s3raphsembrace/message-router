"""Router decision record and construction (Layer 3).

Named `decision` rather than `schema` on purpose -- code/preprocess/schema.py is
already on sys.path by the time this imports, and two modules called `schema`
resolve to whichever directory landed first.

There is one path from a model response to a row: validate strictly, build from a
clean payload, or fall back to the safe default. Nothing here repairs a bad
response -- silent repair would hide exactly the failures the re-ask exists to fix.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "context"))

from prompt import ACTIONS, MESSAGE_TYPES                      # noqa: E402
from retrieve import evidence_ids_for                          # noqa: E402
from validate import MAX_EVIDENCE_IDS                          # noqa: E402

NONE_EVIDENCE = "none"

# Applied when a response is still invalid after the one permitted re-ask.
# Conservative by design: `digest` neither interrupts the user nor suppresses
# something they may have needed, and 0.5 states honestly that this row is a
# fallback rather than a judgement.
SAFE_DEFAULT_ACTION = "digest"
SAFE_DEFAULT_TYPE = "unknown"
SAFE_DEFAULT_CONFIDENCE = 0.5
SAFE_DEFAULT_REASON = (
    "Routing could not be validated for this message, so it was held for later review "
    "rather than interrupting the user."
)

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
    attempts: int = 1              # model calls made for this row (0 = none available)
    fell_back: bool = False
    notes: List[str] = field(default_factory=list)

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
            "confidence": self.format_confidence(self.confidence),
            "evidence_message_ids": self.evidence_field,
        }

    @staticmethod
    def format_confidence(value) -> str:
        text = "%.2f" % float(value)
        return text.rstrip("0").rstrip(".") if "." in text else text


def build_decision(payload, message_id, candidates, attempts=1):
    """Build a decision from a payload that has already passed validation.

    Deduplicates evidence indices and applies the citation ceiling. Validation
    permits at most MAX_EVIDENCE_IDS *distinct* indices, so this can only collapse
    repeats, never discard a considered choice.
    """
    ids = evidence_ids_for(candidates, payload.get("evidence_indices") or [])
    notes = []
    if len(ids) > MAX_EVIDENCE_IDS:
        notes.append("evidence trimmed to %d" % MAX_EVIDENCE_IDS)
        ids = ids[:MAX_EVIDENCE_IDS]
    return RouterDecision(
        message_id=message_id,
        action=payload["action"].strip().lower(),
        message_type=payload["message_type"].strip().lower(),
        reason=" ".join(str(payload["reason"]).split()),
        confidence=float(payload["confidence"]),
        evidence_message_ids=ids,
        attempts=attempts,
        notes=notes,
    )


def safe_default(message_id, attempts=0, notes=None):
    """The fallback row: digest / unknown / 0.5 / none."""
    return RouterDecision(
        message_id=message_id,
        action=SAFE_DEFAULT_ACTION,
        message_type=SAFE_DEFAULT_TYPE,
        reason=SAFE_DEFAULT_REASON,
        confidence=SAFE_DEFAULT_CONFIDENCE,
        evidence_message_ids=[],
        attempts=attempts,
        fell_back=True,
        notes=list(notes or []),
    )
