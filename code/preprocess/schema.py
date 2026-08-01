"""Typed records for Layer 1 (multimodal preprocessing).

Every media file resolves to exactly one MediaInterpretation, whether or not the
interpretation succeeded. Downstream layers must branch on `status`, never on the
presence of a field -- a failed interpretation is still a well-formed record, so a
missing or unreadable file degrades the row to metadata-only routing instead of
killing the run.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional

# ---------------------------------------------------------------------------
# status values
# ---------------------------------------------------------------------------
# Terminal states. `OK` is the only one where the interpretation fields carry
# meaning; every other value tells the router to fall back to metadata.
STATUS_OK = "ok"                      # interpreted successfully
STATUS_MISSING_INDEX = "missing_index"  # media_id absent from images.csv / voice_notes.csv
STATUS_MISSING_FILE = "missing_file"    # indexed, but the file is not on disk
STATUS_UNREADABLE = "unreadable"        # on disk but empty / unopenable / not decodable
STATUS_NO_API_KEY = "no_api_key"        # no credentials configured; nothing attempted
STATUS_API_ERROR = "api_error"          # model call attempted and failed
STATUS_BAD_RESPONSE = "bad_response"    # model replied but not in the expected shape

FAILURE_STATUSES = frozenset({
    STATUS_MISSING_INDEX,
    STATUS_MISSING_FILE,
    STATUS_UNREADABLE,
    STATUS_NO_API_KEY,
    STATUS_API_ERROR,
    STATUS_BAD_RESPONSE,
})

# ---------------------------------------------------------------------------
# image layout classes
# ---------------------------------------------------------------------------
# Requested by the router design: a stock promo poster and a personal screenshot
# carry very different risk/priority priors even when their OCR text is similar.
LAYOUT_STOCK_PROMO = "stock_promo"              # designed marketing poster / ad creative
LAYOUT_PERSONAL_SCREENSHOT = "personal_screenshot"  # phone screenshot, chat capture, receipt
LAYOUT_OTHER = "other"                          # photo, document scan, anything else
LAYOUT_UNKNOWN = "unknown"                      # not determined (any failure status)

VALID_LAYOUTS = frozenset({
    LAYOUT_STOCK_PROMO,
    LAYOUT_PERSONAL_SCREENSHOT,
    LAYOUT_OTHER,
    LAYOUT_UNKNOWN,
})

KIND_VOICE = "voice"
KIND_IMAGE = "image"


@dataclass
class MediaInterpretation:
    """One interpreted media file. Serialises 1:1 into the on-disk cache."""

    media_id: str
    kind: str                      # KIND_VOICE | KIND_IMAGE
    status: str                    # STATUS_*

    # --- voice fields (empty unless kind == KIND_VOICE and status == OK) ---
    transcript: str = ""
    intent_summary: str = ""       # one line: what the sender wants from the listener
    language: str = ""             # best-effort BCP-47-ish tag, may be mixed e.g. "hi-en"

    # --- image fields (empty unless kind == KIND_IMAGE and status == OK) ---
    extracted_text: str = ""       # OCR: poster copy, screenshot text, overlaid text
    visual_description: str = ""   # short description of what is depicted
    layout: str = LAYOUT_UNKNOWN

    # --- provenance, common to both ---
    interp_confidence: float = 0.0  # 0..1; propagates into final routing confidence
    model: str = ""
    file_path: str = ""             # dataset-relative, as given by the index CSV
    sha256: str = ""                # integrity: detects a changed file behind a cached id
    error: str = ""                 # short diagnostic, empty when status == OK

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def has_content(self) -> bool:
        """True when this interpretation contributes any text signal to the router."""
        if not self.ok:
            return False
        if self.kind == KIND_VOICE:
            return bool(self.transcript.strip() or self.intent_summary.strip())
        return bool(self.extracted_text.strip() or self.visual_description.strip())

    def router_text(self) -> str:
        """Flattened text for prompt assembly.

        Deliberately labelled so Layer 3 can tell model-derived text from text the
        sender actually typed -- the two must not be weighed identically.
        """
        if not self.ok:
            return ""
        if self.kind == KIND_VOICE:
            parts = []
            if self.transcript.strip():
                parts.append("[voice transcript] " + self.transcript.strip())
            if self.intent_summary.strip():
                parts.append("[voice intent] " + self.intent_summary.strip())
            return "\n".join(parts)
        parts = []
        if self.extracted_text.strip():
            parts.append("[image text] " + self.extracted_text.strip())
        if self.visual_description.strip():
            parts.append("[image description] " + self.visual_description.strip())
        if self.layout != LAYOUT_UNKNOWN:
            parts.append("[image layout] " + self.layout)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MediaInterpretation":
        known = {f for f in cls.__dataclass_fields__}          # tolerate cache schema drift
        return cls(**{k: v for k, v in d.items() if k in known})

    # -- constructors for the failure paths -------------------------------
    @classmethod
    def failure(
        cls,
        media_id: str,
        kind: str,
        status: str,
        error: str = "",
        file_path: str = "",
        model: str = "",
    ) -> "MediaInterpretation":
        """A typed placeholder. Always well-formed; never raises."""
        return cls(
            media_id=media_id,
            kind=kind if kind in (KIND_VOICE, KIND_IMAGE) else KIND_IMAGE,
            status=status,
            layout=LAYOUT_UNKNOWN,
            interp_confidence=0.0,
            file_path=file_path,
            model=model,
            error=error[:500],
        )
