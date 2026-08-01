"""Gemini-backed media interpretation.

Uses Gemini's native multimodal input: audio and image bytes are passed directly as
inline parts, so there is no separate ASR or OCR dependency.

Key-optional by design. With no credentials configured the interpreter reports
`available == False` and every call returns a typed STATUS_NO_API_KEY placeholder,
which is not cached -- so adding a key later reprocesses those files and nothing
else.
"""

import json
import os
import time
from typing import Optional, Tuple

from media_index import MediaIndex
from prompts import IMAGE_PROMPT, IMAGE_SCHEMA, VOICE_PROMPT, VOICE_SCHEMA
from schema import (
    KIND_IMAGE,
    KIND_VOICE,
    LAYOUT_OTHER,
    LAYOUT_UNKNOWN,
    STATUS_API_ERROR,
    STATUS_BAD_RESPONSE,
    STATUS_NO_API_KEY,
    STATUS_OK,
    VALID_LAYOUTS,
    MediaInterpretation,
)

# See code/router/client.py: gemini-2.5-flash is capped at 20 requests per day on
# a free-tier key. Quota is per model, so 3.5-flash-lite has its own bucket.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def _api_key() -> Optional[str]:
    for name in API_KEY_ENV:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


def _clamp01(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


class MediaInterpreter:
    """Interprets one media file at a time. Never raises on interpretation failure."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or os.environ.get("MEDIA_MODEL", DEFAULT_MODEL)
        self._key = api_key or _api_key()
        self._client = None
        self._init_error = ""
        if self._key:
            try:
                from google import genai                       # imported lazily: key-optional
                self._client = genai.Client(api_key=self._key)
            except Exception as exc:                           # SDK absent or client rejected
                self._init_error = "client init failed: %s" % exc
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str:
        if self._client is not None:
            return ""
        if self._init_error:
            return self._init_error
        return "no API key set (checked %s)" % ", ".join(API_KEY_ENV)

    # -- public -----------------------------------------------------------
    def interpret(
        self,
        media_id: str,
        kind: str,
        abs_path: str,
        rel_path: str,
        data: bytes,
        sha256: str,
    ) -> MediaInterpretation:
        """Interpret one already-read media file. Always returns a typed record."""
        if not self.available:
            rec = MediaInterpretation.failure(
                media_id, kind, STATUS_NO_API_KEY, self.unavailable_reason, rel_path, self.model
            )
            rec.sha256 = sha256
            return rec

        mime = MediaIndex.mime_for(abs_path)
        if not mime:
            rec = MediaInterpretation.failure(
                media_id, kind, STATUS_BAD_RESPONSE,
                "unsupported media extension: %s" % os.path.basename(abs_path),
                rel_path, self.model,
            )
            rec.sha256 = sha256
            return rec

        prompt, schema = (
            (VOICE_PROMPT, VOICE_SCHEMA) if kind == KIND_VOICE else (IMAGE_PROMPT, IMAGE_SCHEMA)
        )

        payload, error = self._call(data, mime, prompt, schema)
        if payload is None:
            rec = MediaInterpretation.failure(
                media_id, kind, STATUS_API_ERROR, error, rel_path, self.model
            )
            rec.sha256 = sha256
            return rec

        try:
            return self._to_record(media_id, kind, payload, rel_path, sha256)
        except Exception as exc:                               # malformed but parseable JSON
            rec = MediaInterpretation.failure(
                media_id, kind, STATUS_BAD_RESPONSE, "unexpected payload: %s" % exc,
                rel_path, self.model,
            )
            rec.sha256 = sha256
            return rec

    # -- internals --------------------------------------------------------
    def _call(self, data: bytes, mime: str, prompt: str, schema: dict) -> Tuple[Optional[dict], str]:
        """Call Gemini with inline media bytes. Returns (parsed_json, error)."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0,                                   # determinism (AGENTS.md 6.3)
            response_mime_type="application/json",
            response_schema=schema,
        )
        contents = [types.Part.from_bytes(data=data, mime_type=mime), prompt]

        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    last = "empty response"
                else:
                    try:
                        parsed = json.loads(text)
                    except ValueError as exc:
                        last = "response was not valid JSON: %s" % exc
                    else:
                        if isinstance(parsed, dict):
                            return parsed, ""
                        last = "response JSON was %s, expected object" % type(parsed).__name__
            except Exception as exc:
                last = "%s: %s" % (type(exc).__name__, exc)

            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)          # linear backoff
        return None, last

    def _to_record(
        self, media_id: str, kind: str, payload: dict, rel_path: str, sha256: str
    ) -> MediaInterpretation:
        conf = _clamp01(payload.get("confidence"), 0.5)

        if kind == KIND_VOICE:
            transcript = (payload.get("transcript") or "").strip()
            summary = (payload.get("intent_summary") or "").strip()
            return MediaInterpretation(
                media_id=media_id,
                kind=KIND_VOICE,
                status=STATUS_OK,
                transcript=transcript,
                intent_summary=summary,
                language=(payload.get("language") or "unknown").strip() or "unknown",
                layout=LAYOUT_UNKNOWN,                         # not applicable to audio
                # An empty transcript is a successful call with nothing in it; force
                # confidence to 0 so the router does not read silence as certainty.
                interp_confidence=conf if transcript else 0.0,
                model=self.model,
                file_path=rel_path,
                sha256=sha256,
            )

        layout = (payload.get("layout") or "").strip()
        if layout not in VALID_LAYOUTS or layout == LAYOUT_UNKNOWN:
            layout = LAYOUT_OTHER                              # model returned something off-enum
        text = (payload.get("extracted_text") or "").strip()
        desc = (payload.get("visual_description") or "").strip()
        return MediaInterpretation(
            media_id=media_id,
            kind=KIND_IMAGE,
            status=STATUS_OK,
            extracted_text=text,
            visual_description=desc,
            layout=layout,
            interp_confidence=conf if (text or desc) else 0.0,
            model=self.model,
            file_path=rel_path,
            sha256=sha256,
        )
