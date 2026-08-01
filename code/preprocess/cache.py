"""Interpretation cache, keyed on media_id.

Media interpretation is the only expensive, non-deterministic, network-dependent
step in the pipeline. Caching it to a committed JSON file is what lets Layers 2-4
replay identically with no API key and no network.

Key is `media_id` (as specified). `sha256` is stored alongside as an integrity
check: if the bytes behind a cached id ever change, `is_stale` reports it rather
than silently serving an interpretation of a different file.

Failure placeholders are cached too, but only the *permanent* ones. A transient
API error or a missing key must not be baked in -- otherwise a later run with
credentials would keep serving the empty placeholder.
"""

import json
import os
import tempfile
from typing import Dict, Iterator, Optional

from schema import (
    FAILURE_STATUSES,
    STATUS_API_ERROR,
    STATUS_BAD_RESPONSE,
    STATUS_NO_API_KEY,
    MediaInterpretation,
)

CACHE_VERSION = 1

# Statuses that reflect the environment rather than the file itself. Never cached,
# so that adding a key (or fixing the network) is enough to reprocess.
TRANSIENT_STATUSES = frozenset({STATUS_NO_API_KEY, STATUS_API_ERROR, STATUS_BAD_RESPONSE})


class InterpretationCache:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._entries: Dict[str, MediaInterpretation] = {}
        self._dirty = False
        self.load()

    # -- io ---------------------------------------------------------------
    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            # A corrupt cache is a performance problem, not a correctness one.
            # Start clean rather than taking the run down.
            self._entries = {}
            return
        for mid, d in (blob.get("entries") or {}).items():
            try:
                self._entries[mid] = MediaInterpretation.from_dict(d)
            except (TypeError, ValueError):
                continue

    def save(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        blob = {
            "cache_version": CACHE_VERSION,
            "entries": {
                mid: rec.to_dict()
                for mid, rec in sorted(self._entries.items())      # sorted -> stable diffs
            },
        }
        # Atomic replace: a crash mid-write must not leave a truncated cache.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(blob, f, indent=2, ensure_ascii=False, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._dirty = False

    # -- access -----------------------------------------------------------
    def get(self, media_id: str) -> Optional[MediaInterpretation]:
        return self._entries.get(media_id)

    def put(self, rec: MediaInterpretation) -> bool:
        """Store an interpretation. Returns True if it was actually cached."""
        if rec.status in TRANSIENT_STATUSES:
            return False
        self._entries[rec.media_id] = rec
        self._dirty = True
        return True

    def is_stale(self, media_id: str, sha256: str) -> bool:
        """True when the cached entry was built from different bytes."""
        rec = self._entries.get(media_id)
        if rec is None or not rec.sha256 or not sha256:
            return False
        return rec.sha256 != sha256

    def __contains__(self, media_id: str) -> bool:
        return media_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[MediaInterpretation]:
        return iter(self._entries.values())

    def stats(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for rec in self._entries.values():
            out[rec.status] = out.get(rec.status, 0) + 1
        return out

    @property
    def failures(self):
        return [r for r in self._entries.values() if r.status in FAILURE_STATUSES]
