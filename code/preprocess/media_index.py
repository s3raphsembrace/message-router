"""media_id -> file path resolution.

`images.csv` and `voice_notes.csv` are the only mapping from a message's `media_id`
to a file on disk. Resolution is split out from interpretation so that an indexing
problem (unknown id, deleted file) is reported as a distinct status rather than
surfacing as an opaque model failure later.
"""

import csv
import hashlib
import os
from typing import Dict, Optional, Tuple

from schema import (
    KIND_IMAGE,
    KIND_VOICE,
    STATUS_MISSING_FILE,
    STATUS_MISSING_INDEX,
    STATUS_OK,
    STATUS_UNREADABLE,
)

IMAGE_INDEX = "images.csv"
VOICE_INDEX = "voice_notes.csv"

# (index file, id column) per media kind
_INDEX_SPEC = {
    KIND_IMAGE: (IMAGE_INDEX, "image_id"),
    KIND_VOICE: (VOICE_INDEX, "voice_note_id"),
}

MIME_BY_EXT = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class MediaIndex:
    """Resolves media ids to on-disk paths, and reports why when it cannot."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self._tables: Dict[str, Dict[str, str]] = {}
        for kind, (fname, id_col) in _INDEX_SPEC.items():
            self._tables[kind] = self._load(fname, id_col)

    def _load(self, fname: str, id_col: str) -> Dict[str, str]:
        path = os.path.join(self.dataset_dir, fname)
        if not os.path.exists(path):
            return {}
        with open(path, newline="", encoding="utf-8") as f:
            return {r[id_col]: r["file_path"] for r in csv.DictReader(f) if r.get(id_col)}

    def known_ids(self, kind: str):
        return set(self._tables.get(kind, {}))

    def resolve(self, media_id: str, kind: str) -> Tuple[str, Optional[str], Optional[str], str]:
        """Return (status, rel_path, abs_path, error).

        status is STATUS_OK only when the file exists and is non-empty.
        Never raises -- every failure mode is encoded in the returned status.
        """
        table = self._tables.get(kind)
        if table is None:
            return STATUS_MISSING_INDEX, None, None, "unknown media kind %r" % kind

        rel = table.get(media_id)
        if rel is None:
            return (
                STATUS_MISSING_INDEX,
                None,
                None,
                "%s not present in %s" % (media_id, _INDEX_SPEC[kind][0]),
            )

        abs_path = os.path.join(self.dataset_dir, rel.replace("/", os.sep))
        if not os.path.exists(abs_path):
            return STATUS_MISSING_FILE, rel, abs_path, "file not found: %s" % rel

        try:
            if os.path.getsize(abs_path) == 0:
                return STATUS_UNREADABLE, rel, abs_path, "file is empty: %s" % rel
        except OSError as exc:                                  # permissions, bad handle
            return STATUS_UNREADABLE, rel, abs_path, "stat failed: %s" % exc

        return STATUS_OK, rel, abs_path, ""

    @staticmethod
    def mime_for(path: str) -> Optional[str]:
        return MIME_BY_EXT.get(os.path.splitext(path)[1].lower())

    @staticmethod
    def read_bytes(path: str) -> Tuple[Optional[bytes], str]:
        """Read a media file. Returns (data, error); data is None on failure."""
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as exc:
            return None, "read failed: %s" % exc
        if not data:
            return None, "file is empty"
        return data, ""

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
