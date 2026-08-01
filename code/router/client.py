"""Gemini client for the router (Layer 3).

Returns raw response text; validation, the re-ask and the fallback all live in
route.py, so this module only has to do one thing.

Determinism, per AGENTS.md 6.3: temperature 0, a structured response schema, and a
response cache keyed on a hash of (model, system, user). A second run over an
unchanged prompt makes no network calls and reproduces output.csv byte for byte.
The re-ask carries a different user turn, so it gets its own cache entry rather
than colliding with the first attempt.

The key is read from the environment only -- GEMINI_API_KEY, GOOGLE_API_KEY or
GOOGLE_GENAI_API_KEY -- and is never logged, cached or written to disk.
"""

import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision import RESPONSE_SCHEMA                          # noqa: E402

# gemini-3.5-flash-lite by default. gemini-2.5-flash is unusable here: on a
# free-tier key it allows 20 requests PER DAY, which cannot route 110 messages.
# Quota is per model, and 3.5-flash-lite is both available and roughly 10x faster
# (~0.5s vs ~9.4s per call measured). Override with ROUTER_MODEL.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
API_KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0

# Free-tier keys also cap requests per minute. Spacing calls is cheaper than
# discovering the limit through 429s, since a rejected request still costs quota.
MIN_INTERVAL_SECONDS = float(os.environ.get("ROUTER_MIN_INTERVAL", "1.0"))

# A per-minute 429 is worth waiting out. A per-day one is not: every retry burns
# quota that is already gone, which is what turned one exhausted key into 89
# failed rows on the first run.
_DAILY_QUOTA_MARKERS = ("PerDay", "per day", "GenerateRequestsPerDay")


def _is_daily_quota_error(message):
    return any(marker in message for marker in _DAILY_QUOTA_MARKERS)


def _suggested_retry_seconds(message, default=0.0):
    """Pull 'Please retry in 28.59s' out of a 429 body."""
    import re
    match = re.search(r"retry in ([0-9.]+)s", message)
    if not match:
        return default
    try:
        return min(float(match.group(1)), 60.0)
    except ValueError:
        return default

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESPONSE_CACHE = os.path.join(REPO_ROOT, "cache", "router_responses.json")


def api_key():
    for name in API_KEY_ENV:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class ResponseCache:
    """Prompt-hash -> raw response text. Committed, so runs are reproducible."""

    def __init__(self, path=DEFAULT_RESPONSE_CACHE):
        self.path = os.path.abspath(path)
        self._entries = {}
        self.hits = 0
        self.misses = 0
        self._dirty = False
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._entries = json.load(f).get("entries", {})
            except (OSError, ValueError):
                self._entries = {}

    @staticmethod
    def key(model, system, user):
        digest = hashlib.sha256()
        for part in (model, system, user):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def get(self, key):
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key, value):
        self._entries[key] = value
        self._dirty = True

    def save(self):
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        blob = {"cache_version": 1, "entries": self._entries}
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

    def __len__(self):
        return len(self._entries)


class RouterClient:
    """Callable as call_model(system, user) -> raw response text."""

    def __init__(self, model=None, cache_path=DEFAULT_RESPONSE_CACHE, use_cache=True,
                 progress=None):
        self.model = model or os.environ.get("ROUTER_MODEL", DEFAULT_MODEL)
        self._key = api_key()
        self.cache = ResponseCache(cache_path) if use_cache else None
        self.calls = 0
        self.errors = 0
        self.quota_exhausted = False
        self._last_call_at = 0.0
        self.progress = progress
        self._client = None
        self._init_error = ""
        if self._key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._key)
            except Exception as exc:
                self._init_error = "client init failed: %s" % exc

    @property
    def available(self):
        return self._client is not None

    @property
    def unavailable_reason(self):
        if self._client is not None:
            return ""
        return self._init_error or ("no API key set (checked %s)" % ", ".join(API_KEY_ENV))

    def __call__(self, system, user):
        if not self.available:
            raise RuntimeError(self.unavailable_reason)

        cache_key = None
        if self.cache is not None:
            cache_key = ResponseCache.key(self.model, system, user)
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.progress:
                    self.progress("cache")
                return cached

        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            system_instruction=system,
        )

        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            try:
                self.calls += 1
                response = self._client.models.generate_content(
                    model=self.model, contents=user, config=config)
                text = (getattr(response, "text", None) or "").strip()
                if text:
                    if cache_key:
                        self.cache.put(cache_key, text)
                    if self.progress:
                        self.progress("ok")
                    return text
                last = "empty response"
            except Exception as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    if _is_daily_quota_error(message):
                        # Nothing to wait for -- the day's allowance is spent.
                        self.quota_exhausted = True
                        self.errors += 1
                        if self.progress:
                            self.progress("error")
                        raise RuntimeError("daily quota exhausted for model %s: %s"
                                           % (self.model, message[:200]))
                    wait = _suggested_retry_seconds(message, BACKOFF_SECONDS * attempt)
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(wait)
                        continue
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)

        self.errors += 1
        if self.progress:
            self.progress("error")
        # route.py turns an exception into a counted fallback for this row.
        raise RuntimeError("router call failed after %d attempts: %s" % (MAX_ATTEMPTS, last))

    def _pace(self):
        """Keep at least MIN_INTERVAL_SECONDS between requests."""
        if MIN_INTERVAL_SECONDS <= 0:
            return
        gap = time.time() - self._last_call_at
        if gap < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - gap)
        self._last_call_at = time.time()

    def save(self):
        if self.cache is not None:
            self.cache.save()
