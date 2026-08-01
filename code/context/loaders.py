"""Dataset loading and indexing.

Everything is read once into memory and indexed for the lookups Layer 2 performs
per message. The whole dataset is small (110 messages, 412 history rows, ~750 daily
load rows), so there is no reason to stream or query lazily -- one load, then pure
dictionary access.
"""

import csv
import datetime as dt
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
DEFAULT_DATASET = os.path.join(REPO_ROOT, "dataset")
DEFAULT_MEDIA_CACHE = os.path.join(REPO_ROOT, "cache", "media_interpretations.json")

sys.path.insert(0, os.path.join(CODE_DIR, "preprocess"))


def parse_ts(value):
    """Parse a dataset timestamp. Returns None rather than raising on junk."""
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def counterpart_of(row):
    """The other party in a conversation, whichever kind it is.

    business_id | group_id | sender_user_id -- exactly one is populated per row.
    This single key is what makes 'have I seen this sender before' a dict lookup
    across all three conversation types.
    """
    return row.get("business_id") or row.get("group_id") or row.get("sender_user_id") or ""


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Dataset:
    """All participant-facing tables, indexed for per-message assembly."""

    def __init__(self, dataset_dir=DEFAULT_DATASET, media_cache=DEFAULT_MEDIA_CACHE):
        self.dataset_dir = os.path.abspath(dataset_dir)

        self.messages = _read(os.path.join(self.dataset_dir, "messages.csv"))
        self.samples = _read(os.path.join(self.dataset_dir, "sample_messages.csv"))
        self.history = _read(os.path.join(self.dataset_dir, "message_history.csv"))
        events = _read(os.path.join(self.dataset_dir, "message_events.csv"))

        self.users = {r["user_id"]: r for r in _read(os.path.join(self.dataset_dir, "users.csv"))}
        self.groups = {r["group_id"]: r for r in _read(os.path.join(self.dataset_dir, "groups.csv"))}
        self.businesses = {
            r["business_id"]: r
            for r in _read(os.path.join(self.dataset_dir, "business_accounts.csv"))
        }

        # (group_id, user_id) -> membership row
        self.membership = {
            (r["group_id"], r["user_id"]): r
            for r in _read(os.path.join(self.dataset_dir, "group_members.csv"))
        }
        # user_id -> set of group_ids (used as a trust proxy for personal senders)
        self.groups_of_user = defaultdict(set)
        for (gid, uid) in self.membership:
            self.groups_of_user[uid].add(gid)

        # (user_id, business_id) -> relationship row
        self.business_history = {
            (r["user_id"], r["business_id"]): r
            for r in _read(os.path.join(self.dataset_dir, "user_business_history.csv"))
        }

        # (user_id, message_id) -> reaction row
        self.events = {(r["user_id"], r["message_id"]): r for r in events}

        # user_id -> their historical messages, newest first
        self.history_by_user = defaultdict(list)
        for row in self.history:
            self.history_by_user[row["user_id"]].append(row)
        for rows in self.history_by_user.values():
            rows.sort(key=lambda r: parse_ts(r["created_at"]) or dt.datetime.min, reverse=True)

        # user_id -> {date -> daily load row}
        self.daily_load = defaultdict(dict)
        for row in _read(os.path.join(self.dataset_dir, "daily_notification_summary.csv")):
            self.daily_load[row["user_id"]][row["date"]] = row

        self.media = self._load_media(media_cache)
        self._baselines = None

    @property
    def baselines(self):
        """Dataset-wide denominators for relative reporting. Computed once."""
        if self._baselines is None:
            from aggregates import build_baselines
            self._baselines = build_baselines(self)
        return self._baselines

    @staticmethod
    def _load_media(cache_path):
        """Layer 1 output, keyed on media_id. Absent cache is normal, not an error --
        it just means every media row routes on metadata alone."""
        try:
            from cache import InterpretationCache
        except ImportError:
            return {}
        try:
            return {rec.media_id: rec for rec in InterpretationCache(cache_path)}
        except Exception:
            return {}

    # -- convenience lookups ------------------------------------------------
    def user(self, user_id):
        return self.users.get(user_id, {})

    def group(self, group_id):
        return self.groups.get(group_id, {})

    def business(self, business_id):
        return self.businesses.get(business_id, {})

    def member(self, group_id, user_id):
        return self.membership.get((group_id, user_id), {})

    def relationship(self, user_id, business_id):
        return self.business_history.get((user_id, business_id), {})

    def event(self, user_id, message_id):
        return self.events.get((user_id, message_id), {})

    def user_history(self, user_id):
        return self.history_by_user.get(user_id, [])
