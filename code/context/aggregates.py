"""Behavioural aggregates.

Two things are computed here, both once per dataset rather than per message:

1. Per (user, counterpart) reaction stats. This is the single most load-bearing
   derived signal in the system -- raw history is weak, but history joined with how
   the user actually reacted to it is close to decisive.

2. Per-user notification load, from daily_notification_summary.

Critically, these aggregates feed BOTH the router (as raw counts it can weigh) and
the deterministic guard (as a boolean it enforces). Computing them in one place is
what stops the rules and the features from drifting apart.
"""

from collections import defaultdict

from loaders import counterpart_of, parse_ts


class ReactionStats:
    """How a user has historically reacted to one counterpart."""

    __slots__ = ("n", "opened", "replied", "dismissed", "reported", "muted_after",
                 "reaction_times")

    def __init__(self):
        self.n = 0
        self.opened = 0
        self.replied = 0
        self.dismissed = 0
        self.reported = 0
        self.muted_after = 0
        self.reaction_times = []

    @property
    def open_rate(self):
        return (self.opened / self.n) if self.n else None

    @property
    def dismiss_rate(self):
        return (self.dismissed / self.n) if self.n else None

    @property
    def median_reaction_minutes(self):
        if not self.reaction_times:
            return None
        vals = sorted(self.reaction_times)
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    # -- the one surviving hard rule -------------------------------------
    @property
    def unanimously_reported(self):
        """Every message from this counterpart was reported and none was ever opened.

        Deliberately narrow. The broader rule ('user has ever reported this sender')
        fires on 54% of messages and contradicts two labelled samples -- see
        code/analysis/rule_validation.py. Only this degenerate case is safe to
        enforce; everything softer is a feature for the router to weigh.
        """
        return self.n > 1 and self.reported == self.n and self.opened == 0

    def summary(self):
        """Compact dict for the prompt. None values are dropped by the assembler."""
        out = {
            "messages_seen": self.n,
            "opened": self.opened,
            "replied": self.replied,
            "dismissed": self.dismissed,
            "reported": self.reported,
            "muted_after": self.muted_after,
        }
        if self.open_rate is not None:
            out["open_rate"] = round(self.open_rate, 2)
        med = self.median_reaction_minutes
        if med is not None:
            out["median_reply_minutes"] = round(med, 1)
        return out


def build_reaction_stats(dataset):
    """(user_id, counterpart) -> ReactionStats, over all of message_history."""
    agg = defaultdict(ReactionStats)
    for row in dataset.history:
        ev = dataset.event(row["user_id"], row["message_id"])
        if not ev:
            continue
        st = agg[(row["user_id"], counterpart_of(row))]
        st.n += 1
        st.opened += int(ev.get("message_opened") or 0)
        st.replied += int(ev.get("message_replied") or 0)
        st.dismissed += int(ev.get("notification_dismissed") or 0)
        st.reported += int(ev.get("message_reported") or 0)
        st.muted_after += int(ev.get("muted_after_message") or 0)
        raw = ev.get("reaction_time_minutes")
        if raw not in (None, ""):
            try:
                st.reaction_times.append(float(raw))
            except ValueError:
                pass
    return agg


def notification_load(dataset, user_id, created_at=None):
    """The user's baseline notification load and how much of it they throw away.

    Treated as a fixed baseline profile rather than a rolling window, because the
    table does not overlap the messages at all: daily_notification_summary covers
    2026-07-04..07-17 while messages.csv spans 07-18..07-31, and every user has
    exactly 14 rows over that same period.

    A relative lookback would therefore hand a message dated 07-18 thirteen days of
    history and a message dated 07-31 none -- an artifact of the table's coverage,
    not a difference in user behaviour. Summarising the whole profile gives every
    message the same, comparable signal.

    Summarised, never dumped: 756 rows become three numbers, since no individual
    day changes a routing decision -- only the level and the dismissal rate do.
    """
    rows = dataset.daily_load.get(user_id) or {}
    if not rows:
        return {}

    dated = []
    for date_str, row in rows.items():
        parsed = parse_ts(date_str)
        if parsed is not None:
            dated.append((parsed, row))
    if not dated:
        return {}
    dated.sort(key=lambda t: t[0])

    sent = sum(int(r.get("notifications_sent") or 0) for _, r in dated)
    dismissed = sum(int(r.get("notifications_dismissed") or 0) for _, r in dated)
    days = len(dated)

    out = {
        "avg_notifications_per_day": round(sent / float(days), 1),
        "baseline_days": days,
        "baseline_period": "%s..%s" % (dated[0][0].strftime("%Y-%m-%d"),
                                       dated[-1][0].strftime("%Y-%m-%d")),
    }
    if sent:
        out["dismiss_rate"] = round(dismissed / float(sent), 2)
    return out
