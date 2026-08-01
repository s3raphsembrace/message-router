"""Tests candidate hard rules against the 30 labelled samples.

A rule only belongs in the deterministic guard (Layer 4) if it decides on a fact
the model cannot verify AND it can only ever be right. This script checks four
candidate rules against sample_messages.csv, which is the only labelled data
available. A rule that would overturn a provided label is not a hard rule.

    python code/analysis/rule_validation.py

Results on the shipped dataset:
  R1 verified opt-out          -> HOLDS. 6 opted-out business messages, all
                                  promotional; none is genuinely transactional.
                                  Must still gate on promotional CONTENT, since
                                  one promo uses transactional vocabulary, and one
                                  is media-only so it depends on Layer 1.
  R2 reported/muted sender     -> FAILS. Fires on 59/110 messages and contradicts
                                  two labels. The discriminator is the open RATIO
                                  (19/21 vs 0/7), not whether a report ever
                                  happened. Demoted to a Layer 2 feature; only the
                                  degenerate case (unanimous reports, zero opens)
                                  survives as a rule.
  R3 quiet-hours downgrade     -> INERT. Zero labelled samples fall inside quiet
                                  hours, so the rule has no ground-truth support in
                                  either direction. Only 8/110 live messages are
                                  affected and every one already routes to digest or
                                  mute on content alone, so the rule changes nothing
                                  and can only ever demote a correct notify.
                                  Demoted to a Layer 2 feature.
  R4 scam signature            -> HOLDS ONLY AS A CONJUNCTION. The labels show
                                  domain mismatch alone -> digest/promotion, while
                                  mismatch + unverified + young domain + elevated
                                  reports -> mute/spam.
"""

import collections
import csv
import datetime as dt
import os
import re
import sys

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset"
)


def load(name):
    with open(os.path.join(DATASET, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def counterpart(r):
    return r["business_id"] or r["group_id"] or r["sender_user_id"]


def parse_ts(s):
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")


def in_quiet_hours(window, created_at):
    """True if created_at falls inside the user's DND window (handles midnight wrap)."""
    if not window or "-" not in window:
        return None
    a, b = window.split("-")
    start = dt.time(*map(int, a.split(":")))
    end = dt.time(*map(int, b.split(":")))
    t = parse_ts(created_at).time()
    return start <= t < end if start <= end else (t >= start or t < end)


def build_reactions(hist, events):
    """Per (user, counterpart) reaction aggregates. Layer 2 computes these too --
    the guard's boolean and the router's raw counts must come from one source."""
    ev = {(r["user_id"], r["message_id"]): r for r in events}
    agg = collections.defaultdict(
        lambda: dict(n=0, reported=0, muted_after=0, dismissed=0, opened=0))
    for h in hist:
        e = ev.get((h["user_id"], h["message_id"]))
        if not e:
            continue
        a = agg[(h["user_id"], counterpart(h))]
        a["n"] += 1
        a["reported"] += int(e["message_reported"])
        a["muted_after"] += int(e["muted_after_message"])
        a["dismissed"] += int(e["notification_dismissed"])
        a["opened"] += int(e["message_opened"])
    return agg


def main():
    msgs = load("messages.csv")
    samp = load("sample_messages.csv")
    users = {r["user_id"]: r for r in load("users.csv")}
    biz = {r["business_id"]: r for r in load("business_accounts.csv")}
    ubh = {(r["user_id"], r["business_id"]): r for r in load("user_business_history.csv")}
    react = build_reactions(load("message_history.csv"), load("message_events.csv"))

    promo_re = re.compile(
        r"\b(offer|sale|deal|discount|coupon|save|off|limited|shop|price|launch|benefit|"
        r"package|fares?)\b", re.I)
    txn_re = re.compile(
        r"\b(order|delivery|deliver|shipped|packed|appointment|prescription|refill|booking|"
        r"ticket|statement|invoice|receipt|otp|due date|claim)\b", re.I)

    # -- R3 -----------------------------------------------------------------
    print("=" * 92)
    print("R3  'quiet hours -> downgrade'   tested against the 30 labelled samples")
    print("=" * 92)
    tab = collections.Counter()
    for s in samp:
        q = in_quiet_hours(users[s["user_id"]]["do_not_disturb_window"], s["created_at"])
        if q is not None:
            tab[(q, s["action"])] += 1
    for (q, a), n in sorted(tab.items(), key=lambda x: (-x[0][0], x[0][1])):
        print("   in_quiet_hours=%-5s labelled=%-7s %d" % (q, a, n))
    inside = sum(n for (q, _), n in tab.items() if q)
    print("\n   labelled samples INSIDE quiet hours: %d  <- no ground truth either way" % inside)
    live = [m for m in msgs
            if in_quiet_hours(users[m["user_id"]]["do_not_disturb_window"], m["created_at"])]
    print("   live messages inside quiet hours: %d / %d" % (len(live), len(msgs)))
    for m in live:
        print("     %-8s u=%-6s %-9s at %s" % (
            m["message_id"], m["user_id"], m["conversation_type"], m["created_at"]))
        print("        %s" % (m["message_text"] or "(media-only)")[:150].replace("\n", " "))
    print("\n   VERDICT: inert. Every affected row already routes to digest or mute on content.")

    # -- R1 -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("R1  'verified opt-out -> mute'   does opt-out ever cover transactional mail?")
    print("=" * 92)
    rows = []
    for m in msgs:
        if m["conversation_type"] != "business":
            continue
        h = ubh.get((m["user_id"], m["business_id"]))
        if h and h["promotions_opted_out_at"]:
            t = m["message_text"] or ""
            rows.append((m, h, bool(promo_re.search(t)), bool(txn_re.search(t))))
    for m, h, p, x in rows:
        tag = "PROMO" if p and not x else ("TRANSACTIONAL" if x and not p else
                                           ("BOTH" if p and x else "MEDIA-ONLY/NEITHER"))
        print("   %-8s u=%-6s %-20s %-19s knows=%s" % (
            m["message_id"], m["user_id"],
            (biz.get(m["business_id"], {}).get("brand_name") or "")[:20], tag,
            h["why_user_knows_account"]))
    print("\n   opted-out messages: %d | purely transactional (would be wrongly muted): %d"
          % (len(rows), sum(1 for r in rows if r[3] and not r[2])))
    print("   VERDICT: holds, but gate on promotional CONTENT, not on business-sender.")

    # -- R2 -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("R2  'user reported/muted this sender -> mute'")
    print("=" * 92)
    fires = [m for m in msgs
             if (react.get((m["user_id"], counterpart(m))) or {}).get("reported", 0)
             or (react.get((m["user_id"], counterpart(m))) or {}).get("muted_after", 0)]
    print("   fires on %d / %d live messages (%.0f%%)" % (
        len(fires), len(msgs), 100.0 * len(fires) / len(msgs)))
    viol = []
    for s in samp:
        a = react.get((s["user_id"], counterpart(s)))
        if a and (a["reported"] or a["muted_after"]) and s["action"] != "mute":
            viol.append((s, a))
    print("\n   labelled samples where the rule fires but the label is NOT mute: %d" % len(viol))
    for s, a in viol:
        print("     %s  label=%s/%s   reported=%d muted_after=%d opened=%d of %d" % (
            s["message_id"], s["action"], s["message_type"],
            a["reported"], a["muted_after"], a["opened"], a["n"]))
        print("        %s" % (s["message_text"] or "")[:130].replace("\n", " "))
    print("\n   contrast with rows where muting IS correct:")
    for m in fires:
        a = react[(m["user_id"], counterpart(m))]
        if a["opened"] == 0 and a["reported"] == a["n"] and a["n"] > 1:
            print("     %-8s reported=%d of %d, opened=%d  ::  %s" % (
                m["message_id"], a["reported"], a["n"], a["opened"],
                (m["message_text"] or "(media-only)")[:70].replace("\n", " ")))
    print("\n   VERDICT: fails as a hard rule. Keep only 'reported == n AND opened == 0'.")

    # -- R4 -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("R4  'clear scam signature'   is any single signal sufficient?")
    print("=" * 92)

    def signature(r):
        b = biz.get(r["business_id"])
        if not b:
            return None
        return (("mismatch", b["official_domain"] != b["domain_used_by_sender"]),
                ("unverified", b["verified"] == "0"),
                ("young_domain", int(b["domain_used_by_sender_age_days"]) < 90),
                ("many_reports", int(b["user_reports_30d"]) >= 20))

    grouped = collections.defaultdict(list)
    for s in samp:
        sig = signature(s)
        if sig:
            grouped[sig].append("%s/%s" % (s["action"], s["message_type"]))
    print("   labelled business samples grouped by trust signature:")
    for sig, labels in grouped.items():
        print("     %s" % dict(sig))
        print("        -> %s" % collections.Counter(labels).most_common())
    trap = [m for m in msgs
            if (signature(m) or ()) and dict(signature(m) or ()).get("mismatch")
            and not dict(signature(m)).get("unverified")]
    print("\n   live messages with domain mismatch but VERIFIED and old (single-signal trap): %d"
          % len(trap))
    for m in trap:
        b = biz[m["business_id"]]
        print("     %-8s %-28s official=%-22s sender=%-20s acct=%sd dom=%sd reports=%s" % (
            m["message_id"], b["brand_name"][:28], b["official_domain"][:22],
            b["domain_used_by_sender"][:20], b["account_age_days"],
            b["domain_used_by_sender_age_days"], b["user_reports_30d"]))
    print("\n   VERDICT: hard-mute only on the FULL conjunction; partial matches are features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
