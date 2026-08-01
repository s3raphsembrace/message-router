"""Evidence that personalization is decidable from the provided data.

The task's core claim is that two identical-looking messages can require opposite
routing depending on who receives them. This script finds those collisions in the
real dataset and shows which file resolves each one. Run before trusting any
routing design:

    python code/analysis/collision_check.py

Findings this produces (as of the shipped dataset):
  1. Two promo pairs share the SAME image file across different users, split
     cleanly by user_business_history (allows_promotions + promotions_opted_out_at
     + opened/dismissed ratio). Two other identical-text pairs do NOT diverge --
     both recipients have live transactional relationships -- so duplicate text is
     not by itself a routing signal.
  2. Money-related business messages separate almost linearly on
     official_domain != domain_used_by_sender combined with account age and
     report count.
  3. 14 group messages target groups the user has muted; mentions are literal
     @user_id tokens, and every mention in messages.csv addresses the recipient.
"""

import collections
import csv
import os
import re
import sys

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset"
)


def load(name):
    with open(os.path.join(DATASET, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())


def toks(t):
    return set(norm(t).split())


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if A and B else 0.0


def counterpart(r):
    """The other party in the conversation, whichever kind it is."""
    return r["business_id"] or r["group_id"] or r["sender_user_id"]


def main():
    msgs = load("messages.csv")
    gm = {(r["group_id"], r["user_id"]): r for r in load("group_members.csv")}
    grps = {r["group_id"]: r for r in load("groups.csv")}
    biz = {r["business_id"]: r for r in load("business_accounts.csv")}
    ubh = {(r["user_id"], r["business_id"]): r for r in load("user_business_history.csv")}

    print("=" * 92)
    print("CASE 1 - same promo, different users: does user_business_history split them?")
    print("=" * 92)
    bmsgs = [m for m in msgs if m["conversation_type"] == "business"]
    seen, found = set(), 0
    for i, a in enumerate(bmsgs):
        for b in bmsgs[i + 1:]:
            if a["user_id"] == b["user_id"]:
                continue
            s = jaccard(a["message_text"], b["message_text"])
            same_media = bool(a["media_id"]) and a["media_id"] == b["media_id"]
            if s < 0.55 and not same_media:
                continue
            key = tuple(sorted([a["message_id"], b["message_id"]]))
            if key in seen:
                continue
            seen.add(key)
            found += 1
            print("\n--- pair  text_sim=%.2f  same_media=%s" % (s, same_media))
            for m in (a, b):
                h = ubh.get((m["user_id"], m["business_id"]))
                print("  %-8s u=%-6s %-20s media=%s" % (
                    m["message_id"], m["user_id"],
                    (biz.get(m["business_id"], {}).get("brand_name") or "")[:20],
                    m["media_id"] or "-"))
                if h:
                    print("      knows=%-28s allows_promo=%s opted_out=%-16s opened30=%-3s dismissed30=%s"
                          % (h["why_user_knows_account"], h["allows_promotions"],
                             h["promotions_opted_out_at"] or "-",
                             h["messages_opened_30d"], h["messages_dismissed_30d"]))
                else:
                    print("      no user_business_history row -- no prior relationship")
    print("\n[case 1] cross-user near-duplicate business pairs: %d" % found)

    print("\n" + "=" * 92)
    print("CASE 2 - payment reminders: does business trust metadata split legit from scam?")
    print("=" * 92)
    money = re.compile(
        r"\b(pay|payment|otp|upi|kyc|verif|fee|due|invoice|account|bank|refund|blocked|"
        r"expire|link|click|wallet|card)\b", re.I)
    rows = []
    for m in msgs:
        if m["conversation_type"] != "business" or not money.search(m["message_text"] or ""):
            continue
        b = biz.get(m["business_id"])
        if b:
            rows.append((b["official_domain"] != b["domain_used_by_sender"], m, b))
    rows.sort(key=lambda r: (not r[0], r[2]["brand_name"]))
    print("\n%-9s %-7s %-20s %-4s %-24s %-24s %-8s %-8s %s" % (
        "msg", "user", "brand", "ver", "official_domain", "sender_domain",
        "acct_age", "dom_age", "reports"))
    for mismatch, m, b in rows:
        print("%-9s %-7s %-20s %-4s %-24s %-24s %-8s %-8s %-7s %s" % (
            m["message_id"], m["user_id"], b["brand_name"][:20], b["verified"],
            b["official_domain"][:24], b["domain_used_by_sender"][:24],
            b["account_age_days"], b["domain_used_by_sender_age_days"],
            b["user_reports_30d"], "<== MISMATCH" if mismatch else ""))
    mm = sum(1 for r in rows if r[0])
    print("\n[case 2] money-related business messages: %d | mismatched: %d | clean: %d"
          % (len(rows), mm, len(rows) - mm))
    print("[case 2] NOTE: mismatch alone is NOT scam -- see rule_validation.py, which shows a")
    print("         verified 4000+ day-old sender using a link shortener labelled digest/promotion.")

    print("\n" + "=" * 92)
    print("CASE 3 - muted groups: is there content signal to override the mute?")
    print("=" * 92)
    gmsgs = [m for m in msgs if m["conversation_type"] == "group"]
    muted = [m for m in gmsgs
             if (gm.get((m["group_id"], m["user_id"])) or {}).get("group_muted_by_user") == "1"]
    print("group messages: %d | to a group THIS user has muted: %d" % (len(gmsgs), len(muted)))
    for m in muted:
        mem = gm[(m["group_id"], m["user_id"])]
        g = grps.get(m["group_id"], {})
        snd = gm.get((m["group_id"], m["sender_user_id"])) or {}
        print("\n%-8s u=%-6s %-26s [%s] sender_role=%s read30=%s dismissed30=%s" % (
            m["message_id"], m["user_id"], (g.get("group_name") or "")[:26],
            g.get("group_type"), snd.get("role", "?"),
            mem["messages_read_30d"], mem["notifications_dismissed_30d"]))
        print("    %s" % (m["message_text"] or "(media-only message)")[:200].replace("\n", " "))

    print("\n--- @mention targeting (mentions are literal @user_id; there is no name field) ---")
    ats = lambda t: set(re.findall(r"@(u_\d+)", t or ""))
    self_m = other_m = 0
    for m in msgs:
        a = ats(m["message_text"])
        if not a:
            continue
        hit = m["user_id"] in a
        self_m += hit
        other_m += not hit
        mem = gm.get((m["group_id"], m["user_id"]), {})
        print("  %-8s to=%-6s mentions=%-10s muted=%-4s %s" % (
            m["message_id"], m["user_id"], ",".join(sorted(a)),
            mem.get("group_muted_by_user", "-"),
            "SELF" if hit else "addresses someone else"))
    print("\n[case 3] self-addressed=%d other-addressed=%d" % (self_m, other_m))
    print("[case 3] NOTE: a self-mention is NOT sufficient to override a mute -- one of the two")
    print("         self-mentions inside muted groups is chain-forward spam. See rule_validation.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
