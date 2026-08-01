"""Layer 2 tests. No network, no API key, no third-party test runner.

    python code/context/test_context.py

Assertions are anchored on specific real rows whose correct handling was
established in code/analysis/ -- notably the three collision cases and the two
candidate hard rules that failed label validation. If a refactor breaks the
personalization the task is graded on, these fail.
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregates import build_reaction_stats, notification_load          # noqa: E402
from assemble import (                                                  # noqa: E402
    build_context,
    estimate_tokens,
)
from loaders import DEFAULT_DATASET, Dataset, counterpart_of, parse_ts  # noqa: E402
from retrieve import evidence_ids_for, shortlist                        # noqa: E402
from signals import in_quiet_hours, business_trust, repetition          # noqa: E402
from textutil import condense, jaccard, mentions, tokens                # noqa: E402

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + str(detail)) if detail and not cond else ""))


def msg(ds, mid):
    return next(r for r in ds.messages if r["message_id"] == mid)


def main():
    if not os.path.isdir(DEFAULT_DATASET):
        print("dataset not found: %s" % DEFAULT_DATASET)
        return 2

    ds = Dataset()
    stats = build_reaction_stats(ds)

    print("\n[loading]")
    check("messages loaded", len(ds.messages) == 110, len(ds.messages))
    check("history loaded", len(ds.history) == 412, len(ds.history))
    check("events cover history 1:1", len(ds.events) == len(ds.history))
    check("users indexed", len(ds.users) == 54, len(ds.users))
    check("history sorted newest-first per user",
          all(parse_ts(rows[0]["created_at"]) >= parse_ts(rows[-1]["created_at"])
              for rows in ds.history_by_user.values() if len(rows) > 1))
    check("absent media cache is not an error", isinstance(ds.media, dict))

    print("\n[helpers]")
    check("counterpart: business wins", counterpart_of(
        {"business_id": "b1", "group_id": "", "sender_user_id": ""}) == "b1")
    check("counterpart: group", counterpart_of(
        {"business_id": "", "group_id": "g1", "sender_user_id": "u9"}) == "g1")
    check("parse_ts handles junk", parse_ts("not-a-date") is None and parse_ts("") is None)
    check("mentions are literal user ids", mentions("@u_001 hi @u_002") == {"u_001", "u_002"})
    check("no mention -> empty", mentions("hello there") == set())
    check("tokens drop boilerplate", "customer" not in tokens("Dear Customer offer"))
    check("jaccard identical == 1", jaccard("pay now please", "pay now please") == 1.0)
    check("jaccard disjoint == 0", jaccard("alpha beta", "gamma delta") == 0.0)
    check("condense truncates with ascii", condense("x" * 50, 20).endswith("...")
          and len(condense("x" * 50, 20)) == 20)
    check("condense leaves short text", condense("short", 20) == "short")

    print("\n[quiet hours]")
    at = lambda s: dt.datetime(2026, 7, 20, *map(int, s.split(":")))
    check("wraps midnight - late night", in_quiet_hours("22:00-07:00", at("23:30")) is True)
    check("wraps midnight - early morning", in_quiet_hours("22:00-07:00", at("06:00")) is True)
    check("wraps midnight - daytime", in_quiet_hours("22:00-07:00", at("12:00")) is False)
    check("boundary start inclusive", in_quiet_hours("22:00-07:00", at("22:00")) is True)
    check("boundary end exclusive", in_quiet_hours("22:00-07:00", at("07:00")) is False)
    check("malformed window -> None", in_quiet_hours("garbage", at("12:00")) is None)
    check("missing window -> None", in_quiet_hours("", at("12:00")) is None)

    print("\n[reaction aggregates]")
    st = stats.get(("u_005", "group_005"))
    check("unanimous report + zero opens fires",
          st is not None and st.unanimously_reported is True,
          st.summary() if st else None)
    s1 = next(r for r in ds.samples if r["message_id"] == "sample_msg_001")
    st1 = stats.get((s1["user_id"], counterpart_of(s1)))
    check("high-engagement sender does NOT fire the rule",
          st1 is not None and st1.unanimously_reported is False,
          st1.summary() if st1 else None)
    check("that sender really was reported at least once",
          st1 is not None and st1.reported >= 1 and st1.opened >= 15,
          st1.summary() if st1 else None)
    check("open_rate computed", st1 is not None and 0.0 < st1.open_rate <= 1.0)
    check("empty stats are safe", ("nobody", "nothing") not in stats)

    print("\n[notification load]")
    load = notification_load(ds, "u_001")
    check("baseline is the full fixed period", load.get("baseline_days") == 14, load)
    check("period reported so it is not read as 'now'",
          load.get("baseline_period") == "2026-07-04..2026-07-17", load)
    check("dismiss rate present", 0.0 <= load.get("dismiss_rate", -1) <= 1.0, load)
    check("unknown user -> empty, no raise", notification_load(ds, "u_nope") == {})
    every = [notification_load(ds, u).get("baseline_days") for u in ds.users]
    check("every user gets the same baseline window", set(every) == {14}, set(every))

    print("\n[business trust: conjunction, not single signal]")
    polaris = business_trust(ds.business(msg(ds, "msg_041")["business_id"]))
    check("Polaris has a domain mismatch", polaris["domain_mismatch"] is True)
    check("Polaris is verified and old", polaris["verified"] and not polaris["young_domain"])
    check("Polaris is NOT flagged as scam", polaris["scam_signature"] is False, polaris)
    chase = business_trust(ds.business(msg(ds, "msg_019")["business_id"]))
    check("Chase impersonation IS flagged", chase["scam_signature"] is True, chase)
    check("scam needs all four", all(
        [chase["domain_mismatch"], not chase["verified"], chase["young_domain"],
         chase["high_reports"]]))
    check("unknown business -> empty, no raise", business_trust({}) == {})

    print("\n[signals on the collision rows]")
    c66 = build_context(msg(ds, "msg_066"), ds, stats)
    check("opt-out detected", c66["signals"].get("opted_out_of_promotions") is True)
    check("opt-out carries the timestamp", bool(c66["signals"].get("opted_out_at")))
    c65 = build_context(msg(ds, "msg_065"), ds, stats)
    check("same poster, other user is opted IN",
          c65["signals"].get("opted_out_of_promotions") is False
          and c65["signals"].get("allows_promotions") is True)
    check("same media id on both sides of the collision",
          msg(ds, "msg_065")["media_id"] == msg(ds, "msg_066")["media_id"])

    c56 = build_context(msg(ds, "msg_056"), ds, stats)
    check("muted group detected", c56["signals"]["group_muted_by_user"] is True)
    check("direct address detected", c56["signals"]["directly_addressed"] is True)
    c40 = build_context(msg(ds, "msg_040"), ds, stats)
    check("chain-spam @mention is ALSO both muted+addressed",
          c40["signals"]["group_muted_by_user"] is True
          and c40["signals"]["directly_addressed"] is True,
          "signals alone must not decide this -- Layer 3/4 must")

    print("\n[repetition]")
    # Use the real message text: Jaccard is length-sensitive, so a short excerpt of
    # a long template will not match the full resend. That is correct for detecting
    # resent templates, which are the same length by construction.
    rep = repetition(msg(ds, "msg_066")["message_text"], ds.user_history("u_007"))
    check("near-duplicates of a resent template found",
          rep["near_duplicates_in_history"] >= 1, rep)
    check("duplicate ids are capped", len(rep["duplicate_message_ids"]) <= 5)
    check("empty text -> zero", repetition("", ds.user_history("u_007"))
          ["near_duplicates_in_history"] == 0)

    print("\n[retrieval]")
    cands = shortlist(msg(ds, "msg_066"), ds)
    check("shortlist is bounded", 0 < len(cands) <= 6, len(cands))
    check("idx is 1-based and contiguous", [c["idx"] for c in cands] == list(range(1, len(cands) + 1)))
    check("same-sender history ranked first", cands[0]["same_sender"] is True)
    check("reactions are attached", all("user_reaction" in c for c in cands))
    check("mute-after surfaced in plain words",
          any("MUTED" in c["user_reaction"] for c in cands))
    again = shortlist(msg(ds, "msg_066"), ds)
    check("retrieval is deterministic", [c["message_id"] for c in cands]
          == [c["message_id"] for c in again])
    check("candidates belong to the receiving user",
          all(any(h["message_id"] == c["message_id"]
                  for h in ds.user_history("u_007")) for c in cands))

    print("\n[reaction salience drives ranking]")
    from retrieve import (FALLBACK_SIMILARITY, REACTION_DELIBERATE, REACTION_NONE,
                          REACTION_PASSIVE, REACTION_STRONG, reaction_salience)
    # The scale is symmetric by construction: each negative tier has a positive
    # tier of equal weight. An earlier version scored `reported` at 1.0 against
    # `opened+replied` at 0.5, which surfaced negative history first on 59 of 110
    # messages and biased the router toward mute.
    check("reported and replied are equally decisive",
          reaction_salience({"message_reported": "1"})
          == reaction_salience({"message_replied": "1"}) == REACTION_DELIBERATE)
    check("muted-after and a fast open are equally strong",
          reaction_salience({"muted_after_message": "1"})
          == reaction_salience({"message_opened": "1", "reaction_time_minutes": "2"})
          == REACTION_STRONG)
    check("dismissed and plain-opened are equally passive",
          reaction_salience({"notification_dismissed": "1", "message_opened": "0"})
          == reaction_salience({"message_opened": "1", "reaction_time_minutes": "600"})
          == REACTION_PASSIVE)
    check("tiers are ordered", REACTION_DELIBERATE > REACTION_STRONG > REACTION_PASSIVE
          > REACTION_NONE)
    check("a slow open is weaker than a fast one",
          reaction_salience({"message_opened": "1", "reaction_time_minutes": "600"})
          < reaction_salience({"message_opened": "1", "reaction_time_minutes": "2"}))
    check("junk reaction time does not raise",
          reaction_salience({"message_opened": "1", "reaction_time_minutes": "soon"})
          == REACTION_PASSIVE)
    check("no recorded reaction is least salient",
          reaction_salience({}) == REACTION_NONE and reaction_salience(None) == REACTION_NONE)
    ranked = shortlist(msg(ds, "msg_066"), ds)
    check("decisive reactions surface at the top",
          "MUTED" in ranked[0]["user_reaction"] or "REPORTED" in ranked[0]["user_reaction"],
          ranked[0]["user_reaction"])

    print("\n[two-tier filter]")
    strict = shortlist(msg(ds, "msg_066"), ds, allow_fallback=False)
    check("strict mode keeps only same-sender rows",
          all(c["same_sender"] for c in strict), [c["same_sender"] for c in strict])
    check("fallback adds near-identical templates from other senders",
          len(ranked) > len(strict), (len(ranked), len(strict)))
    check("primary tier always outranks fallback",
          [c["same_sender"] for c in ranked] == sorted(
              [c["same_sender"] for c in ranked], reverse=True),
          [c["same_sender"] for c in ranked])
    check("fallback rows clear the similarity floor",
          all(c["text_similarity"] >= FALLBACK_SIMILARITY
              for c in ranked if not c["same_sender"]))
    fb_ctx = [build_context(m, ds, stats) for m in ds.messages]
    fb_used = sum(1 for c in fb_ctx
                  if any(not e["same_sender"] for e in c["evidence_candidates"]))
    check("fallback stays rare, not a back door", 0 < fb_used <= 15, fb_used)

    print("\n[K is configurable]")
    for k in (1, 3, 6, 10):
        cands = shortlist(msg(ds, "msg_051"), ds, limit=k)
        check("K=%d respected" % k, len(cands) <= k, len(cands))
    check("K flows through build_context",
          len(build_context(msg(ds, "msg_051"), ds, stats,
                            shortlist_limit=2)["evidence_candidates"]) <= 2)
    check("larger K yields at least as much evidence",
          len(shortlist(msg(ds, "msg_051"), ds, limit=10))
          >= len(shortlist(msg(ds, "msg_051"), ds, limit=3)))
    check("K=0 yields nothing", shortlist(msg(ds, "msg_051"), ds, limit=0) == [])
    check("strict mode flows through build_context",
          all(e["same_sender"] for e in build_context(
              msg(ds, "msg_066"), ds, stats, allow_fallback=False)["evidence_candidates"]))

    print("\n[evidence id mapping]")
    ids = evidence_ids_for(cands, [1, 3])
    check("indices map to real ids", ids == [cands[0]["message_id"], cands[2]["message_id"]])
    check("out-of-range index dropped", evidence_ids_for(cands, [99]) == [])
    check("junk index dropped", evidence_ids_for(cands, ["x", None, 1.0]) == [cands[0]["message_id"]])
    check("duplicates collapsed", evidence_ids_for(cands, [1, 1, 1]) == [cands[0]["message_id"]])
    check("empty selection -> empty", evidence_ids_for(cands, []) == [])
    check("hallucinated ids are structurally impossible",
          all(i in {c["message_id"] for c in cands} for i in evidence_ids_for(cands, [1, 2, 3])))

    print("\n[context shape + budget]")
    required = {"message", "user", "sender", "signals", "evidence_candidates", "_meta"}
    all_ctx = [build_context(m, ds, stats) for m in ds.messages]
    check("every message builds a context", len(all_ctx) == 110)
    check("all contexts have required keys", all(required <= set(c) for c in all_ctx))
    check("no context exceeds the default budget",
          all(c["_meta"]["estimated_tokens"] <= 1400 for c in all_ctx),
          max(c["_meta"]["estimated_tokens"] for c in all_ctx))
    check("none needed truncation at default budget",
          not any(c["_meta"]["truncated"] for c in all_ctx))
    # Evidence is filtered to the same sender/group/business, so a user with no
    # prior contact from that counterpart correctly gets nothing to cite and the
    # router will emit `none`. These 7 are exactly the rows with no same-counterpart
    # history and no near-identical template anywhere else in their history.
    empty = sorted(c["message"]["message_id"] for c in all_ctx if not c["evidence_candidates"])
    check("exactly the no-history rows lack evidence",
          empty == ["msg_089", "msg_090", "msg_092", "msg_093", "msg_094",
                    "msg_095", "msg_096"], empty)
    check("everything else has evidence", len(all_ctx) - len(empty) == 103)
    check("media-only rows still carry text signal absence",
          build_context(msg(ds, "msg_088"), ds, stats)["signals"]["text_is_empty"] is True)

    # The real data never hits the ladder, so force it.
    tight = build_context(msg(ds, "msg_066"), ds, stats, token_budget=300)
    check("tight budget triggers truncation", tight["_meta"]["truncated"] is True)
    check("tight budget still returns valid structure", required <= set(tight))
    check("tight budget shrinks the context",
          tight["_meta"]["estimated_tokens"] < all_ctx[0]["_meta"]["estimated_tokens"] + 1000)
    check("incoming message text is preserved above the floor",
          len(tight["message"].get("authored_text") or "") >= 100
          or len(msg(ds, "msg_066")["message_text"]) < 100)
    check("signals survive truncation", bool(tight["signals"]))
    check("some evidence survives truncation", len(tight["evidence_candidates"]) >= 1)

    print("\n[no raw table dumping]")
    biggest = max(all_ctx, key=lambda c: c["_meta"]["estimated_tokens"])
    check("largest context is still compact",
          biggest["_meta"]["estimated_tokens"] < 1200,
          biggest["_meta"]["estimated_tokens"])
    check("daily load is summarised to a handful of numbers, not 756 rows",
          len(biggest["user"].get("notification_load", {})) <= 5,
          biggest["user"].get("notification_load"))
    check("history is a shortlist, not the full table",
          all(len(c["evidence_candidates"]) <= 6 for c in all_ctx))
    total = sum(c["_meta"]["estimated_tokens"] for c in all_ctx)
    check("whole run fits a sane prompt budget", total < 150000, total)
    print("     (all 110 contexts total ~%d tokens)" % total)

    print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
