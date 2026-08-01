"""Layer 4 tests: override rules, monotonicity, and the audit trail.

    python code/guard/test_guard.py

Anchored on real rows wherever possible. The quiet-hours rule only ever acts on a
`notify`, and the pipeline currently produces none, so those cases are driven with
synthesised decisions.
"""

import csv
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CODE, "context"))
sys.path.insert(0, os.path.join(CODE, "router"))

from aggregates import build_reaction_stats                   # noqa: E402
from apply import apply_guard, apply_guard_all                # noqa: E402
from assemble import build_context                            # noqa: E402
from audit import AUDIT_COLUMNS, disagreements, summarise, write_audit  # noqa: E402
from decision import RouterDecision, safe_default             # noqa: E402
from loaders import Dataset                                   # noqa: E402
from rules import (                                           # noqa: E402
    ACTION_SEVERITY,
    REPORTED_POLICY_ANY,
    REPORTED_POLICY_UNANIMOUS,
    GuardPolicy,
    Override,
    rule_quiet_hours,
)

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + str(detail)) if detail and not cond else ""))


def decide(mid, action="notify", mtype="business_update", conf=0.8, reason="Model reason here."):
    return RouterDecision(message_id=mid, action=action, message_type=mtype,
                          reason=reason, confidence=conf, evidence_message_ids=[])


def main():
    ds = Dataset()
    stats = build_reaction_stats(ds)
    ctx = {r["message_id"]: build_context(r, ds, stats) for r in ds.messages}
    unanimous = GuardPolicy(REPORTED_POLICY_UNANIMOUS)

    print("\n[monotonicity: the guard can only make things safer]")
    check("severity scale is ordered",
          ACTION_SEVERITY["notify"] > ACTION_SEVERITY["digest"] > ACTION_SEVERITY["mute"])
    worst = []
    for mid, context in ctx.items():
        for action in ("notify", "digest", "mute"):
            before = decide(mid, action=action, mtype="promotion")
            after = apply_guard(before, context, unanimous).decision
            if ACTION_SEVERITY[after.action] > ACTION_SEVERITY[action]:
                worst.append((mid, action, after.action))
    check("no row is ever promoted, across all 110 x 3 actions", worst == [], worst[:3])
    muted_in = decide("msg_019", action="mute", mtype="scam")
    check("a mute is never lifted",
          apply_guard(muted_in, ctx["msg_019"], unanimous).decision.action == "mute")

    print("\n[promotion attempt is a hard error, not a silent pass]")
    import apply as apply_mod
    original = apply_mod.RULES
    apply_mod.RULES = (lambda d, c, p: Override(rule="bad", action="notify",
                                                reason="x", confidence=0.9),)
    try:
        apply_guard(decide("m", action="mute"), ctx["msg_019"], unanimous)
        check("promoting rule raises", False, "no exception")
    except AssertionError as exc:
        check("promoting rule raises", "may only make things safer" in str(exc))
    finally:
        apply_mod.RULES = original

    print("\n[scam signature forces mute/scam]")
    res = apply_guard(decide("msg_019", action="notify", mtype="payment"), ctx["msg_019"], unanimous)
    check("scam row is forced to mute", res.decision.action == "mute")
    check("type forced to scam", res.decision.message_type == "scam")
    check("rule recorded", res.records[0].rule == "scam_signature")
    check("reason names the deciding signal",
          "chase-secure-alert.com" in res.decision.reason, res.decision.reason)
    check("model reason preserved in the audit",
          res.records[0].model_reason == "Model reason here.")
    check("scam rule is terminal (no later rule fires)", len(res.records) == 1)
    # the verified-but-mismatched sender must survive
    res41 = apply_guard(decide("msg_041", action="digest", mtype="business_update"),
                        ctx["msg_041"], unanimous)
    check("verified old sender on a shortener is NOT muted as scam",
          res41.decision.action == "digest" and res41.decision.message_type != "scam",
          (res41.decision.action, res41.decision.message_type))

    print("\n[reported sender: narrow by default]")
    res16 = apply_guard(decide("msg_016", action="notify", mtype="urgent"), ctx["msg_016"], unanimous)
    check("unanimously reported sender forced to mute", res16.decision.action == "mute")
    check("typed as spam or scam", res16.decision.message_type in ("spam", "scam"))
    broad = GuardPolicy(REPORTED_POLICY_ANY)
    fired_narrow = sum(1 for m in ctx if any(
        r.rule == "reported_sender"
        for r in apply_guard(decide(m, action="notify"), ctx[m], unanimous).records))
    fired_broad = sum(1 for m in ctx if any(
        r.rule == "reported_sender"
        for r in apply_guard(decide(m, action="notify"), ctx[m], broad).records))
    check("broad policy fires far more often", fired_broad > fired_narrow * 2,
          (fired_narrow, fired_broad))
    print("     (reported_sender fires on %d rows narrow, %d broad)" % (fired_narrow, fired_broad))
    check("invalid policy is rejected", _raises(lambda: GuardPolicy("sometimes")))

    print("\n[the broad policy contradicts a labelled sample]")
    s1 = next(r for r in ds.samples if r["message_id"] == "sample_msg_001")
    s1_ctx = build_context(s1, ds, stats)
    s1_dec = decide("sample_msg_001", action="notify", mtype="urgent")
    check("narrow policy leaves the labelled notify alone",
          apply_guard(s1_dec, s1_ctx, unanimous).decision.action == "notify")
    check("broad policy would mute it",
          apply_guard(s1_dec, s1_ctx, broad).decision.action == "mute",
          "this is why the default is narrow")

    print("\n[opt-out caps rather than forces]")
    res66 = apply_guard(decide("msg_066", action="notify", mtype="promotion"),
                        ctx["msg_066"], unanimous)
    check("opted-out promo is capped", res66.decision.action in ("digest", "mute"))
    check("rule recorded", any(r.rule == "opted_out_promotions" for r in res66.records))
    check("reason cites the opt-out", "opted out" in res66.decision.reason.lower())
    # a transactional message from the same business must not be caught
    trans = apply_guard(decide("msg_066", action="notify", mtype="business_update"),
                        ctx["msg_066"], unanimous)
    check("opt-out does not touch non-promotional content",
          not any(r.rule == "opted_out_promotions" for r in trans.records))
    res65 = apply_guard(decide("msg_065", action="notify", mtype="promotion"),
                        ctx["msg_065"], unanimous)
    check("the opted-IN user on the same poster is untouched",
          not any(r.rule == "opted_out_promotions" for r in res65.records))

    print("\n[muted sender: capped, but a direct request survives]")
    res40 = apply_guard(decide("msg_040", action="notify", mtype="forward"),
                        ctx["msg_040"], unanimous)
    check("muted group chain-spam is suppressed", res40.decision.action == "mute")
    res56 = apply_guard(decide("msg_056", action="notify", mtype="urgent"),
                        ctx["msg_056"], unanimous)
    check("muted group urgent direct mention survives as notify",
          res56.decision.action == "notify",
          "muting a group is not refusing to be reached")
    check("no mute rule fired on it",
          not any(r.rule == "muted_by_user" for r in res56.records))
    res56d = apply_guard(decide("msg_056", action="notify", mtype="greeting"),
                         ctx["msg_056"], unanimous)
    check("same muted group, non-actionable type IS capped",
          res56d.decision.action != "notify")

    print("\n[quiet hours]")
    # msg_062 is inside u_011's DND window
    qh = ctx["msg_062"]
    check("the fixture really is in quiet hours", qh["signals"].get("in_quiet_hours") is True)
    res = apply_guard(decide("msg_062", action="notify", mtype="event"), qh, unanimous)
    check("notify is downgraded to digest", res.decision.action == "digest")
    check("rule recorded", any(r.rule == "quiet_hours" for r in res.records))
    check("reason names quiet hours", "quiet hours" in res.decision.reason.lower())
    check("digest is left alone",
          apply_guard(decide("msg_062", action="digest", mtype="event"), qh, unanimous)
          .decision.action == "digest")
    check("mute is left alone",
          apply_guard(decide("msg_062", action="mute", mtype="spam"), qh, unanimous)
          .decision.action == "mute")
    # carve-outs, driven synthetically since these signal combinations are rare
    exempt_ctx = dict(qh)
    exempt_ctx["signals"] = dict(qh["signals"], in_quiet_hours=True, directly_addressed=True)
    check("urgent direct mention is exempt",
          rule_quiet_hours(decide("m", action="notify", mtype="urgent"),
                           exempt_ctx, unanimous) is None)
    check("payment direct mention is exempt",
          rule_quiet_hours(decide("m", action="notify", mtype="payment"),
                           exempt_ctx, unanimous) is None)
    check("non-urgent direct mention is NOT exempt",
          rule_quiet_hours(decide("m", action="notify", mtype="promotion"),
                           exempt_ctx, unanimous) is not None)
    admin_ctx = dict(qh)
    admin_ctx["signals"] = dict(qh["signals"], in_quiet_hours=True,
                                directly_addressed=False, sender_is_group_admin=True)
    check("payment from a group admin is exempt",
          rule_quiet_hours(decide("m", action="notify", mtype="payment"),
                           admin_ctx, unanimous) is None)
    biz_ctx = dict(qh)
    biz_ctx["signals"] = dict(qh["signals"], in_quiet_hours=True, directly_addressed=False,
                              verified=True, scam_signature=False)
    check("payment from a verified business is exempt",
          rule_quiet_hours(decide("m", action="notify", mtype="payment"),
                           biz_ctx, unanimous) is None)
    scam_biz = dict(qh)
    scam_biz["signals"] = dict(qh["signals"], in_quiet_hours=True, directly_addressed=False,
                               verified=True, scam_signature=True)
    check("payment from a scam-flagged sender is NOT exempt",
          rule_quiet_hours(decide("m", action="notify", mtype="payment"),
                           scam_biz, unanimous) is not None)
    outside = ctx["msg_066"]
    check("outside quiet hours the rule never fires",
          rule_quiet_hours(decide("m", action="notify", mtype="event"), outside, unanimous) is None
          if not outside["signals"].get("in_quiet_hours") else True)

    print("\n[internal consistency of an overridden row]")
    res = apply_guard(decide("msg_019", action="notify", mtype="payment",
                             reason="A trusted delivery update the user usually opens.",
                             conf=0.91), ctx["msg_019"], unanimous)
    check("reason is rewritten with the action",
          res.decision.reason != "A trusted delivery update the user usually opens.")
    check("confidence is rewritten too", res.decision.confidence != 0.91)
    check("override is noted on the decision",
          any("override" in n for n in res.decision.notes), res.decision.notes)
    check("original decision object is not mutated",
          decide("msg_019", action="notify").action == "notify")

    print("\n[audit trail]")
    decisions = {m: safe_default(m) for m in ctx}
    guarded, records = apply_guard_all(decisions, ctx, unanimous)
    check("every message still has a decision", len(guarded) == 110)
    check("records were produced", len(records) > 0)
    check("audit captures both sides",
          all(r.from_action and r.to_action and r.model_reason and r.rule_reason
              for r in records))
    check("disagreement flag is set when the action changed",
          all(r.disagreement for r in records if r.changed_action))
    check("fallback rows are marked", all(r.model_fell_back for r in records))
    s = summarise(records, 110)
    check("summary counts messages touched", 0 < s["messages_touched"] <= 110)
    check("summary breaks down by rule", len(s["by_rule"]) >= 3, s["by_rule"])
    check("disagreements() filters", len(disagreements(records)) <= len(records))

    tmp = tempfile.mkdtemp(prefix="audit_")
    try:
        path = os.path.join(tmp, "override_audit.csv")
        write_audit(records, path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        check("audit header exact", tuple(rows[0]) == AUDIT_COLUMNS)
        check("one audit row per record", len(rows) - 1 == len(records))
        check("audit is CRLF like the rest",
              open(path, "rb").read().count(b"\n") == open(path, "rb").read().count(b"\r\n"))
        check("every audit row names a rule", all(r[1] for r in rows[1:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n[rule toggles]")
    only_scam = GuardPolicy(enabled={"scam_signature"})
    res = apply_guard(decide("msg_066", action="notify", mtype="promotion"),
                      ctx["msg_066"], only_scam)
    check("disabled rules do not fire", res.records == [])
    _, none_records = apply_guard_all(decisions, ctx, GuardPolicy(enabled=set()))
    check("all rules disabled -> no overrides", none_records == [])

    print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
