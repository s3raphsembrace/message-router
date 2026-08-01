"""Eval harness tests.

    python code/evaluation/test_eval.py

Metrics are tested against hand-built cases rather than only against whatever the
pipeline currently produces -- a scorer that is only ever exercised on one set of
predictions can be confidently wrong.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
for _sub in ("context", "router", "guard"):
    sys.path.insert(0, os.path.join(CODE, _sub))

from aggregates import build_reaction_stats                   # noqa: E402
from loaders import Dataset                                   # noqa: E402
from main import LABEL_COLUMNS, leak_check, strip_labels      # noqa: E402
from main import main as eval_main                            # noqa: E402
from metrics import (                                         # noqa: E402
    ACTION_COST,
    accuracy,
    action_cost,
    calibration,
    confusion,
    evidence_scores,
    joint_accuracy,
    parse_evidence,
    per_class_report,
    severe_errors,
)

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + str(detail)) if detail and not cond else ""))


def main():
    print("\n[accuracy]")
    check("perfect", accuracy(["a", "b"], ["a", "b"]) == 1.0)
    check("none right", accuracy(["a", "b"], ["b", "a"]) == 0.0)
    check("half", accuracy(["a", "b"], ["a", "x"]) == 0.5)
    check("empty is 0, not a crash", accuracy([], []) == 0.0)
    check("joint needs both fields",
          joint_accuracy(["notify"], ["notify"], ["urgent"], ["event"]) == 0.0)
    check("joint counts full matches",
          joint_accuracy(["notify"], ["notify"], ["urgent"], ["urgent"]) == 1.0)

    print("\n[confusion matrix]")
    m, unknown = confusion(["notify", "notify", "mute"], ["notify", "mute", "mute"])
    check("diagonal counted", m["notify"]["notify"] == 1 and m["mute"]["mute"] == 1)
    check("off-diagonal counted", m["notify"]["mute"] == 1)
    check("no false entries", m["digest"]["digest"] == 0)
    _, unknown = confusion(["notify"], ["escalate"])
    check("out-of-vocab counted separately", unknown == 1)

    print("\n[asymmetric errors]")
    gold_a = ["notify", "mute", "mute", "digest", "notify"]
    pred_a = ["mute", "notify", "notify", "digest", "digest"]
    gold_t = ["urgent", "scam", "promotion", "event", "personal"]
    sev = severe_errors(gold_a, pred_a, gold_t)
    check("suppression detected", len(sev["suppressed_notify"]) == 1)
    check("urgent suppression flagged", len(sev["suppressed_notify_urgent"]) == 1)
    check("intrusion detected", len(sev["intruding_mute"]) == 2)
    check("risky intrusion isolated", len(sev["intruding_mute_risky"]) == 1)
    check("notify->digest is not severe",
          not any(i == 4 for i, _ in sev["suppressed_notify"]))

    print("\n[cost is asymmetric]")
    check("muting a notify costs more than digesting it",
          ACTION_COST["notify"]["mute"] > ACTION_COST["notify"]["digest"])
    check("notifying a mute costs more than digesting it",
          ACTION_COST["mute"]["notify"] > ACTION_COST["mute"]["digest"])
    check("diagonal is free", all(ACTION_COST[a][a] == 0.0 for a in ACTION_COST))
    perfect = action_cost(["notify", "mute"], ["notify", "mute"], ["urgent", "scam"])
    check("perfect prediction costs nothing", perfect["total"] == 0.0)
    scam_intrusion = action_cost(["mute"], ["notify"], ["scam"])
    promo_intrusion = action_cost(["mute"], ["notify"], ["promotion"])
    check("notifying a scam costs more than notifying a promo",
          scam_intrusion["total"] > promo_intrusion["total"],
          (scam_intrusion["total"], promo_intrusion["total"]))
    near = action_cost(["notify"], ["digest"], ["urgent"])
    far = action_cost(["notify"], ["mute"], ["urgent"])
    check("adjacent mistakes cost less than inversions", near["total"] < far["total"])
    check("per_row is normalised", action_cost(gold_a, pred_a, gold_t)["per_row"] ==
          action_cost(gold_a, pred_a, gold_t)["total"] / 5.0)

    print("\n[evidence parsing]")
    check("'none' is empty", parse_evidence("none") == set())
    check("'NONE' is empty", parse_evidence("NONE") == set())
    check("blank is empty", parse_evidence("") == set() and parse_evidence(None) == set())
    check("semicolons split", parse_evidence("a;b") == {"a", "b"})
    check("whitespace tolerated", parse_evidence(" a ; b ") == {"a", "b"})
    check("trailing separator ignored", parse_evidence("a;") == {"a"})

    print("\n[evidence scoring]")
    ev = evidence_scores(["m1", "m2;m3", "none"], ["m1", "m2;m3", "none"])
    check("perfect exact match", ev["exact_match"] == 1.0)
    check("perfect precision", ev["precision"] == 1.0)
    check("perfect recall", ev["recall"] == 1.0)
    check("none rows counted", ev["gold_none"] == 1 and ev["both_none"] == 1)
    ev = evidence_scores(["m1;m2"], ["m1;m9"])
    check("partial: tp counted", ev["true_positives"] == 1)
    check("partial: fp counted", ev["false_positives"] == 1)
    check("partial: fn counted", ev["false_negatives"] == 1)
    check("partial: precision 0.5", ev["precision"] == 0.5)
    check("partial: not an exact match", ev["exact_match"] == 0.0)
    ev = evidence_scores(["m1"], ["none"])
    check("citing nothing when something was expected -> recall 0", ev["recall"] == 0.0)
    check("precision undefined when nothing was cited", ev["precision"] is None)
    ev = evidence_scores(["none"], ["m1"])
    check("padding when none expected -> fp", ev["false_positives"] == 1)
    check("recall undefined when nothing was expected", ev["recall"] is None)
    ev = evidence_scores(["none", "none"], ["none", "m1"])
    check("none agreement is a rate", abs(ev["none_agreement"] - 0.5) < 1e-9)

    print("\n[calibration]")
    cal = calibration([0.95] * 10, [True] * 10)
    check("perfectly confident and perfectly right -> small ECE", cal["ece"] < 0.06, cal["ece"])
    cal = calibration([0.95] * 10, [False] * 10)
    check("confident and wrong -> large ECE", cal["ece"] > 0.9, cal["ece"])
    check("overconfidence flagged", cal["overconfident_bins"] >= 1)
    cal = calibration([0.55] * 10, [True] * 10)
    check("underconfidence gives a negative gap",
          [b for b in cal["bins"] if b["n"]][0]["gap"] < 0)
    cal = calibration([0.85] * 4 + [0.55] * 6, [True] * 4 + [True, True, True, False, False, False])
    rows = [b for b in cal["bins"] if b["n"]]
    check("bins are separated", len(rows) == 2, len(rows))
    check("per-bin accuracy computed",
          any(abs(r["accuracy"] - 0.5) < 1e-9 for r in rows))
    check("ECE is sample-weighted", 0.0 <= cal["ece"] <= 1.0)
    check("empty bins are reported but skipped",
          any(b["n"] == 0 for b in calibration([0.95], [True])["bins"]))

    print("\n[per-class report]")
    rep = {r["label"]: r for r in per_class_report(
        ["scam", "scam", "promotion"], ["scam", "promotion", "promotion"])}
    check("support counted", rep["scam"]["support"] == 2)
    check("recall computed", abs(rep["scam"]["recall"] - 0.5) < 1e-9)
    check("precision computed", abs(rep["promotion"]["precision"] - 0.5) < 1e-9)
    check("unpredicted class has None precision",
          per_class_report(["a"], ["b"])[0]["precision"] is None)

    print("\n[label isolation]")
    ds = Dataset()
    stats = build_reaction_stats(ds)
    sample = ds.samples[0]
    check("sample rows really do carry labels",
          all(c in sample for c in LABEL_COLUMNS))
    stripped = strip_labels(sample)
    check("stripping removes every label column",
          not (set(stripped) & set(LABEL_COLUMNS)), sorted(set(stripped) & set(LABEL_COLUMNS)))
    check("stripping keeps the input fields",
          {"message_id", "user_id", "message_text", "created_at"} <= set(stripped))
    problems = leak_check(ds, ds.samples, stats)
    check("no label reaches any prompt, across all 30 rows", problems == [], problems[:3])

    print("\n[harness end to end]")
    check("scoring run exits 0", eval_main([]) == 0)
    check("leak-check run exits 0", eval_main(["--leak-check"]) == 0)
    check("no-guard run exits 0", eval_main(["--no-guard"]) == 0)

    print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
