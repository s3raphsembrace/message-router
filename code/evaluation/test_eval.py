"""Eval harness tests.

    python code/evaluation/test_eval.py

Metrics are tested against hand-built cases rather than only against whatever the
pipeline currently produces -- a scorer that is only ever exercised on one set of
predictions can be confidently wrong.
"""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
for _sub in ("context", "router", "guard"):
    sys.path.insert(0, os.path.join(CODE, _sub))

from aggregates import build_reaction_stats                   # noqa: E402
from loaders import Dataset                                   # noqa: E402
import ledger                                                 # noqa: E402
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

    print("\n[ledger: deltas and direction]")
    raw, verdict = ledger.delta("action_acc", 0.50, 0.60)
    check("accuracy up is better", verdict == "better" and abs(raw - 0.10) < 1e-9)
    check("accuracy down is worse", ledger.delta("action_acc", 0.60, 0.50)[1] == "worse")
    check("ECE down is better (lower is better)",
          ledger.delta("ece", 0.20, 0.10)[1] == "better")
    check("ECE up is worse", ledger.delta("ece", 0.10, 0.20)[1] == "worse")
    check("cost down is better", ledger.delta("cost_per_row", 0.6, 0.4)[1] == "better")
    check("severe errors up is worse",
          ledger.delta("severe_suppressed", 0, 2)[1] == "worse")
    check("first run has no baseline", ledger.delta("action_acc", None, 0.5)[1] == "new")

    print("\n[ledger: n=30 noise floor]")
    check("floor is one row of thirty", abs(ledger.ACCURACY_FLOOR - 1.0 / 30) < 1e-9)
    check("sub-one-row change is noise",
          ledger.delta("action_acc", 0.500, 0.520)[1] == "noise",
          "2pp is less than one row")
    check("one full row is not noise",
          ledger.delta("action_acc", 0.500, 0.534)[1] == "better")
    check("tiny ECE change is noise", ledger.delta("ece", 0.100, 0.105)[1] == "noise")
    check("meaningful ECE change is not", ledger.delta("ece", 0.100, 0.140)[1] == "worse")

    print("\n[ledger: trade-off detection]")
    warn = ledger.tradeoffs(ledger.compare(
        {"action_acc": 0.50, "ece": 0.05, "evidence_f1": 0.8, "cost_per_row": 0.4,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4},
        {"action_acc": 0.70, "ece": 0.15, "evidence_f1": 0.8, "cost_per_row": 0.4,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4}))
    check("accuracy up + calibration worse is flagged",
          any("calibration got WORSE" in w for w in warn), warn)
    warn = ledger.tradeoffs(ledger.compare(
        {"action_acc": 0.50, "ece": 0.05, "evidence_f1": 0.80, "cost_per_row": 0.4,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4},
        {"action_acc": 0.70, "ece": 0.05, "evidence_f1": 0.50, "cost_per_row": 0.4,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4}))
    check("accuracy up + evidence worse is flagged",
          any("evidence F1 got WORSE" in w for w in warn), warn)
    warn = ledger.tradeoffs(ledger.compare(
        {"action_acc": 0.50, "ece": 0.05, "evidence_f1": 0.8, "cost_per_row": 0.4,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4},
        {"action_acc": 0.70, "ece": 0.05, "evidence_f1": 0.8, "cost_per_row": 0.4,
         "severe_suppressed": 3, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4}))
    check("accuracy up + severe errors up is flagged",
          any("severe errors increased" in w for w in warn), warn)
    warn = ledger.tradeoffs(ledger.compare(
        {"action_acc": 0.50, "ece": 0.05, "evidence_f1": 0.8, "cost_per_row": 0.4,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4},
        {"action_acc": 0.51, "ece": 0.05, "evidence_f1": 0.8, "cost_per_row": 0.7,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.5, "joint_acc": 0.4}))
    check("cost worse without accuracy dropping is flagged",
          any("expensive kind" in w for w in warn), warn)
    clean = ledger.tradeoffs(ledger.compare(
        {"action_acc": 0.50, "ece": 0.10, "evidence_f1": 0.5, "cost_per_row": 0.6,
         "severe_suppressed": 2, "severe_intruded": 1, "type_acc": 0.5, "joint_acc": 0.4},
        {"action_acc": 0.70, "ece": 0.05, "evidence_f1": 0.8, "cost_per_row": 0.3,
         "severe_suppressed": 0, "severe_intruded": 0, "type_acc": 0.6, "joint_acc": 0.5}))
    check("an across-the-board win raises no warning", clean == [], clean)

    print("\n[ledger: persistence]")
    tmpdir = tempfile.mkdtemp(prefix="ledger_")
    try:
        path = os.path.join(tmpdir, "runs.csv")
        base = {"timestamp": "t", "git_sha": "abc1234", "change": "first", "config": "c",
                "action_acc": 0.5, "type_acc": 0.5, "joint_acc": 0.4, "evidence_f1": 0.6,
                "evidence_precision": 0.6, "evidence_recall": 0.6, "evidence_exact": 0.5,
                "ece": 0.1, "cost_per_row": 0.5, "severe_suppressed": 0,
                "severe_intruded": 0, "n_rows": 30, "notes": ""}
        check("first append is run 1", ledger.append_run(path, base) == 1)
        second = dict(base, change="second", action_acc=0.7)
        check("second append is run 2", ledger.append_run(path, second) == 2)
        runs = ledger.load_runs(path)
        check("both runs persisted", len(runs) == 2)
        check("run numbers are sequential", [r["run"] for r in runs] == [1, 2])
        check("floats are parsed back", isinstance(runs[0]["action_acc"], float))
        check("history never rewrites an earlier run",
              runs[0]["change"] == "first" and runs[0]["action_acc"] == 0.5)
        check("a regression stays in the history",
              ledger.append_run(path, dict(base, change="regression", action_acc=0.2)) == 3
              and ledger.load_runs(path)[2]["action_acc"] == 0.2)
        table = ledger.history_table(ledger.load_runs(path))
        check("table renders every run", table.count("\n") > 3)
        check("worse deltas are marked", "!" in table)
        check("better deltas are marked", "*" in table)
        # scoped to the delta rows: the header underline is a long run of dashes
        delta_rows = [ln for ln in table.splitlines() if "vs prev" in ln]
        check("delta rows exist", len(delta_rows) >= 2, len(delta_rows))
        check("delta sign is not doubled",
              all("--" not in ln and "++" not in ln for ln in delta_rows), delta_rows)
        check("noise floor is documented in the legend", "1 of 30 rows" in table)
        check("empty ledger degrades gracefully",
              ledger.history_table([]) == "no runs recorded yet")
        check("missing ledger file returns no runs",
              ledger.load_runs(os.path.join(tmpdir, "nope.csv")) == [])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n[golden set]")
    import golden as golden_mod
    import run_golden
    ids = [c.message_id for c in golden_mod.GOLDEN]
    check("golden set is small", 4 <= len(golden_mod.GOLDEN) <= 10, len(golden_mod.GOLDEN))
    check("no duplicate rows", len(set(ids)) == len(ids))
    by_id = run_golden.resolve_rows(ds)
    check("every golden row resolves", all(i in by_id for i in ids),
          [i for i in ids if i not in by_id])
    check("golden rows carry no labels",
          all(not (set(by_id[i]) & set(LABEL_COLUMNS)) for i in ids))
    check("every case states a constraint",
          all(c.expect_action or c.forbid_action or c.expect_type or c.forbid_type
              for c in golden_mod.GOLDEN))
    check("every case explains itself", all(len(c.why) > 40 for c in golden_mod.GOLDEN))
    check("asserted-only cases are flagged",
          all(c.ground_truth or c.source == "messages" for c in golden_mod.GOLDEN))
    covered = {c.case for c in golden_mod.GOLDEN}
    for required in ("scam-as-payment", "useful-poster-for-this-user",
                     "muted-group-urgent-mention", "voice-note-only",
                     "high-dismissal-repeat-sender"):
        check("covers %s" % required, required in covered)
    check("the identical-content pair is both present",
          {"useful-poster-for-this-user", "same-poster-hostile-rapport"} <= covered)
    pair = {c.case: c.message_id for c in golden_mod.GOLDEN}
    a = next(s for s in ds.samples if s["message_id"] == pair["useful-poster-for-this-user"])
    b = next(s for s in ds.samples if s["message_id"] == pair["same-poster-hostile-rapport"])
    check("the pair really is identical content",
          a["message_text"] == b["message_text"] and a["media_id"] == b["media_id"])
    check("the pair really is labelled differently", a["action"] != b["action"])

    ok = golden_mod.evaluate(golden_mod.GOLDEN[0], "mute", "scam")
    check("a satisfying row passes", ok == [])
    bad = golden_mod.evaluate(golden_mod.GOLDEN[0], "notify", "urgent")
    check("a violating row fails", len(bad) == 2, bad)
    forbid = next(c for c in golden_mod.GOLDEN if c.forbid_action)
    check("forbidden action is caught",
          golden_mod.evaluate(forbid, sorted(forbid.forbid_action)[0], "promotion"))
    check("golden runner exits 0 while the model is unwired", run_golden.main([]) == 0)

    print("\n[harness end to end]")
    check("scoring run exits 0", eval_main([]) == 0)
    check("leak-check run exits 0", eval_main(["--leak-check"]) == 0)
    check("no-guard run exits 0", eval_main(["--no-guard"]) == 0)
    check("history run exits 0", eval_main(["--history"]) == 0)
    tmpdir = tempfile.mkdtemp(prefix="ledger_e2e_")
    try:
        path = os.path.join(tmpdir, "runs.csv")
        check("recording run exits 0",
              eval_main(["--ledger", path, "--record", "test run"]) == 0)
        check("recorded run landed in the ledger", len(ledger.load_runs(path)) == 1)
        check("second recording computes a delta",
              eval_main(["--ledger", path, "--record", "test run 2", "--no-guard"]) == 0
              and len(ledger.load_runs(path)) == 2)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n%d passed, %d failed" % (len(PASSED), len(FAILED)))
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
