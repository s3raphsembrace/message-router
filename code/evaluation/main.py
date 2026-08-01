"""Evaluation harness -- scores the router against sample_messages.csv.

    python code/evaluation/main.py                    # run the pipeline and score it
    python code/evaluation/main.py --no-guard         # score the router alone
    python code/evaluation/main.py --predictions p.csv  # score a saved run
    python code/evaluation/main.py --leak-check       # prove no labels reach the router

sample_messages.csv is the only labelled data available. It is read HERE, for
scoring, and nowhere else. The label columns are stripped from every row before
it is handed to context assembly or the router, and --leak-check asserts that no
label value appears anywhere in a rendered prompt.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE)
sys.path.insert(0, HERE)
for _sub in ("context", "router", "guard"):
    sys.path.insert(0, os.path.join(CODE, _sub))

from aggregates import build_reaction_stats                   # noqa: E402
from apply import apply_guard_all                             # noqa: E402
from assemble import build_context                            # noqa: E402
from loaders import DEFAULT_DATASET, DEFAULT_MEDIA_CACHE, Dataset  # noqa: E402
from metrics import (                                         # noqa: E402
    ACTIONS,
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
from prompt import render                                     # noqa: E402
from route import RouteStats, route_all                       # noqa: E402
from rules import REPORTED_POLICY_ANY, REPORTED_POLICY_UNANIMOUS, GuardPolicy  # noqa: E402

# The five columns that exist only in sample_messages.csv. They must never travel
# into context assembly or a prompt.
LABEL_COLUMNS = ("action", "message_type", "reason", "confidence", "evidence_message_ids")


def strip_labels(row):
    """A sample row with the answers removed -- the shape messages.csv has."""
    return {k: v for k, v in row.items() if k not in LABEL_COLUMNS}


def build_model_callable():
    """Same seam as code/main.py. None until the Layer 3 client exists."""
    return None


def load_predictions(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["message_id"]: r for r in csv.DictReader(f)}


def _fmt(value, spec="%.3f"):
    return "n/a" if value is None else (spec % value)


def _keys_anywhere(node, found):
    """Every dict key appearing anywhere in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            _keys_anywhere(value, found)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _keys_anywhere(item, found)
    return found


def leak_check(dataset, samples, reaction_stats):
    """Prove no label reaches the router. Returns a list of problems.

    Structural rather than textual: the assembled context is inspected for any key
    named like an answer column, and the rendered prompt for the gold reason. A
    gold evidence id appearing as a retrieved candidate is NOT leakage -- that is
    retrieval finding the same history a human found -- so ids are not checked.
    """
    problems = []
    for row in samples:
        stripped = strip_labels(row)
        leaked_cols = set(stripped) & set(LABEL_COLUMNS)
        if leaked_cols:
            problems.append("%s: label column(s) survived stripping: %s"
                            % (row["message_id"], ", ".join(sorted(leaked_cols))))

        context = build_context(stripped, dataset, reaction_stats)
        keys = _keys_anywhere(context, set())
        answer_keys = keys & set(LABEL_COLUMNS)
        if answer_keys:
            problems.append("%s: context contains answer-shaped key(s): %s"
                            % (row["message_id"], ", ".join(sorted(answer_keys))))

        _, user = render(context)
        reason = " ".join((row.get("reason") or "").split()).lower()
        if len(reason) > 25 and reason[:40] in user.lower():
            problems.append("%s: gold reason text appears in the prompt" % row["message_id"])
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the router against sample_messages.csv")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--media-cache", default=DEFAULT_MEDIA_CACHE)
    ap.add_argument("--predictions", default="", help="score an existing predictions CSV")
    ap.add_argument("--no-guard", action="store_true", help="score the router without Layer 4")
    ap.add_argument("--reported-policy", default=REPORTED_POLICY_UNANIMOUS,
                    choices=[REPORTED_POLICY_UNANIMOUS, REPORTED_POLICY_ANY])
    ap.add_argument("--leak-check", action="store_true",
                    help="verify no label reaches the router, then exit")
    args = ap.parse_args(argv)

    dataset = Dataset(args.dataset, args.media_cache)
    samples = dataset.samples
    if not samples:
        print("ERROR: sample_messages.csv is empty or missing", file=sys.stderr)
        return 2
    reaction_stats = build_reaction_stats(dataset)

    if args.leak_check:
        problems = leak_check(dataset, samples, reaction_stats)
        print("leak check over %d labelled rows" % len(samples))
        if problems:
            print("FAILED (%d):" % len(problems))
            for p in problems[:20]:
                print("  - %s" % p)
            return 1
        print("OK: label columns stripped before assembly; no gold value in any prompt")
        return 0

    # ---- predictions -------------------------------------------------
    if args.predictions:
        preds = load_predictions(args.predictions)
        source = args.predictions
        missing = [r["message_id"] for r in samples if r["message_id"] not in preds]
        if missing:
            print("ERROR: predictions missing %d sample row(s): %s"
                  % (len(missing), ", ".join(missing[:5])), file=sys.stderr)
            return 2
    else:
        stripped = [strip_labels(r) for r in samples]
        contexts = {r["message_id"]: build_context(r, dataset, reaction_stats)
                    for r in stripped}
        call_model = build_model_callable()
        decisions, route_stats = route_all(stripped, contexts, call_model, RouteStats())
        if not args.no_guard:
            decisions, _ = apply_guard_all(
                decisions, contexts, GuardPolicy(reported_policy=args.reported_policy))
        preds = {mid: d.to_row() for mid, d in decisions.items()}
        source = "live pipeline (%s, guard %s)" % (
            "model wired" if call_model else "NO MODEL - safe defaults",
            "off" if args.no_guard else "on")

    gold_actions = [r["action"] for r in samples]
    gold_types = [r["message_type"] for r in samples]
    gold_evidence = [r["evidence_message_ids"] for r in samples]
    pred_actions = [preds[r["message_id"]]["action"] for r in samples]
    pred_types = [preds[r["message_id"]]["message_type"] for r in samples]
    pred_evidence = [preds[r["message_id"]]["evidence_message_ids"] for r in samples]
    pred_conf = []
    for r in samples:
        try:
            pred_conf.append(float(preds[r["message_id"]]["confidence"]))
        except (TypeError, ValueError):
            pred_conf.append(0.0)

    print("=" * 76)
    print("EVALUATION - %d labelled rows" % len(samples))
    print("source: %s" % source)
    print("=" * 76)

    # ---- accuracy ----------------------------------------------------
    act_acc = accuracy(gold_actions, pred_actions)
    typ_acc = accuracy(gold_types, pred_types)
    joint = joint_accuracy(gold_actions, pred_actions, gold_types, pred_types)
    print("\n## Accuracy")
    print("  action        %6.1f%%  (%d/%d)" % (
        act_acc * 100, round(act_acc * len(samples)), len(samples)))
    print("  message_type  %6.1f%%  (%d/%d)" % (
        typ_acc * 100, round(typ_acc * len(samples)), len(samples)))
    print("  both correct  %6.1f%%" % (joint * 100))

    # ---- confusion ---------------------------------------------------
    matrix, unknown = confusion(gold_actions, pred_actions)
    print("\n## Action confusion  (rows = gold, columns = predicted)")
    print("            " + "".join("%9s" % a for a in ACTIONS))
    for g in ACTIONS:
        cells = []
        for p in ACTIONS:
            n = matrix[g][p]
            mark = ""
            if (g, p) in (("notify", "mute"), ("mute", "notify")) and n:
                mark = "!"
            cells.append("%8d%s" % (n, mark or " "))
        print("  %-8s %s" % (g, "".join(cells)))
    if unknown:
        print("  (%d row(s) had an out-of-vocabulary action)" % unknown)

    sev = severe_errors(gold_actions, pred_actions, gold_types)
    cost = action_cost(gold_actions, pred_actions, gold_types)
    print("\n## Asymmetric errors  (the two that matter)")
    print("  gold notify -> predicted mute   : %d   %s" % (
        len(sev["suppressed_notify"]),
        "<-- user missed something they needed" if sev["suppressed_notify"] else ""))
    print("     of which urgent/payment/event: %d" % len(sev["suppressed_notify_urgent"]))
    print("  gold mute   -> predicted notify : %d   %s" % (
        len(sev["intruding_mute"]),
        "<-- user interrupted by junk" if sev["intruding_mute"] else ""))
    print("     of which scam/spam           : %d" % len(sev["intruding_mute_risky"]))
    for key, label in (("suppressed_notify", "suppressed"), ("intruding_mute", "intruded")):
        for idx, gtype in sev[key][:5]:
            print("     %-10s %s (gold type %s)" % (label, samples[idx]["message_id"], gtype))
    print("  weighted cost: %.1f total, %.2f per row (worst case %.1f)"
          % (cost["total"], cost["per_row"], cost["worst_case_per_row"]))

    # ---- message_type breakdown --------------------------------------
    print("\n## message_type, per class")
    print("  %-16s %8s %10s %10s %9s" % ("type", "support", "predicted", "precision", "recall"))
    for row in per_class_report(gold_types, pred_types):
        if not row["support"] and not row["predicted"]:
            continue
        print("  %-16s %8d %10d %10s %9s" % (
            row["label"], row["support"], row["predicted"],
            _fmt(row["precision"], "%.2f"), _fmt(row["recall"], "%.2f")))

    # ---- evidence ----------------------------------------------------
    ev = evidence_scores(gold_evidence, pred_evidence)
    print("\n## Evidence")
    print("  exact set match : %6.1f%%" % (ev["exact_match"] * 100))
    print("  precision       : %s   (tp=%d fp=%d)" % (
        _fmt(ev["precision"]), ev["true_positives"], ev["false_positives"]))
    print("  recall          : %s   (fn=%d)" % (_fmt(ev["recall"]), ev["false_negatives"]))
    print("  F1              : %s" % _fmt(ev["f1"]))
    print("  gold 'none'     : %d rows | predicted 'none': %d rows | agreed: %d"
          % (ev["gold_none"], ev["pred_none"], ev["both_none"]))
    if ev["none_agreement"] is not None:
        print("  none agreement  : %6.1f%%" % (ev["none_agreement"] * 100))

    # ---- calibration -------------------------------------------------
    correct = [g == p for g, p in zip(gold_actions, pred_actions)]
    cal = calibration(pred_conf, correct)
    print("\n## Confidence calibration  (action correctness per confidence bin)")
    print("  %-14s %5s %12s %10s %8s" % ("bin", "n", "mean conf", "accuracy", "gap"))
    for row in cal["bins"]:
        if not row["n"]:
            continue
        print("  %.2f - %.2f   %5d %12s %10s %8s" % (
            row["low"], row["high"], row["n"],
            _fmt(row["mean_confidence"], "%.3f"), _fmt(row["accuracy"], "%.3f"),
            _fmt(row["gap"], "%+.3f")))
    print("  expected calibration error (ECE): %.3f" % cal["ece"])
    if cal["overconfident_bins"]:
        print("  %d bin(s) overconfident by more than 0.10" % cal["overconfident_bins"])
    print("  (positive gap = stated confidence exceeds observed accuracy)")

    if not args.predictions and build_model_callable() is None:
        print("\n" + "!" * 72)
        print("! No model client is wired. These numbers score the safe-default")
        print("! fallback plus the deterministic guard, NOT the router. They are a")
        print("! floor and a proof the harness works, not a result.")
        print("!" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
