"""Run the golden set. Fast regression check for every prompt or rule change.

    python code/evaluation/run_golden.py
    python code/evaluation/run_golden.py --no-guard

Exit code is 1 on any failure once a model client is wired. While the router is
unwired every row is the safe default, so failures are expected and the exit code
stays 0 with a banner -- a check that always fails is a check nobody reads.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
for _sub in ("context", "router", "guard"):
    sys.path.insert(0, os.path.join(CODE, _sub))

from aggregates import build_reaction_stats                   # noqa: E402
from apply import apply_guard_all                             # noqa: E402
from assemble import build_context                            # noqa: E402
from golden import GOLDEN, evaluate                           # noqa: E402
from loaders import DEFAULT_DATASET, DEFAULT_MEDIA_CACHE, Dataset  # noqa: E402
from main import build_model_callable, strip_labels           # noqa: E402
from route import RouteStats, route_all                       # noqa: E402
from rules import REPORTED_POLICY_ANY, REPORTED_POLICY_UNANIMOUS, GuardPolicy  # noqa: E402


def resolve_rows(dataset):
    """message_id -> the raw row, from whichever table the case names."""
    by_id = {}
    for row in dataset.messages:
        by_id[row["message_id"]] = row
    for row in dataset.samples:
        by_id[row["message_id"]] = strip_labels(row)
    return by_id


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the adversarial golden set.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--media-cache", default=DEFAULT_MEDIA_CACHE)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--reported-policy", default=REPORTED_POLICY_UNANIMOUS,
                    choices=[REPORTED_POLICY_UNANIMOUS, REPORTED_POLICY_ANY])
    args = ap.parse_args(argv)

    dataset = Dataset(args.dataset, args.media_cache)
    reaction_stats = build_reaction_stats(dataset)
    by_id = resolve_rows(dataset)

    missing = [c.message_id for c in GOLDEN if c.message_id not in by_id]
    if missing:
        print("ERROR: golden rows not found in the dataset: %s" % ", ".join(missing),
              file=sys.stderr)
        return 2

    rows = [by_id[c.message_id] for c in GOLDEN]
    contexts = {r["message_id"]: build_context(r, dataset, reaction_stats) for r in rows}
    call_model = build_model_callable()
    decisions, _ = route_all(rows, contexts, call_model, RouteStats())
    if not args.no_guard:
        decisions, _ = apply_guard_all(
            decisions, contexts, GuardPolicy(reported_policy=args.reported_policy))

    print("GOLDEN SET - %d adversarial rows  (guard %s)"
          % (len(GOLDEN), "off" if args.no_guard else "on"))
    print("=" * 78)

    failed = []
    for case in GOLDEN:
        decision = decisions[case.message_id]
        problems = evaluate(case, decision.action, decision.message_type)
        status = "PASS" if not problems else "FAIL"
        truth = "" if case.ground_truth else "  [asserted, not labelled]"
        print("%s  %-28s %-16s -> %s/%s%s" % (
            status, case.case, case.message_id, decision.action,
            decision.message_type, truth))
        for problem in problems:
            print("        %s" % problem)
        if problems:
            failed.append(case)

    print("")
    print("%d/%d passed" % (len(GOLDEN) - len(failed), len(GOLDEN)))

    if call_model is None:
        print("")
        print("!" * 74)
        print("! No model client wired: every row enters as the safe default, so these")
        print("! failures are expected. Exit code held at 0 so this stays useful as a")
        print("! smoke test. It becomes a real regression gate once the router lands.")
        print("!" * 74)
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
