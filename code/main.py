"""Message Notification Router - pipeline entry point.

    python code/main.py                  # route all messages, write dataset/output.csv
    python code/main.py --out /tmp/o.csv # write elsewhere
    python code/main.py --verify-only    # re-check an existing output.csv

Layer 1 (media interpretation) is a separate, cached step:

    python code/preprocess/run.py

The model client for Layer 3 is not wired yet. Until it is, every row takes the
validated safe default (digest / unknown / 0.5 / none) and the run says so loudly
-- the file is contract-valid but is NOT a real submission.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "context"))
sys.path.insert(0, os.path.join(HERE, "router"))
sys.path.insert(0, os.path.join(HERE, "guard"))

from aggregates import build_reaction_stats                    # noqa: E402
from apply import apply_guard_all                              # noqa: E402
from assemble import DEFAULT_TOKEN_BUDGET, build_context       # noqa: E402
from audit import report as audit_report, write_audit          # noqa: E402
from loaders import DEFAULT_DATASET, DEFAULT_MEDIA_CACHE, Dataset  # noqa: E402
from retrieve import DEFAULT_SHORTLIST                         # noqa: E402
from route import MAX_REASKS, RouteStats, route_all            # noqa: E402
from rules import REPORTED_POLICY_ANY, REPORTED_POLICY_UNANIMOUS, GuardPolicy  # noqa: E402
from writer import verify_output, write_output                 # noqa: E402

DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "dataset", "output.csv")
DEFAULT_AUDIT = os.path.join(REPO_ROOT, "cache", "override_audit.csv")


def build_model_callable():
    """The Layer 3 client. Returns None until it is built.

    Kept as a seam so the routing loop, validator, re-ask and writer are all
    exercised end to end today, and wiring the client changes nothing else.
    """
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Route WhatsApp messages and write output.csv")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--media-cache", default=DEFAULT_MEDIA_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    ap.add_argument("--top-k", type=int, default=DEFAULT_SHORTLIST)
    ap.add_argument("--same-sender-only", action="store_true")
    ap.add_argument("--max-reasks", type=int, default=MAX_REASKS)
    ap.add_argument("--audit", default=DEFAULT_AUDIT,
                    help="where to write the override audit log")
    ap.add_argument("--reported-policy", default=REPORTED_POLICY_UNANIMOUS,
                    choices=[REPORTED_POLICY_UNANIMOUS, REPORTED_POLICY_ANY],
                    help="'unanimous' (default): mute only when every past message from "
                         "the sender was reported and none opened. 'any': mute if the user "
                         "ever reported the sender -- fires on ~54%% of rows and contradicts "
                         "two labelled samples.")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the deterministic override layer entirely")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify an existing output.csv without rewriting it")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print("ERROR: dataset not found: %s" % args.dataset, file=sys.stderr)
        return 2

    dataset = Dataset(args.dataset, args.media_cache)
    message_ids = [r["message_id"] for r in dataset.messages]

    if args.verify_only:
        problems = verify_output(args.out, message_ids)
        print("verifying %s" % args.out)
        if problems:
            print("FAILED (%d problem(s)):" % len(problems))
            for p in problems[:20]:
                print("  - %s" % p)
            return 1
        print("OK: %d rows, correct columns and vocabulary" % len(message_ids))
        return 0

    reaction_stats = build_reaction_stats(dataset)
    contexts = {
        r["message_id"]: build_context(
            r, dataset, reaction_stats, token_budget=args.budget,
            shortlist_limit=args.top_k, allow_fallback=not args.same_sender_only)
        for r in dataset.messages
    }

    call_model = build_model_callable()
    print("Message Notification Router")
    print("  messages : %d" % len(message_ids))
    print("  media    : %d interpreted" % sum(1 for m in dataset.media.values() if m.ok))
    print("  model    : %s" % ("configured" if call_model else "NOT WIRED - see note below"))
    print("")

    decisions, stats = route_all(dataset.messages, contexts, call_model,
                                 RouteStats(), args.max_reasks)
    print(stats.report())
    print("")

    records = []
    if args.no_guard:
        print("guard: SKIPPED (--no-guard)")
    else:
        policy = GuardPolicy(reported_policy=args.reported_policy)
        decisions, records = apply_guard_all(decisions, contexts, policy)
        print("Layer 4 - deterministic override guard (reported-policy=%s)"
              % args.reported_policy)
        print(audit_report(records, len(message_ids)))
        if records:
            write_audit(records, args.audit)
            print("audit log -> %s" % args.audit)
    print("")

    rows = {mid: d.to_row() for mid, d in decisions.items()}
    write_output(rows, message_ids, args.out)
    print("wrote %s" % args.out)

    problems = verify_output(args.out, message_ids)
    if problems:
        print("CONTRACT CHECK FAILED (%d problem(s)):" % len(problems))
        for p in problems[:20]:
            print("  - %s" % p)
        return 1
    print("contract check: OK (%d rows, exact columns, closed vocabulary)" % len(message_ids))

    if not call_model:
        print("")
        print("!" * 72)
        print("! NOT A SUBMISSION. No model client is wired, so all %d rows are the" % len(message_ids))
        print("! safe default (digest / unknown / 0.5 / none). The file is contract-")
        print("! valid so the plumbing is proven, but it contains no real routing.")
        print("!" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
