"""Submission validator for dataset/output.csv.

    python code/validate_submission.py
    python code/validate_submission.py --out /path/to/output.csv

Independent of the writer on purpose: it re-reads the file from disk and checks it
against the dataset, rather than trusting whatever produced it. Exits non-zero on
any violation.

Checks:
  1. exactly one row per message_id in messages.csv -- 110, no duplicates, no extras
  2. header order is exactly the six required columns
  3. action within {notify, digest, mute}
  4. message_type within the eleven allowed values
  5. confidence parses as a float in [0, 1]
  6. every evidence id is a real message_history id, or the literal "none"
  7. reason non-empty and CSV-safe (no raw newlines or control characters)
Then prints the action and message_type distribution, so a collapsed output is
obvious rather than silently valid.
"""

import argparse
import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "router"))

from prompt import ACTIONS, MESSAGE_TYPES                      # noqa: E402

COLUMNS = ("message_id", "action", "message_type", "reason", "confidence",
           "evidence_message_ids")

DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "dataset", "output.csv")
DEFAULT_DATASET = os.path.join(REPO_ROOT, "dataset")

CONTROL_CHARS = set(chr(c) for c in list(range(0, 32)) + [127]) - {"\t"}


def read_ids(path, column):
    with open(path, newline="", encoding="utf-8") as f:
        return [r[column] for r in csv.DictReader(f)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate output.csv against the contract.")
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    args = ap.parse_args(argv)

    problems = []

    if not os.path.exists(args.out):
        print("FAIL: %s does not exist" % args.out, file=sys.stderr)
        return 1

    expected_ids = read_ids(os.path.join(args.dataset, "messages.csv"), "message_id")
    history_ids = set(read_ids(os.path.join(args.dataset, "message_history.csv"), "message_id"))

    with open(args.out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # -- 2. header ------------------------------------------------------
    if not rows:
        print("FAIL: file is empty", file=sys.stderr)
        return 1
    if tuple(rows[0]) != COLUMNS:
        problems.append("header is %s, expected %s" % (rows[0], list(COLUMNS)))

    data = rows[1:]

    # -- 1. one row per message_id -------------------------------------
    seen = [r[0] for r in data if r]
    counts = Counter(seen)
    duplicates = sorted(m for m, n in counts.items() if n > 1)
    extras = sorted(set(seen) - set(expected_ids))
    absent = sorted(set(expected_ids) - set(seen))
    if len(data) != len(expected_ids):
        problems.append("%d data rows, expected %d" % (len(data), len(expected_ids)))
    if duplicates:
        problems.append("duplicate message_id(s): %s" % ", ".join(duplicates[:5]))
    if extras:
        problems.append("message_id(s) not in messages.csv: %s" % ", ".join(extras[:5]))
    if absent:
        problems.append("missing message_id(s): %s" % ", ".join(absent[:5]))
    if seen and seen != expected_ids and not duplicates and not extras and not absent:
        problems.append("rows are not in messages.csv order")

    actions, types = Counter(), Counter()
    confidences = []

    for line_no, row in enumerate(data, start=2):
        if len(row) != len(COLUMNS):
            problems.append("line %d: %d fields, expected %d" % (line_no, len(row), len(COLUMNS)))
            continue
        mid, action, mtype, reason, confidence, evidence = row

        # -- 3 / 4. closed vocabularies --------------------------------
        if action not in ACTIONS:
            problems.append("line %d (%s): action %r not allowed" % (line_no, mid, action))
        else:
            actions[action] += 1
        if mtype not in MESSAGE_TYPES:
            problems.append("line %d (%s): message_type %r not allowed" % (line_no, mid, mtype))
        else:
            types[mtype] += 1

        # -- 5. confidence ---------------------------------------------
        try:
            value = float(confidence)
        except ValueError:
            problems.append("line %d (%s): confidence %r is not a float" % (line_no, mid, confidence))
        else:
            if not (0.0 <= value <= 1.0):
                problems.append("line %d (%s): confidence %s outside [0,1]" % (line_no, mid, value))
            confidences.append(value)

        # -- 6. evidence ids -------------------------------------------
        text = (evidence or "").strip()
        if not text:
            problems.append("line %d (%s): evidence is blank; use 'none'" % (line_no, mid))
        elif text != "none":
            for part in text.split(";"):
                part = part.strip()
                if not part:
                    problems.append("line %d (%s): empty evidence id in %r" % (line_no, mid, text))
                elif part not in history_ids:
                    problems.append("line %d (%s): evidence %r is not a message_history id"
                                    % (line_no, mid, part))

        # -- 7. reason -------------------------------------------------
        if not reason.strip():
            problems.append("line %d (%s): reason is empty" % (line_no, mid))
        bad = sorted(set(reason) & CONTROL_CHARS)
        if bad:
            problems.append("line %d (%s): reason contains control character(s) %r"
                            % (line_no, mid, bad))

    # -- report ---------------------------------------------------------
    print("VALIDATING %s" % args.out)
    print("=" * 72)
    if problems:
        print("FAILED - %d problem(s):" % len(problems))
        for p in problems[:30]:
            print("  - %s" % p)
        if len(problems) > 30:
            print("  ... and %d more" % (len(problems) - 30))
        return 1

    print("PASS - all checks")
    print("  rows                : %d (one per message_id, correct order)" % len(data))
    print("  header              : exact")
    print("  vocabularies        : action and message_type all in range")
    print("  confidence          : all floats in [0,1]")
    print("  evidence ids        : all real message_history ids or 'none'")
    print("  reason              : all non-empty and CSV-safe")
    print("")
    print("DISTRIBUTION (sanity check -- a collapsed output is a failed one)")
    total = float(len(data)) or 1.0
    print("  action:")
    for name in ACTIONS:
        n = actions.get(name, 0)
        print("    %-16s %4d  %5.1f%%" % (name, n, 100 * n / total))
    print("  message_type:")
    for name, n in types.most_common():
        print("    %-16s %4d  %5.1f%%" % (name, n, 100 * n / total))
    unused = [t for t in MESSAGE_TYPES if t not in types]
    if unused:
        print("    (unused: %s)" % ", ".join(unused))
    if confidences:
        ordered = sorted(confidences)
        print("  confidence: min=%.2f median=%.2f max=%.2f distinct=%d"
              % (ordered[0], ordered[len(ordered) // 2], ordered[-1], len(set(confidences))))
    if len(actions) == 1:
        print("")
        print("WARNING: every row has the same action -- output is collapsed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
