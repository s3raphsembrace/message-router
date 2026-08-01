"""output.csv writer and contract verifier.

The blank template shipped in dataset/output.csv is the contract: UTF-8, no BOM,
CRLF line endings, the six columns in a fixed order, and one row for every
message_id in dataset/messages.csv in that file's order. This module reproduces
it exactly and then re-reads what it wrote to prove it.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import ACTIONS, MESSAGE_TYPES                      # noqa: E402

COLUMNS = ("message_id", "action", "message_type", "reason", "confidence",
           "evidence_message_ids")

NONE_EVIDENCE = "none"


def write_output(rows, message_ids, path):
    """Write predictions for exactly `message_ids`, in that order.

    `rows` maps message_id -> dict with the five prediction fields. A missing id
    is a programming error, not something to paper over: the submission must have
    one row per message, so it raises rather than shipping a short file.
    """
    missing = [mid for mid in message_ids if mid not in rows]
    if missing:
        raise ValueError("no prediction for %d message(s): %s"
                         % (len(missing), ", ".join(missing[:5])))

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # newline="" + lineterminator="\r\n" reproduces the template's CRLF exactly
    # on every platform, rather than inheriting whatever the OS does.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS), lineterminator="\r\n")
        writer.writeheader()
        for mid in message_ids:
            row = rows[mid]
            writer.writerow({
                "message_id": mid,
                "action": row["action"],
                "message_type": row["message_type"],
                # Newlines inside a reason would survive quoting but make the file
                # painful to diff and read; the field is one sentence by contract.
                "reason": " ".join(str(row["reason"]).split()),
                "confidence": row["confidence"],
                "evidence_message_ids": row["evidence_message_ids"] or NONE_EVIDENCE,
            })
    return path


def verify_output(path, message_ids):
    """Re-read a written file and check it against the contract.

    Returns a list of problem strings; empty means the file is submittable.
    """
    problems = []
    if not os.path.exists(path):
        return ["file does not exist: %s" % path]

    raw = open(path, "rb").read()
    if raw[:3] == b"\xef\xbb\xbf":
        problems.append("file has a UTF-8 BOM; the template has none")
    lf_only = raw.count(b"\n") - raw.count(b"\r\n")
    if lf_only:
        problems.append("%d line(s) use bare LF; the template uses CRLF" % lf_only)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return problems + ["file is empty"]

    if tuple(rows[0]) != COLUMNS:
        problems.append("header is %s, expected %s" % (rows[0], list(COLUMNS)))

    data = rows[1:]
    if len(data) != len(message_ids):
        problems.append("%d data rows, expected %d" % (len(data), len(message_ids)))

    seen = []
    for i, row in enumerate(data, start=2):
        if len(row) != len(COLUMNS):
            problems.append("line %d has %d fields, expected %d" % (i, len(row), len(COLUMNS)))
            continue
        mid, action, mtype, reason, conf, evidence = row
        seen.append(mid)
        if action not in ACTIONS:
            problems.append("line %d: action %r not in vocabulary" % (i, action))
        if mtype not in MESSAGE_TYPES:
            problems.append("line %d: message_type %r not in vocabulary" % (i, mtype))
        if not reason.strip():
            problems.append("line %d: empty reason" % i)
        try:
            value = float(conf)
        except ValueError:
            problems.append("line %d: confidence %r is not a number" % (i, conf))
        else:
            if not (0.0 <= value <= 1.0):
                problems.append("line %d: confidence %s outside [0,1]" % (i, conf))
        if not evidence.strip():
            problems.append("line %d: evidence is blank; use 'none'" % i)

    if seen and seen != list(message_ids):
        if sorted(seen) == sorted(message_ids):
            problems.append("rows are not in messages.csv order")
        else:
            extra = set(seen) - set(message_ids)
            absent = set(message_ids) - set(seen)
            if extra:
                problems.append("unexpected message_id(s): %s" % ", ".join(sorted(extra)[:5]))
            if absent:
                problems.append("missing message_id(s): %s" % ", ".join(sorted(absent)[:5]))
        duplicates = len(seen) - len(set(seen))
        if duplicates:
            problems.append("%d duplicate message_id row(s)" % duplicates)

    return problems
