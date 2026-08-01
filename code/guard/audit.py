"""Override audit log.

Every rule firing is recorded with the model's original verdict alongside the
rule's, so a run can be reviewed row by row: which rules fired, which rows the
model and the rules actually disagreed on, and what each disagreement was.

Written as CSV because the natural next step is sorting and filtering it.
"""

import csv
import os
from collections import Counter

AUDIT_COLUMNS = (
    "message_id", "rule", "from_action", "to_action", "from_type", "to_type",
    "from_confidence", "to_confidence", "disagreement", "model_fell_back",
    "model_reason", "rule_reason",
)


def write_audit(records, path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(AUDIT_COLUMNS), lineterminator="\r\n")
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return path


def summarise(records, total_messages=None):
    """Counts per rule, plus how many were genuine disagreements."""
    fired = Counter(r.rule for r in records)
    disagreed = Counter(r.rule for r in records if r.disagreement)
    transitions = Counter("%s->%s" % (r.from_action, r.to_action)
                          for r in records if r.changed_action)
    touched = {r.message_id for r in records}
    contested = {r.message_id for r in records if r.disagreement}
    return {
        "records": len(records),
        "messages_touched": len(touched),
        "messages_contested": len(contested),
        "total_messages": total_messages,
        "by_rule": dict(fired),
        "disagreements_by_rule": dict(disagreed),
        "action_transitions": dict(transitions),
    }


def report(records, total_messages=None):
    s = summarise(records, total_messages)
    lines = ["override records   : %d" % s["records"],
             "messages touched   : %d%s" % (
                 s["messages_touched"],
                 " / %d" % total_messages if total_messages else "")]
    lines.append("messages where the rules contradicted the model: %d" % s["messages_contested"])
    if s["by_rule"]:
        lines.append("by rule:")
        for rule, n in sorted(s["by_rule"].items(), key=lambda kv: -kv[1]):
            lines.append("  %-24s fired=%-4d contradicted_model=%d"
                         % (rule, n, s["disagreements_by_rule"].get(rule, 0)))
    if s["action_transitions"]:
        lines.append("action changes:")
        for transition, n in sorted(s["action_transitions"].items(), key=lambda kv: -kv[1]):
            lines.append("  %-16s %d" % (transition, n))
    return "\n".join(lines)


def disagreements(records):
    """Just the rows where a rule actually changed the model's verdict."""
    return [r for r in records if r.disagreement]
