"""Append-only record of evaluation runs, with deltas between them.

Every prompt or rule change should end with a recorded run, so improvement is a
tracked number rather than a recollection. The ledger is CSV, committed, and never
rewritten -- a run that made things worse stays in the history.

One caveat is enforced rather than left to the reader: there are 30 labelled rows,
so a single row is 3.33 percentage points. Any accuracy delta smaller than that is
one row moving, which is not evidence of anything. Deltas below the floor are
marked `~` instead of being reported as improvements.
"""

import csv
import os
import subprocess
from collections import OrderedDict

LEDGER_COLUMNS = (
    "run", "timestamp", "git_sha", "change",
    "config",
    "action_acc", "type_acc", "joint_acc",
    "evidence_f1", "evidence_precision", "evidence_recall", "evidence_exact",
    "ece", "cost_per_row", "severe_suppressed", "severe_intruded",
    "n_rows", "notes",
)

# Metrics where higher is better; everything else is lower-is-better.
HIGHER_IS_BETTER = frozenset({
    "action_acc", "type_acc", "joint_acc",
    "evidence_f1", "evidence_precision", "evidence_recall", "evidence_exact",
})

# One labelled row out of 30 is 3.33pp. Anything smaller is a single row moving.
SAMPLE_ROWS = 30
ACCURACY_FLOOR = 1.0 / SAMPLE_ROWS
# Calibration is continuous rather than per-row, so it gets its own floor.
ECE_FLOOR = 0.02
COST_FLOOR = 0.05

FLOORS = {
    "ece": ECE_FLOOR,
    "cost_per_row": COST_FLOOR,
    "severe_suppressed": 0.5,
    "severe_intruded": 0.5,
}


def git_sha():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.STDOUT)
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return "unknown"


def load_runs(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key in ("run", "n_rows", "severe_suppressed", "severe_intruded"):
                try:
                    row[key] = int(row[key])
                except (TypeError, ValueError):
                    row[key] = 0
            elif key not in ("timestamp", "git_sha", "change", "config", "notes"):
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    row[key] = None
    return rows


def append_run(path, record):
    """Append one run. Returns the assigned run number."""
    existing = load_runs(path)
    record = dict(record)
    record["run"] = len(existing) + 1
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEDGER_COLUMNS), lineterminator="\r\n")
        if is_new:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in LEDGER_COLUMNS})
    return record["run"]


def _floor_for(metric):
    return FLOORS.get(metric, ACCURACY_FLOOR)


def delta(metric, previous, current):
    """Returns (raw_delta, verdict) where verdict is 'better' | 'worse' | 'noise'."""
    if previous is None or current is None:
        return None, "new"
    raw = current - previous
    if abs(raw) < _floor_for(metric):
        return raw, "noise"
    improved = raw > 0 if metric in HIGHER_IS_BETTER else raw < 0
    return raw, "better" if improved else "worse"


def compare(previous, current):
    """Per-metric deltas between two ledger rows."""
    out = OrderedDict()
    for metric in ("action_acc", "type_acc", "joint_acc", "evidence_f1",
                   "ece", "cost_per_row", "severe_suppressed", "severe_intruded"):
        raw, verdict = delta(metric, (previous or {}).get(metric), current.get(metric))
        out[metric] = {"previous": (previous or {}).get(metric),
                       "current": current.get(metric),
                       "delta": raw, "verdict": verdict}
    return out


def tradeoffs(comparison):
    """Plain-language warnings where one metric improved at another's expense.

    This is the whole point of keeping a ledger: an accuracy gain paid for with
    calibration or evidence quality is not obviously a gain, and it is easy to miss
    when reading one number at a time.
    """
    notes = []
    action = comparison.get("action_acc", {})
    if action.get("verdict") == "better":
        if comparison.get("ece", {}).get("verdict") == "worse":
            notes.append(
                "action accuracy improved (%+.1fpp) but calibration got WORSE "
                "(ECE %+.3f) -- the model is being right more often while its stated "
                "confidence means less" % (action["delta"] * 100,
                                           comparison["ece"]["delta"]))
        if comparison.get("evidence_f1", {}).get("verdict") == "worse":
            notes.append(
                "action accuracy improved (%+.1fpp) but evidence F1 got WORSE "
                "(%+.3f) -- decisions are better, their justifications are not"
                % (action["delta"] * 100, comparison["evidence_f1"]["delta"]))
        for key, label in (("severe_suppressed", "gold notify -> predicted mute"),
                           ("severe_intruded", "gold mute -> predicted notify")):
            if comparison.get(key, {}).get("verdict") == "worse":
                notes.append(
                    "action accuracy improved but severe errors increased (%s: %+d) -- "
                    "aggregate accuracy is hiding a worse failure mode"
                    % (label, int(comparison[key]["delta"])))
    if action.get("verdict") == "worse" and comparison.get("ece", {}).get("verdict") == "better":
        notes.append(
            "action accuracy dropped (%+.1fpp) while calibration improved (ECE %+.3f) -- "
            "check whether the model simply became less confident overall"
            % (action["delta"] * 100, comparison["ece"]["delta"]))
    if comparison.get("cost_per_row", {}).get("verdict") == "worse" and \
            action.get("verdict") in ("better", "noise"):
        notes.append(
            "weighted cost per row got WORSE (%+.2f) despite action accuracy not "
            "dropping -- the mistakes that remain are the expensive kind"
            % comparison["cost_per_row"]["delta"])
    return notes


def _cell(value, metric):
    if value is None:
        return "   -  "
    if metric in ("severe_suppressed", "severe_intruded"):
        return "%6d" % value
    if metric in ("ece", "cost_per_row"):
        return "%6.3f" % value
    return "%5.1f%%" % (value * 100)


def _delta_cell(info, metric):
    """Signed delta with a trailing verdict marker.

    The marker trails rather than leads so it cannot be misread as part of the
    sign: "-20.0!" is unambiguous where "--20.0" is not.
    """
    if info["verdict"] == "new" or info["delta"] is None:
        return "     "
    mark = {"better": "*", "worse": "!", "noise": "~"}[info["verdict"]]
    if metric in ("severe_suppressed", "severe_intruded"):
        return "%+d%s" % (int(info["delta"]), mark)
    if metric in ("ece", "cost_per_row"):
        return "%+.3f%s" % (info["delta"], mark)
    return "%+.1f%s" % (info["delta"] * 100, mark)


def history_table(runs, limit=12):
    """Render the ledger as a table with per-run deltas."""
    if not runs:
        return "no runs recorded yet"
    shown = runs[-limit:]
    lines = []
    header = ("%-4s %-9s %-34s %8s %8s %8s %8s %8s"
              % ("run", "sha", "change", "act-acc", "typ-acc", "ev-F1", "ECE", "cost/row"))
    lines.append(header)
    lines.append("-" * len(header))
    for i, run in enumerate(shown):
        previous = runs[runs.index(run) - 1] if runs.index(run) > 0 else None
        cmp_ = compare(previous, run)
        lines.append("%-4s %-9s %-34s %8s %8s %8s %8s %8s" % (
            run["run"], (run.get("git_sha") or "")[:9], (run.get("change") or "")[:34],
            _cell(run.get("action_acc"), "action_acc"),
            _cell(run.get("type_acc"), "type_acc"),
            _cell(run.get("evidence_f1"), "evidence_f1"),
            _cell(run.get("ece"), "ece"),
            _cell(run.get("cost_per_row"), "cost_per_row")))
        if previous:
            lines.append("%-4s %-9s %-34s %8s %8s %8s %8s %8s" % (
                "", "", "  vs prev",
                _delta_cell(cmp_["action_acc"], "action_acc"),
                _delta_cell(cmp_["type_acc"], "type_acc"),
                _delta_cell(cmp_["evidence_f1"], "evidence_f1"),
                _delta_cell(cmp_["ece"], "ece"),
                _delta_cell(cmp_["cost_per_row"], "cost_per_row")))
    lines.append("")
    lines.append("* better   ! worse   ~ within noise (< %.1fpp accuracy = 1 of %d rows)"
                 % (ACCURACY_FLOOR * 100, SAMPLE_ROWS))
    return "\n".join(lines)
