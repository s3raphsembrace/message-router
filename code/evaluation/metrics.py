"""Scoring metrics for the router.

Pure functions over gold/predicted lists -- no I/O, no dataset access -- so they
can be unit-tested against hand-built cases rather than only against whatever the
pipeline happens to produce today.

The action confusion matrix is not treated as symmetric. Muting a message the user
needed and interrupting them with a scam are the two failures that actually matter;
confusing notify with digest is a nuisance. The cost matrix below encodes that, and
the two severe cells are reported separately from the aggregate.
"""

from collections import Counter, OrderedDict

ACTIONS = ("notify", "digest", "mute")

# cost[gold][predicted]. Zero on the diagonal.
#
#   gold=notify, pred=mute    -> the user missed something they needed      (worst)
#   gold=mute,   pred=notify  -> the user was interrupted by junk or a scam (worst)
#   anything adjacent          -> a nuisance, not a failure
ACTION_COST = {
    "notify": {"notify": 0.0, "digest": 1.0, "mute": 5.0},
    "digest": {"notify": 1.0, "digest": 0.0, "mute": 1.0},
    "mute":   {"notify": 5.0, "digest": 1.0, "mute": 0.0},
}

# Extra weight when the thing wrongly promoted to notify was a scam or spam.
SCAM_INTRUSION_MULTIPLIER = 1.5
RISK_TYPES = frozenset({"scam", "spam"})

# Gold types that make a suppression especially bad.
URGENT_TYPES = frozenset({"urgent", "payment", "event"})

DEFAULT_BINS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def accuracy(gold, pred):
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / float(len(gold))


def joint_accuracy(gold_actions, pred_actions, gold_types, pred_types):
    """Both fields correct on the same row."""
    if not gold_actions:
        return 0.0
    hits = sum(1 for ga, pa, gt, pt in zip(gold_actions, pred_actions, gold_types, pred_types)
               if ga == pa and gt == pt)
    return hits / float(len(gold_actions))


def confusion(gold, pred, labels=ACTIONS):
    """matrix[gold_label][pred_label] = count."""
    matrix = OrderedDict((g, OrderedDict((p, 0) for p in labels)) for g in labels)
    unknown = 0
    for g, p in zip(gold, pred):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
        else:
            unknown += 1
    return matrix, unknown


def severe_errors(gold_actions, pred_actions, gold_types):
    """The two asymmetric failures, itemised.

    suppressed_urgent  -- gold notify, predicted mute
    intruding_risk     -- gold mute, predicted notify (split by whether it was risky)
    """
    suppressed, intruded, intruded_risk = [], [], []
    for i, (ga, pa) in enumerate(zip(gold_actions, pred_actions)):
        gt = gold_types[i] if i < len(gold_types) else ""
        if ga == "notify" and pa == "mute":
            suppressed.append((i, gt))
        elif ga == "mute" and pa == "notify":
            intruded.append((i, gt))
            if gt in RISK_TYPES:
                intruded_risk.append((i, gt))
    return {
        "suppressed_notify": suppressed,
        "suppressed_notify_urgent": [x for x in suppressed if x[1] in URGENT_TYPES],
        "intruding_mute": intruded,
        "intruding_mute_risky": intruded_risk,
    }


def action_cost(gold_actions, pred_actions, gold_types):
    """Total and per-row weighted cost using ACTION_COST."""
    total = 0.0
    for i, (g, p) in enumerate(zip(gold_actions, pred_actions)):
        if g not in ACTION_COST or p not in ACTION_COST[g]:
            continue
        c = ACTION_COST[g][p]
        gt = gold_types[i] if i < len(gold_types) else ""
        if g == "mute" and p == "notify" and gt in RISK_TYPES:
            c *= SCAM_INTRUSION_MULTIPLIER
        total += c
    n = len(gold_actions) or 1
    worst = max(max(row.values()) for row in ACTION_COST.values()) * SCAM_INTRUSION_MULTIPLIER
    return {"total": total, "per_row": total / n, "worst_case_per_row": worst}


def parse_evidence(value):
    """'a;b' -> {'a','b'};  'none' / '' -> set()."""
    if value is None:
        return set()
    text = str(value).strip()
    if not text or text.lower() == "none":
        return set()
    return {p.strip() for p in text.split(";") if p.strip()}


def evidence_scores(gold_values, pred_values):
    """Micro precision/recall/F1 over evidence ids, plus none-handling.

    Micro-averaging is the honest choice here: most rows cite one id, so a macro
    average would be dominated by whether single-id rows happened to match, and
    rows citing nothing would need an arbitrary convention.
    """
    tp = fp = fn = 0
    exact = 0
    gold_none = pred_none = both_none = 0
    rows = 0

    for gv, pv in zip(gold_values, pred_values):
        g, p = parse_evidence(gv), parse_evidence(pv)
        rows += 1
        if g == p:
            exact += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if not g:
            gold_none += 1
        if not p:
            pred_none += 1
        if not g and not p:
            both_none += 1

    # Always numeric, so the ledger records the metric instead of storing a blank.
    # The degenerate cases are resolved explicitly rather than left undefined:
    #   nothing expected AND nothing cited -> 1.0 (perfect agreement on `none`)
    #   something expected but nothing cited -> 0.0 (a real miss, not "undefined")
    denom_p, denom_r = tp + fp, tp + fn
    precision = (tp / float(denom_p)) if denom_p else (1.0 if denom_r == 0 else 0.0)
    recall = (tp / float(denom_r)) if denom_r else (1.0 if denom_p == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "rows": rows,
        "exact_match": exact / float(rows) if rows else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "gold_none": gold_none,
        "pred_none": pred_none,
        "both_none": both_none,
        "none_agreement": both_none / float(gold_none) if gold_none else None,
    }


def calibration(confidences, correct_flags, bins=DEFAULT_BINS):
    """Bin predictions by stated confidence and compare to observed accuracy.

    Returns per-bin rows plus Expected Calibration Error -- the sample-weighted
    mean gap between stated confidence and observed accuracy. ECE near 0 means the
    stated numbers mean something.
    """
    rows = []
    total = len(confidences)
    ece = 0.0
    for low, high in bins:
        idx = [i for i, c in enumerate(confidences) if low <= c < high]
        if not idx:
            rows.append({"low": low, "high": high, "n": 0, "mean_confidence": None,
                         "accuracy": None, "gap": None})
            continue
        mean_conf = sum(confidences[i] for i in idx) / float(len(idx))
        acc = sum(1 for i in idx if correct_flags[i]) / float(len(idx))
        gap = mean_conf - acc
        ece += (len(idx) / float(total)) * abs(gap)
        rows.append({"low": low, "high": high, "n": len(idx),
                     "mean_confidence": mean_conf, "accuracy": acc, "gap": gap})
    over = sum(1 for r in rows if r["gap"] is not None and r["gap"] > 0.1)
    return {"bins": rows, "ece": ece, "overconfident_bins": over, "n": total}


def per_class_report(gold, pred):
    """Precision/recall/support for each message_type actually present."""
    labels = sorted(set(gold) | set(pred))
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    hits = Counter(g for g, p in zip(gold, pred) if g == p)
    out = []
    for label in labels:
        support = gold_counts.get(label, 0)
        predicted = pred_counts.get(label, 0)
        tp = hits.get(label, 0)
        out.append({
            "label": label,
            "support": support,
            "predicted": predicted,
            "precision": (tp / float(predicted)) if predicted else None,
            "recall": (tp / float(support)) if support else None,
        })
    return out
