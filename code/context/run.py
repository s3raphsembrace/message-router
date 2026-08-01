"""Layer 2 entry point: build decision contexts and report on them.

    python code/context/run.py --show msg_066        # inspect one context
    python code/context/run.py --stats               # token budget report
    python code/context/run.py --out contexts.json   # write all 110

Building contexts requires no API key and no network. If Layer 1 has not run, the
media blocks report `not_interpreted` and those rows carry metadata only.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregates import build_reaction_stats                     # noqa: E402
from assemble import DEFAULT_TOKEN_BUDGET, build_all, build_context, estimate_tokens  # noqa: E402
from loaders import DEFAULT_DATASET, DEFAULT_MEDIA_CACHE, Dataset  # noqa: E402
from retrieve import DEFAULT_SHORTLIST                          # noqa: E402


def _percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build routing decision contexts (Layer 2).")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--media-cache", default=DEFAULT_MEDIA_CACHE)
    ap.add_argument("--budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    ap.add_argument("--top-k", type=int, default=DEFAULT_SHORTLIST,
                    help="evidence candidates offered to the router (default %d)"
                         % DEFAULT_SHORTLIST)
    ap.add_argument("--same-sender-only", action="store_true",
                    help="restrict evidence to the same sender/group/business; "
                         "disables the near-duplicate fallback tier")
    ap.add_argument("--show", default="", help="print the context for one message_id")
    ap.add_argument("--stats", action="store_true", help="token budget report")
    ap.add_argument("--out", default="", help="write all contexts to a JSON file")
    ap.add_argument("--samples", action="store_true", help="use sample_messages.csv instead")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print("ERROR: dataset not found: %s" % args.dataset, file=sys.stderr)
        return 2

    dataset = Dataset(args.dataset, args.media_cache)
    stats = build_reaction_stats(dataset)
    rows = dataset.samples if args.samples else dataset.messages

    interpreted = sum(1 for m in dataset.media.values() if m.ok)
    print("Layer 2 - context assembly")
    print("  messages         : %d" % len(rows))
    print("  history rows     : %d" % len(dataset.history))
    print("  reaction pairs   : %d" % len(stats))
    print("  media interpreted: %d / %d in cache" % (interpreted, len(dataset.media)))
    print("  token budget     : %d" % args.budget)
    print("  top-K evidence   : %d%s" % (
        args.top_k, "  (same-sender only)" if args.same_sender_only else ""))
    print("")

    if args.show:
        match = next((r for r in rows if r["message_id"] == args.show), None)
        if match is None:
            print("ERROR: no such message_id: %s" % args.show, file=sys.stderr)
            return 2
        ctx = build_context(match, dataset, stats, token_budget=args.budget,
                            shortlist_limit=args.top_k,
                            allow_fallback=not args.same_sender_only)
        print(json.dumps(ctx, indent=2, ensure_ascii=False))
        return 0

    contexts = build_all(dataset, stats, rows=rows, token_budget=args.budget,
                         shortlist_limit=args.top_k,
                         allow_fallback=not args.same_sender_only)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(contexts, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        print("wrote %d contexts -> %s" % (len(contexts), args.out))

    if args.stats or not args.out:
        sizes = [estimate_tokens(c) for c in contexts.values()]
        over = [mid for mid, c in contexts.items() if estimate_tokens(c) > args.budget]
        truncated = [mid for mid, c in contexts.items() if c["_meta"]["truncated"]]
        no_evidence = [mid for mid, c in contexts.items() if not c["evidence_candidates"]]
        print("estimated tokens  min=%d  p50=%d  p95=%d  max=%d  total=%d"
              % (min(sizes), _percentile(sizes, 50), _percentile(sizes, 95),
                 max(sizes), sum(sizes)))
        print("truncated: %d   over budget: %d" % (len(truncated), len(over)))
        if over:
            print("  OVER: %s" % ", ".join(sorted(over)[:10]))
        counts = [len(c["evidence_candidates"]) for c in contexts.values()]
        fallback_only = [mid for mid, c in contexts.items()
                         if c["evidence_candidates"]
                         and not any(e["same_sender"] for e in c["evidence_candidates"])]
        print("evidence per context: min=%d p50=%d max=%d (K=%d)"
              % (min(counts), _percentile(counts, 50), max(counts), args.top_k))
        print("contexts with no evidence candidates: %d %s"
              % (len(no_evidence), sorted(no_evidence)[:8] if no_evidence else ""))
        print("contexts relying only on fallback (no same-sender history): %d %s"
              % (len(fallback_only), sorted(fallback_only)[:8] if fallback_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
