"""Layer 3 prompt preview.

    python code/router/run.py --show msg_066     # the exact prompt for one message
    python code/router/run.py --system           # just the system prompt
    python code/router/run.py --stats            # prompt sizes across all 110

No API key and no network -- this renders prompts, it does not call a model.
The model client is the next piece of Layer 3.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "context"))

from aggregates import build_reaction_stats                    # noqa: E402
from assemble import build_context, estimate_tokens            # noqa: E402
from loaders import DEFAULT_DATASET, DEFAULT_MEDIA_CACHE, Dataset  # noqa: E402
from prompt import SYSTEM_PROMPT, render                       # noqa: E402


def _percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Preview router prompts (Layer 3).")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--media-cache", default=DEFAULT_MEDIA_CACHE)
    ap.add_argument("--show", default="", help="render the full prompt for one message_id")
    ap.add_argument("--system", action="store_true", help="print the system prompt only")
    ap.add_argument("--stats", action="store_true", help="prompt size report")
    ap.add_argument("--samples", action="store_true", help="use sample_messages.csv")
    args = ap.parse_args(argv)

    if args.system:
        print(SYSTEM_PROMPT)
        return 0

    dataset = Dataset(args.dataset, args.media_cache)
    stats = build_reaction_stats(dataset)
    rows = dataset.samples if args.samples else dataset.messages

    if args.show:
        match = next((r for r in rows if r["message_id"] == args.show), None)
        if match is None:
            print("ERROR: no such message_id: %s" % args.show, file=sys.stderr)
            return 2
        system, user = render(build_context(match, dataset, stats))
        print("=" * 78)
        print("SYSTEM (%d tokens)" % estimate_tokens(system))
        print("=" * 78)
        print(system)
        print("")
        print("=" * 78)
        print("USER (%d tokens)" % estimate_tokens(user))
        print("=" * 78)
        print(user)
        return 0

    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    user_tokens = []
    no_evidence = 0
    for row in rows:
        ctx = build_context(row, dataset, stats)
        _, user = render(ctx)
        user_tokens.append(estimate_tokens(user))
        if not ctx["evidence_candidates"]:
            no_evidence += 1

    print("Layer 3 - prompt sizes")
    print("  messages          : %d" % len(rows))
    print("  system prompt     : %d tokens (cached across all calls)" % sys_tokens)
    print("  user prompt       : min=%d p50=%d p95=%d max=%d"
          % (min(user_tokens), _percentile(user_tokens, 50),
             _percentile(user_tokens, 95), max(user_tokens)))
    print("  total user tokens : %d" % sum(user_tokens))
    print("  per-call total    : ~%d tokens (system + median user)"
          % (sys_tokens + _percentile(user_tokens, 50)))
    print("  forced-none rows  : %d (no evidence candidates offered)" % no_evidence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
