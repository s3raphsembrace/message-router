"""Layer 1 entry point: interpret every media file referenced by the dataset.

    python code/preprocess/run.py                 # process messages + history media
    python code/preprocess/run.py --scope messages
    python code/preprocess/run.py --force         # ignore cache, reprocess everything
    python code/preprocess/run.py --dry-run       # resolve + report, no model calls

Writes/updates cache/media_interpretations.json. Safe to re-run: cached ids are
skipped, so a second run costs nothing and a partial run resumes where it stopped.

The process exit code reflects orchestration health only. An individual file that
cannot be interpreted produces a typed placeholder and does not fail the run --
Layer 2 falls back to message metadata for that row.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache import InterpretationCache                                    # noqa: E402
from gemini import MediaInterpreter                                      # noqa: E402
from media_index import MediaIndex                                       # noqa: E402
from schema import (                                                     # noqa: E402
    KIND_IMAGE,
    KIND_VOICE,
    STATUS_OK,
    MediaInterpretation,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "dataset")
DEFAULT_CACHE = os.path.join(REPO_ROOT, "cache", "media_interpretations.json")

SOURCES = {
    "messages": ["messages.csv"],
    "history": ["message_history.csv"],
    "all": ["messages.csv", "message_history.csv"],
}


def collect_refs(dataset_dir, filenames):
    """Every distinct (media_id, kind) referenced by the given message tables.

    Deduplicated: the same poster is sent to several users, and it must only be
    interpreted once.
    """
    refs = {}
    for fname in filenames:
        path = os.path.join(dataset_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mid = (row.get("media_id") or "").strip()
                kind = (row.get("media_type") or "").strip()
                if not mid or kind not in (KIND_IMAGE, KIND_VOICE):
                    continue
                refs.setdefault(mid, kind)
    return refs


def load_interpretations(cache_path=DEFAULT_CACHE):
    """Helper for downstream layers: media_id -> MediaInterpretation."""
    return {rec.media_id: rec for rec in InterpretationCache(cache_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interpret dataset media (Layer 1).")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--scope", choices=sorted(SOURCES), default="all")
    ap.add_argument("--force", action="store_true", help="reprocess even if cached")
    ap.add_argument("--dry-run", action="store_true", help="resolve only, no model calls")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files")
    ap.add_argument("--only", default="", help="comma-separated media_ids to process")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print("ERROR: dataset directory not found: %s" % args.dataset, file=sys.stderr)
        return 2

    index = MediaIndex(args.dataset)
    cache = InterpretationCache(args.cache)
    interpreter = MediaInterpreter()

    refs = collect_refs(args.dataset, SOURCES[args.scope])
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        refs = {k: v for k, v in refs.items() if k in wanted}

    print("Layer 1 - media preprocessing")
    print("  dataset : %s" % args.dataset)
    print("  cache   : %s (%d existing entries)" % (args.cache, len(cache)))
    print("  model   : %s" % (interpreter.model if interpreter.available else "-"))
    if not interpreter.available and not args.dry_run:
        print("  NOTE    : %s" % interpreter.unavailable_reason)
        print("            Placeholders will be produced and NOT cached; set a key and re-run.")
    print("  scope   : %s -> %d distinct media files" % (args.scope, len(refs)))
    print("")

    processed = skipped = 0
    results = []

    for media_id in sorted(refs):
        kind = refs[media_id]

        cached = cache.get(media_id)
        if cached is not None and not args.force:
            skipped += 1
            results.append(cached)
            if args.verbose:
                print("  skip  %-10s %-6s (cached: %s)" % (media_id, kind, cached.status))
            continue

        if args.limit and processed >= args.limit:
            break

        status, rel, abs_path, error = index.resolve(media_id, kind)
        if status != STATUS_OK:
            rec = MediaInterpretation.failure(media_id, kind, status, error, rel or "")
            cache.put(rec)                                     # permanent: worth caching
            results.append(rec)
            processed += 1
            print("  FAIL  %-10s %-6s %s (%s)" % (media_id, kind, status, error))
            continue

        data, read_error = MediaIndex.read_bytes(abs_path)
        if data is None:
            from schema import STATUS_UNREADABLE
            rec = MediaInterpretation.failure(
                media_id, kind, STATUS_UNREADABLE, read_error, rel
            )
            cache.put(rec)
            results.append(rec)
            processed += 1
            print("  FAIL  %-10s %-6s unreadable (%s)" % (media_id, kind, read_error))
            continue

        sha = MediaIndex.sha256(data)

        if cached is not None and cache.is_stale(media_id, sha):
            print("  stale %-10s file changed since cache was written; reprocessing" % media_id)

        if args.dry_run:
            print("  ok    %-10s %-6s %s (%d bytes) [dry-run]" % (media_id, kind, rel, len(data)))
            processed += 1
            continue

        rec = interpreter.interpret(media_id, kind, abs_path, rel, data, sha)
        cache.put(rec)
        results.append(rec)
        processed += 1

        if rec.status == STATUS_OK:
            preview = (rec.intent_summary or rec.visual_description or "")[:64]
            extra = (" [%s]" % rec.layout) if rec.kind == KIND_IMAGE else ""
            print("  ok    %-10s %-6s conf=%.2f%s %s" % (
                media_id, kind, rec.interp_confidence, extra, preview))
        else:
            print("  ---   %-10s %-6s %s (%s)" % (media_id, kind, rec.status, rec.error[:70]))

    if not args.dry_run:
        cache.save()

    print("")
    print("processed=%d skipped(cached)=%d cache_entries=%d" % (processed, skipped, len(cache)))
    by_status = cache.stats()
    if by_status:
        print("cache by status: %s" % ", ".join(
            "%s=%d" % kv for kv in sorted(by_status.items())))
    unresolved = [m for m in refs if m not in cache]
    if unresolved:
        print("NOT interpreted (%d): %s" % (len(unresolved), ", ".join(sorted(unresolved)[:12])))
        print("  -> these rows will route on metadata alone until a key is configured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
