"""Build code.zip for submission.

    python code/package.py

Includes the runnable solution, prompts, configs, README, evaluation harness and
the caches that make a re-run reproducible offline.

Explicitly EXCLUDES .env, so a real key can never leave the machine inside the
submission. The exclusion is asserted after the archive is written rather than
merely intended.
"""

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_ZIP = os.path.join(REPO_ROOT, "code.zip")

# Directories and files to include, relative to the repo root.
INCLUDE_TREES = ("code", "cache")
INCLUDE_FILES = ("README.md", "problem_statement.md", "AGENTS.md", ".env.example",
                 ".gitignore")

# Never packaged, whatever else matches.
EXCLUDE_NAMES = {".env", ".env.local"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".tmp", ".key")

# Anything matching these must not appear in the archive. Checked after writing.
SECRET_MARKERS = ("/.env", "\\.env")


def _should_skip(path):
    name = os.path.basename(path)
    if name in EXCLUDE_NAMES:
        return True
    if name.endswith(EXCLUDE_SUFFIXES):
        return True
    return False


def collect(root):
    members = []
    for rel in INCLUDE_FILES:
        full = os.path.join(root, rel)
        if os.path.exists(full) and not _should_skip(full):
            members.append((full, rel))
    for tree in INCLUDE_TREES:
        base = os.path.join(root, tree)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for filename in sorted(filenames):
                full = os.path.join(dirpath, filename)
                if _should_skip(full):
                    continue
                members.append((full, os.path.relpath(full, root).replace(os.sep, "/")))
    return sorted(set(members), key=lambda t: t[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build code.zip")
    ap.add_argument("--out", default=DEFAULT_ZIP)
    args = ap.parse_args(argv)

    members = collect(REPO_ROOT)
    if not members:
        print("ERROR: nothing to package", file=sys.stderr)
        return 2

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arcname in members:
            z.write(full, arcname)

    # Verify rather than assume: re-open and prove no secret slipped in.
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
    leaked = [n for n in names
              if os.path.basename(n) in EXCLUDE_NAMES
              or any(m in ("/" + n) for m in SECRET_MARKERS)]
    if leaked:
        os.unlink(args.out)
        print("ERROR: refusing to ship, archive contained %s" % ", ".join(leaked),
              file=sys.stderr)
        return 1

    size = os.path.getsize(args.out)
    print("wrote %s" % args.out)
    print("  files      : %d" % len(names))
    print("  size       : %.1f KB" % (size / 1024.0))
    print("  .env       : excluded and verified absent")
    top = {}
    for n in names:
        top[n.split("/")[0]] = top.get(n.split("/")[0], 0) + 1
    print("  contents   : %s" % ", ".join("%s (%d)" % kv for kv in sorted(top.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
