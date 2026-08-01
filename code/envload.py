"""Minimal .env loader.

Secrets are read from the environment. This adds one convenience: if a .env file
sits at the repo root, its values are loaded into os.environ for keys that are not
already set, so a real shell export always wins over the file.

Deliberately dependency-free and deliberately non-overriding -- a stray .env must
never silently shadow a key the operator exported on purpose.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENV_PATH = os.path.join(REPO_ROOT, ".env")


def load_env(path=DEFAULT_ENV_PATH, override=False):
    """Load KEY=VALUE lines into os.environ. Returns the names it set."""
    if not os.path.exists(path):
        return []
    loaded = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key or (not override and os.environ.get(key)):
            continue
        if not value:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
