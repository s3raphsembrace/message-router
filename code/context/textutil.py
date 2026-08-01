"""Small text helpers shared by retrieval and signal extraction.

Deliberately lexical rather than semantic. The dataset's repeated messages are
near-verbatim resends of the same template, so token overlap identifies them
reliably -- and unlike an embedding model it adds no dependency, no network call,
and no non-determinism.
"""

import re

_WORD = re.compile(r"[a-z0-9]+")
_MENTION = re.compile(r"@(u_\d+)")

# Template boilerplate that appears across unrelated business messages. Left in,
# it makes every "Dear Customer" message look similar to every other one.
_STOPWORDS = frozenset("""
a an the and or but if then this that these those is are was were be been being
to of for in on at by with from as your you we our us i me my it its
dear customer hi hello please pls kindly tap below view details check
""".split())


def tokens(text):
    """Lowercased content words, boilerplate removed."""
    if not text:
        return set()
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def jaccard(a, b):
    """Token-set overlap in [0, 1]."""
    ta, tb = (a if isinstance(a, set) else tokens(a)), (b if isinstance(b, set) else tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def mentions(text):
    """User ids explicitly @-addressed in the text.

    There is no name field anywhere in the dataset -- mentions are literal
    @user_id tokens -- so direct-address detection is exact matching, with no
    name resolution and no ambiguity.
    """
    return set(_MENTION.findall(text or ""))


def condense(text, limit):
    """Collapse whitespace and truncate to `limit` characters."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    # ASCII marker on purpose: this text reaches a Windows console, a JSON prompt,
    # and eventually output.csv, and a U+2026 survives none of those reliably.
    return flat[: max(0, limit - 3)].rstrip() + "..."
