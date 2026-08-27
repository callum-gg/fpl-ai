"""Name normalisation shared by every resolver. Accent-stripped, punctuation-free, lowercase."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_SUFFIXES = {"jr", "junior", "snr", "senior", "ii", "iii"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    s = strip_accents(s).lower().replace("'", "").replace("-", " ")
    s = _PUNCT.sub(" ", s)
    parts = [p for p in _WS.sub(" ", s).strip().split(" ") if p and p not in _SUFFIXES]
    return " ".join(parts)


def name_variants(first: str | None, last: str | None, web: str | None, full: str | None) -> set[str]:
    """Every reasonable surface form for a player, for alias seeding and lookup."""
    out: set[str] = set()
    for v in (full, web, last, f"{first or ''} {last or ''}"):
        n = norm_name(v)
        if n:
            out.add(n)
    if first and last:
        out.add(norm_name(f"{first[0]} {last}"))
    return out
