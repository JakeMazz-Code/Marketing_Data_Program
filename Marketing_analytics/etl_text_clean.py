"""Helpers for normalising text fields used by ETL outputs."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

# Known mojibake patterns observed in Shopify exports (UTF-8 misreads)
_MOJIBAKE_REPLACEMENTS = {
    # "No√≠re" -> "Noire" (common Shopify export artefact)
    "\u221a\u2260": "i",  # \N{SQUARE ROOT}\N{NOT EQUAL TO} -> "i"
    "\u00c2\u00a0": " ",     # non-breaking space artefacts
}

_REPEATING_SPACE = re.compile(r"\s+")


def _apply_mojibake_fixes(text: str) -> str:
    fixed = text
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        fixed = fixed.replace(bad, good)
    fixed = fixed.replace("\ufffd", "")  # striped replacement chars
    return fixed


def normalize_product_title(value: object) -> str:
    """Return a clean, human-friendly product title string."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\ufeff", "")
    text = _apply_mojibake_fixes(text)
    text = unicodedata.normalize("NFKC", text)
    text = _REPEATING_SPACE.sub(" ", text)
    return text.strip()


def split_product_fields(value: object) -> Tuple[str, Optional[str]]:
    """Return (base_title, variant) from a raw product title."""
    title = normalize_product_title(value)
    if not title:
        return "Unnamed Product", None

    base = title
    variant: Optional[str] = None

    if " - " in title:
        candidate_base, candidate_variant = title.rsplit(" - ", 1)
        candidate_variant = candidate_variant.strip()
        if candidate_variant and len(candidate_variant) <= 48:
            base = candidate_base.strip() or base
            variant = candidate_variant

    if variant is None and "(" in base and base.endswith(")"):
        candidate_base, candidate_variant = base.rsplit("(", 1)
        candidate_variant = candidate_variant.strip().strip(")").strip()
        if candidate_variant:
            base = candidate_base.strip() or base
            variant = candidate_variant

    base = base or "Unnamed Product"
    return base, variant
