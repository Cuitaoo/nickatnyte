"""Audience (department) guardrail for the final ranking pass.

The catalog is skewed roughly 2.4:1 toward women's products, so a men's or
boys' query can be outranked by women's products that match every other
signal. Semantic reranking does not fix this: a women's denim skirt is a
plausible neighbour of "men jeans" in embedding space, it is just the wrong
thing to sell.

This module infers the audience the shopper asked for, infers the audience a
product is sold to, and grades the mismatch. It is deliberately conservative:
when either side is unknown, or the shopper named two audiences at once, the
penalty is zero. It never filters, so a strong exact or identifier match can
still outrank the penalty.
"""

from __future__ import annotations

import re
from typing import Any

# (gender, age). Gender "n" is unisex/neutral, which never conflicts.
Audience = tuple[str, str]

UNKNOWN: Audience = ("", "")

_FAMILIES: dict[str, Audience] = {
    "mens": ("m", "adult"),
    "men": ("m", "adult"),
    "man": ("m", "adult"),
    "male": ("m", "adult"),
    "guys": ("m", "adult"),
    "womens": ("f", "adult"),
    "women": ("f", "adult"),
    "woman": ("f", "adult"),
    "female": ("f", "adult"),
    "ladies": ("f", "adult"),
    "boys": ("m", "child"),
    "boy": ("m", "child"),
    "girls": ("f", "child"),
    "girl": ("f", "child"),
    "baby-boys": ("m", "baby"),
    "baby-girls": ("f", "baby"),
    "baby": ("n", "baby"),
    "babies": ("n", "baby"),
    "infant": ("n", "baby"),
    "toddler": ("n", "baby"),
    "unisex": ("n", "adult"),
    "unisex-adult": ("n", "adult"),
    "unisex-child": ("n", "child"),
    "unisex-baby": ("n", "baby"),
    "kids": ("n", "child"),
    "kid": ("n", "child"),
    "children": ("n", "child"),
    "child": ("n", "child"),
    "youth": ("n", "child"),
}

# `details` reaches us as one flattened lowercase string, so the department is
# read positionally rather than from a dict key.
_DEPARTMENT_RE = re.compile(r"department\s+([a-z][a-z\-]*)")

_TOKEN_RE = re.compile(
    r"\b(mens|men|man|male|guys|womens|women|woman|female|ladies"
    r"|boys|boy|girls|girl|baby-boys|baby-girls|babies|baby|infant|toddler"
    r"|unisex-adult|unisex-child|unisex-baby|unisex"
    r"|kids|kid|children|child|youth)\b"
)

# "men's relaxed fit jeans" - the possessive is a reliable last resort.
_POSSESSIVE_RE = re.compile(r"\b(men|women|boy|girl|kid|baby|toddler)'?s\b")

# Style names and colors that contain an audience word but name no audience.
# "panties boy shorts" is womenswear; reading it as boys' clothing inverts the
# guardrail and penalizes every correct result.
_STYLE_NOISE_RE = re.compile(
    r"\b(?:boy\s?shorts?|boy\s?leg|baby\s?doll|baby\s?blue|baby\s?pink"
    r"|baby\s?powder|mom\s?jeans?|girl\s?friend|boy\s?friend)\b"
)


def _strip_style_noise(text: str) -> str:
    return _STYLE_NOISE_RE.sub(" ", text)


def normalize_audience(raw: object) -> Audience:
    """Map a raw department/token string onto a canonical (gender, age) pair."""
    if not raw:
        return UNKNOWN
    key = str(raw).strip().lower().replace("_", "-").replace(" ", "-")
    if key in _FAMILIES:
        return _FAMILIES[key]
    # "baby-girls-clothing" and similar compounds.
    match = _TOKEN_RE.search(key.replace("-", " "))
    if match:
        return _FAMILIES.get(match.group(1), UNKNOWN)
    return UNKNOWN


def _scan(text: object) -> Audience:
    """Return the single audience named in `text`, or UNKNOWN if 0 or 2+."""
    if not text:
        return UNKNOWN
    found = {
        _FAMILIES[token]
        for token in _TOKEN_RE.findall(_strip_style_noise(str(text).lower()))
        if token in _FAMILIES
    }
    if len(found) == 1:
        return next(iter(found))
    return UNKNOWN


def product_audience(meta: dict[str, Any]) -> Audience:
    """Infer who a product is sold to, best signal first."""
    if not meta:
        return UNKNOWN

    details = meta.get("details")
    if details:
        match = _DEPARTMENT_RE.search(str(details).lower())
        if match:
            audience = normalize_audience(match.group(1))
            if audience != UNKNOWN:
                return audience

    audience = _scan(meta.get("categories"))
    if audience != UNKNOWN:
        return audience

    title = _strip_style_noise(str(meta.get("title") or "").lower())
    match = _POSSESSIVE_RE.search(title)
    if match:
        return _FAMILIES.get(match.group(1), UNKNOWN)
    return UNKNOWN


def requested_audience(state: Any, latest_message: str = "") -> Audience:
    """Infer the audience the shopper asked for.

    Re-derived every turn rather than stored: `state.category` already
    persists across turns and is rewritten on intent override, so reading it
    is sticky and self-correcting without a new state field.
    """
    audience = _scan(getattr(state, "category", None))
    if audience != UNKNOWN:
        return audience

    audience = _scan(latest_message)
    if audience != UNKNOWN:
        return audience

    preferences = getattr(state, "preferences", None) or {}
    values: list[str] = []
    for entry in preferences.values():
        if isinstance(entry, (list, tuple)):
            values.extend(str(value) for value in entry)
        else:
            values.append(str(entry))
    return _scan(" ".join(values))


def audience_penalty(requested: Audience, product: Audience) -> float:
    """Grade a mismatch as a fraction of the full penalty.

    1.0 when the gender is known on both sides and differs; 0.5 when the
    gender agrees but the age band does not (mens vs boys); 0.0 otherwise.
    """
    if requested == UNKNOWN or product == UNKNOWN or requested == product:
        return 0.0

    req_gender, req_age = requested
    prod_gender, prod_age = product

    if req_gender and prod_gender and "n" not in (req_gender, prod_gender):
        if req_gender != prod_gender:
            return 1.0
    if req_age and prod_age and req_age != prod_age:
        return 0.5
    return 0.0


def apply_audience_guardrail(
    ordered_ids: list[str],
    metadata: dict[str, dict[str, Any]],
    state: Any,
    latest_message: str = "",
    penalty: float = 0.15,
    top_n: int = 20,
) -> list[str]:
    """Demote wrong-audience products within the head of the ranking.

    Only the first `top_n` entries are reordered; the tail is spliced back
    untouched. Position within the window is scored (N-i)/N, so `penalty` is
    readable as the fraction of the window a full mismatch can fall.
    """
    if not ordered_ids or penalty <= 0 or top_n <= 0:
        return ordered_ids

    requested = requested_audience(state, latest_message)
    if requested == UNKNOWN:
        return ordered_ids

    window = ordered_ids[:top_n]
    tail = ordered_ids[top_n:]

    scored = []
    for index, product_id in enumerate(window):
        # Normalized by top_n, not by the window length, so one position is
        # always worth 1/top_n and `penalty` moves a fixed number of slots
        # however few candidates came back.
        base = (top_n - index) / top_n
        grade = audience_penalty(requested, product_audience(metadata.get(product_id) or {}))
        scored.append((base - grade * penalty, index, product_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [product_id for _, _, product_id in scored] + tail
