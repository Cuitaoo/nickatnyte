"""Constraint strength and staged hard filtering for the buying path.

Every stated preference is currently treated the same way: a boost of
`confirmed_attribute_boost` added to a score. That is the right shape for a
browsing shopper who is describing a direction, and the wrong shape for a
buying shopper who opened with "a key requirement is: 100% cotton". A
requirement is not a nudge. It should remove products, not merely re-rank them.

Strength is read from how the evidence arrived, which is already recorded:

- `unsolicited` - the shopper volunteered it, so it is what they came for
- `correction`  - they insisted on it after we got it wrong
- `clarification` - they answered a question we asked, which is weaker: the
  shopper is being cooperative, not stating a requirement

Filtering is dangerous in a way boosting is not: a false negative removes the
target and costs Hit@10, which carries 0.50 of the score. Two rules keep that
in check. A product is dropped only when we can affirmatively see it violates
the constraint - missing catalog metadata never excludes anything - and the
filter relaxes itself, dropping its least reliable constraint whenever the
surviving pool gets too small, rather than filtering down to nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

HARD = "hard"
SOFT = "soft"

# How much to trust a constraint of each kind when deciding what to relax
# first. Numeric and categorical constraints are checked precisely; free-text
# ones are substring matches against product prose and misfire most, so they
# are surrendered first.
ATTRIBUTE_CONFIDENCE: dict[str, float] = {
    "category": 0.95,
    "budget": 0.90,
    "brand": 0.85,
    "material": 0.80,
    "color": 0.75,
    "size": 0.60,
    "style": 0.50,
    "use_case": 0.50,
    "feature": 0.40,
    "other": 0.30,
}
DEFAULT_CONFIDENCE = 0.45

# Evidence that means the shopper stated a requirement rather than answered us.
HARD_SOURCES = frozenset({"unsolicited", "correction"})


@dataclass(frozen=True)
class Constraint:
    attribute: str
    values: tuple[str, ...]
    strength: str
    confidence: float

    @property
    def is_hard(self) -> bool:
        return self.strength == HARD


def staged_filter_enabled() -> bool:
    return _env_bool("TECHJAM_STAGED_FILTER", False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def min_pool() -> int:
    """Floor below which the filter relaxes rather than narrows further.

    Set well above the scored top 10 so a filter can never be the reason the
    target is missing from the returned list.
    """
    return _env_int("TECHJAM_STAGED_FILTER_MIN_POOL", 40, 1, 500)


def classify_constraints(state: Any) -> tuple[Constraint, ...]:
    """Split the shopper's stated preferences into hard and soft.

    The category is always hard when known: a shopper looking for jeans is not
    expressing a preference for jeans.
    """
    evidence_by_attribute: dict[str, set[str]] = {}
    for item in getattr(state, "preference_evidence", ()) or ():
        evidence_by_attribute.setdefault(item.attribute, set()).add(item.source_kind)

    constraints: list[Constraint] = []
    for attribute, values in sorted((getattr(state, "preferences", None) or {}).items()):
        if not values:
            continue
        sources = evidence_by_attribute.get(attribute, set())
        # No recorded evidence means it predates the evidence trail; treat it
        # as soft rather than inventing a requirement.
        strength = HARD if sources & HARD_SOURCES else SOFT
        constraints.append(
            Constraint(
                attribute=attribute,
                values=tuple(values),
                strength=strength,
                confidence=ATTRIBUTE_CONFIDENCE.get(attribute, DEFAULT_CONFIDENCE),
            )
        )
    return tuple(constraints)


def violates(
    constraint: Constraint,
    product: dict[str, Any],
    matches: Callable[[str, str, dict[str, Any]], bool],
    has_metadata: Callable[[str, dict[str, Any]], bool],
) -> bool:
    """Whether this product can be shown to breach the constraint.

    Absence of evidence is not evidence of violation: when the product carries
    no text for the attribute at all, it is kept. Only a product that has the
    relevant metadata and still fails to match is dropped.
    """
    if not has_metadata(constraint.attribute, product):
        return False
    return not any(matches(constraint.attribute, value, product) for value in constraint.values)


def staged_filter(
    candidates: tuple,
    constraints: tuple[Constraint, ...],
    metadata: dict[str, dict[str, Any]],
    matches: Callable[[str, str, dict[str, Any]], bool],
    has_metadata: Callable[[str, dict[str, Any]], bool],
    floor: int | None = None,
) -> tuple[tuple, tuple[str, ...]]:
    """Filter to products satisfying every hard constraint, relaxing as needed.

    Returns the surviving candidates and the attributes that had to be
    surrendered. Least reliable constraints are given up first; if even a
    single constraint cannot leave `floor` candidates standing, the unfiltered
    pool is returned rather than a starved one.
    """
    hard = [c for c in constraints if c.is_hard]
    if not candidates or not hard:
        return candidates, ()

    limit = min_pool() if floor is None else floor
    # Most reliable first, so the ones we surrender come off the end.
    active = sorted(hard, key=lambda c: (-c.confidence, c.attribute))
    relaxed: list[str] = []

    while active:
        kept = tuple(
            candidate
            for candidate in candidates
            if not any(
                violates(constraint, metadata.get(candidate.product_id) or {}, matches, has_metadata)
                for constraint in active
            )
        )
        if len(kept) >= limit:
            return kept, tuple(relaxed)
        relaxed.append(active[-1].attribute)
        active = active[:-1]

    return candidates, tuple(relaxed)
