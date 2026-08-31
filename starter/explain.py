"""Deterministic recommendation explanations.

Every ranking decision already leaves evidence behind - which preference
attributes a product matched, which profile tags fired, whether an exact
identifier or category hit. `RankedCandidate` has carried
`score_components`, `matched_attributes` and `matched_profile_tags` all along
and nothing has ever read them.

This turns that evidence into a sentence. It is generated from the recorded
match data, never from a model and never from the ranking scores: a score of
2.47 explains nothing to a shopper, and asking an LLM to invent a reason after
the fact produces plausible text with no relationship to why the product
actually ranked.

Ordering is by how strongly the evidence bound the decision - identifier, then
category, then stated preferences, then the profile - so the first clause is
the most load-bearing reason.
"""

from __future__ import annotations

from typing import Any
from starter import config

PREFIX = "Recommended because it matches"
PROFILE_CLAUSE = "your profile's {tags} preference"
NO_EVIDENCE = "Here are the closest matches I found."


def explanations_enabled() -> bool:
    return _env_bool("TECHJAM_EXPLANATIONS", False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = config.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _join(items: list[str]) -> str:
    """Oxford-comma join: a; a and b; a, b and c."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _component_names(candidate: Any) -> set[str]:
    return {
        name
        for name, value in getattr(candidate, "score_components", ()) or ()
        if value
    }


def evidence_for(candidate: Any, state: Any) -> list[str]:
    """The reasons this product ranked, strongest binding first."""
    if candidate is None:
        return []

    reasons: list[str] = []
    components = _component_names(candidate)

    if "exact_identifier" in components:
        reasons.append("the exact model number you gave")

    category = getattr(state, "category", None)
    if category and "category" in components:
        reasons.append(str(category))

    # Stated preferences, in the order the shopper's own attributes are held.
    matched = set(getattr(candidate, "matched_attributes", frozenset()))
    preferences = getattr(state, "preferences", None) or {}
    for attribute, values in preferences.items():
        if attribute == "category" or attribute not in matched:
            continue
        for value in values:
            text = str(value).strip()
            if text and text not in reasons:
                reasons.append(text)

    return reasons


def explain(candidate: Any, state: Any) -> str:
    """One sentence naming why this product was recommended."""
    reasons = evidence_for(candidate, state)
    tags = list(getattr(candidate, "matched_profile_tags", ()) or ())

    if not reasons and not tags:
        return NO_EVIDENCE

    parts: list[str] = []
    if reasons:
        parts.append(f"{PREFIX} {_join(reasons)}")
    if tags:
        clause = PROFILE_CLAUSE.format(tags=_join(tags))
        # The profile is a weaker signal than anything stated, so it is
        # always appended rather than leading.
        parts.append(clause if reasons else f"{PREFIX} {clause}")

    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}."
    return f"{parts[0]}."


def explain_top(
    identifiers: list[str],
    candidates: Any,
    state: Any,
    fallback: str,
) -> str:
    """Explain the product that ended up first, or fall back to `fallback`."""
    if not identifiers:
        return fallback
    leader = identifiers[0]
    for candidate in candidates or ():
        if getattr(candidate, "product_id", None) == leader:
            sentence = explain(candidate, state)
            return sentence if sentence != NO_EVIDENCE else fallback
    return fallback
