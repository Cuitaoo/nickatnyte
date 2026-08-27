from __future__ import annotations

import re
from typing import Any, Iterable

from starter.state import ShoppingState


RECOMMENDATION_MESSAGE = "Here are the closest matches I found."
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ATTRIBUTE_PRIORITY = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
QUESTION_TEMPLATES = {
    "category": "What kind of product are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Is there a color you prefer?",
    "size": "What size or fit should I prioritize?",
    "style": "What style do you have in mind?",
    "brand": "Do you prefer any particular brand?",
    "budget": "What budget range should I use?",
    "feature": "Which product feature matters most to you?",
    "use_case": "What will you mainly use the product for?",
    "other": "Is there another must-have detail I should consider?",
}

MATERIALS = frozenset(
    {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
)
COLORS = frozenset(
    {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
)
USE_CASES = frozenset(
    {"hiking", "running", "gym", "winter", "outdoor", "work", "walking", "travel", "marathon", "sports", "workout"}
)
STYLE_TERMS = frozenset(
    {
        "ankle",
        "bootie",
        "heel",
        "platform",
        "lace",
        "slip",
        "v",
        "neck",
        "sleeve",
        "crew",
        "bifold",
        "low",
        "cut",
        "thigh",
        "high",
        "plus",
        "skinny",
        "straight",
        "relaxed",
    }
)
SIZE_TERMS = frozenset({"size", "fit", "wide", "narrow", "small", "medium", "large", "xl", "plus"})
FEATURE_STOPWORDS = frozenset(
    {
        "clothing",
        "shoes",
        "jewelry",
        "women",
        "men",
        "for",
        "and",
        "with",
        "the",
        "imported",
        "closure",
        "sole",
        "rubber",
        "machine",
        "wash",
        "hand",
        "only",
        "department",
        "made",
        "usa",
        "measures",
        "approximately",
    }
)
QUESTION_PRIORS = {
    "category": 0.20,
    "material": 0.22,
    "color": 0.12,
    "size": 0.10,
    "style": 0.30,
    "brand": 0.06,
    "budget": 0.08,
    "feature": 0.18,
    "use_case": 0.25,
}


def _field_text(product: dict[str, Any], fields: tuple[str, ...]) -> str:
    values = [str(product.get(field, "")) for field in fields]
    if not any(value.strip() for value in values):
        values = [str(product.get("corpus", ""))]
    return " ".join(values).lower()


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value)}


def _signature(product: dict[str, Any], attribute: str) -> str:
    if attribute == "feature":
        text = _field_text(product, ("features", "details", "description"))
    elif attribute == "style":
        text = _field_text(product, ("title", "categories", "details"))
    else:
        text = _field_text(product, ("title", "categories", "corpus"))
    tokens = _tokens(text)
    if attribute == "category":
        return str(product.get("categories", "")).lower()
    if attribute == "material":
        return " ".join(sorted(tokens & MATERIALS))
    if attribute == "color":
        return " ".join(sorted(tokens & COLORS))
    if attribute == "use_case":
        return " ".join(sorted(tokens & USE_CASES))
    if attribute == "style":
        return " ".join(sorted(tokens & STYLE_TERMS))
    if attribute == "size":
        return " ".join(sorted(tokens & SIZE_TERMS))
    if attribute == "brand":
        return " ".join(TOKEN_RE.findall(str(product.get("store", "")).lower())[:3])
    if attribute == "budget":
        price = product.get("price")
        if price is None:
            return ""
        try:
            return f"band{int(float(price) // 25)}"
        except (TypeError, ValueError):
            return ""
    if attribute == "feature":
        tokens = [
            token
            for token in TOKEN_RE.findall(text)
            if len(token) > 3
            and token.lower() not in FEATURE_STOPWORDS
            and token.lower() not in STYLE_TERMS
            and token.lower() not in MATERIALS
            and token.lower() not in COLORS
            and token.lower() not in USE_CASES
        ]
        return " ".join(tokens[:8])
    return ""


def _score_attribute(attribute: str, candidates: list[dict[str, Any]]) -> float:
    signatures = [_signature(product, attribute) for product in candidates]
    present = [signature for signature in signatures if signature]
    if not candidates or not present:
        return 0.0
    coverage = len(present) / len(candidates)
    disagreement = min(1.0, len(set(present)) / max(2, len(present)))
    expected_answer_usefulness = QUESTION_PRIORS[attribute]
    if attribute == "feature":
        distinctive = sum(
            1
            for signature in present
            if any(token not in FEATURE_STOPWORDS for token in signature.split())
        ) / len(present)
        if distinctive < 0.35:
            return 0.0
        coverage = min(coverage, 0.60)
    scenario_relevance = min(1.0, expected_answer_usefulness + 0.20 * coverage)
    return (
        0.40 * disagreement
        + 0.25 * coverage
        + 0.20 * expected_answer_usefulness
        + 0.15 * scenario_relevance
    )


def _eligible_attributes(state: ShoppingState) -> list[str]:
    excluded = set(state.asked_attributes)
    excluded.update(state.no_preference_attributes)
    excluded.update(state.preferences)
    if state.category:
        excluded.add("category")
    return [attribute for attribute in ATTRIBUTE_PRIORITY if attribute not in excluded]


def choose_clarification(
    state: ShoppingState,
    turn: int,
    candidate_products: Iterable[dict[str, Any]] | None = None,
) -> tuple[str, str | None]:
    if turn >= 10:
        return RECOMMENDATION_MESSAGE, None

    eligible = _eligible_attributes(state)
    candidates = list(candidate_products or ())
    if candidates:
        scored = [
            (_score_attribute(attribute, candidates), attribute)
            for attribute in eligible
            if attribute != "other"
        ]
        scored = [(score, attribute) for score, attribute in scored if score > 0.0]
        if scored:
            _, attribute = max(scored, key=lambda item: (item[0], -ATTRIBUTE_PRIORITY.index(item[1])))
            return QUESTION_TEMPLATES[attribute], attribute

    for attribute in eligible:
        if attribute == "other" or not candidates:
            return QUESTION_TEMPLATES[attribute], attribute
    return RECOMMENDATION_MESSAGE, None
