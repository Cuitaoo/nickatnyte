from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from starter.state import ALLOWED_PREFERENCE_ATTRIBUTES, ShoppingState


SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9$]+", re.IGNORECASE)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work", "walking")
ATTRIBUTE_CONSTRAINT_RE = re.compile(
    r"^(?:rubber|textile|synthetic|leather|foam|eva)?\s*"
    r"(?:sole|outsole|upper|closure|lining|footbed|heel|strap|sleeve)$",
    re.IGNORECASE,
)
FALLBACK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "actually",
        "but",
        "earlier",
        "for",
        "have",
        "i",
        "ignore",
        "im",
        "is",
        "looking",
        "me",
        "my",
        "need",
        "please",
        "preference",
        "the",
        "this",
        "to",
        "want",
        "what",
        "with",
    }
)


class PreferenceValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: str
    value: str


class PreferenceRemoval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: str
    value: str | None = None


class PreferencePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_mode: Literal["buying", "browsing", "unknown", "unchanged"] = "unchanged"
    category: str = "unchanged"
    set_preferences: list[PreferenceValue] = Field(default_factory=list)
    remove_preferences: list[PreferenceRemoval] = Field(default_factory=list)
    no_preference_attributes: list[str] = Field(default_factory=list)
    reset_product_preferences: bool = False
    search_terms: list[str] = Field(default_factory=list)


def normalize_value(value: object) -> str:
    return SPACE_RE.sub(" ", str(value)).strip(" \t\r\n,.;:-").lower()


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _looks_like_attribute_constraint(value: str) -> bool:
    normalized = normalize_value(value)
    return (
        normalized in COLORS
        or normalized in MATERIALS
        or normalized in USE_CASES
        or bool(ATTRIBUTE_CONSTRAINT_RE.fullmatch(normalized))
    )


def _canonical_preference_values(attribute: str, value: str) -> list[str]:
    vocabularies = {
        "material": MATERIALS,
        "color": COLORS,
        "use_case": USE_CASES,
    }
    vocabulary = vocabularies.get(attribute)
    if vocabulary is None:
        return [value]
    matches = [
        candidate
        for candidate in vocabulary
        if re.search(rf"\b{re.escape(candidate)}\b", value, re.IGNORECASE)
    ]
    return matches or [value]


def apply_preference_patch(
    state: ShoppingState, patch: PreferencePatch
) -> ShoppingState:
    category = state.category
    preferences = {key: list(values) for key, values in state.preferences.items()}
    removed = {key: list(values) for key, values in state.removed_preferences.items()}
    no_preference = set(state.no_preference_attributes)
    search_terms = list(state.search_terms)

    asked_attributes = state.asked_attributes
    previous_ask_attribute = state.previous_ask_attribute
    latest_recommendations = state.latest_recommendations

    patch_category = normalize_value(patch.category)
    if patch.reset_product_preferences:
        preferences.clear()
        removed.clear()
        no_preference.clear()
        search_terms.clear()
        asked_attributes = ()
        previous_ask_attribute = None
        latest_recommendations = ()

    for removal in patch.remove_preferences:
        attribute = normalize_value(removal.attribute)
        if attribute not in ALLOWED_PREFERENCE_ATTRIBUTES:
            continue
        value = normalize_value(removal.value) if removal.value is not None else ""
        if attribute == "category":
            if not value or category == value:
                category = None
        elif attribute in preferences:
            if value:
                preferences[attribute] = [
                    item for item in preferences[attribute] if item != value
                ]
                if not preferences[attribute]:
                    preferences.pop(attribute)
            else:
                preferences.pop(attribute)
        if value:
            bucket = removed.setdefault(attribute, [])
            _append_unique(bucket, value)
            search_terms = [term for term in search_terms if term != value]

    for raw_attribute in patch.no_preference_attributes:
        attribute = normalize_value(raw_attribute)
        if attribute not in ALLOWED_PREFERENCE_ATTRIBUTES:
            continue
        no_preference.add(attribute)
        if attribute == "category":
            category = None
        else:
            preferences.pop(attribute, None)

    intent_mode = state.intent_mode
    if patch.intent_mode != "unchanged":
        intent_mode = patch.intent_mode

    if patch.reset_product_preferences and _looks_like_attribute_constraint(
        patch_category
    ):
        patch_category = "unchanged"
    if patch_category and patch_category != "unchanged":
        category = patch_category
        no_preference.discard("category")

    for item in patch.set_preferences:
        attribute = normalize_value(item.attribute)
        value = normalize_value(item.value)
        if attribute not in ALLOWED_PREFERENCE_ATTRIBUTES or not value:
            continue
        if attribute == "category":
            category = value
            no_preference.discard("category")
            continue
        no_preference.discard(attribute)
        bucket = preferences.setdefault(attribute, [])
        for canonical_value in _canonical_preference_values(attribute, value):
            if canonical_value in removed.get(attribute, []):
                removed[attribute].remove(canonical_value)
                if not removed[attribute]:
                    removed.pop(attribute)
            _append_unique(bucket, canonical_value)

    rejected_values = {value for values in removed.values() for value in values}
    for raw_term in patch.search_terms:
        term = normalize_value(raw_term)
        if term and term not in rejected_values:
            _append_unique(search_terms, term)

    return replace(
        state,
        intent_mode=intent_mode,
        category=category,
        preferences={key: tuple(values) for key, values in preferences.items()},
        removed_preferences={key: tuple(values) for key, values in removed.items()},
        no_preference_attributes=frozenset(no_preference),
        search_terms=tuple(search_terms),
        asked_attributes=asked_attributes,
        previous_ask_attribute=previous_ask_attribute,
        latest_recommendations=latest_recommendations,
    )


def _first_match(pattern: str, message: str) -> str | None:
    match = re.search(pattern, message, re.IGNORECASE)
    return normalize_value(match.group(1)) if match else None


def parse_preference_fallback(
    message: str, state: ShoppingState
) -> PreferencePatch:
    lowered = normalize_value(message)
    override = bool(
        re.search(
            r"\b(ignore|forget|instead|changed my mind|not .+ anymore)\b", lowered
        )
    )
    browsing = bool(re.search(r"\b(browsing|exploring|not sure|just looking)\b", lowered))
    buying = bool(re.search(r"\b(looking for|i need|i want|requirement|must have)\b", lowered))
    intent_mode: Literal["buying", "browsing", "unknown", "unchanged"] = "unchanged"
    if browsing:
        intent_mode = "browsing"
    elif buying or override:
        intent_mode = "buying"

    category = "unchanged"
    extracted_category = _first_match(
        r"\blooking for\s+(.+?)(?=\s+(?:but|with|under|below|for)\b|[.,;]|$)",
        lowered,
    )
    if override:
        extracted_category = _first_match(
            r"(?:what i need is|instead|changed my mind(?:,|:)?(?: i need)?)\s*[:,-]?\s*(.+?)(?=[.;]|$)",
            lowered,
        ) or extracted_category
    if extracted_category:
        category = extracted_category

    values: list[PreferenceValue] = []
    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", lowered):
            values.append(PreferenceValue(attribute="color", value=color))
    for material in MATERIALS:
        if re.search(rf"\b{re.escape(material)}\b", lowered):
            values.append(PreferenceValue(attribute="material", value=material))
    budget = _first_match(
        r"\b((?:under|below|less than|up to)\s*\$?\s*\d+(?:\.\d{1,2})?)\b",
        lowered,
    )
    if budget:
        budget = re.sub(
            r"\b(under|below|less than|up to)\b\s*\$?\s*(\d+(?:\.\d{1,2})?)",
            r"\1 $\2",
            budget,
        )
        values.append(PreferenceValue(attribute="budget", value=budget))
    for use_case in USE_CASES:
        if re.search(rf"\b{re.escape(use_case)}\b", lowered):
            values.append(PreferenceValue(attribute="use_case", value=use_case))

    no_preference: list[str] = []
    no_preference_match = re.search(
        r"(?:no|don'?t have (?:an? )?)\s*preference\s+for\s+([a-z_]+)", lowered
    )
    if no_preference_match:
        no_preference.append(no_preference_match.group(1))
    elif (
        state.previous_ask_attribute
        and re.search(r"\b(no preference|don'?t care|use your judgment)\b", lowered)
    ):
        no_preference.append(state.previous_ask_attribute)

    search_terms: list[str] = []
    for token in TOKEN_RE.findall(lowered):
        if len(token) > 1 and token not in FALLBACK_STOPWORDS:
            _append_unique(search_terms, token)

    return PreferencePatch(
        intent_mode=intent_mode,
        category=category,
        set_preferences=values,
        no_preference_attributes=no_preference,
        reset_product_preferences=override,
        search_terms=search_terms[:24],
    )
