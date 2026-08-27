from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from starter.state import (
    ALLOWED_PREFERENCE_ATTRIBUTES,
    PreferenceEvidence,
    ShoppingState,
)


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
DIRECT_ANSWER_RE = re.compile(
    r"^(?:for that,?\s*)?what matters is\s*:\s*(.+)$",
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


def _normalized_patch_values(
    patch: PreferencePatch,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in patch.set_preferences:
        attribute = normalize_value(item.attribute)
        value = normalize_value(item.value)
        if (
            attribute not in ALLOWED_PREFERENCE_ATTRIBUTES
            or attribute == "category"
            or not value
        ):
            continue
        bucket = grouped.setdefault(attribute, [])
        for canonical_value in _canonical_preference_values(attribute, value):
            _append_unique(bucket, canonical_value)
    return {attribute: tuple(values) for attribute, values in grouped.items()}


def _evidence_source(
    state: ShoppingState,
    attributes: set[str],
    *,
    correction: bool,
    allow_clarification: bool = True,
) -> Literal["unsolicited", "clarification", "correction"]:
    if correction:
        return "correction"
    if allow_clarification and state.previous_ask_attribute in attributes:
        return "clarification"
    return "unsolicited"


UpdateKind = Literal["ordinary", "preference_correction", "product_change"]


def _classify_update(
    state: ShoppingState,
    patch: PreferencePatch,
    patch_category: str,
    values_by_attribute: dict[str, tuple[str, ...]],
) -> UpdateKind:
    if not patch.reset_product_preferences:
        return "ordinary"
    current_category = normalize_value(state.category) if state.category else ""
    keeps_category = (
        patch_category == "unchanged"
        or patch_category == current_category
        or _looks_like_attribute_constraint(patch_category)
    )
    if state.category and keeps_category and values_by_attribute:
        return "preference_correction"
    return "product_change"


def _without_attributes(
    evidence: list[PreferenceEvidence], attributes: set[str]
) -> tuple[list[PreferenceEvidence], list[PreferenceEvidence]]:
    active = [item for item in evidence if item.attribute not in attributes]
    retired = [item for item in evidence if item.attribute in attributes]
    return active, retired


def _retire_latest_unsolicited(
    evidence: list[PreferenceEvidence],
    excluded_attributes: set[str],
) -> tuple[list[PreferenceEvidence], PreferenceEvidence | None]:
    for index in range(len(evidence) - 1, -1, -1):
        item = evidence[index]
        if item.source_kind == "unsolicited" and item.attribute not in excluded_attributes:
            return evidence[:index] + evidence[index + 1 :], item
    return evidence, None


def _remove_evidence_values(
    evidence: list[PreferenceEvidence],
    attribute: str,
    values: set[str] | None,
) -> tuple[list[PreferenceEvidence], list[PreferenceEvidence]]:
    active: list[PreferenceEvidence] = []
    retired: list[PreferenceEvidence] = []
    for item in evidence:
        if item.attribute != attribute:
            active.append(item)
            continue
        if values is None:
            retired.append(item)
            continue
        removed_values = tuple(value for value in item.values if value in values)
        removed_terms = tuple(
            term
            for term in item.terms
            if any(value == term or value in term for value in values)
        )
        remaining_values = tuple(value for value in item.values if value not in values)
        remaining_terms = tuple(term for term in item.terms if term not in removed_terms)
        if removed_values or removed_terms:
            retired.append(
                replace(item, values=removed_values, terms=removed_terms)
            )
        if remaining_values or remaining_terms:
            active.append(
                replace(item, values=remaining_values, terms=remaining_terms)
            )
    return active, retired


def _drop_retired_support(
    preferences: dict[str, list[str]],
    search_terms: list[str],
    active: list[PreferenceEvidence],
    retired: list[PreferenceEvidence],
) -> tuple[dict[str, list[str]], list[str]]:
    supported_values = {
        (item.attribute, value) for item in active for value in item.values
    }
    supported_terms = {term for item in active for term in item.terms}
    for item in retired:
        if item.attribute in preferences:
            preferences[item.attribute] = [
                value
                for value in preferences[item.attribute]
                if (item.attribute, value) in supported_values
                or value not in item.values
            ]
            if not preferences[item.attribute]:
                preferences.pop(item.attribute)
        search_terms = [
            term
            for term in search_terms
            if term in supported_terms or term not in item.terms
        ]
    return preferences, search_terms


def _keep_active_evidence_support(
    preferences: dict[str, list[str]],
    search_terms: list[str],
    evidence: list[PreferenceEvidence],
) -> tuple[dict[str, list[str]], list[str]]:
    supported_values = {
        (item.attribute, value) for item in evidence for value in item.values
    }
    supported_terms = {term for item in evidence for term in item.terms}
    kept_preferences = {
        attribute: [
            value
            for value in values
            if (attribute, value) in supported_values
        ]
        for attribute, values in preferences.items()
    }
    return (
        {
            attribute: values
            for attribute, values in kept_preferences.items()
            if values
        },
        [term for term in search_terms if term in supported_terms],
    )


def apply_preference_patch(
    state: ShoppingState, patch: PreferencePatch
) -> ShoppingState:
    category = state.category
    preferences = {key: list(values) for key, values in state.preferences.items()}
    removed = {key: list(values) for key, values in state.removed_preferences.items()}
    no_preference = set(state.no_preference_attributes)
    search_terms = list(state.search_terms)
    evidence = list(state.preference_evidence)
    values_by_attribute = _normalized_patch_values(patch)

    asked_attributes = state.asked_attributes
    previous_ask_attribute = state.previous_ask_attribute
    latest_recommendations = state.latest_recommendations

    patch_category = normalize_value(patch.category)
    update_kind = _classify_update(
        state,
        patch,
        patch_category,
        values_by_attribute,
    )
    if update_kind == "product_change":
        category = None
        preferences.clear()
        removed.clear()
        no_preference.clear()
        search_terms.clear()
        evidence.clear()
        asked_attributes = ()
        previous_ask_attribute = None
        latest_recommendations = ()
    elif update_kind == "preference_correction":
        corrected_attributes = set(values_by_attribute)
        evidence, retired = _without_attributes(evidence, corrected_attributes)
        evidence, retired_unsolicited = _retire_latest_unsolicited(
            evidence,
            corrected_attributes,
        )
        if retired_unsolicited is not None:
            retired.append(retired_unsolicited)
        for attribute in corrected_attributes:
            preferences.pop(attribute, None)
            removed.pop(attribute, None)
            no_preference.discard(attribute)
        preferences, search_terms = _drop_retired_support(
            preferences,
            search_terms,
            evidence,
            retired,
        )
        preferences, search_terms = _keep_active_evidence_support(
            preferences,
            search_terms,
            evidence,
        )
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
                canonical_values = _canonical_preference_values(attribute, value)
                preferences[attribute] = [
                    item
                    for item in preferences[attribute]
                    if item not in canonical_values
                ]
                if not preferences[attribute]:
                    preferences.pop(attribute)
            else:
                preferences.pop(attribute)
        canonical_values = (
            _canonical_preference_values(attribute, value) if value else []
        )
        evidence, retired_evidence = _remove_evidence_values(
            evidence,
            attribute,
            set(canonical_values) if canonical_values else None,
        )
        preferences, search_terms = _drop_retired_support(
            preferences,
            search_terms,
            evidence,
            retired_evidence,
        )
        if value:
            bucket = removed.setdefault(attribute, [])
            for canonical_value in canonical_values:
                _append_unique(bucket, canonical_value)
                search_terms = [
                    term for term in search_terms if term != canonical_value
                ]

    for raw_attribute in patch.no_preference_attributes:
        attribute = normalize_value(raw_attribute)
        if attribute not in ALLOWED_PREFERENCE_ATTRIBUTES:
            continue
        no_preference.add(attribute)
        if attribute == "category":
            category = None
        else:
            preferences.pop(attribute, None)
        evidence, retired_evidence = _remove_evidence_values(
            evidence,
            attribute,
            None,
        )
        preferences, search_terms = _drop_retired_support(
            preferences,
            search_terms,
            evidence,
            retired_evidence,
        )

    intent_mode = state.intent_mode
    if patch.intent_mode != "unchanged":
        intent_mode = patch.intent_mode

    if update_kind == "preference_correction" and _looks_like_attribute_constraint(
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
    accepted_patch_terms: list[str] = []
    for raw_term in patch.search_terms:
        term = normalize_value(raw_term)
        if term and term not in rejected_values:
            _append_unique(search_terms, term)
            _append_unique(accepted_patch_terms, term)

    attributes = set(values_by_attribute)
    assigned_terms: set[str] = set()
    terms_by_attribute: dict[str, list[str]] = {
        attribute: [] for attribute in values_by_attribute
    }
    for term in accepted_patch_terms:
        matching_attributes = [
            attribute
            for attribute, values in values_by_attribute.items()
            if any(term in value or value in term for value in values)
        ]
        if len(matching_attributes) == 1:
            terms_by_attribute[matching_attributes[0]].append(term)
            assigned_terms.add(term)
    for attribute, values in values_by_attribute.items():
        source_kind = _evidence_source(
            state,
            {attribute},
            correction=update_kind == "preference_correction",
            allow_clarification=update_kind != "product_change",
        )
        terms = tuple(terms_by_attribute[attribute])
        evidence.append(
            PreferenceEvidence(
                attribute=attribute,
                values=values,
                terms=terms,
                source_turn=state.turn + 1,
                source_kind=source_kind,
            )
        )
    remaining_terms = tuple(
        term for term in accepted_patch_terms if term not in assigned_terms
    )
    if remaining_terms:
        source_kind = _evidence_source(
            state,
            attributes,
            correction=update_kind == "preference_correction",
            allow_clarification=update_kind != "product_change",
        )
        evidence.append(
            PreferenceEvidence(
                attribute="other",
                terms=remaining_terms,
                source_turn=state.turn + 1,
                source_kind=source_kind,
            )
        )

    return replace(
        state,
        intent_mode=intent_mode,
        category=category,
        preferences={key: tuple(values) for key, values in preferences.items()},
        removed_preferences={key: tuple(values) for key, values in removed.items()},
        no_preference_attributes=frozenset(no_preference),
        search_terms=tuple(search_terms),
        preference_evidence=tuple(evidence),
        asked_attributes=asked_attributes,
        previous_ask_attribute=previous_ask_attribute,
        latest_recommendations=latest_recommendations,
    )


def _first_match(pattern: str, message: str) -> str | None:
    match = re.search(pattern, message, re.IGNORECASE)
    return normalize_value(match.group(1)) if match else None


def _direct_answer_values(message: str, attribute: str) -> list[str]:
    match = DIRECT_ANSWER_RE.fullmatch(message.strip())
    if not match:
        return []
    values: list[str] = []
    for part in re.split(r"[;|]", match.group(1)):
        normalized = normalize_value(part)
        if not normalized:
            continue
        for value in _canonical_preference_values(attribute, normalized):
            _append_unique(values, value)
    return values


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
        r"\b((?:under|below|less than|up to|around|about)\s*\$?\s*\d+(?:\.\d{1,2})?)\b",
        lowered,
    )
    if budget:
        budget = re.sub(
            r"\b(under|below|less than|up to|around|about)\b\s*\$?\s*(\d+(?:\.\d{1,2})?)",
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
        and re.search(
            r"\b(no(?: additional)? preference|don'?t care|use your judgment)\b",
            lowered,
        )
    ):
        no_preference.append(state.previous_ask_attribute)

    direct_values: list[str] = []
    direct_attribute = state.previous_ask_attribute
    if (
        direct_attribute in ALLOWED_PREFERENCE_ATTRIBUTES
        and direct_attribute != "category"
        and not no_preference
    ):
        direct_values = _direct_answer_values(lowered, direct_attribute)
        values.extend(
            PreferenceValue(attribute=direct_attribute, value=value)
            for value in direct_values
        )

    search_terms: list[str] = []
    if not direct_values and not no_preference:
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
