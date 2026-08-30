from __future__ import annotations

import os

from starter.retrieval import AttributeDiagnostic
from starter.state import ShoppingState


RECOMMENDATION_MESSAGE = "Here are the closest matches I found."
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


QUESTION_PRIORS = {
    "category": 0.20,
    "material": 0.46,
    "color": 0.36,
    "feature": 0.15,
    "use_case": 0.14,
    "style": 0.12,
    "size": 0.12,
    "brand": 0.05,
    "budget": 0.05,
}
MIN_SPECIFIC_QUESTION_SCORE = 0.28


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


def _should_promote_other(state: ShoppingState) -> bool:
    """Ask the open-ended question before the specific facets are exhausted.

    "other" accepts any undisclosed constraint, while a specific attribute only
    accepts constraints of its own kind, so promoting it harvests details that
    do not fit the standard facets - model numbers, "rubber sole", "tie
    closure".

    Both triggers read structured state rather than the raw message.
    `last_update_type` is set by the state updater, so it cannot be fooled by
    the word "actually" in an ordinary reply, and
    `consecutive_no_preference_turns` resets the moment the shopper does give a
    preference, which is what "two failed attempts" actually means.

    Env-gated so each can be measured apart; both default on, having measured
    +0.014962 public / +0.003921 held-out together.
    """
    if _env_bool("TECHJAM_OTHER_AFTER_OVERRIDE", True) and (
        state.turn > 0
        and state.last_update_type in {"replace_preferences", "product_change"}
    ):
        return True

    threshold = _env_int("TECHJAM_OTHER_AFTER_NO_PREFERENCE", 2, 0, 10)
    if threshold:
        # "cumulative" counts every attribute the shopper has ever declined;
        # "consecutive" resets the moment they do answer. Cumulative is the
        # default: it fires earlier and measured +0.001900 (browsing +0.008333,
        # MTTC 3.615 -> 3.570) with Hit@10 unchanged. The reset in the
        # consecutive counter discards evidence that the facets are exhausted
        # just because one question happened to land.
        if _env_bool("TECHJAM_OTHER_COUNTER_CUMULATIVE", True):
            count = len(state.no_preference_attributes)
        else:
            count = state.consecutive_no_preference_turns
        if count >= threshold:
            return True
    return False


def _question_score(attribute: str, item: AttributeDiagnostic) -> float:
    return (
        0.45 * item.disagreement
        + 0.25 * item.coverage
        + 0.20 * item.relevance
        + QUESTION_PRIORS[attribute]
    )


def choose_clarification(
    state: ShoppingState,
    turn: int,
    diagnostics: dict[str, AttributeDiagnostic],
) -> tuple[str, str | None]:
    if turn >= 10:
        return RECOMMENDATION_MESSAGE, None

    excluded = set(state.asked_attributes)
    # "other" harvests any undisclosed constraint, so repeat it until the
    # shopper says there is nothing left (no_preference below).
    excluded.discard("other")
    excluded.update(state.no_preference_attributes)
    # An attribute whose only evidence came from a correction ("what I need is
    # cotton") was never actually asked; the shopper may still hold a more
    # specific constraint for it, so keep it askable once.
    correction_only = {
        attribute
        for attribute in state.preferences
        if attribute not in state.asked_attributes
        and any(item.attribute == attribute for item in state.preference_evidence)
        and all(
            item.source_kind == "correction"
            for item in state.preference_evidence
            if item.attribute == attribute
        )
    }
    excluded.update(
        attribute for attribute in state.preferences if attribute not in correction_only
    )
    if state.category:
        excluded.add("category")

    category_diagnostic = diagnostics.get("category")
    if (
        "category" not in excluded
        and category_diagnostic is not None
        and category_diagnostic.coverage >= 0.40
        and category_diagnostic.disagreement >= 0.35
    ):
        return QUESTION_TEMPLATES["category"], "category"

    # Placed after the category question so a genuine product change still
    # gets asked first, but ahead of the specific facets so "other" no longer
    # waits for every one of them to score badly.
    if "other" not in excluded and _should_promote_other(state):
        return QUESTION_TEMPLATES["other"], "other"

    scored: list[tuple[float, str]] = []
    for attribute, item in diagnostics.items():
        if (
            attribute not in QUESTION_PRIORS
            or attribute in excluded
            or attribute == "category"
        ):
            continue
        if item.coverage < 0.10:
            continue
        if attribute in {"brand", "budget"} and (
            item.coverage < 0.35 or item.disagreement < 0.15
        ):
            continue
        scored.append((_question_score(attribute, item), attribute))

    if scored:
        best_score, best_attribute = max(scored, key=lambda pair: (pair[0], pair[1]))
        if best_score >= MIN_SPECIFIC_QUESTION_SCORE:
            return QUESTION_TEMPLATES[best_attribute], best_attribute

    if "other" not in excluded:
        return QUESTION_TEMPLATES["other"], "other"
    return RECOMMENDATION_MESSAGE, None
