from __future__ import annotations

from starter.state import ShoppingState


RECOMMENDATION_MESSAGE = "Here are the closest matches I found."
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


def choose_clarification(
    state: ShoppingState, turn: int
) -> tuple[str, str | None]:
    if turn >= 10:
        return RECOMMENDATION_MESSAGE, None

    excluded = set(state.asked_attributes)
    excluded.update(state.no_preference_attributes)
    excluded.update(state.preferences)
    if state.category:
        excluded.add("category")

    for attribute in ATTRIBUTE_PRIORITY:
        if attribute not in excluded:
            return QUESTION_TEMPLATES[attribute], attribute
    return RECOMMENDATION_MESSAGE, None
