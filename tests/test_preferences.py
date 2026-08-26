from __future__ import annotations

import unittest
from dataclasses import replace

from starter.preference_tool import (
    PreferencePatch,
    PreferenceRemoval,
    PreferenceValue,
    apply_preference_patch,
    parse_preference_fallback,
)
from starter.state import ShoppingState


class PreferenceUpdateTest(unittest.TestCase):
    def test_patch_adds_normalized_values_without_mutating_old_state(self) -> None:
        state = ShoppingState.new(
            "s1", {"summary": "likes comfort", "preference_tags": ["fit"]}
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                intent_mode="buying",
                category=" Shoes ",
                set_preferences=[PreferenceValue(attribute="color", value=" Blue ")],
                search_terms=["Running"],
            ),
        )

        self.assertEqual(updated.intent_mode, "buying")
        self.assertEqual(updated.category, "shoes")
        self.assertEqual(updated.preferences, {"color": ("blue",)})
        self.assertEqual(updated.search_terms, ("running",))
        self.assertEqual(state.preferences, {})

    def test_new_state_copies_the_user_profile(self) -> None:
        profile = {"summary": "likes comfort", "preference_tags": ["fit"]}
        state = ShoppingState.new("s1", profile)

        profile["preference_tags"].append("style")

        self.assertEqual(state.user_profile["preference_tags"], ["fit"])

    def test_override_resets_product_specific_preferences(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="shirts",
            preferences={"color": ("red",)},
            removed_preferences={"brand": ("acme",)},
            no_preference_attributes=frozenset({"size"}),
            search_terms=("cotton",),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                category="boots",
                reset_product_preferences=True,
                set_preferences=[
                    PreferenceValue(attribute="material", value="leather")
                ],
            ),
        )

        self.assertEqual(updated.category, "boots")
        self.assertEqual(updated.preferences, {"material": ("leather",)})
        self.assertEqual(updated.removed_preferences, {})
        self.assertEqual(updated.no_preference_attributes, frozenset())
        self.assertEqual(updated.search_terms, ())

    def test_override_clears_question_and_recommendation_context(self) -> None:
        profile = {"summary": "likes comfort", "preference_tags": ["durable"]}
        state = replace(
            ShoppingState.new("s1", profile),
            category="shoes",
            preferences={"material": ("leather",)},
            removed_preferences={"color": ("red",)},
            no_preference_attributes=frozenset({"brand"}),
            search_terms=("trail",),
            asked_attributes=("material", "color"),
            previous_ask_attribute="color",
            latest_recommendations=("OLD_PRODUCT",),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                reset_product_preferences=True,
                category="earrings",
                search_terms=["silver"],
            ),
        )

        self.assertEqual(updated.category, "earrings")
        self.assertEqual(updated.preferences, {})
        self.assertEqual(updated.asked_attributes, ())
        self.assertIsNone(updated.previous_ask_attribute)
        self.assertEqual(updated.latest_recommendations, ())
        self.assertEqual(updated.user_profile, profile)

    def test_reset_without_a_new_category_preserves_product_type(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="accessories belts",
            preferences={"feature": ("hand wash only",)},
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                reset_product_preferences=True,
                set_preferences=[PreferenceValue(attribute="material", value="nylon")],
            ),
        )

        self.assertEqual(updated.category, "accessories belts")
        self.assertEqual(updated.preferences, {"material": ("nylon",)})

    def test_fallback_treats_attribute_override_as_preference_not_category(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="accessories belts",
            search_terms=("hand", "wash"),
        )

        patch = parse_preference_fallback(
            "Actually, ignore my earlier preference. What I need is: nylon.",
            state,
        )
        updated = apply_preference_patch(state, patch)

        self.assertTrue(patch.reset_product_preferences)
        self.assertEqual(updated.category, "accessories belts")
        self.assertEqual(updated.preferences["material"], ("nylon",))

    def test_no_preference_clears_active_attribute(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}), preferences={"color": ("red",)}
        )

        updated = apply_preference_patch(
            state, PreferencePatch(no_preference_attributes=["color"])
        )

        self.assertNotIn("color", updated.preferences)
        self.assertEqual(updated.no_preference_attributes, frozenset({"color"}))

    def test_removal_rejects_one_value_and_keeps_other_values(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            preferences={"color": ("red", "blue")},
            search_terms=("red", "running"),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                remove_preferences=[
                    PreferenceRemoval(attribute="color", value="red")
                ],
                search_terms=["red", "trail"],
            ),
        )

        self.assertEqual(updated.preferences, {"color": ("blue",)})
        self.assertEqual(updated.removed_preferences, {"color": ("red",)})
        self.assertEqual(updated.search_terms, ("running", "trail"))

    def test_fallback_extracts_common_preferences(self) -> None:
        state = ShoppingState.new("s1", {})

        patch = parse_preference_fallback(
            "I'm looking for black leather boots under $90 for hiking.", state
        )
        updated = apply_preference_patch(state, patch)

        self.assertEqual(updated.intent_mode, "buying")
        self.assertEqual(updated.category, "black leather boots")
        self.assertEqual(updated.preferences["color"], ("black",))
        self.assertEqual(updated.preferences["material"], ("leather",))
        self.assertEqual(updated.preferences["budget"], ("under $90",))
        self.assertEqual(updated.preferences["use_case"], ("hiking",))

    def test_fallback_understands_no_preference_reply(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            preferences={"color": ("blue",)},
            previous_ask_attribute="color",
        )

        patch = parse_preference_fallback(
            "I don't have a preference for color; use your judgment.", state
        )
        updated = apply_preference_patch(state, patch)

        self.assertNotIn("color", updated.preferences)
        self.assertIn("color", updated.no_preference_attributes)

    def test_fallback_resets_state_on_explicit_intent_override(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="shirts",
            preferences={"color": ("red",)},
        )

        patch = parse_preference_fallback(
            "Actually, ignore my earlier preference. What I need is: waterproof hiking boots.",
            state,
        )

        self.assertTrue(patch.reset_product_preferences)
        self.assertEqual(patch.intent_mode, "buying")
        self.assertIn("waterproof", patch.search_terms)


if __name__ == "__main__":
    unittest.main()
