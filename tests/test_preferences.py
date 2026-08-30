from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

import starter.state as state_module
from starter.preference_tool import (
    PreferencePatch,
    PreferenceRemoval,
    PreferenceValue,
    apply_preference_patch,
    parse_preference_fallback,
)
from starter.state import PreferenceEvidence, ShoppingState


class PreferenceUpdateTest(unittest.TestCase):
    def test_consecutive_no_preference_counter_tracks_observed_replies(self) -> None:
        state = apply_preference_patch(
            ShoppingState.new("s1", {}),
            PreferencePatch(no_preference_attributes=["color"]),
        )
        self.assertEqual(state.consecutive_no_preference_turns, 1)

        state = apply_preference_patch(
            state,
            PreferencePatch(no_preference_attributes=["brand"]),
        )
        self.assertEqual(state.consecutive_no_preference_turns, 2)

        state = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="water resistant")
                ]
            ),
        )
        self.assertEqual(state.consecutive_no_preference_turns, 0)

    def test_preference_evidence_is_immutable_and_hidden_from_model_prompt(self) -> None:
        evidence_type = getattr(state_module, "PreferenceEvidence", None)
        self.assertIsNotNone(evidence_type)
        evidence = evidence_type(
            attribute="feature",
            values=("machine washable",),
            terms=("machine washable",),
            source_turn=2,
            source_kind="clarification",
        )
        state = replace(
            ShoppingState.new("s1", {}),
            preference_evidence=(evidence,),
        )

        self.assertEqual(state.preference_evidence, (evidence,))
        self.assertNotIn("preference_evidence", state.to_prompt_dict())
        with self.assertRaises(FrozenInstanceError):
            evidence.source_turn = 3

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

    def test_unprompted_preference_records_unsolicited_evidence(self) -> None:
        updated = apply_preference_patch(
            ShoppingState.new("s1", {}),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="machine washable")
                ]
            ),
        )

        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="feature",
                    values=("machine washable",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
        )

    def test_answer_to_previous_question_records_clarification_evidence(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            previous_ask_attribute="material",
            turn=1,
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(
                        attribute="material",
                        value="95% polyester, 5% spandex",
                    )
                ]
            ),
        )

        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="material",
                    values=("polyester", "spandex"),
                    terms=("95% polyester, 5% spandex",),
                    source_turn=2,
                    source_kind="clarification",
                ),
            ),
        )

    def test_compound_material_value_is_canonicalized_for_search(self) -> None:
        state = ShoppingState.new("s1", {})

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(
                        attribute="material",
                        value="95% Polyester, 5% Spandex",
                    )
                ]
            ),
        )

        self.assertEqual(
            updated.preferences,
            {"material": ("polyester", "spandex")},
        )

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

    def test_same_product_correction_retires_only_superseded_evidence(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="accessories belts",
        )
        state = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="hand wash only")
                ],
                search_terms=["hand wash only"],
            ),
        )
        state = replace(
            state,
            turn=1,
            previous_ask_attribute="material",
        )
        state = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="material", value="cotton")
                ]
            ),
        )
        state = replace(
            state,
            turn=2,
            asked_attributes=("material", "color"),
            previous_ask_attribute="color",
            no_preference_attributes=frozenset({"color"}),
            latest_recommendations=("OLD_PRODUCT",),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                reset_product_preferences=True,
                set_preferences=[
                    PreferenceValue(attribute="material", value="nylon")
                ],
            ),
        )

        self.assertEqual(updated.category, "accessories belts")
        self.assertEqual(updated.preferences, {"material": ("nylon",)})
        self.assertEqual(updated.search_terms, ())
        self.assertEqual(updated.asked_attributes, ("material", "color"))
        self.assertEqual(updated.previous_ask_attribute, "color")
        self.assertEqual(updated.no_preference_attributes, frozenset({"color"}))
        self.assertEqual(updated.latest_recommendations, ())
        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="material",
                    values=("nylon",),
                    source_turn=3,
                    source_kind="correction",
                ),
            ),
        )

    def test_correction_preserves_nonconflicting_clarification_evidence(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="accessories belts",
            turn=1,
            previous_ask_attribute="feature",
        )
        state = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="machine washable")
                ]
            ),
        )
        state = replace(state, turn=2, previous_ask_attribute="material")

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                reset_product_preferences=True,
                set_preferences=[
                    PreferenceValue(attribute="material", value="nylon")
                ],
            ),
        )

        self.assertEqual(
            updated.preferences,
            {
                "feature": ("machine washable",),
                "material": ("nylon",),
            },
        )
        self.assertEqual(
            [item.source_kind for item in updated.preference_evidence],
            ["clarification", "correction"],
        )

    def test_explicit_same_category_is_still_a_preference_correction(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="shoes",
            turn=1,
            previous_ask_attribute="feature",
        )
        state = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="waterproof")
                ]
            ),
        )
        state = replace(state, turn=2, previous_ask_attribute="material")

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                reset_product_preferences=True,
                category="shoes",
                set_preferences=[
                    PreferenceValue(attribute="material", value="nylon")
                ],
            ),
        )

        self.assertEqual(updated.category, "shoes")
        self.assertEqual(
            updated.preferences,
            {"feature": ("waterproof",), "material": ("nylon",)},
        )
        self.assertEqual(
            [item.source_kind for item in updated.preference_evidence],
            ["clarification", "correction"],
        )

    def test_product_change_clears_all_prior_product_evidence(self) -> None:
        state = apply_preference_patch(
            replace(ShoppingState.new("s1", {}), category="shoes"),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="waterproof")
                ]
            ),
        )
        state = replace(
            state,
            asked_attributes=("color",),
            previous_ask_attribute="color",
            no_preference_attributes=frozenset({"color"}),
            latest_recommendations=("OLD_PRODUCT",),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                reset_product_preferences=True,
                category="shirts",
                set_preferences=[
                    PreferenceValue(attribute="color", value="red")
                ],
            ),
        )

        self.assertEqual(updated.category, "shirts")
        self.assertEqual(updated.preferences, {"color": ("red",)})
        self.assertEqual(updated.asked_attributes, ())
        self.assertIsNone(updated.previous_ask_attribute)
        self.assertEqual(updated.no_preference_attributes, frozenset())
        self.assertEqual(updated.latest_recommendations, ())
        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="color",
                    values=("red",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
        )

    def test_bare_product_reset_clears_stale_category(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="shirts",
            preferences={"color": ("red",)},
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(reset_product_preferences=True),
        )

        self.assertIsNone(updated.category)
        self.assertEqual(updated.preferences, {})

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

        self.assertEqual(patch.update_type, "replace_preferences")
        self.assertEqual(patch.correction_scope, "latest_unsolicited")
        self.assertEqual(updated.category, "accessories belts")
        self.assertEqual(updated.preferences["material"], ("nylon",))

    def test_unnamed_correction_retires_only_latest_unsolicited_evidence(self) -> None:
        state = replace(
            ShoppingState.new("public_0003", {}),
            category="watches wrist watches",
        )
        state = apply_preference_patch(
            state,
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(
                        attribute="material",
                        value="stainless steel band",
                    )
                ],
                search_terms=["stainless steel band"],
            ),
        )
        state = replace(
            state,
            turn=2,
            previous_ask_attribute="feature",
            no_preference_attributes=frozenset({"color"}),
            latest_recommendations=("STALE_RESULT",),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                update_type="replace_preferences",
                correction_scope="latest_unsolicited",
                set_preferences=[
                    PreferenceValue(attribute="feature", value="water resistant")
                ],
            ),
        )

        self.assertEqual(updated.category, "watches wrist watches")
        self.assertEqual(updated.preferences, {"feature": ("water resistant",)})
        self.assertEqual(updated.search_terms, ())
        self.assertEqual(updated.no_preference_attributes, frozenset({"color"}))
        self.assertEqual(updated.latest_recommendations, ())
        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="feature",
                    values=("water resistant",),
                    source_turn=3,
                    source_kind="correction",
                ),
            ),
        )

    def test_unnamed_correction_preserves_clarification_evidence(self) -> None:
        state = apply_preference_patch(
            replace(
                ShoppingState.new("s1", {}),
                category="watches",
            ),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="material", value="steel")
                ]
            ),
        )
        state = apply_preference_patch(
            replace(state, turn=1, previous_ask_attribute="brand"),
            PreferencePatch(
                set_preferences=[PreferenceValue(attribute="brand", value="casio")]
            ),
        )

        updated = apply_preference_patch(
            replace(state, turn=2, previous_ask_attribute="feature"),
            PreferencePatch(
                update_type="replace_preferences",
                correction_scope="latest_unsolicited",
                set_preferences=[
                    PreferenceValue(attribute="feature", value="water resistant")
                ],
            ),
        )

        self.assertEqual(
            updated.preferences,
            {"brand": ("casio",), "feature": ("water resistant",)},
        )
        self.assertEqual(
            [item.source_kind for item in updated.preference_evidence],
            ["clarification", "correction"],
        )

    def test_no_preference_clears_active_attribute(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}), preferences={"color": ("red",)}
        )

        updated = apply_preference_patch(
            state, PreferencePatch(no_preference_attributes=["color"])
        )

        self.assertNotIn("color", updated.preferences)
        self.assertEqual(updated.no_preference_attributes, frozenset({"color"}))

    def test_no_preference_retires_matching_evidence_and_search_terms(self) -> None:
        state = apply_preference_patch(
            ShoppingState.new("s1", {}),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="machine washable")
                ],
                search_terms=["machine washable"],
            ),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(no_preference_attributes=["feature"]),
        )

        self.assertNotIn("feature", updated.preferences)
        self.assertEqual(updated.search_terms, ())
        self.assertEqual(updated.preference_evidence, ())

    def test_multi_attribute_terms_retire_with_their_matching_attribute(self) -> None:
        state = apply_preference_patch(
            ShoppingState.new("s1", {}),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(attribute="feature", value="machine washable"),
                    PreferenceValue(attribute="color", value="red"),
                ],
                search_terms=["machine washable", "red"],
            ),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(no_preference_attributes=["feature"]),
        )

        self.assertEqual(updated.preferences, {"color": ("red",)})
        self.assertEqual(updated.search_terms, ("red",))
        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="color",
                    values=("red",),
                    terms=("red",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
        )

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

    def test_compound_removal_is_canonicalized_like_compound_addition(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            preferences={"material": ("polyester", "spandex")},
            search_terms=("polyester", "spandex", "running"),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                remove_preferences=[
                    PreferenceRemoval(
                        attribute="material",
                        value="95% polyester, 5% spandex",
                    )
                ]
            ),
        )

        self.assertNotIn("material", updated.preferences)
        self.assertEqual(
            updated.removed_preferences,
            {"material": ("polyester", "spandex")},
        )
        self.assertEqual(updated.search_terms, ("running",))

    def test_value_removal_updates_matching_evidence(self) -> None:
        state = apply_preference_patch(
            ShoppingState.new("s1", {}),
            PreferencePatch(
                set_preferences=[
                    PreferenceValue(
                        attribute="material",
                        value="polyester and spandex",
                    )
                ],
                search_terms=["polyester", "spandex"],
            ),
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                remove_preferences=[
                    PreferenceRemoval(attribute="material", value="polyester")
                ]
            ),
        )

        self.assertEqual(updated.preferences, {"material": ("spandex",)})
        self.assertEqual(updated.search_terms, ("spandex",))
        self.assertEqual(
            updated.preference_evidence,
            (
                PreferenceEvidence(
                    attribute="material",
                    values=("spandex",),
                    terms=("spandex",),
                    source_turn=1,
                    source_kind="unsolicited",
                ),
            ),
        )

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

    def test_fallback_structures_evaluator_style_clarification_answers(self) -> None:
        cases = (
            (
                "material",
                "For that, what matters is: cotton; polyester.",
                ("cotton", "polyester"),
                (),
            ),
            (
                "feature",
                "For that, what matters is: button closure; machine washable.",
                ("button closure", "machine washable"),
                (),
            ),
            (
                "use_case",
                "For that, what matters is: winter hiking.",
                ("hiking", "winter"),
                ("winter hiking",),
            ),
        )

        for attribute, message, expected_values, expected_terms in cases:
            with self.subTest(attribute=attribute):
                state = replace(
                    ShoppingState.new("s1", {}),
                    previous_ask_attribute=attribute,
                    turn=1,
                )
                updated = apply_preference_patch(
                    state,
                    parse_preference_fallback(message, state),
                )

                self.assertEqual(updated.preferences.get(attribute), expected_values)
                self.assertEqual(updated.search_terms, expected_terms)
                self.assertEqual(
                    updated.preference_evidence,
                    (
                        PreferenceEvidence(
                            attribute=attribute,
                            values=expected_values,
                            terms=expected_terms,
                            source_turn=2,
                            source_kind="clarification",
                        ),
                    ),
                )

    def test_fallback_no_additional_preference_adds_no_search_noise(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            previous_ask_attribute="feature",
        )

        updated = apply_preference_patch(
            state,
            parse_preference_fallback(
                "No additional preference; use your judgment.",
                state,
            ),
        )

        self.assertEqual(updated.no_preference_attributes, frozenset({"feature"}))
        self.assertEqual(updated.search_terms, ())
        self.assertEqual(updated.preference_evidence, ())

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

        self.assertEqual(patch.update_type, "product_change")
        self.assertEqual(patch.intent_mode, "buying")
        self.assertIn("waterproof", patch.search_terms)

    def test_explicit_preference_replacement_only_replaces_named_attributes(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="women's boots",
            preferences={
                "material": ("leather",),
                "color": ("black",),
                "feature": ("waterproof",),
            },
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                update_type="replace_preferences",
                set_preferences=[
                    PreferenceValue(attribute="material", value="suede")
                ],
            ),
        )

        self.assertEqual(updated.category, "women's boots")
        self.assertEqual(
            updated.preferences,
            {
                "color": ("black",),
                "feature": ("waterproof",),
                "material": ("suede",),
            },
        )
        self.assertEqual(updated.last_update_type, "replace_preferences")

    def test_product_change_request_is_downgraded_when_only_attribute_changes(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="women's boots",
            preferences={"material": ("leather",), "color": ("black",)},
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                update_type="product_change",
                category="unchanged",
                set_preferences=[
                    PreferenceValue(attribute="material", value="suede")
                ],
            ),
        )

        self.assertEqual(updated.category, "women's boots")
        self.assertEqual(
            updated.preferences,
            {"color": ("black",), "material": ("suede",)},
        )
        self.assertEqual(updated.last_update_type, "replace_preferences")

    def test_attribute_only_value_cannot_replace_category_during_merge(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="women's boots",
            preferences={"color": ("black",)},
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                update_type="merge",
                category="leather",
                set_preferences=[
                    PreferenceValue(attribute="material", value="leather")
                ],
            ),
        )

        self.assertEqual(updated.category, "women's boots")
        self.assertEqual(updated.preferences["material"], ("leather",))

    def test_changed_category_promotes_mislabeled_replacement_to_product_change(self) -> None:
        state = replace(
            ShoppingState.new("s1", {}),
            category="shirts",
            preferences={"material": ("cotton",), "color": ("red",)},
        )

        updated = apply_preference_patch(
            state,
            PreferencePatch(
                update_type="replace_preferences",
                category="waterproof hiking boots",
                set_preferences=[
                    PreferenceValue(attribute="feature", value="waterproof")
                ],
            ),
        )

        self.assertEqual(updated.category, "waterproof hiking boots")
        self.assertEqual(updated.preferences, {"feature": ("waterproof",)})
        self.assertEqual(updated.last_update_type, "product_change")


class CompoundEvidenceTest(unittest.TestCase):
    def test_compound_material_keeps_raw_phrase_as_search_term(self) -> None:
        state = ShoppingState.new("s", {})
        patch = PreferencePatch(
            set_preferences=[
                PreferenceValue(attribute="material", value="90% Cotton, 10% Others")
            ]
        )
        updated = apply_preference_patch(state, patch)
        self.assertEqual(updated.preferences["material"], ("cotton",))
        self.assertIn("90% cotton, 10% others", updated.search_terms)
        material_evidence = [
            item for item in updated.preference_evidence if item.attribute == "material"
        ]
        self.assertTrue(
            any("90% cotton, 10% others" in item.terms for item in material_evidence)
        )

    def test_atomic_material_adds_no_extra_search_term(self) -> None:
        state = ShoppingState.new("s", {})
        patch = PreferencePatch(
            set_preferences=[PreferenceValue(attribute="material", value="cotton")]
        )
        updated = apply_preference_patch(state, patch)
        self.assertNotIn("cotton", updated.search_terms)


class CompoundClarificationFlowTest(unittest.TestCase):
    def test_direct_answer_preserves_raw_compound_phrase(self) -> None:
        state = replace(
            ShoppingState.new("s", {}), previous_ask_attribute="material", turn=1
        )
        patch = parse_preference_fallback(
            "For that, what matters is: cotton; 90% Cotton, 10% Others.", state
        )
        updated = apply_preference_patch(state, patch)
        self.assertEqual(updated.preferences["material"], ("cotton",))
        self.assertIn("90% cotton, 10% others", updated.search_terms)

    def test_correction_keeps_agreeing_clarification_evidence(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            search_terms=("90% cotton, 10% others",),
            asked_attributes=("material",),
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material",
                    values=("cotton",),
                    terms=("90% cotton, 10% others",),
                    source_turn=2,
                    source_kind="clarification",
                ),
            ),
            turn=3,
        )
        patch = PreferencePatch(
            category="cotton",
            set_preferences=[PreferenceValue(attribute="material", value="cotton")],
            reset_product_preferences=True,
        )
        updated = apply_preference_patch(state, patch)
        self.assertIn("90% cotton, 10% others", updated.search_terms)
        self.assertTrue(
            any(
                item.source_kind == "clarification"
                and "90% cotton, 10% others" in item.terms
                for item in updated.preference_evidence
            )
        )

    def test_correction_retires_conflicting_clarification_evidence(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("wool",)},
            search_terms=("100% merino wool",),
            asked_attributes=("material",),
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material",
                    values=("wool",),
                    terms=("100% merino wool",),
                    source_turn=2,
                    source_kind="clarification",
                ),
            ),
            turn=3,
        )
        patch = PreferencePatch(
            category="cotton",
            set_preferences=[PreferenceValue(attribute="material", value="cotton")],
            reset_product_preferences=True,
        )
        updated = apply_preference_patch(state, patch)
        self.assertEqual(updated.preferences["material"], ("cotton",))
        self.assertNotIn("100% merino wool", updated.search_terms)


class NoAdditionalPreferenceTest(unittest.TestCase):
    def test_no_additional_preference_reply_is_not_search_noise(self) -> None:
        state = replace(
            ShoppingState.new("s", {}), previous_ask_attribute="color", turn=2
        )
        patch = parse_preference_fallback(
            "I don't have an additional preference for color.", state
        )
        self.assertEqual(patch.no_preference_attributes, ["color"])
        self.assertEqual(patch.search_terms, [])


class FallbackBudgetTest(unittest.TestCase):
    def test_fallback_captures_around_budget(self) -> None:
        state = ShoppingState.new("s", {})
        patch = parse_preference_fallback(
            "For that, what matters is: budget around $59.99.", state
        )
        budgets = [
            item.value for item in patch.set_preferences if item.attribute == "budget"
        ]
        self.assertEqual(budgets, ["around $59.99"])


if __name__ == "__main__":
    unittest.main()
