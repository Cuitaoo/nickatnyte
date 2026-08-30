from __future__ import annotations

import os
import unittest
from dataclasses import replace

from starter.questions import choose_clarification
from starter.retrieval import AttributeDiagnostic
from starter.state import (
    ALLOWED_PREFERENCE_ATTRIBUTES,
    PreferenceEvidence,
    ShoppingState,
)


def diagnostic(
    attribute: str,
    coverage: float,
    disagreement: float,
    relevance: float,
) -> AttributeDiagnostic:
    return AttributeDiagnostic(attribute, coverage, disagreement, relevance)


def broad_diagnostics() -> dict[str, AttributeDiagnostic]:
    return {
        attribute: diagnostic(attribute, 0.9, 0.8, 0.8)
        for attribute in (
            "category",
            "material",
            "color",
            "size",
            "style",
            "brand",
            "budget",
            "feature",
            "use_case",
        )
    }


class ClarificationTest(unittest.TestCase):
    def test_missing_category_is_asked_first(self) -> None:
        message, attribute = choose_clarification(
            ShoppingState.new("s", {}), 1, broad_diagnostics()
        )

        self.assertEqual(attribute, "category")
        self.assertIn("kind of product", message.lower())

    def test_feature_beats_color_when_it_splits_candidates_more(self) -> None:
        state = replace(ShoppingState.new("s", {}), category="running shoes")
        diagnostics = {
            "feature": diagnostic("feature", 0.9, 0.9, 0.9),
            "color": diagnostic("color", 0.9, 0.1, 0.6),
            "use_case": diagnostic("use_case", 0.4, 0.3, 0.7),
        }

        message, attribute = choose_clarification(state, 1, diagnostics)

        self.assertEqual(attribute, "feature")
        self.assertIn("feature", message.lower())

    def test_material_beats_noisy_feature_signal_for_apparel_pool(self) -> None:
        state = replace(ShoppingState.new("s", {}), category="women dresses")
        diagnostics = {
            "material": diagnostic("material", 0.74, 0.36, 0.76),
            "feature": diagnostic("feature", 0.94, 0.61, 0.99),
            "color": diagnostic("color", 0.45, 0.64, 0.62),
        }

        message, attribute = choose_clarification(state, 1, diagnostics)

        self.assertEqual(attribute, "material")
        self.assertIn("material", message.lower())

    def test_category_is_asked_only_when_missing_and_candidates_disagree(self) -> None:
        diagnostics = {
            "category": diagnostic("category", 1.0, 0.9, 0.9),
            "feature": diagnostic("feature", 0.8, 0.7, 0.8),
        }

        self.assertEqual(
            choose_clarification(ShoppingState.new("s", {}), 1, diagnostics)[1],
            "category",
        )
        known = replace(ShoppingState.new("s", {}), category="shoes")
        self.assertEqual(choose_clarification(known, 1, diagnostics)[1], "feature")

    def test_brand_and_budget_need_substantial_evidence(self) -> None:
        state = replace(ShoppingState.new("s", {}), category="shoes")
        diagnostics = {
            "brand": diagnostic("brand", 0.1, 0.9, 0.9),
            "budget": diagnostic("budget", 0.2, 0.9, 0.9),
            "style": diagnostic("style", 0.7, 0.5, 0.7),
        }

        self.assertEqual(choose_clarification(state, 2, diagnostics)[1], "style")

    def test_other_is_only_used_when_specific_scores_are_too_low(self) -> None:
        state = replace(ShoppingState.new("s", {}), category="shoes")
        diagnostics = {
            "feature": diagnostic("feature", 0.0, 0.0, 0.0),
            "style": diagnostic("style", 0.0, 0.0, 0.0),
        }

        self.assertEqual(choose_clarification(state, 3, diagnostics)[1], "other")

    def test_question_is_valid_and_not_repeated(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            asked_attributes=("feature",),
        )
        diagnostics = {
            "feature": diagnostic("feature", 0.9, 0.9, 0.9),
            "color": diagnostic("color", 0.9, 0.8, 0.6),
        }

        message, attribute = choose_clarification(state, 2, diagnostics)

        self.assertEqual(attribute, "color")
        self.assertIn("color", message.lower())
        self.assertIn(attribute, ALLOWED_PREFERENCE_ATTRIBUTES)

    def test_active_preference_is_not_asked_again(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            preferences={"feature": ("waterproof",), "color": ("black",)},
        )
        diagnostics = {
            "feature": diagnostic("feature", 0.9, 0.9, 0.9),
            "color": diagnostic("color", 0.9, 0.9, 0.9),
            "size": diagnostic("size", 0.9, 0.9, 0.9),
        }

        self.assertEqual(choose_clarification(state, 1, diagnostics)[1], "size")

    def test_no_preference_attribute_is_skipped(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            no_preference_attributes=frozenset({"feature"}),
        )
        diagnostics = {
            "feature": diagnostic("feature", 0.9, 0.9, 0.9),
            "color": diagnostic("color", 0.9, 0.8, 0.6),
        }

        self.assertEqual(choose_clarification(state, 1, diagnostics)[1], "color")

    def test_turn_ten_returns_no_question(self) -> None:
        self.assertEqual(
            choose_clarification(ShoppingState.new("s", {}), 10, broad_diagnostics()),
            ("Here are the closest matches I found.", None),
        )


class ReAskPolicyTest(unittest.TestCase):
    def _diag(self, attribute: str) -> AttributeDiagnostic:
        return AttributeDiagnostic(
            attribute=attribute, coverage=0.8, disagreement=0.6, relevance=0.8
        )

    def test_correction_only_attribute_can_be_asked(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material", values=("cotton",), source_kind="correction"
                ),
            ),
        )
        message, attribute = choose_clarification(
            state, 4, {"material": self._diag("material")}
        )
        self.assertEqual(attribute, "material")

    def test_clarified_attribute_is_not_reasked(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            asked_attributes=("material",),
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material",
                    values=("cotton",),
                    source_kind="clarification",
                ),
            ),
        )
        message, attribute = choose_clarification(
            state, 4, {"material": self._diag("material")}
        )
        self.assertNotEqual(attribute, "material")

    def test_unsolicited_attribute_is_not_reasked(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            preferences={"material": ("cotton",)},
            preference_evidence=(
                PreferenceEvidence(
                    attribute="material", values=("cotton",), source_kind="unsolicited"
                ),
            ),
        )
        message, attribute = choose_clarification(
            state, 4, {"material": self._diag("material")}
        )
        self.assertNotEqual(attribute, "material")

    def test_other_repeats_until_no_preference(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="sweatshirt",
            asked_attributes=("other",),
        )
        message, attribute = choose_clarification(state, 4, {})
        self.assertEqual(attribute, "other")
        exhausted = replace(state, no_preference_attributes=frozenset({"other"}))
        message, attribute = choose_clarification(exhausted, 4, {})
        self.assertIsNone(attribute)

    def test_other_is_asked_immediately_after_preference_override(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="watches",
            preferences={"feature": ("water resistant",)},
            last_update_type="replace_preferences",
            turn=2,
        )

        message, attribute = choose_clarification(
            state,
            3,
            broad_diagnostics(),
        )

        self.assertEqual(attribute, "other")

    def test_initial_product_classification_is_not_treated_as_override(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            last_update_type="product_change",
            turn=0,
        )

        message, attribute = choose_clarification(
            state,
            1,
            broad_diagnostics(),
        )

        self.assertEqual(attribute, "category")

    def test_other_is_asked_after_two_consecutive_no_preference_answers(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            consecutive_no_preference_turns=2,
        )

        message, attribute = choose_clarification(
            state,
            4,
            broad_diagnostics(),
        )

        self.assertEqual(attribute, "other")

    def test_one_no_preference_answer_does_not_force_other(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            consecutive_no_preference_turns=1,
        )

        message, attribute = choose_clarification(
            state,
            3,
            {"material": self._diag("material")},
        )

        self.assertEqual(attribute, "material")


if __name__ == "__main__":
    unittest.main()


class PromoteOtherTest(unittest.TestCase):
    """Env gating over the structured early-`other` triggers.

    Both triggers default on; these cover the switches and the boundaries.
    """

    def setUp(self):
        self._clear()
        self.addCleanup(self._clear)

    def _clear(self):
        for name in (
            "TECHJAM_OTHER_AFTER_OVERRIDE",
            "TECHJAM_OTHER_AFTER_NO_PREFERENCE",
        ):
            os.environ.pop(name, None)

    def _state(self, **kwargs):
        return replace(ShoppingState.new("s", {}), category="jeans", **kwargs)

    def test_override_trigger_on_by_default(self):
        state = self._state(last_update_type="replace_preferences", turn=2)
        _, attribute = choose_clarification(state, 3, broad_diagnostics())
        self.assertEqual(attribute, "other")

    def test_override_trigger_can_be_disabled(self):
        os.environ["TECHJAM_OTHER_AFTER_OVERRIDE"] = "false"
        state = self._state(last_update_type="replace_preferences", turn=2)
        _, attribute = choose_clarification(state, 3, broad_diagnostics())
        self.assertNotEqual(attribute, "other")

    def test_ordinary_update_does_not_trigger(self):
        state = self._state(last_update_type="merge", turn=2)
        _, attribute = choose_clarification(state, 3, broad_diagnostics())
        self.assertNotEqual(attribute, "other")

    def test_no_preference_threshold_boundary(self):
        below = self._state(consecutive_no_preference_turns=1)
        _, attribute = choose_clarification(below, 3, broad_diagnostics())
        self.assertNotEqual(attribute, "other")

        at = self._state(consecutive_no_preference_turns=2)
        _, attribute = choose_clarification(at, 3, broad_diagnostics())
        self.assertEqual(attribute, "other")

    def test_no_preference_threshold_is_configurable(self):
        os.environ["TECHJAM_OTHER_AFTER_NO_PREFERENCE"] = "3"
        state = self._state(consecutive_no_preference_turns=2)
        _, attribute = choose_clarification(state, 3, broad_diagnostics())
        self.assertNotEqual(attribute, "other")

    def test_zero_threshold_disables_the_counter_trigger(self):
        os.environ["TECHJAM_OTHER_AFTER_NO_PREFERENCE"] = "0"
        state = self._state(consecutive_no_preference_turns=5)
        _, attribute = choose_clarification(state, 3, broad_diagnostics())
        self.assertNotEqual(attribute, "other")

    def test_not_asked_again_once_declined(self):
        state = self._state(
            last_update_type="replace_preferences",
            turn=2,
            no_preference_attributes=frozenset({"other"}),
        )
        _, attribute = choose_clarification(state, 3, broad_diagnostics())
        self.assertNotEqual(attribute, "other")

    def test_category_question_still_wins(self):
        state = replace(
            ShoppingState.new("s", {}),
            category=None,
            last_update_type="product_change",
            turn=2,
        )
        _, attribute = choose_clarification(state, 2, broad_diagnostics())
        self.assertEqual(attribute, "category")

    def test_turn_cap_still_applies(self):
        state = self._state(last_update_type="replace_preferences", turn=2)
        _, attribute = choose_clarification(state, 10, broad_diagnostics())
        self.assertIsNone(attribute)
