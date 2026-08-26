from __future__ import annotations

import unittest
from dataclasses import replace

from starter.questions import choose_clarification
from starter.state import ALLOWED_PREFERENCE_ATTRIBUTES, ShoppingState


class ClarificationTest(unittest.TestCase):
    def test_missing_category_is_asked_first(self) -> None:
        message, attribute = choose_clarification(ShoppingState.new("s", {}), turn=1)

        self.assertEqual(attribute, "category")
        self.assertIn("kind of product", message.lower())

    def test_question_is_valid_and_not_repeated(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            asked_attributes=("material",),
        )

        message, attribute = choose_clarification(state, turn=2)

        self.assertEqual(attribute, "color")
        self.assertIn("color", message.lower())
        self.assertIn(attribute, ALLOWED_PREFERENCE_ATTRIBUTES)

    def test_active_preference_is_not_asked_again(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            preferences={"material": ("leather",), "color": ("black",)},
        )

        self.assertEqual(choose_clarification(state, turn=1)[1], "size")

    def test_no_preference_attribute_is_skipped(self) -> None:
        state = replace(
            ShoppingState.new("s", {}),
            category="shoes",
            no_preference_attributes=frozenset({"material"}),
        )

        self.assertEqual(choose_clarification(state, turn=1)[1], "color")

    def test_turn_ten_returns_no_question(self) -> None:
        self.assertEqual(
            choose_clarification(ShoppingState.new("s", {}), turn=10),
            ("Here are the closest matches I found.", None),
        )


if __name__ == "__main__":
    unittest.main()
