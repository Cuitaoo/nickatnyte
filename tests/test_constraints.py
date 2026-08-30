import os
import unittest
from dataclasses import dataclass

from starter.constraints import (
    HARD,
    SOFT,
    Constraint,
    classify_constraints,
    staged_filter,
    violates,
)


@dataclass(frozen=True)
class _Evidence:
    attribute: str
    source_kind: str


@dataclass(frozen=True)
class _Cand:
    product_id: str


class _State:
    def __init__(self, preferences=None, evidence=()):
        self.preferences = preferences or {}
        self.preference_evidence = evidence


def _matches(attribute, value, product):
    return value.lower() in str(product.get(attribute, "")).lower()


def _has_meta(attribute, product):
    return bool(str(product.get(attribute, "")).strip())


class ClassifyTest(unittest.TestCase):
    def test_volunteered_evidence_is_hard(self):
        state = _State(
            {"material": ("cotton",)}, (_Evidence("material", "unsolicited"),)
        )
        self.assertEqual(classify_constraints(state)[0].strength, HARD)

    def test_answering_our_question_is_soft(self):
        state = _State(
            {"color": ("black",)}, (_Evidence("color", "clarification"),)
        )
        self.assertEqual(classify_constraints(state)[0].strength, SOFT)

    def test_correction_is_hard(self):
        """Insisting after we got it wrong is a requirement, not a hint."""
        state = _State(
            {"material": ("wool",)}, (_Evidence("material", "correction"),)
        )
        self.assertEqual(classify_constraints(state)[0].strength, HARD)

    def test_missing_evidence_defaults_to_soft(self):
        state = _State({"style": ("casual",)}, ())
        self.assertEqual(classify_constraints(state)[0].strength, SOFT)

    def test_empty_values_are_skipped(self):
        self.assertEqual(classify_constraints(_State({"color": ()})), ())

    def test_confidence_orders_reliable_above_freetext(self):
        state = _State(
            {"budget": ("under $50",), "feature": ("rubber sole",)},
            (_Evidence("budget", "unsolicited"), _Evidence("feature", "unsolicited")),
        )
        by = {c.attribute: c.confidence for c in classify_constraints(state)}
        self.assertGreater(by["budget"], by["feature"])


class ViolatesTest(unittest.TestCase):
    def test_missing_metadata_never_violates(self):
        """Absence of evidence is not evidence of violation."""
        c = Constraint("material", ("cotton",), HARD, 0.8)
        self.assertFalse(violates(c, {"material": ""}, _matches, _has_meta))
        self.assertFalse(violates(c, {}, _matches, _has_meta))

    def test_present_and_contradicting_violates(self):
        c = Constraint("material", ("cotton",), HARD, 0.8)
        self.assertTrue(violates(c, {"material": "100% polyester"}, _matches, _has_meta))

    def test_present_and_matching_does_not(self):
        c = Constraint("material", ("cotton",), HARD, 0.8)
        self.assertFalse(violates(c, {"material": "100% cotton"}, _matches, _has_meta))

    def test_any_value_satisfies(self):
        c = Constraint("color", ("black", "navy"), HARD, 0.75)
        self.assertFalse(violates(c, {"color": "navy blue"}, _matches, _has_meta))


class StagedFilterTest(unittest.TestCase):
    def setUp(self):
        self.meta = {}
        for i in range(100):
            self.meta[f"c{i}"] = {
                "material": "cotton" if i < 50 else "polyester",
                "color": "black" if i < 10 else "red",
            }
        self.cands = tuple(_Cand(f"c{i}") for i in range(100))

    def _filter(self, constraints, floor):
        return staged_filter(self.cands, constraints, self.meta, _matches, _has_meta, floor)

    def test_filters_to_satisfying_products(self):
        c = (Constraint("material", ("cotton",), HARD, 0.8),)
        kept, relaxed = self._filter(c, 40)
        self.assertEqual(len(kept), 50)
        self.assertEqual(relaxed, ())

    def test_soft_constraints_never_filter(self):
        c = (Constraint("material", ("cotton",), SOFT, 0.8),)
        kept, relaxed = self._filter(c, 40)
        self.assertEqual(len(kept), 100)
        self.assertEqual(relaxed, ())

    def test_relaxes_least_reliable_first(self):
        """color would leave 10, below the floor, so color is surrendered."""
        c = (
            Constraint("material", ("cotton",), HARD, 0.80),
            Constraint("color", ("black",), HARD, 0.75),
        )
        kept, relaxed = self._filter(c, 40)
        self.assertEqual(relaxed, ("color",))
        self.assertEqual(len(kept), 50)

    def test_falls_back_to_unfiltered_rather_than_starving(self):
        c = (Constraint("color", ("black",), HARD, 0.75),)
        kept, relaxed = self._filter(c, 40)
        self.assertEqual(len(kept), 100)
        self.assertEqual(relaxed, ("color",))

    def test_never_returns_fewer_than_the_floor_unless_unfiltered(self):
        for floor in (5, 10, 40, 80):
            kept, _ = self._filter((Constraint("color", ("black",), HARD, 0.75),), floor)
            self.assertTrue(len(kept) >= floor or len(kept) == 100)

    def test_no_constraints_is_a_no_op(self):
        kept, relaxed = self._filter((), 40)
        self.assertEqual(len(kept), 100)
        self.assertEqual(relaxed, ())

    def test_empty_candidates(self):
        kept, relaxed = staged_filter((), (Constraint("m", ("x",), HARD, 0.5),), {}, _matches, _has_meta, 10)
        self.assertEqual(kept, ())


if __name__ == "__main__":
    unittest.main()
