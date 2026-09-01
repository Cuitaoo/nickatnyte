import os
import unittest
from dataclasses import dataclass

from starter.confidence import (
    ASK_ONLY,
    BROADEN_RETRIEVAL,
    RECOMMEND_NOW,
    RECOMMEND_WHILE_ASKING,
    ConfidenceSignals,
    assess,
    choose_strategy,
    withholds_recommendations,
)


@dataclass(frozen=True)
class _Cand:
    product_id: str
    score: float
    route_ranks: tuple = ()
    matched_attributes: frozenset = frozenset()


class _State:
    def __init__(self, preferences=None):
        self.preferences = preferences or {}


class AssessTest(unittest.TestCase):
    def test_margin_is_the_top_two_gap(self):
        s = assess((_Cand("a", 2.5), _Cand("b", 2.0)), _State())
        self.assertAlmostEqual(s.margin, 0.5)

    def test_route_agreement_counts_the_leader_routes(self):
        s = assess((_Cand("a", 2.5, (("category", 1), ("attribute", 2))), _Cand("b", 1.0)), _State())
        self.assertEqual(s.route_agreement, 2)

    def test_fallback_anywhere_in_the_head_is_flagged(self):
        s = assess((_Cand("a", 2.5, (("category", 1),)), _Cand("b", 1.0, (("fallback", 1),))), _State())
        self.assertTrue(s.has_fallback)

    def test_constraint_coverage(self):
        state = _State({"material": ("cotton",), "color": ("black",)})
        top = _Cand("a", 2.5, (("category", 1),), frozenset({"material"}))
        self.assertAlmostEqual(assess((top, _Cand("b", 1.0)), state).constraint_coverage, 0.5)

    def test_category_is_not_counted_as_a_constraint(self):
        state = _State({"category": ("jeans",)})
        s = assess((_Cand("a", 2.5), _Cand("b", 1.0)), state)
        self.assertEqual(s.constraint_coverage, 0.0)

    def test_empty_candidates(self):
        s = assess((), _State())
        self.assertEqual(s.pool_size, 0)
        self.assertEqual(s.margin, 0.0)


class ConfidenceThresholdTest(unittest.TestCase):
    def tearDown(self):
        for n in ("TECHJAM_CONFIDENCE_MIN_MARGIN", "TECHJAM_CONFIDENCE_MIN_ROUTES"):
            os.environ.pop(n, None)

    def test_wide_margin_with_route_agreement_is_confident(self):
        self.assertTrue(ConfidenceSignals(margin=0.5, route_agreement=2, pool_size=50).is_confident)

    def test_narrow_margin_is_not(self):
        """Ambiguous sessions measured margins around 0.003."""
        self.assertFalse(ConfidenceSignals(margin=0.003, route_agreement=3, pool_size=50).is_confident)

    def test_single_route_is_not_confident(self):
        self.assertFalse(ConfidenceSignals(margin=0.5, route_agreement=1, pool_size=50).is_confident)

    def test_fallback_is_never_confident(self):
        self.assertFalse(
            ConfidenceSignals(margin=0.9, route_agreement=3, pool_size=50, has_fallback=True).is_confident
        )

    def test_starved_pool(self):
        self.assertTrue(ConfidenceSignals(pool_size=2).is_starved)
        self.assertFalse(ConfidenceSignals(pool_size=40).is_starved)


class ChooseStrategyTest(unittest.TestCase):
    def test_confident_turn_overrides_the_deferral_rule(self):
        """The whole point: a separated ranking should not be withheld."""
        signals = ConfidenceSignals(margin=0.5, route_agreement=3, pool_size=50)
        got = choose_strategy(signals, "color", rule_would_defer=True)
        self.assertEqual(got, RECOMMEND_WHILE_ASKING)
        self.assertFalse(withholds_recommendations(got))

    def test_unconfident_turn_still_defers(self):
        signals = ConfidenceSignals(margin=0.003, route_agreement=1, pool_size=50)
        got = choose_strategy(signals, "color", rule_would_defer=True)
        self.assertEqual(got, ASK_ONLY)
        self.assertTrue(withholds_recommendations(got))

    def test_never_withholds_what_the_rule_would_have_released(self):
        """The controller may only release, never withhold - MTTC cannot worsen."""
        for margin in (0.0, 0.003, 0.5):
            for routes in (1, 3):
                signals = ConfidenceSignals(margin=margin, route_agreement=routes, pool_size=50)
                got = choose_strategy(signals, "color", rule_would_defer=False)
                self.assertFalse(withholds_recommendations(got), f"{margin}/{routes}")

    def test_starved_pool_broadens(self):
        signals = ConfidenceSignals(pool_size=1, margin=0.9, route_agreement=3)
        self.assertEqual(choose_strategy(signals, "color", rule_would_defer=False), BROADEN_RETRIEVAL)

    def test_no_question_means_recommend(self):
        signals = ConfidenceSignals(margin=0.5, route_agreement=3, pool_size=50)
        self.assertEqual(choose_strategy(signals, None, rule_would_defer=False), RECOMMEND_NOW)


if __name__ == "__main__":
    unittest.main()
