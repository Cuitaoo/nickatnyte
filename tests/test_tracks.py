import os
import unittest
from dataclasses import dataclass

from starter.tracks import (
    BROWSING,
    BUYING,
    NEUTRAL,
    apply_track,
    leaf_category,
    resolve_track,
    select_diverse_recommendations,
)


@dataclass(frozen=True)
class _Weights:
    confirmed_attribute_boost: float = 1.0
    exact_phrase_boost: float = 2.0
    category_boost: float = 1.0
    route_relaxed: float = 1.0
    route_synonym: float = 1.0
    route_attribute: float = 2.0


@dataclass(frozen=True)
class _Candidate:
    product_id: str


class ResolveTrackTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("TECHJAM_DUAL_TRACK_BROWSING", None)
        self.addCleanup(os.environ.pop, "TECHJAM_DUAL_TRACK_BROWSING", None)

    def test_maps_intent_to_track(self):
        self.assertIs(resolve_track("buying"), BUYING)
        self.assertIs(resolve_track("unknown"), NEUTRAL)
        self.assertIs(resolve_track(""), NEUTRAL)

    def test_browsing_is_opt_in(self):
        """Browsing is the strongest scenario, so it is left alone by default."""
        self.assertIs(resolve_track("browsing"), NEUTRAL)
        os.environ["TECHJAM_DUAL_TRACK_BROWSING"] = "true"
        self.assertIs(resolve_track("browsing"), BROWSING)

    def test_override_turns_stay_neutral(self):
        """An override is classified as buying, but the context just changed."""
        self.assertIs(resolve_track("buying", is_override=True), NEUTRAL)
        os.environ["TECHJAM_DUAL_TRACK_BROWSING"] = "true"
        self.assertIs(resolve_track("browsing", is_override=True), NEUTRAL)


class ApplyTrackTest(unittest.TestCase):
    def test_neutral_returns_weights_untouched(self):
        weights = _Weights()
        self.assertIs(apply_track(weights, NEUTRAL), weights)

    def test_buying_tightens(self):
        scaled = apply_track(_Weights(), BUYING)
        self.assertGreater(scaled.confirmed_attribute_boost, 1.0)
        self.assertGreater(scaled.exact_phrase_boost, 2.0)
        self.assertLess(scaled.route_relaxed, 1.0)
        self.assertLess(scaled.route_synonym, 1.0)

    def test_browsing_widens(self):
        scaled = apply_track(_Weights(), BROWSING)
        self.assertLess(scaled.confirmed_attribute_boost, 1.0)
        self.assertGreater(scaled.route_relaxed, 1.0)
        self.assertGreater(scaled.route_synonym, 1.0)

    def test_does_not_mutate_the_original(self):
        weights = _Weights()
        apply_track(weights, BUYING)
        self.assertEqual(weights.confirmed_attribute_boost, 1.0)
        self.assertEqual(weights.route_relaxed, 1.0)

    def test_only_browsing_carries_a_category_cap(self):
        self.assertEqual(BUYING.category_cap, 0)
        self.assertEqual(NEUTRAL.category_cap, 0)
        self.assertGreater(BROWSING.category_cap, 0)


class LeafCategoryTest(unittest.TestCase):
    def test_takes_the_tail_of_the_flattened_path(self):
        meta = {"categories": "clothing, shoes & jewelry men clothing jeans"}
        self.assertEqual(leaf_category(meta), "clothing jeans")

    def test_falls_back_to_store(self):
        self.assertEqual(leaf_category({"categories": "", "store": "levis"}), "levis")

    def test_empty_metadata(self):
        self.assertEqual(leaf_category({}), "")


class DiversityTest(unittest.TestCase):
    def setUp(self):
        # Six jeans then two bags, so a cap must reach past the jeans run.
        self.metadata = {
            f"J{i}": {"categories": f"clothing, shoes & jewelry men clothing jeans"}
            for i in range(1, 7)
        }
        self.metadata.update(
            {
                "B1": {"categories": "clothing, shoes & jewelry handbags shoulder bags"},
                "B2": {"categories": "clothing, shoes & jewelry handbags shoulder bags"},
            }
        )
        self.candidates = tuple(
            _Candidate(p) for p in ["J1", "J2", "J3", "J4", "J5", "J6", "B1", "B2"]
        )

    def test_cap_zero_is_plain_truncation(self):
        got = select_diverse_recommendations(self.candidates, 4, self.metadata, 0)
        self.assertEqual(got, ("J1", "J2", "J3", "J4"))

    def test_no_metadata_is_plain_truncation(self):
        got = select_diverse_recommendations(self.candidates, 4, None, 3)
        self.assertEqual(got, ("J1", "J2", "J3", "J4"))

    def test_cap_spreads_across_categories(self):
        got = select_diverse_recommendations(self.candidates, 4, self.metadata, 2)
        self.assertEqual(got, ("J1", "J2", "B1", "B2"))

    def test_capped_items_backfill_rather_than_shortening(self):
        """Variety must never cost the shopper results."""
        got = select_diverse_recommendations(self.candidates, 6, self.metadata, 2)
        self.assertEqual(len(got), 6)
        self.assertEqual(got[:4], ("J1", "J2", "B1", "B2"))
        self.assertEqual(got[4:], ("J3", "J4"))

    def test_backfill_preserves_original_order(self):
        got = select_diverse_recommendations(self.candidates, 8, self.metadata, 1)
        self.assertEqual(got, ("J1", "B1", "J2", "J3", "J4", "J5", "J6", "B2"))

    def test_relevance_order_held_within_a_category(self):
        got = select_diverse_recommendations(self.candidates, 8, self.metadata, 3)
        jeans = [p for p in got if p.startswith("J")]
        self.assertEqual(jeans, ["J1", "J2", "J3", "J4", "J5", "J6"])

    def test_empty_and_zero_k(self):
        self.assertEqual(select_diverse_recommendations((), 5, self.metadata, 2), ())
        self.assertEqual(
            select_diverse_recommendations(self.candidates, 0, self.metadata, 2), ()
        )

    def test_legacy_two_argument_call_still_works(self):
        got = select_diverse_recommendations(self.candidates, 3)
        self.assertEqual(got, ("J1", "J2", "J3"))


if __name__ == "__main__":
    unittest.main()
